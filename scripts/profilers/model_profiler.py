"""Model profiler to monitor network parameters and memory consumption"""

import argparse
import os
import sys
import tensorflow as tf

from tensorflow.python.framework import ops

import avod
import avod.builders.config_builder_util as config_builder

from avod.builders.dataset_builder import DatasetBuilder
from avod.builders import optimizer_builder
from avod.core import trainer_utils
from avod.core.models.avod_model import AvodModel
from avod.core.models.avod_ssd_model import AvodSSDModel
from avod.core.models.rpn_model import RpnModel

slim = tf.contrib.slim


def set_up_model_train_mode(pipeline_config_path, data_split):
    """Returns the model and its train_op."""

    model_config, train_config, _,  dataset_config = \
        config_builder.get_configs_from_pipeline_file(
            pipeline_config_path, is_training=True)

    dataset = DatasetBuilder.build_kitti_dataset(dataset_config,
                                                 use_defaults=False)

    model_name = model_config.model_name
    if model_name == 'rpn_model':
        model = RpnModel(model_config,
                         train_val_test=data_split,
                         dataset=dataset)
    elif model_name == 'avod_model':
        model = AvodModel(model_config,
                          train_val_test=data_split,
                          dataset=dataset)
    elif model_name == 'avod_ssd_model':
        model = AvodSSDModel(model_config,
                             train_val_test=data_split,
                             dataset=dataset)
    else:
        raise ValueError('Invalid model_name')

    prediction_dict = model.build()
    losses_dict, total_loss = model.loss(prediction_dict)

    # These parameters are required to set up the optimizer
    global_summaries = set([])
    global_step_tensor = tf.Variable(0, trainable=False)
    training_optimizer = optimizer_builder.build(
        train_config.optimizer, global_summaries, global_step_tensor)

    # Set up the train op
    train_op = slim.learning.create_train_op(
        total_loss,
        training_optimizer)

    return model, train_op


def set_up_model_test_mode(pipeline_config_path, data_split):
    """Returns the model and its config in test mode."""

    model_config, _, _,  dataset_config = \
        config_builder.get_configs_from_pipeline_file(
            pipeline_config_path, is_training=False)

    dataset = DatasetBuilder.build_kitti_dataset(dataset_config,
                                                 use_defaults=False)

    # Overwrite the defaults
    dataset_config = config_builder.proto_to_obj(dataset_config)

    # Use the validation set
    dataset_config.data_split = data_split
    dataset_config.data_split_dir = 'training'
    if data_split == 'test':
        dataset_config.data_split_dir = 'testing'

    # Remove augmentation when in test mode
    dataset_config.aug_list = []

    # Build the dataset object
    dataset = DatasetBuilder.build_kitti_dataset(dataset_config,
                                                 use_defaults=False)

    model_name = model_config.model_name
    if model_name == 'rpn_model':
        model = RpnModel(model_config,
                         train_val_test='test',
                         dataset=dataset)
    elif model_name == 'avod_model':
        model = AvodModel(model_config,
                          train_val_test='test',
                          dataset=dataset)
    elif model_name == 'avod_ssd_model':
        model = AvodSSDModel(model_config,
                             train_val_test='test',
                             dataset=dataset)
    else:
        raise ValueError('Invalid model_name')

    return model, model_config


def run_profiler(pipeline_config_path, run_mode,
                 data_split, ckpt_index):

    avod_top_dir = avod.top_dir()
    # Timeline results logfile
    file_name = avod_top_dir + '/scripts/profilers/tf_profiler/' + \
        'tf_timeline_output.json'

    with tf.Session() as sess:

        if run_mode == 'train':
            # In train mode, data_split should not be 'test' as the test
            # split does not have gt.
            if data_split == 'test':
                raise ValueError('Data split can only be train or val'
                                 'in train mode.')
            model, train_op = set_up_model_train_mode(
                pipeline_config_path, data_split)
            init = tf.global_variables_initializer()
            sess.run(init)
        elif run_mode == 'test':
            model, model_config = set_up_model_test_mode(
                pipeline_config_path, data_split)
            paths_config = model_config.paths_config

            checkpoint_dir = paths_config.checkpoint_dir
            prediction_dict = model.build()

            # Load the weights
            saver = tf.train.Saver()
            trainer_utils.load_checkpoints(checkpoint_dir,
                                           saver)
            if not saver.last_checkpoints:
                raise ValueError('Need existing checkpoints to run'
                                 'in test_mode')
            checkpoint_to_restore = saver.last_checkpoints[ckpt_index]
            saver.restore(sess, checkpoint_to_restore)

        else:
            raise ValueError('Invalid run_mode {}'.format(run_mode))

        feed_dict = model.create_feed_dict()

        ############################################
        # Parameters and Shapes
        ############################################

        graph = tf.get_default_graph()
        # Print trainable variable parameter statistics to stdout.
        ProfileOptionBuilder = tf.profiler.ProfileOptionBuilder

        # Gives the total number of trainable parameters
        param_stats = tf.profiler.profile(
            graph,
            options=ProfileOptionBuilder.trainable_variables_parameter())

        # Gives the FLOPS for the ops
        tf.profiler.profile(
            graph,
            options=tf.profiler.ProfileOptionBuilder.float_operation())

        run_metadata = tf.RunMetadata()
        if run_mode == 'train':
            sess.run([train_op],
                     options=tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE),
                     run_metadata=run_metadata,
                     feed_dict=feed_dict)
        else:
            # Run in test mode
            sess.run(prediction_dict,
                     options=tf.RunOptions(
                            trace_level=tf.RunOptions.FULL_TRACE),
                     run_metadata=run_metadata,
                     feed_dict=feed_dict)

        # The profiler gives us rounded FLOP counts
        # So instead query it directly and count the total
        op_missing_shape = 0
        # op_missing_shape_names = []
        total_flops = 0
        for op in graph.get_operations():
            try:
                stats = ops.get_stats_for_node_def(
                    graph, op.node_def, 'flops')
                if stats.value:
                    total_flops += stats.value
            except ValueError:
                op_missing_shape += 1
                # op_missing_shape_names.append(op.name)
        print('=============================================================')
        print('Number of ops with missing shape: ', op_missing_shape)
        print('=============================================================')

        ############################################
        # Log Time and Memory
        ############################################
        # Log the analysis to file
        # 'code' view organizes profile using Python call stack
        opts = ProfileOptionBuilder(
            ProfileOptionBuilder.time_and_memory()).with_timeline_output(
            file_name).build()

        tf.profiler.profile(
            graph,
            run_meta=run_metadata,
            cmd='code',
            options=opts)

        ############################################
        # Show Time and Memory on the console
        ############################################
        tf.profiler.profile(
            graph,
            run_meta=run_metadata,
            cmd='op',
            options=tf.profiler.ProfileOptionBuilder.time_and_memory())

        # print the total number of parameters
        print('Total params: %d' % param_stats.total_parameters)
        print('Total FLOPs: ', total_flops)
        print('=============================================================')


def main(_):
    parser = argparse.ArgumentParser()

    # Example usage
    # --pipeline_config=avod/configs/avod_exp_example.config
    # --run_mode='test'
    # --data_split='val'
    # Optional args:
    # --ckpt_index=0
    # --device=0

    parser.add_argument('--pipeline_config',
                        type=str,
                        dest='pipeline_config_path',
                        required=True,
                        help='Path to the pipeline config')

    parser.add_argument('--run_mode',
                        type=str,
                        dest='run_mode',
                        required=True,
                        help='Run mode must be specified as a str->\
                        train or test')

    parser.add_argument('--data_split',
                        type=str,
                        dest='data_split',
                        required=True,
                        help='Data split must be specified e.g. val or test')

    # By default, grabs the first available checkpoint in 'test' mode
    # This is not used in 'train' mode
    parser.add_argument(
        '--ckpt_index',
        type=int,
        dest='ckpt_index',
        default='0',
        help='Checkpoint index must be an integer like 0 10 etc')

    parser.add_argument('--device',
                        type=str,
                        dest='device',
                        default='0',
                        help='CUDA device id')

    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    run_profiler(args.pipeline_config_path,
                 args.run_mode,
                 args.data_split,
                 args.ckpt_index)


if __name__ == '__main__':
    tf.app.run()
