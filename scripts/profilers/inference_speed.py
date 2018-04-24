"""Detection model profiler.
"""

import argparse
import os
import time
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

import avod
import avod.builders.config_builder_util as config_builder
from avod.builders.dataset_builder import DatasetBuilder
from avod.core.models.avod_model import AvodModel
from avod.core.models.avod_ssd_model import AvodSSDModel
from avod.core.models.rpn_model import RpnModel

# Small hack to run even with @profile tags
import builtins
builtins.profile = lambda x: x


def set_up_model(pipeline_config,
                 data_split):

    model_config, train_config, _, dataset_config = \
        config_builder.get_configs_from_pipeline_file(
            pipeline_config, is_training=False)

    dataset_config = config_builder.proto_to_obj(dataset_config)

    train_val_test = data_split
    # Always run in test mode
    dataset_config.data_split = 'test'
    dataset_config.data_split_dir = 'testing'
    dataset_config.has_labels = False
    dataset_config.aug_list = []

    dataset = DatasetBuilder.build_kitti_dataset(dataset_config,
                                                 use_defaults=False)

    model_name = model_config.model_name
    if model_name == 'rpn_model':
        model = RpnModel(model_config,
                         train_val_test=train_val_test,
                         dataset=dataset)
    elif model_name == 'avod_model':
        model = AvodModel(model_config,
                          train_val_test=train_val_test,
                          dataset=dataset)
    elif model_name == 'avod_ssd_model':
        model = AvodSSDModel(model_config,
                            train_val_test=train_val_test,
                            dataset=dataset)
    else:
        raise ValueError('Invalid model_name')

    return model


@profile
def main():

    parser = argparse.ArgumentParser()

    default_pipeline_config_path = avod.root_dir() + \
        '/configs/avod_ssd_cars_example.config'

    parser.add_argument('--pipeline_config',
                        type=str,
                        dest='pipeline_config_path',
                        default=default_pipeline_config_path,
                        help='Path to the pipeline config')

    parser.add_argument('--data_split',
                        type=str,
                        dest='data_split',
                        default='test',
                        help='Data split for evaluation')

    parser.add_argument('--device',
                        type=str,
                        dest='device',
                        default='0',
                        help='CUDA device id')

    parser.add_argument('--output',
                        type=str,
                        dest='output',
                        default='model_timeline.json',
                        help='CUDA device id')

    args = parser.parse_args()

    model = set_up_model(args.pipeline_config_path,
                         args.data_split)

    prediction_dict = model.build()

    # Set CUDA device id
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    # Set session config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Set run options
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    # Create session
    sess = tf.Session(config=config)

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    all_feed_dict_times = []
    all_inference_times = []

    # Run sess a few times since it is usually slow at the start
    for i in range(5):
        sys.stdout.write('\r{}'.format(i))
        feed_dict = model.create_feed_dict()
        sess.run(prediction_dict, feed_dict)

    for i in range(95):
        sys.stdout.write('\r{}'.format(i + 5))
        feed_dict_start_time = time.time()
        feed_dict = model.create_feed_dict()
        all_feed_dict_times.append(time.time() - feed_dict_start_time)

        inference_start_time = time.time()
        sess.run(prediction_dict, feed_dict)
        all_inference_times.append(time.time() - inference_start_time)

    print('\n')
    print('feed_dict mean', np.mean(all_feed_dict_times))
    print('feed_dict median', np.median(all_feed_dict_times))
    print('feed_dict min', np.min(all_feed_dict_times))
    print('feed_dict max', np.max(all_feed_dict_times))

    print('inference mean', np.mean(all_inference_times))
    print('inference median', np.median(all_inference_times))
    print('inference min', np.min(all_inference_times))
    print('inference max', np.max(all_inference_times))

    # Run once with full timing
    sess.run(prediction_dict, feed_dict,
             options=run_options, run_metadata=run_metadata)

    inference_start_time = time.time()
    sess.run(prediction_dict, feed_dict)
    print('Time:', time.time() - inference_start_time)

    tf_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace_fmt = tf_timeline.generate_chrome_trace_format()

    outputfile = args.output
    with open(outputfile, 'w') as f:
        f.write(chrome_trace_fmt)

    print('Done')


if __name__ == '__main__':
    main()
