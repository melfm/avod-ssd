import numpy as np
import tensorflow as tf

from avod.builders import feature_extractor_builder
from avod.builders import avod_fc_layers_builder
from avod.builders import avod_loss_builder
from avod.core import anchor_encoder
from avod.core import anchor_filter
from avod.core import anchor_projector
from avod.core import constants
from avod.core import model
from avod.core.anchor_generators import grid_anchor_3d_generator
from avod.datasets.kitti import kitti_aug

from avod.core import box_list
from avod.core import box_list_ops
from avod.core import box_3d_encoder
from avod.core import box_8c_encoder
from avod.core import box_4c_encoder
from avod.core import orientation_encoder


class AvodSSDModel(model.DetectionModel):
    ##############################
    # Keys for Placeholders
    ##############################
    PL_BEV_INPUT = 'bev_input_pl'
    PL_IMG_INPUT = 'img_input_pl'
    PL_ANCHORS = 'anchors_pl'

    PL_BEV_ANCHORS = 'bev_anchors_pl'
    PL_BEV_ANCHORS_NORM = 'bev_anchors_norm_pl'
    PL_IMG_ANCHORS = 'img_anchors_pl'
    PL_IMG_ANCHORS_NORM = 'img_anchors_norm_pl'
    PL_LABEL_ANCHORS = 'label_anchors_pl'
    PL_LABEL_BOXES_3D = 'label_boxes_3d_pl'
    PL_LABEL_CLASSES = 'label_classes_pl'

    PL_ANCHOR_IOUS = 'anchor_ious_pl'
    PL_ANCHOR_OFFSETS = 'anchor_offsets_pl'
    PL_ANCHOR_CLASSES = 'anchor_classes_pl'

    # Mini batch (mb) predictions
    PRED_MB_CLASSIFICATION_LOGITS = 'avod_mb_classification_logits'
    PRED_MB_CLASSIFICATION_SOFTMAX = 'avod_mb_classification_softmax'
    PRED_MB_OFFSETS = 'avod_mb_offsets'
    PRED_MB_ANGLE_VECTORS = 'avod_mb_angle_vectors'

    # Top predictions after BEV NMS
    PRED_TOP_CLASSIFICATION_LOGITS = 'avod_top_classification_logits'
    PRED_TOP_CLASSIFICATION_SOFTMAX = 'avod_top_classification_softmax'

    PRED_TOP_PREDICTION_ANCHORS = 'avod_top_prediction_anchors'
    PRED_TOP_PREDICTION_BOXES_3D = 'avod_top_prediction_boxes_3d'
    PRED_TOP_ORIENTATIONS = 'avod_top_orientations'

    # Other box representations
    PRED_TOP_BOXES_8C = 'avod_top_regressed_boxes_8c'
    PRED_TOP_BOXES_4C = 'avod_top_prediction_boxes_4c'

    # Sample info, including keys for projection to image space
    # (e.g. camera matrix, image index, etc.)
    PL_CALIB_P2 = 'frame_calib_p2'
    PL_IMG_IDX = 'current_img_idx'
    PL_GROUND_PLANE = 'ground_plane'

    ##############################
    # Keys for Predictions
    ##############################
    PRED_MB_CLASSIFICATIONS_GT = 'avod_mb_classifications_gt'
    PRED_MB_OFFSETS_GT = 'avod_mb_offsets_gt'
    PRED_MB_ORIENTATIONS_GT = 'avod_mb_orientations_gt'

    ##############################
    # Keys for Loss
    ##############################
    LOSS_FINAL_CLASSIFICATION = 'avod_classification_loss'
    LOSS_FINAL_REGRESSION = 'avod_regression_loss'

    LOSS_FINAL_ORIENTATION = 'avod_orientation_loss'
    LOSS_FINAL_LOCALIZATION = 'avod_localization_loss'

    def __init__(self, model_config, train_val_test, dataset):
        """
        Args:
            model_config: configuration for the model
            train_val_test: "train", "val", or "test"
            dataset: the dataset that will provide samples and ground truth
        """

        # Sets model configs (_config)
        super(AvodSSDModel, self).__init__(model_config)

        if train_val_test not in ["train", "val", "test"]:
            raise ValueError('Invalid train_val_test value,'
                             'should be one of ["train", "val", "test"]')
        self._train_val_test = train_val_test

        self._is_training = (self._train_val_test == 'train')

        # Input config
        input_config = self._config.input_config
        self._bev_pixel_size = np.asarray([input_config.bev_dims_h,
                                           input_config.bev_dims_w])
        self._bev_depth = input_config.bev_depth

        self._img_pixel_size = np.asarray([input_config.img_dims_h,
                                           input_config.img_dims_w])
        self._img_depth = input_config.img_depth

        # AVOD config
        avod_config = self._config.model_type.avod_ssd.avod_config
        self._proposal_roi_crop_size = \
            [avod_config.avod_proposal_roi_crop_size] * 2
        self._positive_selection = avod_config.avod_positive_selection
        self._nms_size = avod_config.avod_nms_size
        self._nms_iou_threshold = avod_config.avod_nms_iou_thresh
        self._path_drop_probabilities = self._config.path_drop_probabilities
        self._box_rep = avod_config.avod_box_representation

        # Feature Extractor Nets
        self._bev_feature_extractor = \
            feature_extractor_builder.get_extractor(
                self._config.layers_config.bev_feature_extractor)
        self._img_feature_extractor = \
            feature_extractor_builder.get_extractor(
                self._config.layers_config.img_feature_extractor)

        # Network input placeholders
        self.placeholders = dict()

        # Inputs to network placeholders
        self._placeholder_inputs = dict()

        # Information about the current sample
        self.sample_info = dict()

        # Dataset
        self.dataset = dataset
        self.dataset.train_val_test = self._train_val_test
        self._area_extents = self.dataset.kitti_utils.area_extents
        self._bev_extents = self.dataset.kitti_utils.bev_extents
        self._cluster_sizes, _ = self.dataset.get_cluster_info()
        self._anchor_strides = self.dataset.kitti_utils.anchor_strides
        self._anchor_generator = \
            grid_anchor_3d_generator.GridAnchor3dGenerator()

        self._path_drop_probabilities = self._config.path_drop_probabilities
        self._train_on_all_samples = self._config.train_on_all_samples
        self._eval_all_samples = self._config.eval_all_samples
        # Overwrite the dataset's variable with the config
        self.dataset.train_on_all_samples = self._train_on_all_samples

        # Dataset config
        self._num_final_classes = self.dataset.num_classes + 1

        if self._train_val_test in ["val", "test"]:
            # Disable path-drop, this should already be disabled inside the
            # evaluator, but just in case.
            self._path_drop_probabilities[0] = 1.0
            self._path_drop_probabilities[1] = 1.0

    def _add_placeholder(self, dtype, shape, name):
        placeholder = tf.placeholder(dtype, shape, name)
        self.placeholders[name] = placeholder
        return placeholder

    def _set_up_input_pls(self):
        """Sets up input placeholders by adding them to self._placeholders.
        Keys are defined as self.PL_*.
        """
        # Combine config data
        bev_dims = np.append(self._bev_pixel_size, self._bev_depth)

        with tf.variable_scope('bev_input'):
            # Placeholder for BEV image input, to be filled in with feed_dict
            bev_input_placeholder = self._add_placeholder(tf.float32, bev_dims,
                                                          self.PL_BEV_INPUT)

            self._bev_input_batches = tf.expand_dims(
                bev_input_placeholder, axis=0)

            self._bev_preprocessed = \
                self._bev_feature_extractor.preprocess_input(
                    self._bev_input_batches, self._bev_pixel_size)

            # Summary Images
            bev_summary_images = tf.split(
                bev_input_placeholder, self._bev_depth, axis=2)
            tf.summary.image("bev_maps", bev_summary_images,
                             max_outputs=self._bev_depth)

        with tf.variable_scope('img_input'):
            # Take variable size input images
            img_input_placeholder = self._add_placeholder(
                tf.float32,
                [None, None, self._img_depth],
                self.PL_IMG_INPUT)

            self._img_input_batches = tf.expand_dims(
                img_input_placeholder, axis=0)

            self._img_preprocessed = \
                self._img_feature_extractor.preprocess_input(
                    self._img_input_batches, self._img_pixel_size)

            tf.summary.image("rgb_image", self._img_preprocessed,
                             max_outputs=2)

        with tf.variable_scope('pl_labels'):
            self._add_placeholder(tf.float32, [None, 6],
                                  self.PL_LABEL_ANCHORS)
            self._add_placeholder(tf.float32, [None, 7],
                                  self.PL_LABEL_BOXES_3D)
            self._add_placeholder(tf.float32, [None],
                                  self.PL_LABEL_CLASSES)

        # Placeholders for anchors
        with tf.variable_scope('pl_anchors'):
            self._add_placeholder(tf.float32, [None, 6],
                                  self.PL_ANCHORS)
            self._add_placeholder(tf.float32, [None],
                                  self.PL_ANCHOR_IOUS)
            self._add_placeholder(tf.float32, [None, 6],
                                  self.PL_ANCHOR_OFFSETS)
            self._add_placeholder(tf.float32, [None],
                                  self.PL_ANCHOR_CLASSES)

            with tf.variable_scope('bev_anchor_projections'):
                self._add_placeholder(tf.float32, [None, 4],
                                      self.PL_BEV_ANCHORS)
                self._bev_anchors_norm_pl = self._add_placeholder(
                    tf.float32, [None, 4], self.PL_BEV_ANCHORS_NORM)

            with tf.variable_scope('img_anchor_projections'):
                self._add_placeholder(tf.float32, [None, 4],
                                      self.PL_IMG_ANCHORS)
                self._img_anchors_norm_pl = self._add_placeholder(
                    tf.float32, [None, 4], self.PL_IMG_ANCHORS_NORM)

            with tf.variable_scope('saample_info'):
                # the calib matrix shape is (3 x 4)
                self._add_placeholder(
                    tf.float32, [3, 4], self.PL_CALIB_P2)
                self._add_placeholder(tf.int32,
                                      shape=[1],
                                      name=self.PL_IMG_IDX)
                self._add_placeholder(tf.float32, [4], self.PL_GROUND_PLANE)

    def _set_up_feature_extractors(self):
        """Sets up feature extractors and stores feature maps and as member variables.
        """

        self.bev_feature_maps, self.bev_end_points = \
            self._bev_feature_extractor.build(
                self._bev_preprocessed,
                self._bev_pixel_size,
                self._is_training)

        self.img_feature_maps, self.img_end_points = \
            self._img_feature_extractor.build(
                self._img_preprocessed,
                self._img_pixel_size,
                self._is_training)

    def build(self):

        # Setup input placeholders
        self._set_up_input_pls()

        # Setup feature extractors
        self._set_up_feature_extractors()

        bev_proposal_input = self.bev_feature_maps
        img_proposal_input = self.img_feature_maps

        fusion_mean_div_factor = 2.0

        # If both img and bev probabilites are set to 1.0, don't do
        # path drop.
        if not (self._path_drop_probabilities[0] ==
                self._path_drop_probabilities[1] == 1.0):
            with tf.variable_scope('rpn_path_drop'):

                random_values = tf.random_uniform(shape=[3],
                                                  minval=0.0,
                                                  maxval=1.0)

                img_mask, bev_mask = self.create_path_drop_masks(
                    self._path_drop_probabilities[0],
                    self._path_drop_probabilities[1],
                    random_values)

                img_proposal_input = tf.multiply(img_proposal_input,
                                                 img_mask)

                bev_proposal_input = tf.multiply(bev_proposal_input,
                                                 bev_mask)

                self.img_path_drop_mask = img_mask
                self.bev_path_drop_mask = bev_mask

                # Overwrite the division factor
                fusion_mean_div_factor = img_mask + bev_mask

        with tf.variable_scope('proposal_roi_pooling'):

            with tf.variable_scope('box_indices'):
                def get_box_indices(boxes):
                    proposals_shape = boxes.get_shape().as_list()
                    if any(dim is None for dim in proposals_shape):
                        proposals_shape = tf.shape(boxes)
                    ones_mat = tf.ones(proposals_shape[:2], dtype=tf.int32)
                    multiplier = tf.expand_dims(
                        tf.range(start=0, limit=proposals_shape[0]), 1)
                    return tf.reshape(ones_mat * multiplier, [-1])

                bev_boxes_norm_batches = tf.expand_dims(
                    self._bev_anchors_norm_pl, axis=0)

                # These should be all 0's since there is only 1 image
                tf_box_indices = get_box_indices(bev_boxes_norm_batches)

            # Do ROI Pooling on BEV
            bev_proposal_rois = tf.image.crop_and_resize(
                bev_proposal_input,
                self._bev_anchors_norm_pl,
                tf_box_indices,
                self._proposal_roi_crop_size)
            # Do ROI Pooling on image
            img_proposal_rois = tf.image.crop_and_resize(
                img_proposal_input,
                self._img_anchors_norm_pl,
                tf_box_indices,
                self._proposal_roi_crop_size)

        # Fully connected layers (Box Predictor)
        avod_layers_config = self.model_config.layers_config.avod_config

        with tf.variable_scope('proposal_roi_fusion'):
            feat_fusion_out = None
            fc_layers_type = avod_layers_config.WhichOneof('fc_layers')
            if fc_layers_type == 'basic_fc_layers':
                fusion_method = \
                    avod_layers_config.basic_fc_layers.fusion_method
            elif fc_layers_type == 'fusion_fc_layers':
                fusion_method = \
                    avod_layers_config.fusion_fc_layers.fusion_method

            if fusion_method == 'mean':
                tf_features_sum = tf.add(bev_proposal_rois, img_proposal_rois)
                feat_fusion_out = tf.divide(tf_features_sum,
                                            fusion_mean_div_factor)
            elif fusion_method == 'concat':
                feat_fusion_out = tf.concat(
                    [bev_proposal_rois, img_proposal_rois], axis=3)
            else:
                raise ValueError('Invalid fusion method', self._fusion_method)

        all_anchors = self.placeholders[self.PL_ANCHORS]
        ground_plane = self.placeholders[self.PL_GROUND_PLANE]

        fc_output_layers = \
            avod_fc_layers_builder.build(
                layers_config=avod_layers_config,
                input_rois=[feat_fusion_out],
                input_weights=[1.0],
                num_final_classes=self._num_final_classes,
                box_rep=self._box_rep,
                top_anchors=all_anchors,
                ground_plane=ground_plane,
                is_training=self._is_training)

        all_cls_logits = \
            fc_output_layers[avod_fc_layers_builder.KEY_CLS_LOGITS]
        all_offsets = fc_output_layers[avod_fc_layers_builder.KEY_OFFSETS]

        # This may be None
        all_angle_vectors = \
            fc_output_layers.get(avod_fc_layers_builder.KEY_ANGLE_VECTORS)

        with tf.variable_scope('softmax'):
            all_cls_softmax = tf.nn.softmax(
                all_cls_logits)

        ######################################################
        # Subsample mini_batch for the loss function
        ######################################################
        # Get the ground truth tensors
        anchors_gt = self.placeholders[self.PL_LABEL_ANCHORS]
        if self._box_rep in ['box_3d', 'box_4ca']:
            boxes_3d_gt = self.placeholders[self.PL_LABEL_BOXES_3D]
            orientations_gt = boxes_3d_gt[:, 6]
        elif self._box_rep in ['box_8c', 'box_8co', 'box_4c']:
            boxes_3d_gt = self.placeholders[self.PL_LABEL_BOXES_3D]
        else:
            raise NotImplementedError('Ground truth tensors not implemented')

        if self._train_val_test in ['train', 'val']:
            with tf.variable_scope('bev'):
                # Project all anchors into bev and image spaces
                bev_proposal_boxes, bev_proposal_boxes_norm = \
                    anchor_projector.project_to_bev(
                        all_anchors,
                        self.dataset.kitti_utils.bev_extents)

                # Reorder projected boxes into [y1, x1, y2, x2]
                bev_proposal_boxes_tf_order = \
                    anchor_projector.reorder_projected_boxes(
                        bev_proposal_boxes)

            with tf.variable_scope('img'):
                image_shape = tf.cast(tf.shape(
                    self.placeholders[self.PL_IMG_INPUT])[0:2],
                    tf.float32)
                img_proposal_boxes, img_proposal_boxes_norm = \
                    anchor_projector.tf_project_to_image_space(
                        all_anchors,
                        self.placeholders[self.PL_CALIB_P2],
                        image_shape)

            # Project anchor_gts to 2D bev
            with tf.variable_scope('avod_gt_projection'):
                bev_anchor_boxes_gt, _ = anchor_projector.project_to_bev(
                    anchors_gt, self.dataset.kitti_utils.bev_extents)

                bev_anchor_boxes_gt_tf_order = \
                    anchor_projector.reorder_projected_boxes(
                        bev_anchor_boxes_gt)

            with tf.variable_scope('avod_box_list'):
                # Convert to box_list format
                anchor_box_list_gt = \
                    box_list.BoxList(bev_anchor_boxes_gt_tf_order)
                anchor_box_list = \
                    box_list.BoxList(bev_proposal_boxes_tf_order)

            class_labels = self.placeholders[self.PL_LABEL_CLASSES]
            mb_mask, mb_class_label_indices, mb_gt_indices = \
                self.sample_mini_batch(
                    anchor_box_list_gt=anchor_box_list_gt,
                    anchor_box_list=anchor_box_list,
                    class_labels=class_labels)

            # Create classification one_hot vector
            with tf.variable_scope('avod_one_hot_classes'):
                mb_classification_gt = tf.one_hot(
                    mb_class_label_indices,
                    depth=self._num_final_classes,
                    on_value=1.0 - self._config.label_smoothing_epsilon,
                    off_value=(self._config.label_smoothing_epsilon /
                               self.dataset.num_classes))

            # Mask predictions
            with tf.variable_scope('avod_apply_mb_mask'):
                # Classification
                mb_classifications_logits = tf.boolean_mask(
                    all_cls_logits, mb_mask)
                mb_classifications_softmax = tf.boolean_mask(
                    all_cls_softmax, mb_mask)

                # Offsets
                mb_offsets = tf.boolean_mask(all_offsets, mb_mask)

                # Angle Vectors
                if all_angle_vectors is not None:
                    mb_angle_vectors = tf.boolean_mask(
                        all_angle_vectors, mb_mask)
                else:
                    mb_angle_vectors = None

            # Encode anchor offsets
            with tf.variable_scope('avod_encode_mb_anchors'):
                mb_anchors = tf.boolean_mask(all_anchors, mb_mask)

                if self._box_rep == 'box_3d':
                    # Gather corresponding ground truth anchors for each mb
                    # sample
                    mb_anchors_gt = tf.gather(anchors_gt, mb_gt_indices)
                    mb_offsets_gt = anchor_encoder.tf_anchor_to_offset(
                        mb_anchors, mb_anchors_gt)

                    # Gather corresponding ground truth orientation for each
                    # mb sample
                    mb_orientations_gt = tf.gather(orientations_gt,
                                                   mb_gt_indices)
                elif self._box_rep in ['box_8c', 'box_8co']:

                    # Get boxes_3d ground truth mini-batch and convert to box_8c
                    mb_boxes_3d_gt = tf.gather(boxes_3d_gt, mb_gt_indices)
                    if self._box_rep == 'box_8c':
                        mb_boxes_8c_gt = \
                            box_8c_encoder.tf_box_3d_to_box_8c(mb_boxes_3d_gt)
                    elif self._box_rep == 'box_8co':
                        mb_boxes_8c_gt = \
                            box_8c_encoder.tf_box_3d_to_box_8co(mb_boxes_3d_gt)

                    # Convert proposals: anchors -> box_3d -> box8c
                    proposal_boxes_3d = \
                        box_3d_encoder.anchors_to_box_3d(all_anchors,
                                                         fix_lw=True)
                    proposal_boxes_8c = \
                        box_8c_encoder.tf_box_3d_to_box_8c(proposal_boxes_3d)

                    # Get mini batch offsets
                    mb_boxes_8c = tf.boolean_mask(proposal_boxes_8c, mb_mask)
                    mb_offsets_gt = box_8c_encoder.tf_box_8c_to_offsets(
                        mb_boxes_8c, mb_boxes_8c_gt)

                    # Flatten the offsets to a (N x 24) vector
                    mb_offsets_gt = tf.reshape(mb_offsets_gt, [-1, 24])

                elif self._box_rep in ['box_4c', 'box_4ca']:

                    # Get ground plane for box_4c conversion
                    ground_plane = self.placeholders[
                        self.PL_GROUND_PLANE]

                    # Convert gt boxes_3d -> box_4c
                    mb_boxes_3d_gt = tf.gather(boxes_3d_gt, mb_gt_indices)
                    mb_boxes_4c_gt = box_4c_encoder.tf_box_3d_to_box_4c(
                        mb_boxes_3d_gt, ground_plane)

                    # Convert proposals: anchors -> box_3d -> box_4c
                    proposal_boxes_3d = \
                        box_3d_encoder.anchors_to_box_3d(all_anchors,
                                                         fix_lw=True)
                    proposal_boxes_4c = \
                        box_4c_encoder.tf_box_3d_to_box_4c(proposal_boxes_3d,
                                                           ground_plane)

                    # Get mini batch
                    mb_boxes_4c = tf.boolean_mask(proposal_boxes_4c, mb_mask)
                    mb_offsets_gt = box_4c_encoder.tf_box_4c_to_offsets(
                        mb_boxes_4c, mb_boxes_4c_gt)

                    if self._box_rep == 'box_4ca':
                        # Gather corresponding ground truth orientation for each
                        # mb sample
                        mb_orientations_gt = tf.gather(orientations_gt,
                                                       mb_gt_indices)

                else:
                    raise NotImplementedError(
                        'Anchor encoding not implemented for', self._box_rep)

        elif self._train_val_test in ['test']:
            # In test-mode, skip mini-batch processing and just calculate
            # box conversions.
            if self._box_rep in ['box_4c', 'box_4ca']:
                # Convert proposals: anchors -> box_3d -> box_4c
                proposal_boxes_3d = \
                    box_3d_encoder.anchors_to_box_3d(all_anchors, fix_lw=True)
                proposal_boxes_4c = \
                    box_4c_encoder.tf_box_3d_to_box_4c(proposal_boxes_3d,
                                                       ground_plane)

            elif self._box_rep in ['box_8c', 'box_8co']:
                # Convert proposals: anchors -> box_3d -> box8c
                proposal_boxes_3d = \
                    box_3d_encoder.anchors_to_box_3d(all_anchors, fix_lw=True)
                proposal_boxes_8c = \
                    box_8c_encoder.tf_box_3d_to_box_8c(proposal_boxes_3d)

        ######################################################
        # Final Predictions
        ######################################################
        # Get orientations from angle vectors
        if all_angle_vectors is not None:
            with tf.variable_scope('avod_orientation'):
                all_orientations = \
                    orientation_encoder.tf_angle_vector_to_orientation(
                        all_angle_vectors)

        # Apply offsets to regress proposals
        with tf.variable_scope('avod_regression'):
            if self._box_rep == 'box_3d':
                prediction_anchors = \
                    anchor_encoder.offset_to_anchor(all_anchors,
                                                    all_offsets)

            elif self._box_rep in ['box_8c', 'box_8co']:
                # Reshape the 24-dim regressed offsets to (N x 3 x 8)
                reshaped_offsets = tf.reshape(all_offsets,
                                              [-1, 3, 8])
                # Given the offsets, get the boxes_8c
                prediction_boxes_8c = \
                    box_8c_encoder.tf_offsets_to_box_8c(proposal_boxes_8c,
                                                        reshaped_offsets)
                # Convert corners back to box3D
                prediction_boxes_3d = \
                    box_8c_encoder.box_8c_to_box_3d(prediction_boxes_8c)

                # Convert the box_3d to anchor format for nms
                prediction_anchors = \
                    box_3d_encoder.tf_box_3d_to_anchor(prediction_boxes_3d)

            elif self._box_rep in ['box_4c', 'box_4ca']:
                # Convert predictions box_4c -> box_3d
                prediction_boxes_4c = \
                    box_4c_encoder.tf_offsets_to_box_4c(proposal_boxes_4c,
                                                        all_offsets)

                prediction_boxes_3d = \
                    box_4c_encoder.tf_box_4c_to_box_3d(prediction_boxes_4c,
                                                       ground_plane)

                # Convert to anchor format for nms
                prediction_anchors = \
                    box_3d_encoder.tf_box_3d_to_anchor(prediction_boxes_3d)

            else:
                raise NotImplementedError('Regression not implemented for',
                                          self._box_rep)

        # Apply Non-oriented NMS in BEV
        with tf.variable_scope('avod_nms'):
            bev_extents = self.dataset.kitti_utils.bev_extents

            with tf.variable_scope('bev_projection'):
                # Project predictions into BEV
                avod_bev_boxes, _ = anchor_projector.project_to_bev(
                    prediction_anchors, bev_extents)
                avod_bev_boxes_tf_order = \
                    anchor_projector.reorder_projected_boxes(
                        avod_bev_boxes)

            # Get top score from second column onward
            all_top_scores = tf.reduce_max(all_cls_logits[:, 1:], axis=1)

            # Apply NMS in BEV
            nms_indices = tf.image.non_max_suppression(
                avod_bev_boxes_tf_order,
                all_top_scores,
                max_output_size=self._nms_size,
                iou_threshold=self._nms_iou_threshold)

            # Gather predictions from NMS indices
            top_classification_logits = tf.gather(all_cls_logits,
                                                  nms_indices)
            top_classification_softmax = tf.gather(all_cls_softmax,
                                                   nms_indices)
            top_prediction_anchors = tf.gather(prediction_anchors,
                                               nms_indices)

            if self._box_rep == 'box_3d':
                top_orientations = tf.gather(
                    all_orientations, nms_indices)

            elif self._box_rep in ['box_8c', 'box_8co']:
                top_prediction_boxes_3d = tf.gather(
                    prediction_boxes_3d, nms_indices)
                top_prediction_boxes_8c = tf.gather(
                    prediction_boxes_8c, nms_indices)

            elif self._box_rep == 'box_4c':
                top_prediction_boxes_3d = tf.gather(
                    prediction_boxes_3d, nms_indices)
                top_prediction_boxes_4c = tf.gather(
                    prediction_boxes_4c, nms_indices)

            elif self._box_rep == 'box_4ca':
                top_prediction_boxes_3d = tf.gather(
                    prediction_boxes_3d, nms_indices)
                top_prediction_boxes_4c = tf.gather(
                    prediction_boxes_4c, nms_indices)
                top_orientations = tf.gather(
                    all_orientations, nms_indices)

            else:
                raise NotImplementedError('NMS gather not implemented for',
                                          self._box_rep)

        prediction_dict = dict()

        if self._train_val_test in ['train', 'val']:
            # Additional entries are added to the shared prediction_dict
            # Mini batch predictions
            prediction_dict[self.PRED_MB_CLASSIFICATION_LOGITS] = \
                mb_classifications_logits
            prediction_dict[self.PRED_MB_CLASSIFICATION_SOFTMAX] = \
                mb_classifications_softmax
            prediction_dict[self.PRED_MB_OFFSETS] = mb_offsets

            # Mini batch ground truth
            prediction_dict[self.PRED_MB_CLASSIFICATIONS_GT] = \
                mb_classification_gt
            prediction_dict[self.PRED_MB_OFFSETS_GT] = mb_offsets_gt

            # Top NMS predictions
            prediction_dict[self.PRED_TOP_CLASSIFICATION_LOGITS] = \
                top_classification_logits
            prediction_dict[self.PRED_TOP_CLASSIFICATION_SOFTMAX] = \
                top_classification_softmax

            prediction_dict[self.PRED_TOP_PREDICTION_ANCHORS] = \
                top_prediction_anchors

        else:
            # self._train_val_test == 'test'
            prediction_dict[self.PRED_TOP_CLASSIFICATION_SOFTMAX] = \
                top_classification_softmax
            prediction_dict[self.PRED_TOP_PREDICTION_ANCHORS] = \
                top_prediction_anchors

        if self._box_rep == 'box_3d':
            if self._train_val_test in ['train', 'val']:
                prediction_dict[self.PRED_MB_ANCHORS_GT] = mb_anchors_gt
                prediction_dict[self.PRED_MB_ORIENTATIONS_GT] = \
                    mb_orientations_gt
                prediction_dict[self.PRED_MB_ANGLE_VECTORS] = mb_angle_vectors

            prediction_dict[self.PRED_TOP_ORIENTATIONS] = top_orientations

            # For debugging
            prediction_dict[self.PRED_ALL_ANGLE_VECTORS] = all_angle_vectors

        elif self._box_rep in ['box_8c', 'box_8co']:
            prediction_dict[self.PRED_TOP_PREDICTION_BOXES_3D] = \
                top_prediction_boxes_3d

            # Store the corners before converting for visualization purposes
            prediction_dict[self.PRED_TOP_BOXES_8C] = top_prediction_boxes_8c

        elif self._box_rep == 'box_4c':
            prediction_dict[self.PRED_TOP_PREDICTION_BOXES_3D] = \
                top_prediction_boxes_3d
            prediction_dict[self.PRED_TOP_BOXES_4C] = top_prediction_boxes_4c

        elif self._box_rep == 'box_4ca':
            if self._train_val_test in ['train', 'val']:
                prediction_dict[self.PRED_MB_ORIENTATIONS_GT] = \
                    mb_orientations_gt
                prediction_dict[self.PRED_MB_ANGLE_VECTORS] = mb_angle_vectors

            prediction_dict[self.PRED_TOP_PREDICTION_BOXES_3D] = \
                top_prediction_boxes_3d
            prediction_dict[self.PRED_TOP_BOXES_4C] = top_prediction_boxes_4c
            prediction_dict[self.PRED_TOP_ORIENTATIONS] = top_orientations

        else:
            raise NotImplementedError('Prediction dict not implemented for',
                                      self._box_rep)

        return prediction_dict

    def sample_mini_batch(self, anchor_box_list_gt, anchor_box_list,
                          class_labels):

        with tf.variable_scope('avod_create_mb_mask'):
            # Get IoU for every anchor
            all_ious = box_list_ops.iou(anchor_box_list_gt, anchor_box_list)
            max_ious = tf.reduce_max(all_ious, axis=0)
            max_iou_indices = tf.argmax(all_ious, axis=0)

            # Sample a pos/neg mini-batch from anchors with highest IoU match
            mini_batch_utils = self.dataset.kitti_utils.mini_batch_utils
            mb_mask, mb_pos_mask = mini_batch_utils.sample_avod_mini_batch(
                max_ious)
            mb_class_label_indices = mini_batch_utils.mask_class_label_indices(
                mb_pos_mask, mb_mask, max_iou_indices, class_labels)

            mb_gt_indices = tf.boolean_mask(max_iou_indices, mb_mask)

        return mb_mask, mb_class_label_indices, mb_gt_indices

    def create_feed_dict(self, sample_index=None):
        """ Fills in the placeholders with the actual input values.
            Currently, only a batch size of 1 is supported

        Args:
            sample_index: optional, only used when train_val_test == 'test',
                a particular sample index in the dataset
                sample list to build the feed_dict for

        Returns:
            a feed_dict dictionary that can be used in a tensorflow session
        """

        if self._train_val_test in ["train", "val"]:

            # sample_index should be None
            if sample_index is not None:
                raise ValueError('sample_index should be None. Do not load '
                                 'particular samples during train or val')

            # During training/validation, we need a valid sample
            # with anchor info for loss calculation
            sample = None
            anchors_info = []

            valid_sample = False
            while not valid_sample:
                if self._train_val_test == "train":
                    # Get the a random sample from the remaining epoch
                    samples = self.dataset.next_batch(batch_size=1)

                else:  # self._train_val_test == "val"
                    # Load samples in order for validation
                    samples = self.dataset.next_batch(batch_size=1,
                                                      shuffle=False)

                # Only handle one sample at a time for now
                sample = samples[0]
                anchors_info = sample.get(constants.KEY_ANCHORS_INFO)

                # When training, if the mini batch is empty, go to the next
                # sample. Otherwise carry on with found the valid sample.
                # For validation, even if 'anchors_info' is empty, keep the
                # sample (this will help penalize false positives.)
                # We will substitue the necessary info with zeros later on.
                # Note: Training/validating all samples can be switched off.
                train_cond = (self._train_val_test == "train" and
                              self._train_on_all_samples)
                eval_cond = (self._train_val_test == "val" and
                             self._eval_all_samples)
                if anchors_info or train_cond or eval_cond:
                    valid_sample = True
        else:
            # For testing, any sample should work
            if sample_index is not None:
                samples = self.dataset.load_samples([sample_index])
            else:
                samples = self.dataset.next_batch(batch_size=1, shuffle=False)

            # Only handle one sample at a time for now
            sample = samples[0]
            anchors_info = sample.get(constants.KEY_ANCHORS_INFO)

        sample_name = sample.get(constants.KEY_SAMPLE_NAME)
        sample_augs = sample.get(constants.KEY_SAMPLE_AUGS)

        # Get ground truth data
        label_anchors = sample.get(constants.KEY_LABEL_ANCHORS)
        label_classes = sample.get(constants.KEY_LABEL_CLASSES)
        # We only need orientation from box_3d
        label_boxes_3d = sample.get(constants.KEY_LABEL_BOXES_3D)

        # Network input data
        image_input = sample.get(constants.KEY_IMAGE_INPUT)
        bev_input = sample.get(constants.KEY_BEV_INPUT)

        # Image shape (h, w)
        image_shape = [image_input.shape[0], image_input.shape[1]]

        ground_plane = sample.get(constants.KEY_GROUND_PLANE)
        stereo_calib_p2 = sample.get(constants.KEY_STEREO_CALIB_P2)

        # Fill the placeholders for anchor information
        self._fill_anchor_pl_inputs(anchors_info=anchors_info,
                                    ground_plane=ground_plane,
                                    image_shape=image_shape,
                                    stereo_calib_p2=stereo_calib_p2,
                                    sample_name=sample_name,
                                    sample_augs=sample_augs)

        # this is a list to match the explicit shape for the placeholder
        self._placeholder_inputs[self.PL_IMG_IDX] = [int(sample_name)]

        # Fill in the rest
        self._placeholder_inputs[self.PL_BEV_INPUT] = bev_input
        self._placeholder_inputs[self.PL_IMG_INPUT] = image_input

        self._placeholder_inputs[self.PL_LABEL_ANCHORS] = label_anchors
        self._placeholder_inputs[self.PL_LABEL_BOXES_3D] = label_boxes_3d
        self._placeholder_inputs[self.PL_LABEL_CLASSES] = label_classes

        # Sample Info
        # img_idx is a list to match the placeholder shape
        self._placeholder_inputs[self.PL_IMG_IDX] = [int(sample_name)]
        self._placeholder_inputs[self.PL_CALIB_P2] = stereo_calib_p2
        self._placeholder_inputs[self.PL_GROUND_PLANE] = ground_plane

        # Temporary sample info for debugging
        self.sample_info.clear()
        self.sample_info['sample_name'] = sample_name
        self.sample_info['rpn_mini_batch'] = anchors_info

        # Create a feed_dict and fill it with input values
        feed_dict = dict()
        for key, value in self.placeholders.items():
            feed_dict[value] = self._placeholder_inputs[key]

        return feed_dict

    def _fill_anchor_pl_inputs(self,
                               anchors_info,
                               ground_plane,
                               image_shape,
                               stereo_calib_p2,
                               sample_name,
                               sample_augs):
        """
        Fills anchor placeholder inputs with corresponding data

        Args:
            anchors_info: anchor info from mini_batch_utils
            ground_plane: ground plane coefficients
            image_shape: image shape (h, w), used for projecting anchors
            sample_name: name of the sample, e.g. "000001"
            sample_augs: list of sample augmentations
        """

        # Lists for merging anchors info
        all_anchor_boxes_3d = []
        anchors_ious = []
        anchor_offsets = []
        anchor_classes = []

        # Create anchors for each class
        if len(self.dataset.classes) > 1:
            for class_idx in range(len(self.dataset.classes)):
                # Generate anchors for all classes
                grid_anchor_boxes_3d = self._anchor_generator.generate(
                    area_3d=self._area_extents,
                    anchor_3d_sizes=self._cluster_sizes[class_idx],
                    anchor_stride=self._anchor_strides[class_idx],
                    ground_plane=ground_plane)
                all_anchor_boxes_3d.append(grid_anchor_boxes_3d)
            all_anchor_boxes_3d = np.concatenate(all_anchor_boxes_3d)
        else:
            # Don't loop for a single class
            class_idx = 0
            grid_anchor_boxes_3d = self._anchor_generator.generate(
                area_3d=self._area_extents,
                anchor_3d_sizes=self._cluster_sizes[class_idx],
                anchor_stride=self._anchor_strides[class_idx],
                ground_plane=ground_plane)
            all_anchor_boxes_3d = grid_anchor_boxes_3d

        # Filter empty anchors
        # Skip if anchors_info is []
        sample_has_labels = True
        if self._train_val_test in ['train', 'val']:
            # Read in anchor info during training / validation
            if anchors_info:
                anchor_indices, anchors_ious, anchor_offsets, \
                    anchor_classes = anchors_info

                anchor_boxes_3d_to_use = all_anchor_boxes_3d[anchor_indices]
            else:
                train_cond = (self._train_val_test == "train" and
                              self._train_on_all_samples)
                eval_cond = (self._train_val_test == "val" and
                             self._eval_all_samples)
                if train_cond or eval_cond:
                    sample_has_labels = False
        else:
            sample_has_labels = False

        if not sample_has_labels:
            # During testing, or validation with no anchor info, manually
            # filter empty anchors
            # TODO: share voxel_grid_2d with BEV generation if possible
            voxel_grid_2d = \
                self.dataset.kitti_utils.create_sliced_voxel_grid_2d(
                    sample_name, self.dataset.bev_source,
                    image_shape=image_shape)

            # Convert to anchors and filter
            anchors_to_use = box_3d_encoder.box_3d_to_anchor(
                all_anchor_boxes_3d)
            empty_filter = anchor_filter.get_empty_anchor_filter_2d(
                anchors_to_use, voxel_grid_2d, density_threshold=1)

            anchor_boxes_3d_to_use = all_anchor_boxes_3d[empty_filter]

        # Convert lists to ndarrays
        anchor_boxes_3d_to_use = np.asarray(anchor_boxes_3d_to_use)
        anchors_ious = np.asarray(anchors_ious)
        anchor_offsets = np.asarray(anchor_offsets)
        anchor_classes = np.asarray(anchor_classes)

        # Flip anchors and centroid x offsets for augmented samples
        if kitti_aug.AUG_FLIPPING in sample_augs:
            anchor_boxes_3d_to_use = kitti_aug.flip_boxes_3d(
                anchor_boxes_3d_to_use, flip_ry=False)
            if anchors_info:
                anchor_offsets[:, 0] = -anchor_offsets[:, 0]

        # Convert to anchors
        anchors_to_use = box_3d_encoder.box_3d_to_anchor(
            anchor_boxes_3d_to_use)
        num_anchors = len(anchors_to_use)

        # Project anchors into bev
        bev_anchors, bev_anchors_norm = anchor_projector.project_to_bev(
            anchors_to_use, self._bev_extents)

        # Project box_3d anchors into image space
        img_anchors, img_anchors_norm = \
            anchor_projector.project_to_image_space(
                anchors_to_use, stereo_calib_p2, image_shape)

        # Reorder into [y1, x1, y2, x2] for tf.crop_and_resize op
        self._bev_anchors_norm = bev_anchors_norm[:, [1, 0, 3, 2]]
        self._img_anchors_norm = img_anchors_norm[:, [1, 0, 3, 2]]

        # Fill in placeholder inputs
        self._placeholder_inputs[self.PL_ANCHORS] = anchors_to_use

        # If we are in train/validation mode, and the anchor infos
        # are not empty, store them. Checking for just anchors_ious
        # to be non-empty should be enough.
        if self._train_val_test in ['train', 'val'] and \
                len(anchors_ious) > 0:
            self._placeholder_inputs[self.PL_ANCHOR_IOUS] = anchors_ious
            self._placeholder_inputs[self.PL_ANCHOR_OFFSETS] = anchor_offsets
            self._placeholder_inputs[self.PL_ANCHOR_CLASSES] = anchor_classes

        # During test, or val when there is no anchor info
        elif self._train_val_test in ['test'] or \
                len(anchors_ious) == 0:
            # During testing, or validation with no gt, fill these in with 0s
            self._placeholder_inputs[self.PL_ANCHOR_IOUS] = \
                np.zeros(num_anchors)
            self._placeholder_inputs[self.PL_ANCHOR_OFFSETS] = \
                np.zeros([num_anchors, 6])
            self._placeholder_inputs[self.PL_ANCHOR_CLASSES] = \
                np.zeros(num_anchors)
        else:
            raise ValueError('Got run mode {}, and non-empty anchor info'.
                             format(self._train_val_test))

        self._placeholder_inputs[self.PL_BEV_ANCHORS] = bev_anchors
        self._placeholder_inputs[self.PL_BEV_ANCHORS_NORM] = \
            self._bev_anchors_norm
        self._placeholder_inputs[self.PL_IMG_ANCHORS] = img_anchors
        self._placeholder_inputs[self.PL_IMG_ANCHORS_NORM] = \
            self._img_anchors_norm

    def loss(self, prediction_dict):

        loss_dict = {}

        # Note: The loss should be using mini-batch values only
        losses_output = avod_loss_builder.build(self, prediction_dict)

        classification_loss = \
            losses_output[avod_loss_builder.KEY_CLASSIFICATION_LOSS]

        final_reg_loss = losses_output[avod_loss_builder.KEY_REGRESSION_LOSS]

        avod_loss = losses_output[avod_loss_builder.KEY_AVOD_LOSS]

        offset_loss_norm = \
            losses_output[avod_loss_builder.KEY_OFFSET_LOSS_NORM]

        loss_dict.update({self.LOSS_FINAL_CLASSIFICATION: classification_loss})
        loss_dict.update({self.LOSS_FINAL_REGRESSION: final_reg_loss})

        # Add localization and orientation losses to loss dict for plotting
        loss_dict.update({self.LOSS_FINAL_LOCALIZATION: offset_loss_norm})

        ang_loss_loss_norm = losses_output.get(
            avod_loss_builder.KEY_ANG_LOSS_NORM)
        if ang_loss_loss_norm is not None:
            loss_dict.update({self.LOSS_FINAL_ORIENTATION: ang_loss_loss_norm})

        with tf.variable_scope('model_total_loss'):
            total_loss = avod_loss

        return loss_dict, total_loss

    def create_path_drop_masks(self,
                               p_img,
                               p_bev,
                               random_values):
        """Determines global path drop decision based on given probabilities.

        Args:
            p_img: A tensor of float32, probability of keeping image branch
            p_bev: A tensor of float32, probability of keeping bev branch
            random_values: A tensor of float32 of shape [3], the results
                of coin flips, values should range from 0.0 - 1.0.

        Returns:
            final_img_mask: A constant tensor mask containing either one or zero
                depending on the final coin flip probability.
            final_bev_mask: A constant tensor mask containing either one or zero
                depending on the final coin flip probability.
        """

        def keep_branch(): return tf.constant(1.0)

        def kill_branch(): return tf.constant(0.0)

        # The logic works as follows:
        # We have flipped 3 coins, first determines the chance of keeping
        # the image branch, second determines keeping bev branch, the third
        # makes the final decision in the case where both branches were killed
        # off, otherwise the initial img and bev chances are kept.

        img_chances = tf.case([(tf.less(random_values[0], p_img),
                                keep_branch)], default=kill_branch)

        bev_chances = tf.case([(tf.less(random_values[1], p_bev),
                                keep_branch)], default=kill_branch)

        # Decision to determine whether both branches were killed off
        third_flip = tf.logical_or(tf.cast(img_chances, dtype=tf.bool),
                                   tf.cast(bev_chances, dtype=tf.bool))
        third_flip = tf.cast(third_flip, dtype=tf.float32)

        # Make a second choice, for the third case
        # Here we use a 50/50 chance to keep either image or bev
        # If its greater than 0.5, keep the image
        img_second_flip = tf.case([(tf.greater(random_values[2], 0.5),
                                    keep_branch)],
                                  default=kill_branch)
        # If its less than or equal to 0.5, keep bev
        bev_second_flip = tf.case([(tf.less_equal(random_values[2], 0.5),
                                    keep_branch)],
                                  default=kill_branch)

        # Use lambda since this returns another condition and it needs to
        # be callable
        final_img_mask = tf.case([(tf.equal(third_flip, 1),
                                   lambda: img_chances)],
                                 default=lambda: img_second_flip)

        final_bev_mask = tf.case([(tf.equal(third_flip, 1),
                                   lambda: bev_chances)],
                                 default=lambda: bev_second_flip)

        return final_img_mask, final_bev_mask
