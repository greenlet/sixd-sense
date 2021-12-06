import collections
import dataclasses as dcls
import functools
import math
import os
import string
import sys
from typing import Tuple, List, Dict, Any, Optional, Callable

from keras_applications.imagenet_utils import _obtain_input_shape
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

print(sys.path)

from sds.model.efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, \
    EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from sds.model.bifpn import build_BiFPN
from sds.model.params import ScaledParams
from sds.utils.anchors import anchors_for_shape

MOMENTUM = 0.997
EPSILON = 1e-4


class PriorProbability(tf.keras.initializers.Initializer):
    """ Apply a prior probability to the weights.
    """

    def __init__(self, probability=0.01):
        self.probability = probability

    def get_config(self):
        return {
            'probability': self.probability
        }

    def __call__(self, shape, dtype=None):
        # set bias to -log((1 - p)/p) for foreground
        result = np.ones(shape, dtype=np.float32) * -math.log((1 - self.probability) / self.probability)

        return result


class BoxNet(tf.keras.models.Model):
    def __init__(self, width: int, depth: int, num_anchors: int = 9, freeze_bn: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.num_values = 4
        
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'bias_initializer': 'zeros',
        }

        kernel_initializer = {
            'depthwise_initializer': tf.initializers.VarianceScaling(),
            'pointwise_initializer': tf.initializers.VarianceScaling(),
        }
        options.update(kernel_initializer)
        
        self.convs = [tf.keras.layers.SeparableConv2D(filters=self.width, name=f'{self.name}/box-{i}', **options)
            for i in range(self.depth)]
        self.head = tf.keras.layers.SeparableConv2D(filters=self.num_anchors * self.num_values, name=f'{self.name}/box-predict', **options)
        
        self.bns = [[tf.keras.layers.BatchNormalization(trainable=not freeze_bn, momentum=MOMENTUM, epsilon=EPSILON, name=f'{self.name}/box-{i}-bn-{j}')
            for j in range(3, 8)] for i in range(self.depth)]
        self.activation = tf.keras.layers.Activation('swish')
        self.reshape = tf.keras.layers.Reshape((-1, self.num_values))
        self.level = 0

    def call(self, inputs, **kwargs):
        feature, level = inputs
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.bns[i][self.level](feature)
            feature = self.activation(feature)
        outputs = self.head(feature)
        outputs = self.reshape(outputs)
        self.level += 1
        return outputs


class ClassNet(tf.keras.models.Model):
    def __init__(self, width: int, depth: int, num_classes: int = 8, num_anchors: int = 9, freeze_bn: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
        }
        kernel_initializer = {
            'depthwise_initializer': tf.keras.initializers.VarianceScaling(),
            'pointwise_initializer': tf.keras.initializers.VarianceScaling(),
        }
        options.update(kernel_initializer)

        self.convs = [tf.keras.layers.SeparableConv2D(filters=self.width, bias_initializer='zeros', name=f'{self.name}/class-{i}', **options)
            for i in range(self.depth)]
        self.head = tf.keras.layers.SeparableConv2D(filters=self.num_classes * self.num_anchors, bias_initializer=PriorProbability(probability=0.01), name = f'{self.name}/class-predict', **options)

        self.bns = [[tf.keras.layers.BatchNormalization(trainable=not freeze_bn, momentum=MOMENTUM, epsilon=EPSILON, name=f'{self.name}/class-{i}-bn-{j}')
            for j in range(3, 8)] for i in range(self.depth)]
        self.activation = tf.keras.layers.Activation('swish')
        self.reshape = tf.keras.layers.Reshape((-1, self.num_classes))
        self.activation_sigmoid = tf.keras.layers.Activation('sigmoid')
        self.level = 0

    def call(self, inputs, **kwargs):
        feature, level = inputs
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.bns[i][self.level](feature)
            feature = self.activation(feature)
        outputs = self.head(feature)
        outputs = self.reshape(outputs)
        outputs = self.activation_sigmoid(outputs)
        self.level += 1
        return outputs
    

class IterativeRotationSubNet(tf.keras.models.Model):
    def __init__(self, width: int, depth: int, num_values: int, num_iteration_steps: int, num_anchors: int = 9,
            freeze_bn: bool = False, use_group_norm: bool = True, num_groups_gn: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.num_values = num_values
        self.num_iteration_steps = num_iteration_steps
        self.use_group_norm = use_group_norm
        self.num_groups_gn = num_groups_gn
        
        if tf.keras.backend.image_data_format() == 'channels_first':
            gn_channel_axis = 1
        else:
            gn_channel_axis = -1
            
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'bias_initializer': 'zeros',
        }
        kernel_initializer = {
            'depthwise_initializer': tf.keras.initializers.VarianceScaling(),
            'pointwise_initializer': tf.keras.initializers.VarianceScaling(),
        }
        options.update(kernel_initializer)

        self.convs = [tf.keras.layers.SeparableConv2D(filters=width, name=f'{self.name}/iterative-rotation-sub-{i}', **options)
            for i in range(self.depth)]
        self.head = tf.keras.layers.SeparableConv2D(filters=self.num_anchors * self.num_values, name=f'{self.name}/iterative-rotation-sub-predict', **options)
        
        if self.use_group_norm:
            self.norm_layer = [[[tfa.layers.GroupNormalization(groups=self.num_groups_gn, axis=gn_channel_axis, name=f'{self.name}/iterative-rotation-sub-{k}-{i}-gn-{j}')
                for j in range(3, 8)] for i in range(self.depth)] for k in range(self.num_iteration_steps)]
        else: 
            self.norm_layer = [[[tf.keras.layers.BatchNormalization(trainable=not freeze_bn, momentum=MOMENTUM, epsilon=EPSILON, name=f'{self.name}/iterative-rotation-sub-{k}-{i}-bn-{j}')
                for j in range(3, 8)] for i in range(self.depth)] for k in range(self.num_iteration_steps)]

        self.activation = tf.keras.layers.Activation('swish')

    def call(self, inputs, **kwargs):
        feature, level = inputs
        level_py = kwargs["level_py"]
        iter_step_py = kwargs["iter_step_py"]
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.norm_layer[iter_step_py][i][level_py](feature)
            feature = self.activation(feature)
        outputs = self.head(feature)
        
        return outputs
    
    
class RotationNet(tf.keras.models.Model):
    def __init__(self, width: int, depth: int, num_values: int, num_iteration_steps: int, num_anchors: int = 9, freeze_bn: bool = False,
            use_group_norm: bool = True, num_groups_gn: Optional[bool] = None, **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.num_values = num_values
        self.num_iteration_steps = num_iteration_steps
        self.use_group_norm = use_group_norm
        self.num_groups_gn = num_groups_gn
        
        if tf.keras.backend.image_data_format() == 'channels_first':
            channel_axis = 1
            gn_channel_axis = 1
        else:
            channel_axis = -1
            gn_channel_axis = -1
            
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'bias_initializer': 'zeros',
        }
        kernel_initializer = {
            'depthwise_initializer': tf.keras.initializers.VarianceScaling(),
            'pointwise_initializer': tf.keras.initializers.VarianceScaling(),
        }
        options.update(kernel_initializer)

        self.convs = [tf.keras.layers.SeparableConv2D(filters=self.width, name=f'{self.name}/rotation-{i}', **options)
            for i in range(self.depth)]
        self.initial_rotation = tf.keras.layers.SeparableConv2D(filters=self.num_anchors * self.num_values, name=f'{self.name}/rotation-init-predict', **options)
    
        if self.use_group_norm:
            self.norm_layer = [[tfa.layers.GroupNormalization(groups=self.num_groups_gn, axis=gn_channel_axis, name=f'{self.name}/rotation-{i}-gn-{j}')
                for j in range(3, 8)] for i in range(self.depth)]
        else: 
            self.norm_layer = [[tf.keras.layers.BatchNormalization(trainable=not freeze_bn, momentum=MOMENTUM, epsilon=EPSILON, name=f'{self.name}/rotation-{i}-bn-{j}')
                for j in range(3, 8)] for i in range(self.depth)]
        
        self.iterative_submodel = IterativeRotationSubNet(width=self.width,
                                                          depth=self.depth - 1,
                                                          num_values=self.num_values,
                                                          num_iteration_steps=self.num_iteration_steps,
                                                          num_anchors=self.num_anchors,
                                                          freeze_bn=freeze_bn,
                                                          use_group_norm=self.use_group_norm,
                                                          num_groups_gn=self.num_groups_gn,
                                                          name='iterative_rotation_subnet')

        self.activation = tf.keras.layers.Activation('swish')
        self.reshape = tf.keras.layers.Reshape((-1, num_values))
        self.level = 0
        self.add = tf.keras.layers.Add()
        self.concat = tf.keras.layers.Concatenate(axis=channel_axis)

    def call(self, inputs, **kwargs):
        feature, level = inputs
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.norm_layer[i][self.level](feature)
            feature = self.activation(feature)
            
        rotation = self.initial_rotation(feature)
        
        for i in range(self.num_iteration_steps):
            iterative_input = self.concat([feature, rotation])
            delta_rotation = self.iterative_submodel([iterative_input, level], level_py=self.level, iter_step_py=i)
            rotation = self.add([rotation, delta_rotation])
        
        outputs = self.reshape(rotation)
        self.level += 1
        return outputs


class IterativeTranslationSubNet(tf.keras.models.Model):
    def __init__(self, width: int, depth: int, num_iteration_steps: int, num_anchors: int = 9, freeze_bn: bool = False,
            use_group_norm: bool = True, num_groups_gn: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.num_iteration_steps = num_iteration_steps
        self.use_group_norm = use_group_norm
        self.num_groups_gn = num_groups_gn
        
        if tf.keras.backend.image_data_format() == 'channels_first':
            gn_channel_axis = 1
        else:
            gn_channel_axis = -1
            
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'bias_initializer': 'zeros',
        }
        kernel_initializer = {
            'depthwise_initializer': tf.keras.initializers.VarianceScaling(),
            'pointwise_initializer': tf.keras.initializers.VarianceScaling(),
        }
        options.update(kernel_initializer)

        self.convs = [tf.keras.layers.SeparableConv2D(filters=self.width, name=f'{self.name}/iterative-translation-sub-{i}', **options)
            for i in range(self.depth)]
        self.head_xy = tf.keras.layers.SeparableConv2D(filters=self.num_anchors * 2, name=f'{self.name}/iterative-translation-xy-sub-predict', **options)
        self.head_z = tf.keras.layers.SeparableConv2D(filters=self.num_anchors, name=f'{self.name}/iterative-translation-z-sub-predict', **options)

        if self.use_group_norm:
            self.norm_layer = [[[tfa.layers.GroupNormalization(groups=self.num_groups_gn, axis=gn_channel_axis, name=f'{self.name}/iterative-translation-sub-{k}-{i}-gn-{j}')
                for j in range(3, 8)] for i in range(self.depth)] for k in range(self.num_iteration_steps)]
        else:
            self.norm_layer = [[[tf.keras.layers.BatchNormalization(trainable=not freeze_bn, momentum=MOMENTUM, epsilon=EPSILON, name=f'{self.name}/iterative-translation-sub-{k}-{i}-bn-{j}')
                for j in range(3, 8)] for i in range(self.depth)] for k in range(self.num_iteration_steps)]

        self.activation = tf.keras.layers.Activation('swish')

    def call(self, inputs, **kwargs):
        feature, level = inputs
        level_py = kwargs["level_py"]
        iter_step_py = kwargs["iter_step_py"]
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.norm_layer[iter_step_py][i][level_py](feature)
            feature = self.activation(feature)
        outputs_xy = self.head_xy(feature)
        outputs_z = self.head_z(feature)

        return outputs_xy, outputs_z
    
    
class TranslationNet(tf.keras.models.Model):
    def __init__(self, width: int, depth: int, num_iteration_steps: int, num_anchors: int = 9, freeze_bn: bool = False,
            use_group_norm: bool = True, num_groups_gn: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.num_iteration_steps = num_iteration_steps
        self.use_group_norm = use_group_norm
        self.num_groups_gn = num_groups_gn
        
        if tf.keras.backend.image_data_format() == 'channels_first':
            channel_axis = 1
            gn_channel_axis = 1
        else:
            channel_axis = -1
            gn_channel_axis = -1
            
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'bias_initializer': 'zeros',
        }
        kernel_initializer = {
            'depthwise_initializer': tf.keras.initializers.VarianceScaling(),
            'pointwise_initializer': tf.keras.initializers.VarianceScaling(),
        }
        options.update(kernel_initializer)

        self.convs = [tf.keras.layers.SeparableConv2D(filters=self.width, name=f'{self.name}/translation-{i}', **options)
            for i in range(self.depth)]
        self.initial_translation_xy = tf.keras.layers.SeparableConv2D(filters=self.num_anchors * 2, name=f'{self.name}/translation-xy-init-predict', **options)
        self.initial_translation_z = tf.keras.layers.SeparableConv2D(filters=self.num_anchors, name=f'{self.name}/translation-z-init-predict', **options)

        if self.use_group_norm:
            self.norm_layer = [[tfa.layers.GroupNormalization(groups=self.num_groups_gn, axis=gn_channel_axis, name=f'{self.name}/translation-{i}-gn-{j}')
                for j in range(3, 8)] for i in range(self.depth)]
        else: 
            self.norm_layer = [[tf.keras.layers.BatchNormalization(trainable=not freeze_bn, momentum=MOMENTUM, epsilon=EPSILON, name=f'{self.name}/translation-{i}-bn-{j}')
                for j in range(3, 8)] for i in range(self.depth)]
        
        self.iterative_submodel = IterativeTranslationSubNet(width=self.width,
                                                             depth=self.depth - 1,
                                                             num_iteration_steps=self.num_iteration_steps,
                                                             num_anchors=self.num_anchors,
                                                             freeze_bn=freeze_bn,
                                                             use_group_norm=self.use_group_norm,
                                                             num_groups_gn=self.num_groups_gn,
                                                             name='iterative_translation_subnet')

        self.activation = tf.keras.layers.Lambda(lambda x: tf.nn.swish(x))
        self.reshape_xy = tf.keras.layers.Reshape((-1, 2))
        self.reshape_z = tf.keras.layers.Reshape((-1, 1))
        self.level = 0
        self.add = tf.keras.layers.Add()
        self.concat = tf.keras.layers.Concatenate(axis=channel_axis)
            
        self.concat_output = tf.keras.layers.Concatenate(axis=-1) #always last axis after reshape

    def call(self, inputs, **kwargs):
        feature, level = inputs
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.norm_layer[i][self.level](feature)
            feature = self.activation(feature)
            
        translation_xy = self.initial_translation_xy(feature)
        translation_z = self.initial_translation_z(feature)
        
        for i in range(self.num_iteration_steps):
            iterative_input = self.concat([feature, translation_xy, translation_z])
            delta_translation_xy, delta_translation_z = self.iterative_submodel([iterative_input, level], level_py=self.level, iter_step_py=i)
            translation_xy = self.add([translation_xy, delta_translation_xy])
            translation_z = self.add([translation_z, delta_translation_z])
        
        outputs_xy = self.reshape_xy(translation_xy)
        outputs_z = self.reshape_z(translation_z)
        outputs = self.concat_output([outputs_xy, outputs_z])
        self.level += 1
        return outputs
    

def build_subnets(num_classes: int, subnet_width: int, subnet_depth: int, subnet_num_iteration_steps: int,
        num_groups_gn: int, num_rotation_parameters: int, freeze_bn: int, num_anchors: int):
    """
    Builds the EfficientPose subnetworks
    Args:
        num_classes: Number of classes for the classification network output
        subnet_width: The number of channels used in the subnetwork layers
        subnet_depth: The number of layers used in the subnetworks
        subnet_num_iteration_steps: The number of iterative refinement steps used in the rotation and translation subnets
        num_groups_gn: The number of groups per group norm layer used in the rotation and translation subnets
        num_rotation_parameters: Number of rotation parameters, e.g. 3 for axis angle representation
        freeze_bn: Boolean indicating if the batch norm layers should be freezed during training or not.
        num_anchors: The number of anchors, usually 3 scales and 3 aspect ratios resulting in 3 * 3 = 9 anchors
    
    Returns:
       The subnetworks
    """
    box_net = BoxNet(subnet_width,
                      subnet_depth,
                      num_anchors=num_anchors,
                      freeze_bn=freeze_bn,
                      name='box_net')
    
    class_net = ClassNet(subnet_width,
                          subnet_depth,
                          num_classes=num_classes,
                          num_anchors=num_anchors,
                          freeze_bn=freeze_bn,
                          name='class_net')
    
    rotation_net = RotationNet(subnet_width,
                                subnet_depth,
                                num_values=num_rotation_parameters,
                                num_iteration_steps=subnet_num_iteration_steps,
                                num_anchors=num_anchors,
                                freeze_bn=freeze_bn,
                                use_group_norm=True,
                                num_groups_gn=num_groups_gn,
                                name='rotation_net')
    
    translation_net = TranslationNet(subnet_width,
                                    subnet_depth,
                                    num_iteration_steps=subnet_num_iteration_steps,
                                    num_anchors=num_anchors,
                                    freeze_bn=freeze_bn,
                                    use_group_norm=True,
                                    num_groups_gn=num_groups_gn,
                                    name='translation_net')

    return box_net, class_net, rotation_net, translation_net     


def translation_transform_inv(translation_anchors, deltas, scale_factors = None):
    """ Applies the predicted 2D translation center point offsets (deltas) to the translation_anchors

    Args
        translation_anchors : Tensor of shape (B, N, 3), where B is the batch size, N the number of boxes and 2 values for (x, y) +1 value with the stride.
        deltas: Tensor of shape (B, N, 3). The first 2 deltas (d_x, d_y) are a factor of the stride +1 with Tz.

    Returns
        A tensor of the same shape as translation_anchors, but with deltas applied to each translation_anchors and the last coordinate is the concatenated (untouched) Tz value from deltas.
    """

    stride  = translation_anchors[:, :, -1]

    if scale_factors:
        x = translation_anchors[:, :, 0] + (deltas[:, :, 0] * scale_factors[0] * stride)
        y = translation_anchors[:, :, 1] + (deltas[:, :, 1] * scale_factors[1] * stride)
    else:
        x = translation_anchors[:, :, 0] + (deltas[:, :, 0] * stride)
        y = translation_anchors[:, :, 1] + (deltas[:, :, 1] * stride)
        
    Tz = deltas[:, :, 2]

    pred_translations = tf.stack([x, y, Tz], axis = 2) #x,y 2D Image coordinates and Tz

    return pred_translations


class RegressTranslation(tf.keras.layers.Layer):
    """ 
    Keras layer for applying regression offset values to translation anchors to get the 2D translation centerpoint and Tz.
    """

    def __init__(self, *args, **kwargs):
        """Initializer for the RegressTranslation layer.
        """
        super().__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        translation_anchors, regression_offsets = inputs
        return translation_transform_inv(translation_anchors, regression_offsets)

    def compute_output_shape(self, input_shape):
        # return input_shape[0]
        return input_shape[1]

    def get_config(self):
        config = super(RegressTranslation, self).get_config()

        return config
    

class CalculateTxTy(tf.keras.layers.Layer):
    """ Keras layer for calculating the Tx- and Ty-Components of the Translationvector with a given 2D-point and the intrinsic camera parameters.
    """

    def __init__(self, *args, **kwargs):
        """ Initializer for an CalculateTxTy layer.
        """
        super().__init__(*args, **kwargs)

    def call(self, inputs, fx: float, fy: float, px: float, py: float, tz_scale: float, image_scale: float, **kwargs):
        # Tx = (cx - px) * Tz / fx
        # Ty = (cy - py) * Tz / fy
        
        fx = tf.expand_dims(fx, axis=-1)
        fy = tf.expand_dims(fy, axis=-1)
        px = tf.expand_dims(px, axis=-1)
        py = tf.expand_dims(py, axis=-1)
        tz_scale = tf.expand_dims(tz_scale, axis=-1)
        image_scale = tf.expand_dims(image_scale, axis=-1)
        
        x = inputs[:, :, 0] / image_scale
        y = inputs[:, :, 1] / image_scale
        tz = inputs[:, :, 2] * tz_scale
        
        x = x - px
        y = y - py
        
        tx = tf.math.multiply(x, tz) / fx
        ty = tf.math.multiply(y, tz) / fy
        
        output = tf.stack([tx, ty, tz], axis=-1)
        
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()

        return config


def bbox_transform_inv(boxes, deltas, scale_factors = None):
    """
    Reconstructs the 2D bounding boxes using the anchor boxes and the predicted deltas of the anchor boxes to the bounding boxes
    Args:
        boxes: Tensor containing the anchor boxes with shape (..., 4)
        deltas: Tensor containing the offsets of the anchor boxes to the bounding boxes with shape (..., 4)
        scale_factors: optional scaling factor for the deltas
    Returns:
        Tensor containing the reconstructed 2D bounding boxes with shape (..., 4)

    """
    cxa = (boxes[..., 0] + boxes[..., 2]) / 2
    cya = (boxes[..., 1] + boxes[..., 3]) / 2
    wa = boxes[..., 2] - boxes[..., 0]
    ha = boxes[..., 3] - boxes[..., 1]
    ty, tx, th, tw = deltas[..., 0], deltas[..., 1], deltas[..., 2], deltas[..., 3]
    if scale_factors:
        ty *= scale_factors[0]
        tx *= scale_factors[1]
        th *= scale_factors[2]
        tw *= scale_factors[3]
    w = tf.exp(tw) * wa
    h = tf.exp(th) * ha
    cy = ty * ha + cya
    cx = tx * wa + cxa
    ymin = cy - h / 2.
    xmin = cx - w / 2.
    ymax = cy + h / 2.
    xmax = cx + w / 2.
    return tf.stack([xmin, ymin, xmax, ymax], axis=-1)


class RegressBoxes(tf.keras.layers.Layer):
    """ 
    Keras layer for applying regression offset values to anchor boxes to get the 2D bounding boxes.
    """
    def __init__(self, *args, **kwargs):
        super(RegressBoxes, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        anchors, regression = inputs
        return bbox_transform_inv(anchors, regression)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(RegressBoxes, self).get_config()
        return config

    
class ClipBoxes(tf.keras.layers.Layer):
    """
    Layer that clips 2D bounding boxes so that they are inside the image
    """
    def call(self, inputs, **kwargs):
        image, boxes = inputs
        shape = tf.keras.backend.cast(tf.keras.backend.shape(image), tf.keras.backend.floatx())
        height = shape[1]
        width = shape[2]
        x1 = tf.clip_by_value(boxes[:, :, 0], 0, width - 1)
        y1 = tf.clip_by_value(boxes[:, :, 1], 0, height - 1)
        x2 = tf.clip_by_value(boxes[:, :, 2], 0, width - 1)
        y2 = tf.clip_by_value(boxes[:, :, 3], 0, height - 1)

        return tf.keras.backend.stack([x1, y1, x2, y2], axis=2)

    def compute_output_shape(self, input_shape):
        return input_shape[1]


def apply_subnets_to_feature_maps(box_net, class_net, rotation_net, translation_net, fpn_feature_maps, image_input, camera_parameters_input, input_size, anchor_parameters):
    """
    Applies the subnetworks to the BiFPN feature maps
    Args:
        box_net, class_net, rotation_net, translation_net: Subnetworks
        fpn_feature_maps: Sequence of the BiFPN feature maps of the different levels (P3, P4, P5, P6, P7)
        image_input, camera_parameters_input: The image and camera parameter input layer
        input size: Integer representing the input image resolution
        anchor_parameters: Struct containing anchor parameters. If None, default values are used.
    
    Returns:
       classification: Tensor containing the classification outputs for all anchor boxes. Shape (batch_size, num_anchor_boxes, num_classes)
       bbox_regression: Tensor containing the deltas of anchor boxes to the GT 2D bounding boxes for all anchor boxes. Shape (batch_size, num_anchor_boxes, 4)
       rotation: Tensor containing the rotation outputs for all anchor boxes. Shape (batch_size, num_anchor_boxes, num_rotation_parameters)
       translation: Tensor containing the translation outputs for all anchor boxes. Shape (batch_size, num_anchor_boxes, 3)
       transformation: Tensor containing the concatenated rotation and translation outputs for all anchor boxes. Shape (batch_size, num_anchor_boxes, num_rotation_parameters + 3)
                       Rotation and Translation are concatenated because the Keras Loss function takes only one GT and prediction tensor respectively as input but the transformation loss needs both
       bboxes: Tensor containing the 2D bounding boxes for all anchor boxes. Shape (batch_size, num_anchor_boxes, 4)
    """
    classification = [class_net([feature, i]) for i, feature in enumerate(fpn_feature_maps)]
    classification = tf.keras.layers.Concatenate(axis=1, name='classification')(classification)
    
    bbox_regression = [box_net([feature, i]) for i, feature in enumerate(fpn_feature_maps)]
    bbox_regression = tf.keras.layers.Concatenate(axis=1, name='regression')(bbox_regression)
    
    rotation = [rotation_net([feature, i]) for i, feature in enumerate(fpn_feature_maps)]
    rotation = tf.keras.layers.Concatenate(axis = 1, name='rotation')(rotation)
    
    translation_raw = [translation_net([feature, i]) for i, feature in enumerate(fpn_feature_maps)]
    translation_raw = tf.keras.layers.Concatenate(axis = 1, name='translation_raw_outputs')(translation_raw)
    
    #get anchors and apply predicted translation offsets to translation anchors
    anchors, translation_anchors = anchors_for_shape((input_size, input_size), anchor_params=anchor_parameters)
    translation_anchors_input = np.expand_dims(translation_anchors, axis=0)
    
    translation_xy_Tz = RegressTranslation(name='translation_regression')([translation_anchors_input, translation_raw])
    translation = CalculateTxTy(name='translation')(translation_xy_Tz,
                                                    fx=camera_parameters_input[:, 0],
                                                    fy=camera_parameters_input[:, 1],
                                                    px=camera_parameters_input[:, 2],
                                                    py=camera_parameters_input[:, 3],
                                                    tz_scale=camera_parameters_input[:, 4],
                                                    image_scale=camera_parameters_input[:, 5])
    
    # apply predicted 2D bbox regression to anchors
    anchors_input = np.expand_dims(anchors, axis=0)
    bboxes = RegressBoxes(name='boxes')([anchors_input, bbox_regression[..., :4]])
    bboxes = ClipBoxes(name='clipped_boxes')([image_input, bboxes])
    
    #concat rotation and translation outputs to transformation output to have a single output for transformation loss calculation
    #standard concatenate layer throws error that shapes does not match because translation shape dim 2 is known via translation_anchors and rotation shape dim 2 is None
    #so just use lambda layer with tf concat
    transformation = tf.keras.layers.Lambda(lambda input_list: tf.concat(input_list, axis=-1), name="transformation")([rotation, translation])

    return classification, bbox_regression, rotation, translation, transformation, bboxes
    

def filter_detections(
        boxes,
        classification,
        rotation,
        translation,
        num_rotation_parameters,
        num_translation_parameters = 3,
        class_specific_filter = True,
        nms = True,
        score_threshold = 0.01,
        max_detections = 100,
        nms_threshold = 0.5,
):
    """
    Filter detections using the boxes and classification values.

    Args
        boxes: Tensor of shape (num_boxes, 4) containing the boxes in (x1, y1, x2, y2) format.
        classification: Tensor of shape (num_boxes, num_classes) containing the classification scores.
        rotation: Tensor of shape (num_boxes, num_rotation_parameters) containing the rotations.
        translation: Tensor of shape (num_boxes, 3) containing the translation vectors.
        num_rotation_parameters: Number of rotation parameters, usually 3 for axis angle representation
        num_translation_parameters: Number of translation parameters, usually 3 
        class_specific_filter: Whether to perform filtering per class, or take the best scoring class and filter those.
        nms: Flag to enable/disable non maximum suppression.
        score_threshold: Threshold used to prefilter the boxes with.
        max_detections: Maximum number of detections to keep.
        nms_threshold: Threshold for the IoU value to determine when a box should be suppressed.

    Returns
        A list of [boxes, scores, labels, rotation, translation].
        boxes is shaped (max_detections, 4) and contains the (x1, y1, x2, y2) of the non-suppressed boxes.
        scores is shaped (max_detections,) and contains the scores of the predicted class.
        labels is shaped (max_detections,) and contains the predicted label.
        rotation is shaped (max_detections, num_rotation_parameters) and contains the rotations of the non-suppressed predictions.
        translation is shaped (max_detections, num_translation_parameters) and contains the translations of the non-suppressed predictions.
        In case there are less than max_detections detections, the tensors are padded with -1's.
    """

    def _filter_detections(scores_, labels_):
        # threshold based on score
        # (num_score_keeps, 1)
        indices_ = tf.where(tf.keras.backend.greater(scores_, score_threshold))

        if nms:
            # (num_score_keeps, 4)
            filtered_boxes = tf.gather_nd(boxes, indices_)
            # In [4]: scores = np.array([0.1, 0.5, 0.4, 0.2, 0.7, 0.2])
            # In [5]: tf.greater(scores, 0.4)
            # Out[5]: <tf.Tensor: id=2, shape=(6,), dtype=bool, numpy=array([False,  True, False, False,  True, False])>
            # In [6]: tf.where(tf.greater(scores, 0.4))
            # Out[6]:
            # <tf.Tensor: id=7, shape=(2, 1), dtype=int64, numpy=
            # array([[1],
            #        [4]])>
            #
            # In [7]: tf.gather(scores, tf.where(tf.greater(scores, 0.4)))
            # Out[7]:
            # <tf.Tensor: id=15, shape=(2, 1), dtype=float64, numpy=
            # array([[0.5],
            #        [0.7]])>
            filtered_scores = tf.keras.backend.gather(scores_, indices_)[:, 0]

            # perform NMS
            # filtered_boxes = tf.concat([filtered_boxes[..., 1:2], filtered_boxes[..., 0:1],
            #                             filtered_boxes[..., 3:4], filtered_boxes[..., 2:3]], axis=-1)
            nms_indices = tf.image.non_max_suppression(filtered_boxes, filtered_scores, max_output_size=max_detections,
                                                       iou_threshold=nms_threshold)

            # filter indices based on NMS
            # (num_score_nms_keeps, 1)
            indices_ = tf.keras.backend.gather(indices_, nms_indices)

        # add indices to list of all indices
        # (num_score_nms_keeps, )
        labels_ = tf.gather_nd(labels_, indices_)
        # (num_score_nms_keeps, 2)
        indices_ = tf.keras.backend.stack([indices_[:, 0], labels_], axis=1)

        return indices_

    if class_specific_filter:
        all_indices = []
        # perform per class filtering
        for c in range(int(classification.shape[1])):
            scores = classification[:, c]
            # labels = c * tf.ones((tf.keras.backend.shape(scores)[0],), dtype='int32')
            labels = c * tf.ones(tf.keras.backend.shape(scores)[0], dtype='int64')
            all_indices.append(_filter_detections(scores, labels))

        # concatenate indices to single tensor
        # (concatenated_num_score_nms_keeps, 2)
        indices = tf.keras.backend.concatenate(all_indices, axis=0)
    else:
        scores = tf.keras.backend.max(classification, axis=1)
        labels = tf.keras.backend.argmax(classification, axis=1)
        indices = _filter_detections(scores, labels)

    # select top k
    scores = tf.gather_nd(classification, indices)
    labels = indices[:, 1]
    scores, top_indices = tf.nn.top_k(scores, k=tf.keras.backend.minimum(max_detections, tf.keras.backend.shape(scores)[0]))

    # filter input using the final set of indices
    indices = tf.keras.backend.gather(indices[:, 0], top_indices)
    boxes = tf.keras.backend.gather(boxes, indices)
    labels = tf.keras.backend.gather(labels, top_indices)
    rotation = tf.keras.backend.gather(rotation, indices)
    translation = tf.keras.backend.gather(translation, indices)

    # zero pad the outputs
    pad_size = tf.keras.backend.maximum(0, max_detections - tf.keras.backend.shape(scores)[0])
    boxes = tf.pad(boxes, [[0, pad_size], [0, 0]], constant_values=-1)
    scores = tf.pad(scores, [[0, pad_size]], constant_values=-1)
    labels = tf.pad(labels, [[0, pad_size]], constant_values=-1)
    labels = tf.keras.backend.cast(labels, 'int32')
    rotation = tf.pad(rotation, [[0, pad_size], [0, 0]], constant_values=-1)
    translation = tf.pad(translation, [[0, pad_size], [0, 0]], constant_values=-1)

    # set shapes, since we know what they are
    boxes.set_shape([max_detections, 4])
    scores.set_shape([max_detections])
    labels.set_shape([max_detections])
    rotation.set_shape([max_detections, num_rotation_parameters])
    translation.set_shape([max_detections, num_translation_parameters])

    return [boxes, scores, labels, rotation, translation]


class FilterDetections(tf.keras.layers.Layer):
    """
    Keras layer for filtering detections using score threshold and NMS.
    """

    def __init__(
            self,
            num_rotation_parameters: int,
            num_translation_parameters: int = 3,
            nms: bool = True,
            class_specific_filter: bool = True,
            nms_threshold: float = 0.5,
            score_threshold: float = 0.01,
            max_detections: int = 100,
            parallel_iterations: int = 32,
            **kwargs
    ):
        """
        Filters detections using score threshold, NMS and selecting the top-k detections.

        Args
            num_rotation_parameters: Number of rotation parameters, usually 3 for axis angle representation
            num_translation_parameters: Number of translation parameters, usually 3 
            nms: Flag to enable/disable NMS.
            class_specific_filter: Whether to perform filtering per class, or take the best scoring class and filter those.
            nms_threshold: Threshold for the IoU value to determine when a box should be suppressed.
            score_threshold: Threshold used to prefilter the boxes with.
            max_detections: Maximum number of detections to keep.
            parallel_iterations: Number of batch items to process in parallel.
        """
        self.nms = nms
        self.class_specific_filter = class_specific_filter
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.parallel_iterations = parallel_iterations
        self.num_rotation_parameters = num_rotation_parameters
        self.num_translation_parameters = num_translation_parameters
        super(FilterDetections, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """
        Constructs the NMS graph.

        Args
            inputs : List of [boxes, classification, rotation, translation] tensors.
        """
        boxes = inputs[0]
        classification = inputs[1]
        rotation = inputs[2]
        translation = inputs[3]

        # wrap nms with our parameters
        def _filter_detections(args):
            boxes_ = args[0]
            classification_ = args[1]
            rotation_ = args[2]
            translation_ = args[3]

            return filter_detections(
                boxes_,
                classification_,
                rotation_,
                translation_,
                self.num_rotation_parameters,
                self.num_translation_parameters,
                nms=self.nms,
                class_specific_filter=self.class_specific_filter,
                score_threshold=self.score_threshold,
                max_detections=self.max_detections,
                nms_threshold=self.nms_threshold,
            )

        # call filter_detections on each batch item
        outputs = tf.map_fn(
            _filter_detections,
            elems=[boxes, classification, rotation, translation],
            dtype=['float32', 'float32', 'int32', 'float32', 'float32'],
            parallel_iterations=self.parallel_iterations
        )

        return outputs

    def compute_output_shape(self, input_shape):
        """
        Computes the output shapes given the input shapes.

        Args
            input_shape : List of input shapes [boxes, classification, rotation, translation].

        Returns
            List of tuples representing the output shapes:
            [filtered_boxes.shape, filtered_scores.shape, filtered_labels.shape, filtered_rotation.shape, filtered_translation.shape]
        """
        return [
            (input_shape[0][0], self.max_detections, 4),
            (input_shape[1][0], self.max_detections),
            (input_shape[1][0], self.max_detections),
            (input_shape[2][0], self.max_detections, self.num_rotation_parameters),
            (input_shape[3][0], self.max_detections, self.num_translation_parameters),
        ]

    def compute_mask(self, inputs, mask = None):
        """
        This is required in Keras when there is more than 1 output.
        """
        return (len(inputs) + 1) * [None]

    def get_config(self):
        """
        Gets the configuration of this layer.

        Returns
            Dictionary containing the parameters of this layer.
        """
        config = super().get_config()
        config.update({
            'nms': self.nms,
            'class_specific_filter': self.class_specific_filter,
            'nms_threshold': self.nms_threshold,
            'score_threshold': self.score_threshold,
            'max_detections': self.max_detections,
            'parallel_iterations': self.parallel_iterations,
            'num_rotation_parameters': self.num_rotation_parameters,
            'num_translation_parameters': self.num_translation_parameters,
        })

        return config


def build_EfficientPose(phi: int,
                        num_classes: int = 8,
                        num_anchors: int = 9,
                        freeze_bn: bool = False,
                        score_threshold: float = 0.5,
                        anchor_parameters = None,
                        num_rotation_parameters: int = 3,
                        print_architecture: bool = True):
    """
    Builds an EfficientPose model
    Args:
        phi: EfficientPose scaling hyperparameter phi
        num_classes: Number of classes,
        num_anchors: The number of anchors, usually 3 scales and 3 aspect ratios resulting in 3 * 3 = 9 anchors
        freeze_bn: Boolean indicating if the batch norm layers should be freezed during training or not.
        score_threshold: Minimum score threshold at which a prediction is not filtered out
        anchor_parameters: Struct containing anchor parameters. If None, default values are used.
        num_rotation_parameters: Number of rotation parameters, e.g. 3 for axis angle representation
        print_architecture: Boolean indicating if the model architecture should be printed or not
    
    Returns:
        efficientpose_train: EfficientPose model without NMS used for training
        efficientpose_prediction: EfficientPose model including NMS used for evaluating and inferencing
        all_layers: List of all layers in the EfficientPose model to load weights. Otherwise it can happen that a subnet is considered as a single unit when loading weights and if the output dimension doesn't match with the weight file, the whole subnet weight loading is skipped
    """

    #select parameters according to the given phi
    params = ScaledParams(phi)
    
    # input_size = scaled_parameters["input_size"]
    # input_shape = (input_size, input_size, 3)
    # bifpn_width = subnet_width = scaled_parameters["bifpn_width"]
    # bifpn_depth = scaled_parameters["bifpn_depth"]
    # subnet_depth = scaled_parameters["subnet_depth"]
    # subnet_num_iteration_steps = scaled_parameters["subnet_num_iteration_steps"]
    # num_groups_gn = scaled_parameters["num_groups_gn"]
    # backbone_class = scaled_parameters["backbone_class"]
    
    #input layers
    image_input = tf.keras.layers.Input(params.input_shape)
    camera_parameters_input = tf.keras.layers.Input((6,)) #camera parameters and image scale for calculating the translation vector from 2D x-, y-coordinates
    
    #build EfficientNet backbone
    backbone_model, backbone_feature_maps = params.backbone_class(input_tensor=image_input, freeze_bn=freeze_bn)
    
    #build BiFPN
    fpn_feature_maps = build_BiFPN(backbone_feature_maps, params.bifpn_depth, params.bifpn_width, freeze_bn)
    
    #build subnets
    box_net, class_net, rotation_net, translation_net = build_subnets(num_classes,
                                                                      params.subnet_width,
                                                                      params.subnet_depth,
                                                                      params.subnet_num_iteration_steps,
                                                                      params.num_groups_gn,
                                                                      num_rotation_parameters,
                                                                      freeze_bn,
                                                                      num_anchors)
    
    #apply subnets to feature maps
    classification, bbox_regression, rotation, translation, transformation, bboxes = \
        apply_subnets_to_feature_maps(box_net,
                                      class_net,
                                      rotation_net,
                                      translation_net,
                                      fpn_feature_maps,
                                      image_input,
                                      camera_parameters_input,
                                      params.input_size,
                                      anchor_parameters)
    
    
    #get the EfficientPose model for training without NMS and the rotation and translation output combined in the transformation output because of the loss calculation
    efficientpose_train = tf.keras.models.Model(inputs=[image_input, camera_parameters_input], outputs=[classification, bbox_regression, transformation], name='efficientpose')

    # filter detections (apply NMS / score threshold / select top-k)
    filtered_detections = FilterDetections(num_rotation_parameters=num_rotation_parameters,
                                           num_translation_parameters=3,
                                           name='filtered_detections',
                                           score_threshold=score_threshold
                                           )([bboxes, classification, rotation, translation])

    efficientpose_prediction = tf.keras.models.Model(inputs=[image_input, camera_parameters_input], outputs=filtered_detections, name='efficientpose_prediction')
    
    if print_architecture:
        for model in efficientpose_train, box_net, class_net, rotation_net, translation_net:
            print(f'\n\n{model.summary()}')
        
    #create list with all layers to be able to load all layer weights because sometimes the whole subnet weight loading is skipped if the output shape does not match instead of skipping just the output layer
    all_layers = list(set(efficientpose_train.layers + box_net.layers + class_net.layers + rotation_net.layers + translation_net.layers))
    
    return efficientpose_train, efficientpose_prediction, all_layers



if __name__ == '__main__':
    num_classes = 30
    for phi in range(7):
        model, pred_model, all_layers = build_EfficientPose(phi, num_classes=num_classes, score_threshold=0.7)
        print(phi, model)
