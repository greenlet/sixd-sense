from typing import List

import tensorflow as tf

from sds.model.params import ScaledParams
from sds.utils.utils import compose

MOMENTUM = 0.997
EPSILON = 1e-4


class wBiFPNAdd(tf.keras.layers.Layer):
    """
    Layer that computes a weighted sum of BiFPN feature maps
    """
    def __init__(self, epsilon=1e-4, **kwargs):
        super(wBiFPNAdd, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.w = None

    def build(self, input_shape):
        num_in = len(input_shape)
        self.w = self.add_weight(name=self.name,
                                 shape=(num_in,),
                                 initializer=tf.keras.initializers.constant(1 / num_in),
                                 trainable=True,
                                 dtype=tf.float32)

    @tf.function
    def call(self, inputs, **kwargs):
        w = tf.keras.activations.relu(self.w)
        x = tf.reduce_sum([w[i] * inputs[i] for i in range(len(inputs))], axis=0)
        x = x / (tf.reduce_sum(w) + self.epsilon)
        return x

    def compute_output_shape(self, input_shape):
        # Input is a list of similar shapes: [TensorShape[m, n], TensorShape[m, n], ...]
        return input_shape[0]

    def get_config(self):
        config = super(wBiFPNAdd, self).get_config()
        config.update({
            'epsilon': self.epsilon
        })
        return config


def prepare_feature_maps_for_BiFPN(C3, C4, C5, num_channels, freeze_bn):
    """
    Prepares the backbone feature maps for the first BiFPN layer
    Args:
        C3, C4, C5: The EfficientNet backbone feature maps of the different levels
        num_channels: Number of channels used in the BiFPN
        freeze_bn: Boolean indicating if the batch norm layers should be freezed during training or not.
    
    Returns:
       The prepared input feature maps for the first BiFPN layer
    """

    def conv_bn(name_prefix: str, x):
        x = tf.keras.layers.Conv2D(num_channels, kernel_size=1, padding='same',
            name=f'{name_prefix}/conv2d')(x)
        x = tf.keras.layers.BatchNormalization(trainable=not freeze_bn, momentum=MOMENTUM, epsilon=EPSILON,
            name=f'{name_prefix}/bn')(x)
        return x
    
    def maxpool(name_prefix: str, x):
        return tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same',
            name=f'{name_prefix}/maxpool')(x)

    P3_in = conv_bn('fpn_cells/cell_0/fnode3/resample_0_0_8', C3)
    
    P4_in_1 = conv_bn('fpn_cells/cell_0/fnode2/resample_0_1_7', C4)
    P4_in_2 = conv_bn('fpn_cells/cell_0/fnode4/resample_0_1_9', C4)
    
    P5_in_1 = conv_bn('fpn_cells/cell_0/fnode1/resample_0_2_6', C5)
    P5_in_2 = conv_bn('fpn_cells/cell_0/fnode5/resample_0_2_10', C5)

    P6_in = conv_bn('resample_p6', C5)
    P6_in = maxpool('resample_p6', P6_in)
    
    P7_in = maxpool('resample_p7', P6_in)
    
    return P3_in, P4_in_1, P4_in_2, P5_in_1, P5_in_2, P6_in, P7_in


def SeparableConvBlock(num_channels, kernel_size, strides, name, freeze_bn = False):
    """
    Builds a small block consisting of a depthwise separable convolution layer and a batch norm layer
    Args:
        num_channels: Number of channels used in the BiFPN
        kernel_size: Kernel site of the depthwise separable convolution layer
        strides: Stride of the depthwise separable convolution layer
        name: Name of the block
        freeze_bn: Boolean indicating if the batch norm layers should be freezed during training or not.
    
    Returns:
       The depthwise separable convolution block
    """
    f1 = tf.keras.layers.SeparableConv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
        use_bias=True, name=f'{name}/conv')
    f2 = tf.keras.layers.BatchNormalization(trainable=not freeze_bn, momentum=MOMENTUM, epsilon=EPSILON, name=f'{name}/bn')
    return compose(f1, f2)


def single_BiFPN_merge_step(feature_map_other_level, feature_maps_current_level, upsampling, num_channels, idx_BiFPN_layer, node_idx, op_idx):
    """
    Merges two feature maps of different levels in the BiFPN
    Args:
        feature_map_other_level: Input feature map of a different level. Needs to be resized before merging.
        feature_maps_current_level: Input feature map of the current level
        upsampling: Boolean indicating wheter to upsample or downsample the feature map of the different level to match the shape of the current level
        num_channels: Number of channels used in the BiFPN
        idx_BiFPN_layer: The index of the BiFPN layer to build
        node_idx, op_idx: Integers needed to set the correct layer names
    
    Returns:
       The merged feature map
    """
    if upsampling:
        feature_map_resampled = tf.keras.layers.UpSampling2D()(feature_map_other_level)
    else:
        feature_map_resampled = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(feature_map_other_level)
    
    merged_feature_map = wBiFPNAdd(name = f'fpn_cells/cell_{idx_BiFPN_layer}/fnode{node_idx}/add')(
        feature_maps_current_level + [feature_map_resampled])
    merged_feature_map = tf.keras.activations.swish(merged_feature_map)
    merged_feature_map = SeparableConvBlock(num_channels=num_channels,
                                            kernel_size=3,
                                            strides=1,
                                            name=f'fpn_cells/cell_{idx_BiFPN_layer}/fnode{node_idx}/op_after_combine{op_idx}')(merged_feature_map)

    return merged_feature_map


def top_down_pathway_BiFPN(input_feature_maps_top_down, num_channels, idx_BiFPN_layer):
    """
    Computes the top-down-pathway in a single BiFPN layer
    Args:
        input_feature_maps_top_down: Sequence containing the input feature maps of the BiFPN layer (P3, P4, P5, P6, P7)
        num_channels: Number of channels used in the BiFPN
        idx_BiFPN_layer: The index of the BiFPN layer to build
    
    Returns:
       Sequence with the output feature maps of the top-down-pathway
    """
    feature_map_P7 = input_feature_maps_top_down[0]
    output_top_down_feature_maps = [feature_map_P7]
    for level in range(1, 5):
        merged_feature_map = single_BiFPN_merge_step(feature_map_other_level=output_top_down_feature_maps[-1],
                                                    feature_maps_current_level=[input_feature_maps_top_down[level]],
                                                    upsampling=True,
                                                    num_channels=num_channels,
                                                    idx_BiFPN_layer=idx_BiFPN_layer,
                                                    node_idx=level - 1,
                                                    op_idx=4 + level)
        
        output_top_down_feature_maps.append(merged_feature_map)
        
    return output_top_down_feature_maps


def bottom_up_pathway_BiFPN(input_feature_maps_bottom_up, num_channels, idx_BiFPN_layer):
    """
    Computes the bottom-up-pathway in a single BiFPN layer
    Args:
        input_feature_maps_top_down: Sequence containing a list of feature maps serving as input for each level of the BiFPN layer (P3, P4, P5, P6, P7)
        num_channels: Number of channels used in the BiFPN
        idx_BiFPN_layer: The index of the BiFPN layer to build
    
    Returns:
       Sequence with the output feature maps of the bottom-up-pathway
    """
    feature_map_P3 = input_feature_maps_bottom_up[0][0]
    output_bottom_up_feature_maps = [feature_map_P3]
    for level in range(1, 5):
        merged_feature_map = single_BiFPN_merge_step(feature_map_other_level=output_bottom_up_feature_maps[-1],
                                                    feature_maps_current_level=input_feature_maps_bottom_up[level],
                                                    upsampling=False,
                                                    num_channels=num_channels,
                                                    idx_BiFPN_layer=idx_BiFPN_layer,
                                                    node_idx=3 + level,
                                                    op_idx=8 + level)
        
        output_bottom_up_feature_maps.append(merged_feature_map)
        
    return output_bottom_up_feature_maps


def build_BiFPN_layer(features: List[tf.Tensor], num_channels: int, idx_BiFPN_layer: int, freeze_bn: bool = False) -> List[tf.Tensor]:
    """
    Builds a single layer of the bidirectional feature pyramid
    Args:
        features: Sequence containing the feature maps of the previous BiFPN layer (P3, P4, P5, P6, P7) or the EfficientNet backbone feature maps of the different levels (C1, C2, C3, C4, C5)
        num_channels: Number of channels used in the BiFPN
        idx_BiFPN_layer: The index of the BiFPN layer to build
        freeze_bn: Boolean indicating if the batch norm layers should be freezed during training or not.
    
    Returns:
       BiFPN layers of the different levels (P3, P4, P5, P6, P7)
    """
    if idx_BiFPN_layer == 0:
        _, _, C3, C4, C5 = features
        P3_in, P4_in_1, P4_in_2, P5_in_1, P5_in_2, P6_in, P7_in = prepare_feature_maps_for_BiFPN(C3, C4, C5, num_channels, freeze_bn)
    else:
        P3_in, P4_in, P5_in, P6_in, P7_in = features
        
    #top down pathway
    input_feature_maps_top_down = [P7_in,
                                   P6_in,
                                   P5_in_1 if idx_BiFPN_layer == 0 else P5_in,
                                   P4_in_1 if idx_BiFPN_layer == 0 else P4_in,
                                   P3_in]
    
    P7_in, P6_td, P5_td, P4_td, P3_out = top_down_pathway_BiFPN(input_feature_maps_top_down, num_channels, idx_BiFPN_layer)
    
    #bottom up pathway
    input_feature_maps_bottom_up = [[P3_out],
                                    [P4_in_2 if idx_BiFPN_layer == 0 else P4_in, P4_td],
                                    [P5_in_2 if idx_BiFPN_layer == 0 else P5_in, P5_td],
                                    [P6_in, P6_td],
                                    [P7_in]]
    
    P3_out, P4_out, P5_out, P6_out, P7_out = bottom_up_pathway_BiFPN(input_feature_maps_bottom_up, num_channels, idx_BiFPN_layer)
    
    return P3_out, P4_out, P5_out, P6_out, P7_out


def build_BiFPN(backbone_feature_maps: List[tf.Tensor], bifpn_depth: int, bifpn_width: int, freeze_bn: bool) -> List[tf.Tensor]:
    """
    Building the bidirectional feature pyramid as described in https://arxiv.org/abs/1911.09070
    Args:
        backbone_feature_maps: Sequence containing the EfficientNet backbone feature maps of the different levels (C1, C2, C3, C4, C5)
        bifpn_depth: Number of BiFPN layer
        bifpn_width: Number of channels used in the BiFPN
        freeze_bn: Boolean indicating if the batch norm layers should be freezed during training or not.
    
    Returns:
       fpn_feature_maps: Sequence of BiFPN layers of the different levels (P3, P4, P5, P6, P7)
    """
    fpn_feature_maps = backbone_feature_maps
    for i in range(bifpn_depth):
        fpn_feature_maps = build_BiFPN_layer(fpn_feature_maps, bifpn_width, i, freeze_bn=freeze_bn)
        
    return fpn_feature_maps


if __name__ == '__main__':
    freeze_bn = True
    for phi in range(7):
        params = ScaledParams(phi)
        image_input = tf.keras.layers.Input(params.input_shape)
        cam_params_input = tf.keras.layers.Input((6,))
        bb_model, bb_features = params.backbone_class(input_tensor=image_input, freeze_bn=freeze_bn)
        print('-' * 20)
        print(f'{bb_model.name} input shape: {bb_model.input_shape}, output_shape: {bb_model.output_shape}')
        for bb_feat in bb_features:
            print(f'{bb_feat.name} {bb_feat.shape}')
        fpn_features = build_BiFPN(bb_features, params.bifpn_depth, params.bifpn_width, freeze_bn=freeze_bn)
        for fpn_feat in fpn_features:
            print(f'{fpn_feat.name} {fpn_feat.shape}')


