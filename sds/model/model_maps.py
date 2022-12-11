from typing import Any, List

import tensorflow as tf

from sds.model.model import bifpn_init, bifpn_layer, final_upscale
from sds.model.utils import prefixer
from sds.model.params import ScaledParams


def bifpn_layers_seq(fpn_feature_maps: List[Any], params: ScaledParams, layers_num: int, name: str):
    pname = prefixer(name)
    for i_layer in range(layers_num):
        fpn_feature_maps = bifpn_layer(fpn_feature_maps, params.bifpn_width, name=pname(f'Bifpn{i_layer + 1}'))
    return fpn_feature_maps


def build_model(n_classes: int, phi: int = 0, freeze_bn: bool = False, bifpn_layers_num: int = 1):
    params = ScaledParams(phi)
    image_input = tf.keras.Input(params.input_shape)
    _, bb_feature_maps = params.backbone_class(input_tensor=image_input, freeze_bn=freeze_bn)

    fpn_feature_maps_init = bifpn_init(bb_feature_maps, params.bifpn_width, name='BifpnInit')
    fpn_feature_maps_seg = bifpn_layers_seq(fpn_feature_maps_init, params, bifpn_layers_num, name='BifpnSeqSeg')
    fpn_feature_maps_noc = bifpn_layers_seq(fpn_feature_maps_init, params, bifpn_layers_num, name='BifpnSeqNOC')
    fpn_feature_maps_normals = bifpn_layers_seq(fpn_feature_maps_init, params, bifpn_layers_num, name='BifpnSeqNormals')

    features_seg = final_upscale(fpn_feature_maps_seg, n_classes + 1, name='Segmentation')
    features_noc = final_upscale(fpn_feature_maps_noc, 3, name='NOC')
    features_normals = final_upscale(fpn_feature_maps_normals, 3, name='Normals')

    model = tf.keras.models.Model(inputs=[image_input], outputs=[features_seg, features_noc, features_normals])
    return model


def _test_model():
    n_classes = 30
    model = build_model(n_classes)
    print(model.summary(line_length=250))
    print(model.inputs)
    print(model.outputs)
    print(f'Output layer names:')
    for out in model.outputs:
        print(f'  {out.name}')


if __name__ == '__main__':
    _test_model()

