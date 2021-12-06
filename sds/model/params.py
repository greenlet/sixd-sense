from sds.model.efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, \
    EfficientNetB4, EfficientNetB5, EfficientNetB6


class ScaledParams:
    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    bifpn_widths = (64, 88, 112, 160, 224, 288, 384)
    bifpn_depths = (3, 4, 5, 6, 7, 7, 8)
    subnet_depths = (3, 3, 3, 4, 4, 4, 5)
    subnet_iteration_steps_list = (1, 1, 1, 2, 2, 2, 3)
    num_groups_gn_list = (4, 4, 7, 10, 14, 18, 24) #try to get 16 channels per group
    backbones = (EfficientNetB0,
                 EfficientNetB1,
                 EfficientNetB2,
                 EfficientNetB3,
                 EfficientNetB4,
                 EfficientNetB5,
                 EfficientNetB6)

    def __init__(self, phi: int):
        assert 0 <= phi < len(self.image_sizes)
        self.phi = phi
        self.input_size = self.image_sizes[phi]
        self.input_shape = (self.input_size, self.input_size, 3)
        self.bifpn_width = self.bifpn_widths[phi]
        self.bifpn_depth = self.bifpn_depths[phi]
        self.subnet_width = self.bifpn_width
        self.subnet_depth = self.subnet_depths[phi]
        self.subnet_num_iteration_steps = self.subnet_iteration_steps_list[phi]
        self.num_groups_gn = self.num_groups_gn_list[phi]
        self.backbone_class = self.backbones[phi]

