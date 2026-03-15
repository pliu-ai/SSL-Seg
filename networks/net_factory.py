from networks.unet import UNet, UNet_DS, UNet_URPC, UNet_CCT
from networks.unet_2d_condition import UNet2DCondition

try:
    from networks.efficientunet import Effi_UNet
except ImportError:
    Effi_UNet = None
try:
    from networks.enet import ENet
except ImportError:
    ENet = None
try:
    from networks.pnet import PNet2D
except ImportError:
    PNet2D = None
try:
    from networks.nnunet import initialize_network
except ImportError:
    initialize_network = None


def net_factory(net_type="unet", in_chns=1, class_num=3, device=None):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).to(device)
    elif net_type == "enet":
        net = ENet(in_channels=in_chns, num_classes=class_num).cuda()
    elif net_type == "unet_ds":
        net = UNet_DS(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_cct":
        net = UNet_CCT(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_urpc":
        net = UNet_URPC(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "efficient_unet":
        net = Effi_UNet('efficientnet-b3', encoder_weights='imagenet',
                        in_channels=in_chns, classes=class_num).cuda()
    elif net_type == "pnet":
        net = PNet2D(in_chns, class_num, 64, [1, 2, 4, 8, 16]).cuda()
    elif net_type == "nnUNet":
        net = initialize_network(num_classes=class_num).cuda()
    elif net_type == "unet_condition":
        net = UNet2DCondition(in_chns=in_chns, class_num=class_num).to(device)
    else:
        net = None
    return net
