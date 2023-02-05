from models import vit
import torch


def dino_vits16(pretrained=True, **kwargs):
    """
    ViT-Small/16x16 pre-trained with DINO.
    Achieves 74.5% top-1 accuracy on ImageNet with k-NN classification.
    """
    model = vit.__dict__["vit_small"](patch_size=16, num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.load('/home/yuliu/Projects/LG-SLOT/dino_ckpts/dino_deitsmall16_pretrain.pth', map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
    return model


def dino_vits8(pretrained=True, **kwargs):
    """
    ViT-Small/8x8 pre-trained with DINO.
    Achieves 74.5% top-1 accuracy on ImageNet with k-NN classification.
    """
    model = vit.__dict__["vit_small"](patch_size=8, num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.load('/home/yuliu/Projects/LG-SLOT/dino_ckpts/dino_deitsmall8_pretrain.pth', map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
    return model


def dino_vitb16(pretrained=True, **kwargs):
    """
    ViT-B/16x16 pre-trained with DINO.
    Achieves 76.5% top-1 accuracy on ImageNet with k-NN classification.
    """
    model = vit.__dict__["vit_base"](patch_size=16, num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.load('/home/yuliu/Projects/LG-SLOT/dino_ckpts/dino_vitbase16_pretrain.pth', map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
    return model


def dino_vitb8(pretrained=True, **kwargs):
    """
    ViT-B/8x8 pre-trained with DINO.
    Achieves 76.5% top-1 accuracy on ImageNet with k-NN classification.
    """
    model = vit.__dict__["vit_base"](patch_size=8, num_classes=0, **kwargs)
    if pretrained:
        state_dict = torch.load('/home/yuliu/Projects/LG-SLOT/dino_ckpts/dino_vitbase8_pretrain.pth', map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
    return model


dino_models = {
    "vits16": dino_vits16,
    "vits8": dino_vits8,
    "vitb16": dino_vitb16,
    "vitb8": dino_vitb8,
}