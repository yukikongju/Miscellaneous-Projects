import segmentation_models_pytorch as smp
from utils.registry import register, LOSSES_REGISTRY


@register(LOSSES_REGISTRY, "dice_binary")
def get_dice_metric(cfg):
    return smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
