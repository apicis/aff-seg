""" This script tests that Mask2Former is correctly loaded."""
import os
import torch
import numpy as np

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from Mask2Former.mask2former import add_maskformer2_config
from detectron2.modeling import build_model


def load_mask2former(train_dataset="UMD"):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(os.path.join(os.getcwd(), "src/configs/maskformer2_R50_bs16_160k_{}.yaml".format(
        "UMD" if train_dataset == "UMD" else "CHOC")))
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    model = build_model(cfg)
    return model


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_mask2former(train_dataset="UMD")
    model.to(device)

    print("Model loaded correctly!!")