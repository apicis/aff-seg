""" This script tests that AffordanceNet is correctly loaded. Follow README.md instructions before running this script.
    Modified from https://github.com/HuchieWuchie/affnetDR/blob/main/utils.py
"""

import torch
import affnetDR.lib.mask_rcnn as mask_rcnn

from affnetDR.lib.mask_rcnn import MaskAffordancePredictor, MaskRCNNHeads
from affnetDR.lib.faster_rcnn import FastRCNNPredictor


def load_affordancenet(num_classes, num_affordances):
    # load an instance segmentation model pre-trained on COCO
    model = mask_rcnn.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace mask branch
    mask_layers = [256]
    mask_dilation = 1
    out_channels = model.backbone.out_channels
    model.roi_heads.mask_head = MaskRCNNHeads(out_channels, mask_layers, mask_dilation)

    # and replace the mask predictor with a new one
    in_features_mask = 256
    hidden_layer = 128

    model.roi_heads.mask_predictor = MaskAffordancePredictor(in_features_mask,
                                                             hidden_layer,
                                                             num_affordances)
    return model


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_affordancenet(num_classes=18, num_affordances=8)
    model.to(device)

    print("Model loaded correctly!!")
