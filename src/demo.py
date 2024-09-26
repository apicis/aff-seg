""" This script performs the inference phase of affordance segmentation models.
The user can decide whether to visualise the predictions as overlays, save predictions or save the visualisation."""

import argparse
import os
import torch

from torchvision.transforms import transforms
from datasets.occluded_affordance_dataset import OccludedAffordanceSegmentationDataset
from torch.utils.data import DataLoader
from models.acanet import acanet
from models.acanet import acanet50
from models.resnet_unet import resnet_unet
from models.mask2former.test_mask2former_load import load_mask2former
from models.drnatt import drn_att
from models.resnet_fcn.test_resnet_fcn_load import load_resnet_fcn
from tester import Tester


def get_args():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--model_name', type=str,
                        default="...",
                        )
    parser.add_argument('--train_dataset', type=str, default="CHOC-AFF")
    parser.add_argument('--data_dir', type=str,
                        default="...",
                        )
    parser.add_argument('--checkpoint_path', type=str,
                        default="...",
                        )
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--visualise_overlay', type=bool, default=False)
    parser.add_argument('--save_res', type=bool, default=False)
    parser.add_argument('--save_overlay', type=bool, default=False)
    parser.add_argument('--dest_dir', type=str,
                        default="..."
                        )
    return parser.parse_args()


def get_model(model_name, classes_num, train_dataset):
    model = None
    assert model_name in ["ACANet", "ACANet50", "RN18U", "Mask2Former", "DRNAtt", "RN50F"], \
        "Currently, supported models are ACANet, ACANet50, RN18U, DRNAtt, Mask2Former, RN50F."

    if model_name == "ACANet":
        model = acanet.ACANet(n_class=classes_num, pretrained=True, freeze_back=False)
    elif model_name == "ACANet50":
        model = acanet50.ACANet(n_class=classes_num, pretrained=True, freeze_back=False)
    elif model_name == "RN18U":
        model = resnet_unet.ResNet18Unet(n_class=classes_num, pretrained=True, freeze_back=False)
    elif model_name == "Mask2Former":
        model = load_mask2former(train_dataset=train_dataset)
    elif model_name == "DRNAtt":
        model = drn_att.DRNatt(n_class=classes_num, pretrained=True)
    elif model_name == "RN50F":
        model = load_resnet_fcn(n_classes=classes_num)
    return model


if __name__ == '__main__':
    # Load args
    args = get_args()
    gpu_id = args.gpu_id
    model_name = args.model_name
    train_dataset = args.train_dataset
    data_dir = args.data_dir
    checkpoint_path = args.checkpoint_path
    batch_size = args.batch_size
    visualise_overlay = args.visualise_overlay
    save_overlay = args.save_overlay
    save_res = args.save_res
    dest_dir = args.dest_dir

    # Check args are correct
    assert train_dataset in ["CHOC-AFF", "UMD"], "Allowed training datasets are CHOC-AFF and UMD"
    classes_num = 0
    if train_dataset == "CHOC-AFF":
        classes_num = 4  # 0:background, 1: grasp, 2: contain, 3: arm
        if model_name == "RN50F":
            classes_num = 3 # 0:background, 1: grasp, 2: contain
    elif train_dataset == "UMD":
        classes_num = 8  # 0: background, 1: grasp, 2:cut, 3: scoop, 4: contain, 5: pound, 6: support, 7: wrap-grasp

    # Select device
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = get_model(model_name=model_name, classes_num=classes_num, train_dataset=train_dataset)
    model.to(device)
    input_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    model.load_state_dict(torch.load(checkpoint_path, map_location='cuda:{}'.format(gpu_id)))
    model.eval()

    # Load dataset
    images_dir = os.path.join(data_dir, "rgb")
    test_dataset = OccludedAffordanceSegmentationDataset(
        images_dir,
        augmentation=None,
        preprocessing=input_preprocess,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    tester = Tester(test_loader=test_loader,
                    model_name=model_name,
                    model=model,
                    train_dataset=train_dataset,
                    device=device,
                    visualise_overlay=visualise_overlay,
                    save_res=save_res,
                    save_overlay=save_overlay,
                    dest_dir=dest_dir)
    tester.test_loop()
