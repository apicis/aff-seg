""" This script performs the inference phase of models. The user can decide whether to visualise the predictions as overlays, save predictions or save the visualisation."""

import argparse
import os
import torch

from torchvision.transforms import transforms
from datasets.occluded_affordance_dataset import OccludedAffordanceSegmentationDataset
from torch.utils.data import DataLoader
from models.acanet import acanet
from models.acanet import acanet50
from models.resnet_unet import resnet_unet
from tester import Tester


def get_args():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--model_name', type=str,
                        default="...",
                        )
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
                        default=""
                        )
    return parser.parse_args()


def get_model(model_name):
    model = None
    assert model_name in ["acanet"], "Currently, no other model is supported"

    if model_name == "acanet":
        CLASSES_NUM = 4
        model = acanet.ACANet(n_class=CLASSES_NUM, pretrained=True, freeze_back=False)
    return model


if __name__ == '__main__':
    # Load args
    args = get_args()
    gpu_id = args.gpu_id
    model_name = args.model_name
    data_dir = args.data_dir
    checkpoint_path = args.checkpoint_path
    batch_size = args.batch_size
    visualise_overlay = args.visualise_overlay
    save_overlay = args.save_overlay
    save_res = args.save_res

    # Select device
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    CLASSES_NUM = 4  # 0:background, 1: grasp, 2: contain, 3: arm
    model = get_model(model_name=model_name)
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
                    model=model,
                    checkpoint_dir=args.checkpoint_path,
                    device=device,
                    visualise_overlay=visualise_overlay,
                    save_res=save_res,
                    save_overlay=save_overlay,
                    dest_dir=args.dest_dir)
    tester.test_loop()
