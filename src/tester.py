""" This script contains the class to perform inference of affordance segmentation models. """
import cv2
import torch
import os
import torch.nn.functional as F

from utils.utils_file import make_dir
from tqdm import tqdm
from utils.utils_model import save_prediction_overlay_batch, visualise_prediction_batch, save_prediction_batch


class Tester:
    def __init__(self, test_loader, model_name, model, train_dataset, device, visualise_overlay, save_res, save_overlay,
                 dest_dir):
        self.test_loader = test_loader
        self.model_name = model_name
        self.model = model
        self.train_dataset = train_dataset
        self.device = device
        self.visualise_overlay = visualise_overlay
        self.save_res = save_res
        self.save_overlay = save_overlay
        if save_overlay:
            self.dest_dir_vis = os.path.join(dest_dir, "vis")
            make_dir(self.dest_dir_vis)
        if save_res:
            self.dest_dir_pred = os.path.join(dest_dir, "pred")
            make_dir(self.dest_dir_pred)

    def test_loop(self):
        for i, sample_batch in enumerate(tqdm(self.test_loader)):
            imgs = sample_batch['image']

            # Load to device
            imgs = imgs.to(self.device)

            imgs = self.prepare_inputs(imgs, model_name=self.model_name)

            # Inference
            outputs = self.model(imgs)
            aff_pred = self.prepare_outputs(outputs, model_name=self.model_name)

            if self.save_overlay:
                imgs = sample_batch['rgb']
                filename = sample_batch['filename']
                if self.train_dataset == "CHOC-AFF":
                    save_prediction_overlay_batch(imgs, affs_pred=aff_pred, dest_dir=self.dest_dir_vis,
                                                  filename=filename)

            if self.save_res:
                filename = sample_batch['filename']
                save_prediction_batch(affs_pred=aff_pred, dest_dir=self.dest_dir_pred, filename=filename)

            if self.visualise_overlay:
                imgs = sample_batch['rgb']
                if self.train_dataset == "CHOC-AFF":
                    visualise_prediction_batch(imgs, affs_pred=aff_pred)
                cv2.waitKey(0)
        print("Finished testing loop!")

    def prepare_inputs(self, imgs, model_name):
        new_inputs = None
        if model_name == "ACANet" or model_name == "ACANet50" or model_name == "RN18U" or model_name == "RN50F" or model_name == "DRNAtt":
            new_inputs = ...

        elif model_name == "Mask2Former":
            new_inputs = []

            for ind, _ in enumerate(imgs):
                img_final = imgs[ind]
                height, width = img_final.shape[-2], img_final.shape[-1]

                new_inputs.append(
                    {
                        "image": img_final,
                        "height": height,
                        "width": width,
                    })
        return new_inputs

    def prepare_outputs(self, outputs, model_name):
        aff_pred = None
        if model_name == "ACANet" or model_name == "ACANet50":
            # Affordance output
            probs_aff = torch.softmax(outputs[0], dim=1)
            aff_pred = torch.argmax(probs_aff, dim=1)
            # Object segmentation output
            probs_obj = F.sigmoid(outputs[1].squeeze(1))
            obj_pred = torch.round(probs_obj)
            # Hand segmentation output
            probs_hand = F.sigmoid(outputs[2].squeeze(1))
            hand_pred = torch.round(probs_hand)
        elif model_name == "RN18U" or model_name == "RN50F" or model_name == "DRNAtt":
            aff_pred = ...
        elif model_name == "Mask2Former":
            aff_pred = ...
        return aff_pred
