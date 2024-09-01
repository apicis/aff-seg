""" This script contains the class to perform inference of ACANet. """
import cv2
import torch
import os
import torch.nn.functional as F

from utils.utils_file import make_dir
from tqdm import tqdm
from utils.utils_model import save_prediction_overlay_batch, visualise_prediction_batch, save_prediction_batch, visualise_object_batch


class Tester:
    def __init__(self, test_loader, model, checkpoint_dir, device, visualise_overlay, save_res, save_overlay, dest_dir):
        self.test_loader = test_loader
        self.model = model
        self.checkpoint_dir = checkpoint_dir
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

            # Inference
            outputs = self.model(imgs)
            probs_aff = torch.softmax(outputs[0], dim=1)
            aff_pred = torch.argmax(probs_aff, dim=1)

            probs_obj = F.sigmoid(outputs[1].squeeze(1))
            obj_pred = torch.round(probs_obj)

            probs_hand = F.sigmoid(outputs[2].squeeze(1))
            hand_pred = torch.round(probs_hand)

            if self.save_overlay:
                imgs = sample_batch['rgb']
                filename = sample_batch['filename']
                save_prediction_overlay_batch(imgs, affs_pred=aff_pred, dest_dir=self.dest_dir_vis, filename=filename)

            if self.save_res:
                filename = sample_batch['filename']
                save_prediction_batch(affs_pred=aff_pred, dest_dir=self.dest_dir_pred, filename=filename)

            if self.visualise_overlay:
                imgs = sample_batch['rgb']
                visualise_prediction_batch(imgs, affs_pred=aff_pred)
                visualise_object_batch(imgs, obj_preds=obj_pred, name="Object prediction")
                visualise_object_batch(imgs, obj_preds=hand_pred, name="Hand prediction")
                cv2.waitKey(0)
        print("Finished testing loop!")
