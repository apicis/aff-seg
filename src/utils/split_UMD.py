import argparse
import os
import numpy as np
import scipy
import cv2
from tqdm import tqdm
import random

from src.utils.utils_file import make_dir


def get_args():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str,
                        default="...")
    parser.add_argument('--file_path', type=str,
                        default="...")
    parser.add_argument('--visualise', type=bool,
                        default=False)
    parser.add_argument('--save', type=bool,
                        default=False)
    parser.add_argument('--dst_dir', type=str,
                        default="...")
    return parser.parse_args()


if __name__ == '__main__':
    # Load args
    args = get_args()
    src_dir = args.src_dir
    dst_dir = args.dst_dir
    file_path = args.file_path
    visualise = args.visualise
    save = args.save

    # Read split file
    file = open(file_path, "r")
    folders_split = file.read().splitlines()
    split_id = [int(line.split(" ")[0]) for line in folders_split]
    folders = [line.split(" ")[1] for line in folders_split]

    # Create destination directories
    if save:
        assert os.path.exists(dst_dir), "Destination directory does not exist"
        dst_dir_training = os.path.join(dst_dir, "training")
        make_dir(dst_dir_training)
        make_dir(os.path.join(dst_dir_training, "rgb"))
        make_dir(os.path.join(dst_dir_training, "affordance"))
        make_dir(os.path.join(dst_dir_training, "vis"))
        dst_dir_testing = os.path.join(dst_dir, "testing")
        make_dir(dst_dir_testing)
        make_dir(os.path.join(dst_dir_testing, "rgb"))
        make_dir(os.path.join(dst_dir_testing, "affordance"))
        make_dir(os.path.join(dst_dir_testing, "vis"))


    for split, folder in tqdm(zip(split_id, folders), total=len(folders)):
        # Retrieve file names from object instance folder
        filenames = os.listdir(os.path.join(src_dir, folder))
        filenames = [folder + "_" + f.split("_")[2] for f in filenames]
        filenames = list(set(filenames))
        filenames.sort()

        for file_name in filenames:
            # Load RGB
            src_rgb_path = os.path.join(src_dir, folder, file_name + ("_rgb.jpg"))
            img = cv2.imread(src_rgb_path)
            if img is None:
                print("Skipped: {}".format(file_name))
                continue

            # Load mask
            src_mask_path = os.path.join(src_dir, folder, file_name + ("_label.mat"))
            aff_mask = scipy.io.loadmat(src_mask_path)['gt_label']
            if aff_mask is None:
                print("Skipped: {}".format(file_name))
                continue

            colormap = np.zeros_like(img)
            if 1 in aff_mask:  # 1 - 'grasp'
                colormap[aff_mask == 1] = np.array([0, 0, 255])
            if 2 in aff_mask:  # 2 - 'cut'
                colormap[aff_mask == 2] = np.array([255, 0, 0])
            if 3 in aff_mask:  # 3 - 'scoop'
                colormap[aff_mask == 3] = np.array([0, 255, 255])
            if 4 in aff_mask:  # 4 - 'contain'
                colormap[aff_mask == 4] = np.array([0, 255, 0])
            if 5 in aff_mask:  # 5 - 'pound'
                colormap[aff_mask == 5] = np.array([215, 80, 128])
            if 6 in aff_mask:  # 6 - 'support'
                colormap[aff_mask == 6] = np.array([255, 255, 0])
            if 7 in aff_mask:  # 7 - 'wrap-grasp'
                colormap[aff_mask == 7] = np.array([50, 127, 205])
            mask_vis = cv2.addWeighted(img.copy(), 0.6, colormap, 1, 0.0)

            if save:
                assert split in [1, 2], "Split {} currently not supported".format(split)
                if split == 1:
                    dest_dir_final = dst_dir_training
                elif split == 2:
                    dest_dir_final = dst_dir_testing

                # Copy RGB
                rgb_path_dst = os.path.join(dest_dir_final, "rgb", file_name + (".png"))
                cv2.imwrite(rgb_path_dst, img)

                # Copy mask
                mask_path_dst = os.path.join(dest_dir_final, "affordance", file_name + (".png"))
                cv2.imwrite(mask_path_dst, aff_mask)

                mask_path_dst = os.path.join(dest_dir_final, "vis", file_name + (".png"))
                cv2.imwrite(mask_path_dst, mask_vis)

            if visualise:
                cv2.imshow("Overlay affordance", mask_vis)
                cv2.waitKey(0)