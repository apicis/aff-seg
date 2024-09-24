import argparse
import glob
import numpy as np
import cv2
import os

from tqdm import tqdm


def image_difference_rgb(img1, img2):
    img1_gr_float = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(float)
    img2_gr_float = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(float)
    diff = np.abs(img1_gr_float - img2_gr_float)
    return diff


def image_difference_mask(mask1, mask2):
    mask1_float = mask1.astype(float)
    mask2_float = mask2.astype(float)
    diff = np.abs(mask1_float - mask2_float)
    return diff


def to_uint8(img):
    min_val = np.amin(img)
    max_val = np.amax(img)
    return ((255 - 0) / (max_val - min_val)) * (img.copy() - min_val).astype(np.uint8)


def get_args():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir1', type=str, default="...")
    parser.add_argument('--data_dir2', type=str, default="...")
    parser.add_argument('--visualise', type=bool, default=False)
    return parser.parse_args()


if __name__ == '__main__':
    # Load args
    args = get_args()
    path_to_original_rgb = os.path.join(args.data_dir1, "rgb")
    path_to_original_aff = os.path.join(args.data_dir1, "affordance")
    path_to_my_rgb = os.path.join(args.data_dir2, "rgb")
    path_to_my_aff = os.path.join(args.data_dir2, "affordance")
    visualise = args.visualise

    rgb_diff_list = []
    aff_diff_list = []

    my_rgb_list = glob.glob(os.path.join(path_to_my_rgb, '*.png'))
    my_rgb_list.sort()
    for i in tqdm(range(len(my_rgb_list))):
        filename = os.path.basename(my_rgb_list[i])
        # Load original data
        original_rgb = cv2.imread(os.path.join(path_to_original_rgb, filename), -1)
        original_aff = cv2.imread(os.path.join(path_to_original_aff, filename), -1)

        # Load my data
        my_rgb = cv2.imread(os.path.join(path_to_my_rgb, filename), -1)
        my_aff = cv2.imread(os.path.join(path_to_my_aff, filename), -1)

        # Check differences
        diff_rgb = image_difference_rgb(original_rgb, my_rgb)
        rgb_diff_list.append(np.sum(diff_rgb, dtype=np.float32))

        diff_aff = image_difference_mask(original_aff, my_aff)
        aff_diff_list.append(np.sum(diff_aff, dtype=np.float32))

        if visualise:
            dim = (int(original_rgb.shape[1] * 0.75), int(original_rgb.shape[0] * 0.75))
            cv2.imshow("RGB diff", to_uint8(diff_rgb))
            cv2.imshow("Aff diff", to_uint8(diff_aff))
            cv2.waitKey(0)
        assert diff_rgb.sum() == 0, "RGB diff is not zero"
        assert diff_aff.sum() == 0, "Aff diff is not zero"

    print("Number of rgb with nonzero difference: ", np.count_nonzero(np.asarray(rgb_diff_list)))
    print("Number of affordance with nonzero difference: ", np.count_nonzero(np.asarray(aff_diff_list)))
