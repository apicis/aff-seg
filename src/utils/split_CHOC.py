""" This script splits the CHOC-AFF dataset into training, validation, and 2 testing sets. """
import argparse
import glob
import json
import os
import shutil


def get_args():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str,
                        default="...")
    parser.add_argument('--dest_dir', type=str,
                        default="...")
    return parser.parse_args()


if __name__ == '__main__':
    # Load args
    args = get_args()

    # Source path
    path_to_src = args.src_dir
    path_to_src_rgb = os.path.join(path_to_src, "rgb")
    path_to_src_mask = os.path.join(path_to_src, "mask")
    path_to_src_aff = os.path.join(path_to_src, "affordance")
    path_to_ann = os.path.join(path_to_src, "annotations")

    # Destination paths
    path_to_dst = args.dest_dir

    # Create destination directories
    dst_dir_list = ["training", "test_background", "validation", "test_instances"]
    for dst_dir in dst_dir_list:
        if not os.path.exists(os.path.join(path_to_dst, dst_dir)):
            os.mkdir(os.path.join(path_to_dst, dst_dir))
        if not os.path.exists(os.path.join(path_to_dst, dst_dir, "rgb")):
            os.mkdir(os.path.join(path_to_dst, dst_dir, "rgb"))
        if not os.path.exists(os.path.join(path_to_dst, dst_dir, "mask")):
            os.mkdir(os.path.join(path_to_dst, dst_dir, "mask"))
        if not os.path.exists(os.path.join(path_to_dst, dst_dir, "affordance")):
            os.mkdir(os.path.join(path_to_dst, dst_dir, "affordance"))

    # Initialise the object instances and background splits
    obj_instances_training = [0, 3, 5, 6, 7, 8, 9,
                              12, 13, 14, 15, 16, 17, 18, 19, 20,
                              22, 24, 25, 26, 27, 28, 29, 30,
                              31, 34, 35, 36, 38, 39, 40,
                              41, 42, 43, 44, 47]
    assert len(obj_instances_training) == 36, "Training object instances are not 36!"
    background_training = ["000000.png", "000002.png", "000003.png", "000004.png",
                           "000006.png", "000007.png", "000008.png", "000010.png", "000011.png",
                           "000012.png", "000013.png", "000014.png", "000015.png", "000016.png", "000017.png",
                           "000018.png", "000019.png", "000020.png", "000021.png", "000022.png", "000023.png",
                           "000024.png", "000025.png", "000026.png", "000028.png", "000029.png"]
    background_test1 = ["000001.png", "000005.png", "000009.png", "000027.png"]
    obj_instances_validation = [4, 21, 33, 23, 1, 46]  # box: 4, 21 | stem: 33, 23 | nonstem: 1, 46
    obj_instances_test2 = [10, 32, 11, 2, 37, 45]  # box: 10, 32 | stem: 11, 2 | nonstem: 37, 45

    # Retrieve folder structure
    folders = os.listdir(path_to_ann)
    folders.sort()
    for folder in folders:
        # Retrieve path to files in each folder
        ann_list = glob.glob(os.path.join(path_to_ann, folder, '*.json'))
        ann_list.sort()
        for i in range(0, len(ann_list)):
            # Retrieve file name
            filename = os.path.basename(ann_list[i])
            file_path = ann_list[i]

            # Read annotation json
            f = open(file_path)
            ann_file = json.load(f)

            # Retrieve object instance id
            obj_instance_id = ann_file["object_id"]

            # Retrieve background image name
            background_name = ann_file["background_id"]

            # Check what splits the files belong to
            if obj_instance_id in obj_instances_training and background_name in background_training:
                dst_folder = dst_dir_list[0]
            elif obj_instance_id in obj_instances_training and background_name in background_test1:
                dst_folder = dst_dir_list[1]
            elif obj_instance_id in obj_instances_validation:
                dst_folder = dst_dir_list[2]
            elif obj_instance_id in obj_instances_test2:
                dst_folder = dst_dir_list[3]
            else:
                # continue
                assert False, "Case not supported!"

            # Compose the destination directory
            path_to_dst_rgb = os.path.join(path_to_dst, dst_folder, "rgb")
            path_to_dst_mask = os.path.join(path_to_dst, dst_folder, "mask")
            path_to_dst_aff = os.path.join(path_to_dst, dst_folder, "affordance")

            # Copy RGB in the right destination directory
            src = os.path.join(path_to_src_rgb, folder, filename.replace(".json", ".png"))
            dst = os.path.join(path_to_dst_rgb, filename.replace(".json", ".png"))
            shutil.copyfile(src, dst)

            # Copy mask in the right destination directory
            src = os.path.join(path_to_src_mask, folder, filename.replace(".json", ".png"))
            dst = os.path.join(path_to_dst_mask, filename.replace(".json", ".png"))
            shutil.copyfile(src, dst)

            # Copy affordance in the right destination directory
            src = os.path.join(path_to_src_aff, folder, filename.replace(".json", ".png"))
            dst = os.path.join(path_to_dst_aff, filename.replace(".json", ".png"))
            shutil.copyfile(src, dst)
            print("Copied: ", filename.replace(".json", ".png"))
    print("Finished!")
