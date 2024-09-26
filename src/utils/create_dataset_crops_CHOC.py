""" This script performs the window cropping procedure described in "Affordance segmentation of hand-occluded containers from exocentric images", IEEE/CVF ICCVW 2023."""
import argparse
import glob
import numpy as np
import cv2
import os

from tqdm import tqdm


def retrieve_bbox_from_mask(mask):
    """
    :param
    :return: bounding box [xmin, ymin, xmax, ymax]
    """
    # Retrieve box coordinates
    ymin = np.min(np.where(mask > 0)[0])
    xmin = np.min(np.where(mask > 0)[1])
    ymax = np.max(np.where(mask > 0)[0])
    xmax = np.max(np.where(mask > 0)[1])
    bbox = [int(xmin), int(ymin), int(xmax), int(ymax)]
    return bbox


def clip_bbox(bbox, height, width):
    """
        Clip boxes to image boundaries.
    :param bboxes: bboxes not normalized = [xmin, ymin, xmax, ymax]
    :param height: image height
    :param width: image width
    :returns: clipped_bboxes = [xmin, ymin, xmax, ymax]
    """
    # y1 >= 0, x1 >= 0
    ymin = max(min(bbox[1], height), 0)
    xmin = max(min(bbox[0], width), 0)
    # y2 < height, x2 < width
    ymax = max(min(bbox[3], height), 0)
    xmax = max(min(bbox[2], width), 0)
    return [int(xmin), int(ymin), int(xmax), int(ymax)]


def expand_clip_bbox_margin(box, height, width, margin):
    [xmin, ymin, xmax, ymax] = [box[0], box[1], box[2], box[3]]
    [xmin, ymin, xmax, ymax] = [int(xmin - margin), int(ymin - margin), int(xmax + margin), int(ymax + margin)]
    [xmin, ymin, xmax, ymax] = clip_bbox([xmin, ymin, xmax, ymax], height=height, width=width)
    return [xmin, ymin, xmax, ymax]


def resize_bbox(object_bbox, final_height, final_width, img_width, img_height):
    """

    """
    # Whether the object bounding box is smaller than the final dimensions
    smaller = True

    # Compute the size of the bounding box enclosing the object mask
    bbox_width = object_bbox[2] - object_bbox[0]
    bbox_height = object_bbox[3] - object_bbox[1]

    # Compute the center of the bounding box
    center_x = object_bbox[0] + bbox_width // 2
    center_y = object_bbox[1] + bbox_height // 2

    # If the bounding box width is smaller than the final width
    if bbox_width < final_width:
        # Compute the starting x of the window
        start_x = center_x - final_width // 2
        # Compute the ending x of the window
        end_x = center_x + final_width // 2

        # Check the starting and ending point wrt the image dimension
        if start_x < 0 and end_x > img_width:
            # Set the edges of the window as the ones of the image
            start_x = 0
            end_x = img_width

        if start_x < 0:
            # Add the eccess on the ending side
            if (end_x - start_x) <= img_width:
                end_x = end_x - start_x
            else:
                end_x = img_width
            start_x = 0
        if end_x > img_width:
            # Add the eccess on the starting side
            if (start_x - (end_x - img_width)) >= 0:
                start_x = start_x - (end_x - img_width)
            else:
                start_x = 0
            end_x = img_width
    else:
        smaller = False

    # If the bounding box height is smaller than the final height
    if bbox_height < final_height:
        # Compute the starting y of the window
        start_y = center_y - final_height // 2
        # Compute the ending y of the window
        end_y = center_y + final_height // 2

        # Check the starting and ending point wrt the image dimension
        if start_y < 0 and end_y > img_height:
            start_y = 0
            end_y = img_height

        if start_y < 0:
            # Add the eccess on the ending side
            if (end_y - start_y) <= img_height:
                end_y = end_y - start_y
            else:
                end_y = img_height
            start_y = 0

        if end_y > img_height:
            # Add the eccess on the starting side
            if (start_y - (end_y - img_height)) >= 0:
                start_y = start_y - (end_y - img_height)
            else:
                start_y = 0
            end_y = img_height
    else:
        smaller = False

    if not smaller:
        return [0, 0, 0, 0]

    return [start_x, start_y, end_x, end_y]


def combine_mask_affordance(mask, aff_mask):
    """ 0: background
        1: grasp
        2: contain
        3: hand
        """
    final_mask = np.zeros_like(aff_mask)
    final_mask[aff_mask == 1] = 1
    final_mask[aff_mask == 2] = 2
    final_mask[mask == 200] = 3
    return final_mask


def get_args():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default="...")
    parser.add_argument('--visualise', type=bool, default=False)
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--resolution', type=list, default=[480, 480])
    parser.add_argument('--dest_dir', type=str,
                        default="...")
    return parser.parse_args()


if __name__ == '__main__':
    # Load args
    args = get_args()
    path_to_rgb = os.path.join(args.data_dir, "rgb")
    path_to_masks = os.path.join(args.data_dir, "mask")
    path_to_affordances = os.path.join(args.data_dir, "affordance")
    final_dim = args.resolution
    dest_dir = args.dest_dir
    visualise = args.visualise
    save = args.save

    if not os.path.exists(os.path.join(dest_dir, "rgb")):
        os.mkdir(os.path.join(dest_dir, "rgb"))
    if not os.path.exists(os.path.join(dest_dir, "affordance")):
        os.mkdir(os.path.join(dest_dir, "affordance"))

    resized_num = 0
    rgb_list = glob.glob(os.path.join(path_to_rgb, '*.png'))
    rgb_list.sort()
    aff_list = glob.glob(os.path.join(path_to_affordances, '*.png'))
    aff_list.sort()
    mask_list = glob.glob(os.path.join(path_to_masks, '*.png'))
    mask_list.sort()
    for i, _ in enumerate(tqdm(rgb_list)):
        filename = os.path.basename(rgb_list[i])
        image = cv2.imread(rgb_list[i])
        height, width = image.shape[:2]

        mask = cv2.imread(mask_list[i], -1)
        aff_mask = cv2.imread(aff_list[i], -1)

        object_bbox = retrieve_bbox_from_mask(aff_mask)
        object_bbox = expand_clip_bbox_margin(object_bbox, height, width, margin=0)
        resized_bbox = resize_bbox(object_bbox, final_height=final_dim[0], final_width=final_dim[1], img_width=width,
                                   img_height=height)

        seg_mask = combine_mask_affordance(mask, aff_mask)

        colormap = np.zeros_like(image.copy())
        colormap[seg_mask == 1] = np.array([0, 0, 255])
        colormap[seg_mask == 2] = np.array([0, 255, 0])
        colormap[seg_mask == 3] = np.array([255, 255, 255])
        aff_vis = colormap

        if visualise:
            image_vis = image.copy()
            start_point = (object_bbox[0], object_bbox[1])
            end_point = (object_bbox[2], object_bbox[3])
            # Blue color in BGR
            color = (255, 0, 0)
            # Line thickness of 2 px
            thickness = 2
            image_vis = cv2.rectangle(image_vis, start_point, end_point, color, thickness)
            start_point = (resized_bbox[0], resized_bbox[1])
            end_point = (resized_bbox[2], resized_bbox[3])
            color = (0, 0, 255)
            image_vis = cv2.rectangle(image_vis, start_point, end_point, color, thickness)
            cv2.imshow("Object bounding box", image_vis)

            aff_vis = cv2.addWeighted(image.copy(), 0.6, colormap, 1, 0.0)
            if resized_bbox != [0, 0, 0, 0]:
                rgb_crop = image.copy()[resized_bbox[1]:resized_bbox[3], resized_bbox[0]:resized_bbox[2]].astype(
                    np.uint8)
                aff_crop = aff_vis.copy()[resized_bbox[1]:resized_bbox[3], resized_bbox[0]:resized_bbox[2]]
            else:
                rgb_crop = cv2.resize(image.copy()[object_bbox[1]:object_bbox[3], object_bbox[0]:object_bbox[2]].astype(
                    np.uint8), (final_dim[0], final_dim[1]), interpolation=cv2.INTER_LINEAR)
                aff_crop = cv2.resize(
                    aff_vis.copy()[object_bbox[1]:object_bbox[3], object_bbox[0]:object_bbox[2]].astype(
                        np.uint8), (final_dim[0], final_dim[1]), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("RGB crop", rgb_crop)
            cv2.imshow("Aff crop", aff_crop)
            cv2.waitKey(0)
        if save:
            filename = os.path.basename(rgb_list[i])
            if resized_bbox != [0, 0, 0, 0]:
                cv2.imwrite(os.path.join(dest_dir, "rgb", filename),
                            image.copy()[resized_bbox[1]:resized_bbox[3], resized_bbox[0]:resized_bbox[2]].astype(
                                np.uint8))
                cv2.imwrite(os.path.join(dest_dir, "affordance", filename),
                            seg_mask.copy()[resized_bbox[1]:resized_bbox[3], resized_bbox[0]:resized_bbox[2]].astype(
                                np.uint8))
            else:
                resized_num += 1
                cv2.imwrite(os.path.join(dest_dir, "rgb", filename), cv2.resize(
                    image.copy()[object_bbox[1]:object_bbox[3], object_bbox[0]:object_bbox[2]].astype(
                        np.uint8), (final_dim[0], final_dim[1]), interpolation=cv2.INTER_LINEAR))
                cv2.imwrite(os.path.join(dest_dir, "affordance", filename), cv2.resize(
                    seg_mask.copy()[object_bbox[1]:object_bbox[3], object_bbox[0]:object_bbox[2]].astype(
                        np.uint8), (final_dim[0], final_dim[1]), interpolation=cv2.INTER_NEAREST))
    print("Number of resized images: ", resized_num)
    print("Finished!")
