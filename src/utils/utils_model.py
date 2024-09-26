import cv2
import numpy as np
import os


def save_prediction_overlay_batch(imgs, affs_pred, dest_dir, filename):
    """ Saves the color-coded prediction overlayed with the input image in the specified directory and with the specified filename. 

    Args:
        imgs: batch of RGB images (Tensor)
        affs_pred: batch of predictions (Tensor)
        dest_dir: destination directory
        filename: name of the file

    Returns: 

    """
    imgs = imgs.squeeze().cpu().detach().numpy()
    affs_pred = affs_pred.squeeze().cpu().detach().numpy()
    if len(imgs.shape) == 3:
        imgs = np.expand_dims(imgs, axis=0)
        affs_pred = np.expand_dims(affs_pred, axis=0)
    for index, img in enumerate(imgs):
        image_vis = img.copy()
        image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)

        # Save affordance overlay
        aff_pred = affs_pred[index].copy()
        colormap = np.zeros_like(image_vis)
        colormap[aff_pred == 1] = np.array([0, 0, 255])
        colormap[aff_pred == 2] = np.array([0, 255, 0])
        colormap[aff_pred == 3] = np.array([255, 255, 255])
        aff_vis = cv2.addWeighted(image_vis.copy(), 0.6, colormap, 1.0, 0.0)
        cv2.imwrite(os.path.join(dest_dir, filename[index]), aff_vis)


def save_prediction_object_batch(pred, dest_dir, filename):
    """ Saves the color-coded prediction overlayed with the input image in the specified directory and with the specified filename.

    Args:
        imgs: batch of RGB images (Tensor)
        affs_pred: batch of predictions (Tensor)
        dest_dir: destination directory
        filename: name of the file

    Returns:

    """
    pred = pred.squeeze().cpu().detach().numpy()
    if len(pred.shape) == 2:
        pred = np.expand_dims(pred, axis=0)
    for index, p in enumerate(pred):
        # Save prediction
        seg_pred = p.copy()
        colormap = np.zeros([pred.shape[1], pred.shape[2], 3])
        if seg_pred.dtype == int:
            colormap[seg_pred == 1] = np.array([0, 0, 255])
            colormap[seg_pred == 2] = np.array([0, 255, 0])
            colormap[seg_pred == 3] = np.array([255, 255, 255])
        else:
            colormap[:, :, 0] = p * 255
            colormap[:, :, 1] = p * 255
            colormap[:, :, 2] = p * 255
        cv2.imwrite(os.path.join(dest_dir, filename[index]), colormap)


def visualise_prediction_batch(imgs, affs_pred):
    """ Visualises the color-coded prediction overlayed with the input image.

     Args:
        imgs: batch of RGB images (Tensor)
        affs_pred: batch of predictions (Tensor)

    Returns:

    """
    imgs = imgs.cpu().detach().numpy()
    affs_pred = affs_pred.cpu().detach().numpy()
    for index, img in enumerate(imgs):
        image_vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Visualise affordance overlay
        colormap = np.zeros_like(img)
        aff_pred = affs_pred[index].copy()
        colormap[aff_pred == 1] = np.array([0, 0, 255])
        colormap[aff_pred == 2] = np.array([0, 255, 0])
        colormap[aff_pred == 3] = np.array([255, 255, 255])
        aff_vis = cv2.addWeighted(image_vis.copy(), 0.3, colormap, 1.0, 0.0)
        stack_vis = np.hstack([image_vis, aff_vis])

        cv2.imshow("Prediction: {}".format(index), stack_vis)


def visualise_object_batch(imgs, obj_preds, name="Object prediction"):
    """ Visualises the color-coded prediction overlayed with the input image.

     Args:
        imgs: batch of RGB images (Tensor)
        affs_pred: batch of predictions (Tensor)

    Returns:

    """
    imgs = imgs.cpu().detach().numpy()
    obj_preds = obj_preds.cpu().detach().numpy()
    for index, img in enumerate(imgs):
        image_vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Visualise affordance overlay
        colormap = np.zeros_like(img)
        obj_pred = obj_preds[index].copy()
        # obj_pred_uint8 = (obj_pred * 255).astype(np.uint8)
        # colormap = cv2.applyColorMap(obj_pred_uint8, cv2.COLORMAP_JET)
        colormap[obj_pred == 1] = np.array([0, 0, 255])
        colormap[obj_pred == 2] = np.array([0, 255, 0])
        obj_vis = cv2.addWeighted(image_vis.copy(), 0.3, colormap, 1.0, 0.0)
        stack_vis = np.hstack([image_vis, obj_vis])

        cv2.imshow(name + "{}: ".format(index), stack_vis)


def save_prediction_batch(affs_pred, dest_dir, filename):
    """ Saves the prediction in the specified directory and with the specified filename. 

    Args:
        affs_pred: batch of predictions (Tensor)
        dest_dir: destination directory
        filename: name of the file

    Returns:

    """

    # Save affordance prediction
    affs_pred = affs_pred.cpu().detach().numpy()
    if len(affs_pred.shape) == 2:
        affs_pred = np.expand_dims(affs_pred, axis=0)
    for index, aff_pred in enumerate(affs_pred):
        cv2.imwrite(os.path.join(dest_dir, filename[index]), aff_pred)
