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


def save_prediction_overlay_batch_umd(imgs, affs_pred, dest_dir, filename):
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
        aff_pred = cv2.resize(aff_pred, (image_vis.shape[1], image_vis.shape[0]), interpolation=cv2.INTER_NEAREST)
        colormap = np.zeros_like(image_vis)
        if 1 in aff_pred:  # 1 - 'grasp'
            colormap[aff_pred == 1] = np.array([0, 0, 255])
        if 2 in aff_pred:  # 2 - 'cut'
            colormap[aff_pred == 2] = np.array([255, 0, 0])
        if 3 in aff_pred:  # 3 - 'scoop'
            colormap[aff_pred == 3] = np.array([0, 255, 255])
        if 4 in aff_pred:  # 4 - 'contain'
            colormap[aff_pred == 4] = np.array([0, 255, 0])
        if 5 in aff_pred:  # 5 - 'pound'
            colormap[aff_pred == 5] = np.array([215, 80, 128])
        if 6 in aff_pred:  # 6 - 'support'
            colormap[aff_pred == 6] = np.array([255, 255, 0])
        if 7 in aff_pred:  # 7 - 'wrap-grasp'
            colormap[aff_pred == 7] = np.array([50, 127, 205])
        aff_vis = cv2.addWeighted(image_vis.copy(), 0.6, colormap, 1.0, 0.0)
        cv2.imwrite(os.path.join(dest_dir, filename[index]), aff_vis)


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


def visualise_prediction_batch_umd(imgs, affs_pred):
    imgs = imgs.cpu().detach().numpy()
    affs_pred = affs_pred.cpu().detach().numpy()
    if len(affs_pred.shape) == 2:
        affs_pred = np.expand_dims(affs_pred, axis=0)
    if len(imgs.shape) == 3:
        imgs = np.expand_dims(imgs, axis=0)
    for index, img in enumerate(imgs):
        image_vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        image_vis = cv2.resize(image_vis, (image_vis.shape[1] // 2, image_vis.shape[0] // 2))

        # Visualise affordance overlay
        colormap = np.zeros_like(image_vis)
        aff_pred = affs_pred[index].copy()
        aff_pred = cv2.resize(aff_pred, (image_vis.shape[1], image_vis.shape[0]), interpolation=cv2.INTER_NEAREST)
        if 1 in aff_pred:  # 1 - 'grasp'
            colormap[aff_pred == 1] = np.array([0, 0, 255])
        if 2 in aff_pred:  # 2 - 'cut'
            colormap[aff_pred == 2] = np.array([255, 0, 0])
        if 3 in aff_pred:  # 3 - 'scoop'
            colormap[aff_pred == 3] = np.array([0, 255, 255])
        if 4 in aff_pred:  # 4 - 'contain'
            colormap[aff_pred == 4] = np.array([0, 255, 0])
        if 5 in aff_pred:  # 5 - 'pound'
            colormap[aff_pred == 5] = np.array([215, 80, 128])
        if 6 in aff_pred:  # 6 - 'support'
            colormap[aff_pred == 6] = np.array([255, 255, 0])
        if 7 in aff_pred:  # 7 - 'wrap-grasp'
            colormap[aff_pred == 7] = np.array([50, 127, 205])
        aff_vis = cv2.addWeighted(image_vis.copy(), 0.6, colormap, 1.0, 0.0)
        stack_vis = np.hstack([image_vis, aff_vis])

        cv2.imshow("Prediction", stack_vis)


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
