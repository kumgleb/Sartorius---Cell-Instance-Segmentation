import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import morphology


def ins2rle(ins):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    ins = np.array(ins)
    pixels = ins.flatten()
    pad = np.array([0])
    pixels = np.concatenate([pad, pixels, pad])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def mask2rle(mask, cutoff=0, min_object_size=0, keep_large=False):
    """Return run length encoding of mask.
    """
    # segment image and label different objects
    lab_mask = morphology.label(mask > cutoff)

    # Keep only objects that are large enough.
    if keep_large == True:
        (mask_labels, mask_sizes) = np.unique(lab_mask, return_counts=True)
        if (mask_sizes < min_object_size).any():
            mask_labels = mask_labels[mask_sizes < min_object_size]
            for n in mask_labels:
                lab_mask[lab_mask == n] = 0
            lab_mask = morphology.label(lab_mask > cutoff)

    # Loop over each object excluding the background labeled by 0.
    for i in range(1, lab_mask.max() + 1):
        yield ins2rle(lab_mask == i)


def predict_sample(model, sample_img, device):
    slices_x = [0, 148, 296]
    slices_y = [0, 124, 248, 372, 480]
    prediction = torch.zeros((2, 520, 704), device=device)
    for sx in slices_x:
        for sy in slices_y:
            sample_pad = sample_img[:, sx : sx + 224, sy : sy + 224]
            pad_mask = model(sample_pad.unsqueeze(0).type(torch.float32).to(device))
            prediction[:, sx : sx + 224, sy : sy + 224] += pad_mask[0]
    prediction = prediction.argmax(dim=0).detach().cpu().numpy()
    return prediction


@torch.no_grad()
def infer(model, test_loader, device="cuda"):
    pred_ids = []
    pred_strings = []
    val_iter = iter(test_loader)
    for idx in tqdm(range(len(test_loader))):
        sample, img_name = next(val_iter)
        sample_img = sample["image"][0]
        prd_mask = predict_sample(model, sample_img, device)
        # prd = model(sample['image'].unsqueeze(0).type(torch.float32).to(device))
        # prd_mask = prd.argmax(dim=1)[0, ...].detach().cpu().numpy()
        rle = list(mask2rle(prd_mask))
        pred_strings.extend(rle)
        pred_ids.extend([img_name[0]] * len(rle))
    return pred_strings, pred_ids


def rles_to_mask(encs, shape):
    """
    Decodes a rle.

    Args:
        encs (list of str): Rles for each class.
        shape (tuple [2]): Mask size.

    Returns:
        np array [shape]: Mask.
    """
    img = np.zeros(shape[0] * shape[1], dtype=np.uint)
    for m, enc in enumerate(encs):
        if isinstance(enc, np.float) and np.isnan(enc):
            continue
        enc_split = enc.split()
        for i in range(len(enc_split) // 2):
            start = int(enc_split[2 * i]) - 1
            length = int(enc_split[2 * i + 1])
            img[start : start + length] = 1 + m
    return img.reshape(shape)


def compute_iou(labels, y_pred):
    """
    Computes the IoU for instance labels and predictions.

    Args:
        labels (np array): Labels.
        y_pred (np array): predictions

    Returns:
        np array: IoU matrix, of size true_objects x pred_objects.
    """

    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))

    # Compute intersection between all objects
    intersection = np.histogram2d(
        labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects)
    )[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection
    iou = intersection / union

    return iou[1:, 1:]  # exclude background


def precision_at(threshold, iou):
    """
    Computes the precision at a given threshold.

    Args:
        threshold (float): Threshold.
        iou (np array [n_truths x n_preds]): IoU matrix.

    Returns:
        int: Number of true positives,
        int: Number of false positives,
        int: Number of false negatives.
    """
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) >= 1  # Correct objects
    false_negatives = np.sum(matches, axis=1) == 0  # Missed objects
    false_positives = np.sum(matches, axis=0) == 0  # Extra objects
    tp, fp, fn = (
        np.sum(true_positives),
        np.sum(false_positives),
        np.sum(false_negatives),
    )
    return tp, fp, fn


def iou_map(truths, preds, verbose=0):
    """
    Computes the metric for the competition.
    Masks contain the segmented pixels where each object has one value associated,
    and 0 is the background.

    Args:
        truths (list of masks): Ground truths.
        preds (list of masks): Predictions.
        verbose (int, optional): Whether to print infos. Defaults to 0.

    Returns:
        float: mAP.
    """
    ious = [compute_iou(truth, pred) for truth, pred in zip(truths, preds)]

    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        tps, fps, fns = 0, 0, 0
        for iou in ious:
            tp, fp, fn = precision_at(t, iou)
            tps += tp
            fps += fp
            fns += fn

        p = tps / (tps + fps + fns)
        prec.append(p)

    return np.mean(prec)


def score_val_sets(pred_df, val_df_gt):

    scores = []
    shape = (520, 704)

    for i in tqdm(range(len(val_df_gt))):
        rles = val_df_gt["annotation"][i]
        masks_gt = rles_to_mask(rles, shape).astype(np.uint32)
        rles = pred_df["predicted"][i]
        masks_prd = rles_to_mask(rles, shape).astype(np.uint32)
        score = iou_map(masks_gt, masks_prd)
        scores.append(score)

    return np.mean(scores)


def score_model(model, val_dataset, gt_df):

    pred_strings, pred_paths = infer(model, val_dataset)
    ids = list(map(str, pred_paths))
    pred_df = pd.DataFrame({"id": ids, "predicted": pred_strings})
    pred_df = pred_df.groupby("id").agg(list).reset_index()

    score = score_val_sets(pred_df, gt_df)

    return score
