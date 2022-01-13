import os
import numpy as np
from skimage import morphology
import random
import torch
import matplotlib.pyplot as plt


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(weights, model, device):
    model = model.to(device)
    checkpoint = torch.load(weights, map_location=torch.device(device))
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


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
    inst_masks = morphology.label(prediction)
    return prediction, inst_masks


def plot_samples(model, dataloader, N=4, device="cuda"):

    fig, ax = plt.subplots(N, 3, figsize=(12, 12), sharey=True)

    for k in range(N):
        sample, img_name = next(dataloader)
        sample_img = sample["image"][0]
        prd_mask, inst_masks = predict_sample(model, sample_img, device)
        ax[k][0].imshow(sample_img[0])
        ax[k][1].imshow(sample["mask"][0])
        ax[k][2].imshow(inst_masks)

    ax[0][0].set_title("Input image")
    ax[0][1].set_title("GT mask")
    ax[0][2].set_title("Predicted instance mask")
    fig.tight_layout()
