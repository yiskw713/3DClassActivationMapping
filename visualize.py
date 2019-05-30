import torch
import torch.nn.functional as F

import numpy as np
import cv2


def reverse_normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    x[:, 0, :, :] = x[:, 0, :, :] * std[0] + mean[0]
    x[:, 1, :, :] = x[:, 1, :, :] * std[1] + mean[1]
    x[:, 2, :, :] = x[:, 2, :, :] * std[2] + mean[2]
    return x


def visualize(clip, cam):
    """
    Synthesize an image with CAM to make a result image.
    Args:
        clip: (Tensor) shape => (1, 3, T, H, W)
        cam: (Tensor) shape => (1, 1, T, H', W')
    Return:
        synthesized image (Tensor): shape =>(1, 3, T, H, W)
    """

    _, _, T, H, W = clip.shape
    cam = F.interpolate(
        cam, size=(T, H, W), mode='trilinear', align_corners=False)
    cam = 255 * cam.squeeze()
    heatmaps = []
    for t in range(T):
        c = cam[t]
        heatmap = cv2.applyColorMap(np.uint8(c), cv2.COLORMAP_JET)
        heatmap = torch.from_numpy(heatmap.transpose(2, 0, 1))
        heatmap = heatmap.float() / 255
        b, g, r = heatmap.split(1)
        heatmap = torch.cat([r, g, b])
        heatmaps.append(heatmap)

    heatmaps = torch.stack(heatmaps)
    heatmaps = heatmaps.transpose(1, 0).unsqueeze(0)
    result = heatmaps + clip.cpu()
    result = result.div(result.max())

    return result
