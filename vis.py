import argparse
import glob
import h5py
import numpy as np
import io
import skimage.transform
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
from matplotlib.pyplot import imshow
from torchvision import models, transforms
from torchvision.utils import save_image
from cam import GradCAM, CAM
from visualize import visualize, reverse_normalize
from model.resnet import resnet50


def get_arguments():
    """
    parse all the arguments from command line inteface
    return a list of parsed arguments
    """

    parser = argparse.ArgumentParser(description="visualization")
    parser.add_argument(
        "--video_dir", type=str, default="./videos", help="path of a config file"
    )
    parser.add_argument(
        "--save_dir", type=str, default="./cams", help="path of a config file"
    )

    return parser.parse_args()


def load_video(path):
    with h5py.File(path, "r") as f:
        video = f["video"]
        clip = []
        n_frames = len(video)
        for i in range(n_frames):
            img = Image.open(io.BytesIO(video[i]))
            img = transforms.functional.center_crop(img, (224, 224))
            img = transforms.functional.to_tensor(img)
            img = transforms.functional.normalize(
                img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            clip.append(img)

    clip = torch.stack(clip)
    clip = clip.unsqueeze(0).transpose(1, 2)

    return clip


def main():
    args = get_arguments()

    model = resnet50()
    state_dict = torch.load(
        "./best_model.prm", map_location=lambda storage, loc: storage
    )
    model.load_state_dict(state_dict)

    target_layer = model.layer4[-1].conv3

    wrapped_model = CAM(model, target_layer)

    model.eval()

    dirs = glob.glob(os.path.join(args.video_dir, "*"))
    print(dirs)
    for d in dirs:
        videos = glob.glob(os.path.join(d, "*.hdf5"))
        print(videos)
        for video in videos:
            with torch.no_grad():
                clip = load_video(video)
                cam = wrapped_model(clip)
                clip = reverse_normalize(clip)
                heatmaps = visualize(clip, cam)

            save_path = video[:-5]
            os.makedirs(save_path)
            for i in range(clip.shape[2]):
                heatmap = heatmaps[:, :, i].squeeze()
                save_image(
                    heatmap, os.path.join(save_path, "{:0>3}.png".format(str(i)))
                )

            print("Done")


if __name__ == "__main__":
    main()
