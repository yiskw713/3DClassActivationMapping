import argparse
import cv2
import glob
import os
import shutil
import tqdm


def get_arguments():
    '''
    parse all the arguments from command line inteface
    return a list of parsed arguments
    '''

    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument(
        'cam_dir', type=str, help='path to a directory where cams are saved.')
    parser.add_argument(
        '--keep_orig', action='store_true',
        help='Add --keep_orig option if you keep original .png files.')

    return parser.parse_args()


def main():
    args = get_arguments()

    frame_paths = glob.glob(
        os.path.join(args.cam_dir, "*.png")
    )
    frame_paths = sorted(frame_paths)

    img = cv2.imread(frame_paths[0])
    h, w = img.shape[:2]

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(args.cam_dir + '.mp4', fourcc, 15.0, (w, h))
    video.write(img)

    for frame_path in frame_paths[1:]:
        img = cv2.imread(frame_path)
        video.write(img)

    video.release()

    if not args.keep_orig:
        shutil.rmtree(args.cam_dir)

    print("Done.")


if __name__ == "__main__":
    main()
