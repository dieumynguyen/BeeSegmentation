# General
import os
import sys
import cv2
import glob
import json
import argparse
import numpy as np
import pandas as pd

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks")
plt.rcParams["font.family"] = "Arial"

'''
Given images and corresponding annotations text files, create cropped images and
segmentation masks, with bees drawn as ellipses. 
'''

# Define fixed variables
BEE_OBJECT_SIZES = {1: (20, 35),  # bee class is labeled 1
                    2: (20, 20)}  # butt class is labeled 2

def ellipse_around_point(image, xc, yc, angle, r1, r2, value):

    image_size = image.shape

    ind0 = np.arange(-xc, image_size[0] - xc)[:, np.newaxis] * np.ones((1, image_size[1]))
    ind1 = np.arange(-yc, image_size[1] - yc)[np.newaxis, :] * np.ones((image_size[0], 1))
    ind = np.concatenate([ind0[np.newaxis], ind1[np.newaxis]], axis=0)

    sin_a = np.sin(angle)
    cos_a = np.cos(angle)

    image[((ind[0, :, :] * sin_a + ind[1, :, :] * cos_a) ** 2 / r1 ** 2 + (
            ind[1, :, :] * sin_a - ind[0, :, :] * cos_a) ** 2 / r2 ** 2) <= 1] = value

    return image

def make_crop_mask(cropped_image, df, min_x, min_y):

    cropped_mask = np.zeros(cropped_image.shape[:2])

    for i, row in df.iterrows():
        sys.stdout.write(f'\rProcessing bee: {i+1}/{len(df)}')
        sys.stdout.flush()

        topleft_x = row.x_offset
        topleft_y = row.y_offset
        local_xoffset = row.x_offset - min_x
        local_yoffset = row.y_offset - min_y
        xc = row.x + local_xoffset
        yc = row.y + local_yoffset
        value = row['class']
        angle = row.angle
        r1 = BEE_OBJECT_SIZES[value][0]
        r2 = BEE_OBJECT_SIZES[value][1]

        ellipse_around_point(cropped_mask, yc, xc, angle, r1, r2, value)
    print('\n')
    return cropped_mask

def setup_args():
    parser = argparse.ArgumentParser(description='Preprocess training images')
    parser.add_argument('-r', '--data_root', dest='data_root', type=str, default='data',
                        help='Set root path to data [default: data]')
    parser.add_argument('-o', '--overwrite', dest='overwrite', type=int, default=0,
                        help='Overwrite files [default: 0]')
    args = parser.parse_args()
    return args

def main(args):
    replace = lambda x : x.replace('/frames/', '/frames_txt/').replace('.png', '.txt')

    frames_root = os.path.join(args.data_root, 'frames')
    frame_paths = np.sort(glob.glob(f'{frames_root}/*'))
    txt_paths = [replace(frame_path) for frame_path in frame_paths]

    assert np.all([os.path.exists(txt_path) for txt_path in txt_paths]), 'No'

    for i, (frame_path, txt_path) in enumerate(zip(frame_paths, txt_paths)):
        print(f'Frame {i+1}/{len(frame_paths)}')

        # Check if file exists
        frame_name = frame_path.split('/')[-1][:-4]
        frame_savepath = os.path.join(args.data_root, 'images', f'{frame_name}.npy')
        mask_savepath = os.path.join(args.data_root, 'masks', f'{frame_name}.npy')
        if os.path.exists(frame_savepath) and not args.overwrite:
            print('File exists! Skipping.')
            continue

        # Read image
        img = cv2.imread(frame_path)

        # Read txt into dataframe
        df = pd.read_csv(txt_path, sep='\t', header=None)
        df.columns = ['x_offset', 'y_offset', 'class', 'x', 'y', 'angle']

        # Group offset columns
        df['offset'] = list(zip(df.x_offset, df.y_offset))

        # Get min, max offsets
        max_x = df.x_offset.max()
        min_x = df.x_offset.min()

        max_y = df.y_offset.max()
        min_y = df.y_offset.min()

        # Make cropped image
        cropped_image = img[min_y:max_y, min_x:max_x]

        # Make cropped mask
        cropped_mask = make_crop_mask(cropped_image, df, min_x, min_y)

        # Optimize storage
        # RGB -> grayscale
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)

        # Remove float64 --> go to uint8
        cropped_mask = cropped_mask.astype(np.uint8)

        # Save arrays
        np.save(frame_savepath, cropped_image)
        np.save(mask_savepath, cropped_mask)

if __name__ == '__main__':
    args = setup_args()
    main(args)
