import torch
import numpy as np
import cv2

class RandomCropper():
    def __init__(self, crop_dim):
        self.crop_dim = crop_dim

    def __call__(self, data):
        img, mask = data
        img_shape = img.shape

        x1 = np.random.randint(0, img_shape[1]-self.crop_dim)
        y1 = np.random.randint(0, img_shape[0]-self.crop_dim)

        x2 = x1 + self.crop_dim
        y2 = y1 + self.crop_dim

        img = img[y1:y2, x1:x2]
        mask = mask[y1:y2, x1:x2]

        return (img, mask)

class LRFlipper():
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, data):
        img, mask = data

        draw = np.random.uniform()
        if draw < self.prob:
            img = np.fliplr(img)
            mask = np.fliplr(mask)

        return (img, mask)

class Rotator():
    def __init__(self, prob=0.75):
        self.prob = prob

    def __call__(self, data):
        img, mask = data

        draw = np.random.uniform()
        if draw < self.prob:
            rand_k = np.random.randint(1,4)
            img = np.rot90(img, k=rand_k)
            mask = np.rot90(mask, k=rand_k)
        return (img, mask)

class ToTensor():
    def __call__(self, data):
        img, mask = data

        img = np.ascontiguousarray(img)
        mask = np.ascontiguousarray(mask)

        img = torch.tensor(img).unsqueeze(dim=0)
        mask = torch.tensor(mask).unsqueeze(dim=0)
        return (img, mask)
    
class Normalizer():
    def __call__(self, data):
        img, mask = data
        img = img.float()
        # img = (img - img.min()) / (img.max() - img.min())
        img = (img - 0) / (255 - 0)
        img = img * 2 - 1
        return (img, mask)
    
class Resize():
    def __init__(self, resize_dim):
        self.resize_dim = resize_dim
        
    def __call__(self, data):
        img, mask = data
        img = cv2.resize(img, self.resize_dim)
        mask = cv2.resize(mask, self.resize_dim)
        return (img, mask)
