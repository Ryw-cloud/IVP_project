
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import PIL

import torch.utils.data as data
from PIL import Image
import os
import os.path
import random
import sys
import numpy as np
import torch
import cv2
import torch.nn.functional as F

def _make_dataset(dir):
    framesPath = []

    # Find and loop over all the clips in root `dir`.
    for index, folder in enumerate(os.listdir(dir)):
        clipsFolderPath = os.path.join(dir, folder)

        # Skip items which are not folders.
        if not (os.path.isdir(clipsFolderPath)):
            continue
        framesPath.append([])

        for image in sorted(os.listdir(clipsFolderPath)):
            # Add path to list.
            framesPath[index].append(os.path.join(clipsFolderPath, image))

    return framesPath



def _pil_loader(path, cropArea=None, resizeDim=None, frameFlip=0):
    transform = transforms.Compose([
                transforms.ToTensor()
    ])
    with open(path, 'rb') as f:
        img = Image.open(f)
        # Resize image if specified.
        resized_img = img.resize(resizeDim, Image.ANTIALIAS) if (resizeDim != None) else img
        # cv2.imwrite(resize)
        # Crop image
        cropped_img = resized_img
        # Flip image horizontally if specified.
        flipped_img = cropped_img.transpose(Image.FLIP_LEFT_RIGHT) if frameFlip else cropped_img
        ret = transform(flipped_img.convert('RGB'))
        #print("ret size", ret.shape)


        return transform(flipped_img.convert('RGB'))


    
    
class AniTriplet(data.Dataset):
    def __init__(self, root, transform=None, resizeSize=(448, 256), randomCropSize=(352, 352), train=True, shift=0):
        # Populate the list with image paths for all the
        # frame in `root`.
        framesPath = _make_dataset(root)
        # Raise error if no images found in root.
        if len(framesPath) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"))
                
        self.randomCropSize = randomCropSize
        self.cropX0         = resizeSize[0] - randomCropSize[0]
        self.cropY0         = resizeSize[1] - randomCropSize[1]
        self.root           = root
        self.transform      = transform
        self.train          = train
        self.resizeSize     = resizeSize
        self.framesPath     = framesPath
        self.shift          = shift

    def __getitem__(self, index):
        first_last = []
        mid = []
        inter = None
        cropArea = []
        shifts = []
        
        if (self.train):
            ### Data Augmentation ###
            # To select random 9 frames from 12 frames in a clip
            firstFrame = 0
            # Apply random crop on the 9 input frames

            
            shiftX = random.randint(0, self.shift)//2 * 2
            shiftY = random.randint(0, self.shift)//2 * 2
            shiftX = shiftX * -1 if random.randint(0, 1) > 0 else shiftX
            shiftY = shiftY * -1 if random.randint(0, 1) > 0 else shiftY
    

            
            shifts.append((shiftX, shiftY))
            shifts.append((-shiftX, -shiftY))

            inter = 1
            reverse = random.randint(0, 1)
            if reverse:
                frameRange = [2, 1, 0]
                inter = 1

            else:
                frameRange = [0, 1, 2]
            randomFrameFlip = random.randint(0, 1)

        else:
            frameRange = [0, 1, 2]
            randomFrameFlip = 0
            inter = 1
            shifts = [(0, 0), (0, 0)]
        
        # Loop over for all frames corresponding to the `index`.
        for frameIndex in frameRange:
            # Open image using pil and augment the image.

            image = _pil_loader(self.framesPath[index][frameIndex], cropArea=None,  resizeDim=self.resizeSize, frameFlip=randomFrameFlip)
            if frameIndex == 1:
                mid.append(image)
            else:
                first_last.append(image)
        sample = {'first_last_frames': first_last, 'middle_frame': mid[0]}

        return sample

    def __len__(self):
        return len(self.framesPath)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class AnimeDataSet(Dataset):
    def __init__(self, video_dir, text_split, transform=None):
        """
        Dataset class for the Vimeo-90k dataset, available at http://toflow.csail.mit.edu/.

        Args:
            video_dir (string): Vimeo-90k sequences directory.
            text_split (string): Text file path in the Vimeo-90k folder, either `tri_trainlist.txt` or `tri_testlist.txt`.
            transform (callable, optional): Optional transform to be applied samples.
        """
        self.video_dir = video_dir
        self.text_split = text_split
        # default transform as per RRIN, convert images to tensors, with values between 0 and 1
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        self.middle_frame = []
        self.first_last_frames = []

        # open the given text file path that gives file names for train or test subsets
        with open(self.text_split, 'r') as f:
            filenames = f.readlines()
            f.close()
        final_filenames = []
        for i in filenames:
            final_filenames.append(os.path.join(self.video_dir, i.split('\n')[0]))

        for length in range(0, len(final_filenames) - 2, 3):
            try:
                frames = [final_filenames[length], final_filenames[length + 1], final_filenames[length + 2]]
                print(frames)
                #frames = [os.path.join(f, i) for i in os.listdir(f)]
            except:
                continue
            # make sure images are in order, i.e. im1.png, im2.png, im3.png
            frames = sorted(frames)
            # make sure there are only 3 images in the Vimeo-90k triplet's folder for it to be a valid dataset sample
            if len(frames) == 3:
                self.first_last_frames.append([frames[0], frames[2]])
                self.middle_frame.append(frames[1])

    def __len__(self):
        return len(self.first_last_frames)

    def __getitem__(self, idx):
        first_last = [PIL.Image.open(self.first_last_frames[idx][0]).convert("RGB").resize((256, 448)), PIL.Image.open(self.first_last_frames[idx][1]).convert("RGB").resize((256, 448))]
        mid = PIL.Image.open(self.middle_frame[idx]).convert("RGB").resize((256, 448))

        if self.transform:
            first_last = [self.transform(first_last[0]), self.transform(first_last[1])]
            mid = self.transform(mid)

        sample = {'first_last_frames': first_last, 'middle_frame': mid}

        return sample