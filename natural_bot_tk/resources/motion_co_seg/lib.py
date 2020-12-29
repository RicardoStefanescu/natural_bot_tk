from random import choice
from string import digits
from copy import copy
import warnings
warnings.filterwarnings("ignore")

import imageio
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
%matplotlib inline

from IPython.display import HTML

import torch
import torch.nn.functional as F
import cv2
import face_recognition

from part_swap import load_checkpoints, face_swap

def attempt_transform(photo_to_transform, mugshot, threshold=0.96):
    ''' Takes two open cv2 pictures and attempts to transform one face into
    the mugshot
    '''

    # Transform pictures to usable data
    source_image = imageio.imread(source_path)
    source_image = resize(source_image, (256, 256)).astype('float32')[..., :3]

    target_image = imageio.imread(target_path)
    target_image = resize(target_image, target_image.shape[:2]).astype('float32')[..., :3]