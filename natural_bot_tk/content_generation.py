import warnings
warnings.filterwarnings("ignore")
import os

import numpy as np
import cv2
import imageio
from skimage.transform import resize

import torch
import torch.nn.functional as F

import face_recognition

from .resources.motion_co_seg.part_swap import load_checkpoints, face_swap


def find_similar_faces(mugshot_path, target_img_path):
    # Find faces in target
    target_img_cv2 = cv2.cvtColor(cv2.imread(target_img_path), cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(target_img_cv2, model="hog")

    # Make bounding face boxes slightly bigger
    fixed_faces = []
    for (y_0, x_1, y_1, x_0) in face_locations:
        height = abs(y_0 - y_1) 
        width = abs(x_0 - x_1)

        x_0 -= width//2
        y_0 -= height//2

        x_1 += width//2
        y_1 += height//2
        
        fixed_faces += [[y_0,x_1,y_1,x_0]]

    # Get encoding of source img
    source_image_rgb = cv2.cvtColor(cv2.imread(mugshot_path), cv2.COLOR_BGR2RGB)
    source_encoding = face_recognition.face_encodings(source_image_rgb, model='ccn')[0]

    # Get the encoding of each face and compare with the source
    #target_img_cv2 = cv2.cvtColor(cv2.imread(target_img_path), cv2.COLOR_BGR2RGB)
    target_encodings = face_recognition.face_encodings(target_img_cv2, fixed_faces, model="cnn")

    face_distances = face_recognition.face_distance(target_encodings, source_encoding)

    # Add results into arrays [similarity, [position]] and sort
    faces = [[sim, pos] for sim, pos in zip(face_distances, fixed_faces)]
    faces.sort(key=lambda x: x[0], reverse=True)

    return faces

def replace_face(mugshot_path, target_img_path, face_location, gpu=False):
    lib_dir = os.path.dirname(os.path.realpath(__file__))
    try:
        assert os.path.exists(os.path.join(lib_dir, "resources/models/vox-10segments.pth.tar"))
    except Exception as e:
        print(f"{e}\n\n >> Download the models before using this functionality, you can do so by running: natural_bot_tk.download_models()")
    
    # Load modules
    reconstruction_module, segmentation_module = load_checkpoints(config=os.path.join(lib_dir, "resources/motion_co_seg/config/vox-256-sem-10segments.yaml"), 
                                               checkpoint=os.path.join(lib_dir, "resources/models/vox-10segments.pth.tar"),
                                               blend_scale=1,
                                               cpu=not gpu)

    # Load picture
    mugshot_img = imageio.imread(mugshot_path)
    mugshot_img = resize(mugshot_img, (256, 256)).astype('float32')[..., :3]

    target_img = imageio.imread(target_img_path)
    target_img_cv2 = cv2.cvtColor(cv2.imread(target_img_path), cv2.COLOR_BGR2RGB)
    target_img = resize(target_img, target_img.shape[:2]).astype('float32')[..., :3]

    y_0, x_1, y_1, x_0 = face_location
    w = abs(x_1-x_0)
    h = abs(y_1-y_0)

    crop_img = target_img_cv2[y_0:y_1,x_0:x_1]
    crop_img = resize(crop_img, (256, 256))[..., :3]

    # Segment
    parts_to_swap = [1,2,3,4,5,7,9,10]
    predictions = face_swap(swap_index=parts_to_swap, 
                            source_image=mugshot_img, 
                            target_image=crop_img,
                            segmentation_module=segmentation_module, 
                            reconstruction_module=reconstruction_module, 
                            cpu=not gpu)

    # Replace
    result_image = imageio.imread(target_img_path)
    result_image = resize(result_image, result_image.shape[:2])[..., :3]
    mask_image = resize(predictions, (h, w))[..., :3]
    result_image[y_0:y_1, x_0:x_1] = mask_image

    return result_image


def replace_best_face(mugshot_path, target_img_path, threshold=0.9, gpu=False):
    similar_faces = find_similar_faces(mugshot_path, target_img_path)

    if not similar_faces or similar_faces[0][0] < threshold:
        return None

    return replace_face(mugshot_path, target_img_path, similar_faces[0][1], gpu=gpu)