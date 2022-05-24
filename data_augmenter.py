import albumentations as A
import cv2
import argparse
import numpy as np
import shutil
import math
import os
from matplotlib import pyplot as plt
import random
from albumentations.pytorch import ToTensorV2
import random
import time
import datetime


train_folder_original = os.path.join(".","tmp", "train_set")
train_folder_final = os.path.join(".","dataset", "train")

final_image_w = 416
images_per_class = 4000

def get_files_from_folder(path):

    files = os.listdir(path)
    return np.asarray(files)

def get_pictures_of_class(targetClass):
    path_to_original = os.path.join(train_folder_original, targetClass)
    classPath = path_to_original
    return classPath, get_files_from_folder(path_to_original)


def get_random_square_subpicture(original_image, final_width:int):
    input_h, input_w = original_image.shape[:2]

    normalized_min_width = 1
    minimum_width = min(input_h*normalized_min_width, input_w*normalized_min_width)
    minimum_width = max(math.ceil(minimum_width), final_width)
    random.seed(time.time())

    new_image_w = random.randint(minimum_width,input_w)
    new_image_h = random.randint(minimum_width,input_h)
    square_width = min(new_image_h,new_image_w)
    new_image_starting_w = random.randint(0,input_w - square_width)
    new_image_starting_h = random.randint(0,input_h - square_width)
    image = original_image[new_image_starting_h:new_image_starting_h+square_width, new_image_starting_w:new_image_starting_w+square_width,:]
    
    new_dim = (final_width, final_width)
    # resize image
    resized = cv2.resize(image, new_dim, interpolation = cv2.INTER_AREA)
    return resized

def augument_pictures(class_to_augment, requested_images:int):
    classPath, pictures = get_pictures_of_class(class_to_augment)
    current_number = len(pictures)
    diff = current_number - requested_images
    
    if diff > 0:
        image_indexes = np.random.choice(range(current_number), size=requested_images,replace=False)

    if(diff < 0):
        #so if there are less images then how many we want
        #we have to add new ones.
        print(f"adding {-diff} images..")
        initial_images = np.array(range(current_number),dtype=int)
        to_add_indexes = np.random.choice(range(current_number), size=-diff,replace=True)
        image_indexes = np.concatenate((initial_images,to_add_indexes), axis=0)

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.RandomContrast(p=0.4),
        A.Blur(blur_limit=3),
        A.OpticalDistortion(),
        A.GridDistortion(),
        ])

    print(f"Modifying {len(image_indexes)} images..")
    saving_index = 0
    for i in image_indexes:
        imageName = os.path.join(classPath,pictures[i])
        image = cv2.imread(imageName)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = get_random_square_subpicture(image,final_image_w)

        image = transform(image=image)["image"]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(train_folder_final,class_to_augment, f"{saving_index}.jpg"), image)
        saving_index += 1



if os.path.exists(train_folder_final):
        shutil.rmtree(train_folder_final)	
os.makedirs(train_folder_final)

startTime = datetime.datetime.now()
_, classes, _ = next(os.walk(train_folder_original))
for i in range(len(classes)):
    print(f"Working on class: {classes[i]}")
    classPath, pictures = get_pictures_of_class(classes[i])
    os.makedirs(os.path.join(train_folder_final,classes[i]))
    augument_pictures(classes[i],images_per_class)
print(f"Total time: {datetime.datetime.now() - startTime}")