import os
import cv2
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw
import cv2
import os

def earse_augmentation(data_dir:str) \
    -> None:
    '''
    remove augmented images and labels in train directory
    '''
    folders = ['train']

    for folder in folders:
        image_folder_path = os.path.join(data_dir, folder, 'images')
        label_folder_path = os.path.join(data_dir, folder, 'labels')
        
        for label_file in tqdm(os.listdir(label_folder_path)):
            
            label_path = os.path.join(label_folder_path, label_file)
            image_file = label_file.replace('.txt', '.jpg')
            image_path = os.path.join(image_folder_path, image_file)


            if "_new" in image_path:
                os.remove(image_path); os.remove(label_path)
                if os.path.exists(image_path) or os.path.exists(label_path):
                    print("삭제 안된 증강 데이터가 있습니다.")
                    import pdb;pdb.set_trace()
    print("증강된 데이터 삭제 완료.")

def check_dataset(data_dir:str) \
    -> None:

    folders = ['train']

    images = []
    labels = []

    tik = time.time()
    for folder in folders:
        image_folder_path = os.path.join(data_dir, folder, 'images')
        label_folder_path = os.path.join(data_dir, folder, 'labels')
        
        for label_file in os.listdir(label_folder_path):
            
            label_path = os.path.join(label_folder_path, label_file)
            labels.append(label_path)

        
        for image_file in os.listdir(image_folder_path):
            image_path = os.path.join(image_folder_path, image_file)
            images.append(image_path)

    if len(images) == len(labels):
        print(f"========= check_dataset speaking ================")
        print(f"[images]: {len(images)}")
        print(f"[labels]: {len(labels)}")
        print(f"==============================================")
        print()

    else:
        print('[warning]: number of images and nubmer of labels are different')


if __name__ == "__main__":
    
    ##############################################################################
    # 데이터 root폴더(datasets folder)경로만 잘 지정해줄 것.
    data_root = "/mnt/raid6/aa007878/choi/datasets"
    ##############################################################################
    
    check_dataset(data_root)

    earse_augmentation(data_root)

    check_dataset(data_root)