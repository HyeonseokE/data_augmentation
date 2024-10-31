import os
import shutil
import random

import os
import shutil

def move_val_to_train(val_folder):
    images_folder = os.path.join(val_folder, 'images')
    labels_folder = os.path.join(val_folder, 'labels')
    
    # train 폴더 경로 설정 (val 폴더의 상위 폴더를 기준으로)
    train_folder = os.path.join(os.path.dirname(val_folder), 'train')
    train_images_folder = os.path.join(train_folder, 'images')
    train_labels_folder = os.path.join(train_folder, 'labels')

    # train 폴더와 하위 폴더가 없는 경우 생성
    os.makedirs(train_images_folder, exist_ok=True)
    os.makedirs(train_labels_folder, exist_ok=True)

    # val 폴더의 이미지와 라벨 파일 리스트 가져오기
    images = os.listdir(images_folder)
    labels = os.listdir(labels_folder)

    # 파일 이동
    for image in images:
        shutil.move(os.path.join(images_folder, image), os.path.join(train_images_folder, image))

    for label in labels:
        shutil.move(os.path.join(labels_folder, label), os.path.join(train_labels_folder, label))

    # 이동이 완료되면 val 폴더의 images와 labels 폴더 삭제
    shutil.rmtree(val_folder)

if __name__ == "__main__":
    # ===================================================
    # 아래 val 폴더의 절대 경로를 입력해야 합니다.
    # 예) /mnt/raid6/aa007878/choi/datasets/val
    val_folder_dir = "/mnt/raid6/aa007878/choi/datasets/val"
    # ===================================================
    # 함수 호출
    move_val_to_train(val_folder_dir)