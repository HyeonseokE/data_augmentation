import os
import shutil
import random

def create_val_set(train_folder):
    images_folder = os.path.join(train_folder, 'images')
    labels_folder = os.path.join(train_folder, 'labels')
    # train의 상위 폴더에 val 폴더 생성
    val_folder = os.path.join(os.path.dirname(train_folder), 'val')  

    # val 폴더 및 하위 폴더 생성
    os.makedirs(os.path.join(val_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(val_folder, 'labels'), exist_ok=True)

    # 이미지 파일 리스트 가져오기
    images = os.listdir(images_folder)

    # 랜덤 샘플링
    sampled_images = images[:10]

    # 파일 이동
    for image in sampled_images:
        label_file = image.replace('.jpg', '.txt')  # assuming labels are in .txt format
        shutil.move(os.path.join(images_folder, image), os.path.join(val_folder, 'images', image))
        shutil.move(os.path.join(labels_folder, label_file), os.path.join(val_folder, 'labels', label_file))

if __name__ == "__main__":
	# ===================================================
	# 정호야!!! 아래 train폴더의 절대경로를 넣어줘야해!
	# ex) /mnt/raid6/aa007878/choi/datasets/train
	train_folder_dir = "/mnt/raid6/aa007878/choi/datasets/train"
	# ===================================================
	# 함수 호출
	create_val_set(train_folder_dir)