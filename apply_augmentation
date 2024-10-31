import os
import cv2
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw
import cv2
import os


# ped_go 2구 (class: 7)
def ped_2_go2noSign(image_path, label_path):
    
    # 이미지 로드
    image = Image.open(image_path)
    img_w, img_h = image.size
    
    # 라벨 파일 읽기
    with open(label_path, "r") as file:
        labels = file.readlines()
        
    # 새로운 라벨 데이터 저장용 리스트
    new_labels = []

    # 바운딩박스 처리
    for label in labels:
        # 라벨 데이터를 공백으로 분리하여 클래스와 바운딩박스 정보 추출
        data = label.strip().split()
        class_id = int(data[0])
        x_center, y_center, width, height = map(float, data[1:])
        
        # 클래스가 7인 바운딩박스를 찾기
        if class_id == 7:

            # original scale로 변환
            box_w, box_h = int(width * img_w), int(height * img_h)

            # original scale 좌상단 좌표
            x1, y1 = int((x_center - width / 2) * img_w), int((y_center - height / 2) * img_h)

            # original scale 우하단 좌표
            x2, y2 = x1 + box_w, y1 + box_h
            
            # numpy 배열로 변환
            image = np.array(image)

            # RGB를 BGR로 변환
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 바운딩박스의 위쪽 1/2 영역 추출
            top_half = image[y1: y1 + box_h // 2, x1:x1 + box_w]

            # 아래쪽 1/2 영역에 위쪽 1/2 덮어쓰기
            # 아래쪽 영역의 높이 계산
            lower_half_height = y2 - (y1 + box_h // 2)
            top_half_height = top_half.shape[0]

            # 두 높이 중 작은 것만큼 잘라서 사용
            height_to_use = min(lower_half_height, top_half_height)
            try:
                image[y1 + box_h // 2:y1 + box_h // 2 + height_to_use, x1:x2] = top_half[:height_to_use, :]
            except:
                import pdb;pdb.set_trace()

            # 클래스 7을 8로 변경
            class_id = 8

        # 새로운 라벨 데이터에 추가 (클래스 변경된 상태로)
        new_labels.append(f"{class_id} {x_center} {y_center} {width} {height}\n")

    # 원본 파일명에서 "_new" 붙인 새 파일명 생성
    image_output_path = os.path.splitext(image_path)[0] + "_new.jpg"
    label_output_path = os.path.splitext(label_path)[0] + "_new.txt"

    # 수정된 이미지 저장
    val = cv2.imwrite(image_output_path, image)
    if not val:
        import pdb;pdb.set_trace()
    # 수정된 라벨 파일
    with open(label_output_path, "w") as file:
        file.writelines(new_labels)


# ped_go 3구 (class: 7)
def ped_3_go2noSign(image_path, label_path):
    # 이미지 로드
    image = Image.open(image_path)
    img_w, img_h = image.size

    # 라벨 파일 읽기
    with open(label_path, "r") as file:
        labels = file.readlines()

    # 새로운 라벨 데이터 저장용 리스트
    new_labels = []

    # 바운딩박스 처리
    for label in labels:
        # 라벨 데이터를 공백으로 분리하여 클래스와 바운딩박스 정보 추출
        data = label.strip().split()
        class_id = int(data[0])
        x_center, y_center, width, height = map(float, data[1:])
        
        # 클래스가 10인 바운딩박스를 찾기
        if class_id == 7:

            box_w, box_h = int(width * img_w), int(height * img_h)
            x1, y1 = int((x_center - width / 2) * img_w), int((y_center - height / 2) * img_h)
            x2, y2 = x1 + box_w, y1 + box_h

            # 세로로 3등분한 각 영역의 높이 계산
            third_h = box_h // 3
            top_third_y = y1
            middle_third_y = y1 + third_h
            bottom_third_y = y1 + 2 * third_h

            # numpy 배열로 변환
            image = np.array(image)

            # RGB를 BGR로 변환
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 위쪽 1/3 영역 추출
            top_third = image[top_third_y:middle_third_y, x1:x2]

            # 위쪽 1/3 영역을 중간과 아래 1/3 영역 크기에 맞게 재조정 후 덮어쓰기
            middle_third_resized = cv2.resize(top_third, (x2 - x1, third_h))
            bottom_third_resized = cv2.resize(top_third, (x2 - x1, y2 - bottom_third_y))

            # 덮어쓰기
            image[middle_third_y:bottom_third_y, x1:x2] = middle_third_resized
            image[bottom_third_y:y2, x1:x2] = bottom_third_resized

            # 클래스 7을 8로 변경
            class_id = 8

        # 새로운 라벨 데이터에 추가 (클래스 변경된 상태로)
        new_labels.append(f"{class_id} {x_center} {y_center} {width} {height}\n")

    # 원본 파일명에서 "_new" 붙인 새 파일명 생성
    image_output_path = os.path.splitext(image_path)[0] + "_new.jpg"
    label_output_path = os.path.splitext(label_path)[0] + "_new.txt"

    # 수정된 이미지 저장
    cv2.imwrite(image_output_path, image)

    # 수정된 라벨 파일
    with open(label_output_path, "w") as file:
        file.writelines(new_labels)



# bus_go 3구 (class: 10)
def bus_3_go2noSign(image_path, label_path):
    # 이미지 로드
    image = Image.open(image_path)
    img_w, img_h = image.size

    # 라벨 파일 읽기
    with open(label_path, "r") as file:
        labels = file.readlines()

    # 새로운 라벨 데이터 저장용 리스트
    new_labels = []

    # 바운딩박스 처리
    for label in labels:
        # 라벨 데이터를 공백으로 분리하여 클래스와 바운딩박스 정보 추출
        data = label.strip().split()
        class_id = int(data[0])
        x_center, y_center, width, height = map(float, data[1:])
        
        # 클래스가 10인 바운딩박스를 찾기
        if class_id == 10:

            box_w, box_h = int(width * img_w), int(height * img_h)
            x1, y1 = int((x_center - width / 2) * img_w), int((y_center - height / 2) * img_h)
            x2, y2 = x1 + box_w, y1 + box_h

            # 중간 1/3과 오른쪽 1/3 영역 좌표 계산
            middle_third_start = x1 + box_w // 3
            middle_third_end = x1 + 2 * (box_w // 3)
            right_third = x2 - box_w // 3

            # numpy 배열로 변환
            image = np.array(image)

            # RGB를 BGR로 변환
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 중간 1/3 영역을 오른쪽 1/3에 덮어쓰기
            image[y1:y2, right_third:x2] = image[y1:y2, middle_third_start:middle_third_end]

            # 클래스 10을 11로 변경
            class_id = 11

        # 새로운 라벨 데이터에 추가 (클래스 변경된 상태로)
        new_labels.append(f"{class_id} {x_center} {y_center} {width} {height}\n")

    # 원본 파일명에서 "_new" 붙인 새 파일명 생성
    image_output_path = os.path.splitext(image_path)[0] + "_new.jpg"
    label_output_path = os.path.splitext(label_path)[0] + "_new.txt"
    
    # 수정된 이미지 저장
    cv2.imwrite(image_output_path, image)

    # 수정된 라벨 파일
    with open(label_output_path, "w") as file:
        file.writelines(new_labels)

def augmentation(data_dir:str) \
    -> None:
    '''
    1. find bbox_ratio for entire images in train_folder
    2. make list of data formated below (for entire train set)
    data = {"image_id": image_id,
            "category": current_class_id,
            "aspect_ratio": round(aspect_ratio, 2),
            "label_path": label_path,
            "image_path":image_path}
    '''
    folders = ['train']
    count_images = 0

    # aspect ratio
    car_traffic_infos = []
    ped_traffic_infos = []
    bus_traffic_infos = []


    # data type = {image_id: ratio}
    tik = time.time()
    for folder in folders:
        image_folder_path = os.path.join(data_dir, folder, 'images')
        label_folder_path = os.path.join(data_dir, folder, 'labels')
        
        for label_file in tqdm(os.listdir(label_folder_path)):
            
            label_path = os.path.join(label_folder_path, label_file)
            image_file = label_file.replace('.txt', '.jpg')
            image_path = os.path.join(image_folder_path, image_file)

            # already augmentation has been applied
            try:
                width, height = Image.open(image_path).size
            except:
                if "_new" in image_path:
                    continue
            
            # 라벨 파일 읽기
            with open(label_path, 'r') as file:
                count_images += 1
                image_id = label_path.split('\\')[-1].split(".")[0]
                lines = file.readlines()

                for line in lines:
                    elements = line.strip().split()
                    current_class_id = int(elements[0])
                    x_center, y_center, box_width, box_height = map(float, elements[1:])
                    original_width = box_width*width
                    original_height = box_height*height
                    aspect_ratio = original_height/original_width

                    data = {"image_id": image_id,
                            "category": current_class_id,
                            "aspect_ratio": round(aspect_ratio, 2),
                            "label_path": label_path,
                            "image_path":image_path}
                    
                    # "car" category
                    if current_class_id in [0,1,2,3,4,5,6]:
                        car_traffic_infos.append(data)
                    # "pedestrian" category
                    elif current_class_id in [7,8,9]:
                        ped_traffic_infos.append(data)
                    # "bus" category
                    elif current_class_id in [10,11,12,13]:
                        bus_traffic_infos.append(data)

    print(f"========= find_ratio speaking ================")
    print()
    print("[before augmentation]")
    print(f"[find_ratio running time]: {time.time() - tik}s")
    print(f"[total number of images]: {count_images}장")
    print(f"[total number of car object]: {len(car_traffic_infos)}개") 
    print(f"[total number of ped object]: {len(ped_traffic_infos)}개") 
    print(f"[total number of bus object]: {len(bus_traffic_infos)}개")
    print(f"==============================================")
    print()

    cnt_ped2, cnt_ped3, cnt_bus3 = apply_augmentation(car_traffic_infos,
                       ped_traffic_infos,
                       bus_traffic_infos)
    
    # confirm
    after_count_images = 0
    for folder in folders:
        image_folder_path = os.path.join(data_dir, folder, 'images')
        label_folder_path = os.path.join(data_dir, folder, 'labels')
        
        for label_file in tqdm(os.listdir(label_folder_path)):
            after_count_images += 1

    print(f"========= find_ratio speaking ================")
    print("[after augmentation]")
    print(f"[total number of images]: {after_count_images}장")
    print(f"[total number of car object]: {len(car_traffic_infos)}개") 
    print(f"[total number of ped object]: {len(ped_traffic_infos)+cnt_ped2+cnt_ped3}개") 
    print(f"[total number of bus object]: {len(bus_traffic_infos)+cnt_bus3}개")
    print(f"==============================================")
    print()
    
   


def apply_augmentation(car_traffic_infos:list, 
                       ped_traffic_infos:list, 
                       bus_traffic_infos:list) \
-> int:
    '''
    apply custom augmentation
    target : ped2_go -> ped2_noSign
             ped3_go -> ped3_noSign
             bus3_go -> bus3_noSign
    [functions] 
    1. save augmented_label.txt file in labels folder
    2. save augmented_image.jpg file in images folder
    3. print number of objects that had been applied custom augmentation
    '''    
    cnt_ped2 = 0
    cnt_ped3 = 0
    cnt_bus3 = 0
    count_augmented_images = 0

    for info in ped_traffic_infos:
        if info['category'] == 7:
            if info["aspect_ratio"] <= 2.0:
                ped_2_go2noSign(info["image_path"], info["label_path"])
                cnt_ped2 += 1
            elif info["aspect_ratio"] >= 2.5:
                ped_3_go2noSign(info["image_path"], info["label_path"])
                cnt_ped3 += 1
    for info in bus_traffic_infos:
        if info['category'] == 10:            
            # if bus_traffic_ratios[k]["aspect_ratio"]:
            bus_3_go2noSign(info["image_path"], info["label_path"])
            cnt_bus3 += 1        
    
    print(f"========= apply_augmentation speaking ========")
    print("augmentation이 적용된 객체 수를 출력합니다.")
    print(f"[ped_2]: {cnt_ped2}개, [ped_3]: {cnt_ped3}개, [bus_3]: {cnt_bus3}개")
    print(f"==============================================")
    print()

    return cnt_ped2, cnt_ped3, cnt_bus3

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

    # 데이터 root폴더(datasets folder)경로만 잘 지정해줄 것.
    ########################################################################
    data_root = "C:\\Users\\H\\Desktop\\kakao_project\\datasets\\tld_db"
    ########################################################################

    # check before augmentation
    check_dataset(data_root)

    # applay
    augmentation(data_root)

    # check after augmentation
    check_dataset(data_root)
    
