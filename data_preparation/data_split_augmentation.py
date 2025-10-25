import os
import random
import shutil
import cv2
import albumentations as A
import yaml
from tqdm import tqdm

# Paths
src_folder = r"D:\HocTapUTC\DATN\Detect_mu_bao_hiem\datasets_doi_mu_bao_hiem"
# dst_folder = r"C:\Users\admin\PycharmProjects\Hoc_DL_CV\DATN_Demo_Chat_BienBao\dataset_yolov8n_mubaohiem"
dst_folder = r"D:\HocTapUTC\DATN\Detect_mu_bao_hiem\datasets_doi_mu_bao_hiem_new"

# Parameters
train_ratio = 0.7
valid_ratio = 0.2
img_size = 640

for split in ['train', 'valid', 'test']:
    os.makedirs(os.path.join(dst_folder, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(dst_folder, split, 'labels'), exist_ok=True)

with open(os.path.join(src_folder, 'classes_vie.txt'), 'r', encoding='utf-8') as f:
    classes = [line.strip() for line in f.readlines()]

all_images = [os.path.join(src_folder, 'images', f) for f in os.listdir(os.path.join(src_folder, 'images')) if
              f.endswith('.jpg')]
random.shuffle(all_images)

# Split dataset
train_size = int(len(all_images) * train_ratio)
valid_size = int(len(all_images) * valid_ratio)

train_images = all_images[:train_size]
valid_images = all_images[train_size:train_size + valid_size]
test_images = all_images[train_size + valid_size:]

# Comprehensive augmentation pipeline
augmentations = A.Compose([
    A.Resize(640, 640),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.05),
    A.RandomRotate90(p=0.1),
    A.Rotate(limit=20, p=0.6),
    A.Affine(shear=10, p=0.4),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.RandomBrightnessContrast(p=0.4),
    A.RandomGamma(p=0.2),
    A.GaussNoise(p=0.2),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
], bbox_params=A.BboxParams(format='yolo'))


def process_and_save(images, split):
    for img_path in tqdm(images, desc=f'Processing {split}'):
        filename = os.path.basename(img_path)
        label_filename = filename.replace('.jpg', '.txt')

        # Read image and labels
        image = cv2.imread(img_path)
        h, w = image.shape[:2]

        label_path = os.path.join(src_folder, 'labels', label_filename)
        with open(label_path, 'r') as f:
            labels = [list(map(float, line.split())) for line in f.readlines()]

        bboxes = [label[1:] + [int(label[0])] for label in labels]  # bbox + class_id

        invalid_bbox = False
        for bbox in bboxes:
            x_center, y_center, bbox_w, bbox_h, _ = bbox
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= bbox_w <= 1 and 0 <= bbox_h <= 1):
                invalid_bbox = True
                break

        if invalid_bbox:
            print(f"Skipping {filename} due to invalid bbox values.")
            continue

        try:
            if split == 'train':
                original_img_resized = cv2.resize(image, (640, 640))
                cv2.imwrite(os.path.join(dst_folder, split, 'images', 'original_' + filename), original_img_resized)
                with open(os.path.join(dst_folder, split, 'labels', 'original_' + label_filename), 'w') as f:
                    for bbox in bboxes:
                        cls_id = bbox[-1]
                        bbox_coords = bbox[:4]
                        f.write(f"{cls_id} {' '.join(map(str, bbox_coords))}\n")

                augmented = augmentations(image=image, bboxes=bboxes)
                img_processed, bboxes_processed = augmented['image'], augmented['bboxes']
            else:
                img_processed = cv2.resize(image, (640, 640))
                bboxes_processed = bboxes
        except ValueError as e:
            print(f"Skipping {filename} due to augmentation error: {e}")
            continue

        cv2.imwrite(os.path.join(dst_folder, split, 'images', filename), img_processed)

        with open(os.path.join(dst_folder, split, 'labels', label_filename), 'w') as f:
            for bbox in bboxes_processed:
                cls_id = bbox[-1]
                bbox_coords = bbox[:4]
                f.write(f"{cls_id} {' '.join(map(str, bbox_coords))}\n")



process_and_save(train_images, 'train')
process_and_save(valid_images, 'valid')
process_and_save(test_images, 'test')

# dataset.yaml
dataset_yaml = {
    'path': os.path.abspath(dst_folder),
    'train': 'train/images',
    'val': 'valid/images',
    'test': 'test/images',
    'names': classes
}

with open(os.path.join(dst_folder, 'dataset.yaml'), 'w', encoding='utf-8') as f:
    yaml.dump(dataset_yaml, f, allow_unicode=True)
