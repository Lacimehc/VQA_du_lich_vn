# preprocess_images.py

import os
import glob
import random
import torch
import numpy as np
import cv2
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

random.seed(42)

RAW_DIR = 'data/images/raw'
PROCESSED = 'data/images/processed'
CATEGORIES = ['ben_nha_rong','buu_dien_thanh_pho','chua_mot_cot','dinh_doc_lap','ganh_da_dia','lang_bac','nha_hat_lon','nha_tho_duc_ba','pho_co_hoi_an','thac_ban_gioc']

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def create_processed_dirs():
    for category in CATEGORIES:
        for split in ['train', 'val', 'test']:
            dir_path = os.path.join(PROCESSED, category, split)
            os.makedirs(dir_path, exist_ok=True)

def split_dataset(image_files):
    random.shuffle(image_files)
    n = len(image_files)
    return {
        'train': image_files[:int(0.8 * n)],
        'val': image_files[int(0.8 * n):int(0.9 * n)],
        'test': image_files[int(0.9 * n):]
    }

def normalize_img(img):
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img

def process_one(args):
    file_path, split, category = args
    try:
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = normalize_img(img)
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        tensor_img = torch.tensor(img, dtype=torch.float)

        file_name = os.path.splitext(os.path.basename(file_path))[0] + '.pt'
        save_path = os.path.join(PROCESSED, category, split, file_name)
        torch.save(tensor_img, save_path)
    except Exception as e:
        print(f"[ERROR] {file_path}: {e}")

def process_category(category):
    raw_dir = os.path.join(RAW_DIR, category)
    image_files = glob.glob(os.path.join(raw_dir, '*.jpg'))
    if not image_files:
        print(f"[!] Không tìm thấy ảnh trong: {raw_dir}")
        return

    splits = split_dataset(image_files)
    tasks = []
    for split, files in splits.items():
        for file_path in files:
            tasks.append((file_path, split, category))

    print(f"👉 Đang xử lý {len(tasks)} ảnh cho '{category}' bằng {cpu_count()} luồng...")
    with Pool(processes=cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(process_one, tasks), total=len(tasks)))

def main():
    create_processed_dirs()
    for category in CATEGORIES:
        print(f"🚀 Bắt đầu xử lý ảnh cho địa danh: {category}")
        process_category(category)

if __name__ == '__main__':
    main()
