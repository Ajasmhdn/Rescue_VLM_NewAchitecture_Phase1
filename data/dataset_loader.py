# ==========================================================
# FINAL DATASET LOADER (FIXED PATH VERSION)
# ==========================================================

import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class DisasterDataset(Dataset):

    def __init__(self, json_path, preprocess):

        with open(json_path, "r") as f:
            self.data = json.load(f)

        self.preprocess = preprocess

        # IMPORTANT: base folder = where JSON is located
        self.base_dir = os.path.dirname(json_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        item = self.data[idx]

        # FIX PATH HERE
        pre_path = os.path.join(self.base_dir, item["pre_image_path"])
        post_path = os.path.join(self.base_dir, item["post_image_path"])

        pre_img = self.preprocess(Image.open(pre_path))
        post_img = self.preprocess(Image.open(post_path))

        label = item["training_answer"]

        return pre_img, post_img, label