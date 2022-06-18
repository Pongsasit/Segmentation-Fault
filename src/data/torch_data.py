import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, data_root="", image_size=100, in_memory=True, feature_list = ["AOT", "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL", "TCI", "WVP"]):
        self.data_root = data_root
        self.date_list = os.listdir(os.path.join(self.data_root, "2021"))
        self.date_list.sort()
        self.train_size = len(os.listdir(os.path.join(self.data_root, "2021", self.date_list[0])))
        self.feature_list = feature_list
        self.in_memory = in_memory
        self.image_size = image_size
        self.label_map = {
            "1": 0,
            "2": 1,
            "3": 2,
            "4": 3,
            "X": -1
        }
        self.data = []
        self.labels = []

        if in_memory:
            # prepare data
            for i in tqdm(range(self.train_size)):
                # 11, 8 come from cv2 imread size of tif
                output_np = np.zeros((len(self.feature_list), len(self.date_list), self.image_size, self.image_size))
                label = 0
                for j in range(len(self.date_list)):
                    for k in range(len(self.feature_list)):
                        date = self.date_list[j]
                        feature = self.feature_list[k]
                        img_root_path = os.path.join(self.data_root, "2021", date, i)
                        image_name = "*_47PQS_{}_{}.tif".format(date, feature)
                        img_path = glob(os.path.join(img_root_path, image_name))[0]
                        label = img_path.split("/")[-1][0]
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        img = self.image_resize(img, desired_size=self.image_size)
                        output_np[j, k, :, :] = img
                self.data.append(output_np)
                self.labels.append(self.label_map[label])

    def __len__(self):
        return self.train_size

    def image_resize(self, im, desired_size=100):
        old_size = im.shape[:2] # old_size is in (height, width) format

        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])

        # new_size should be in (width, height) format

        im = cv2.resize(im, (new_size[1], new_size[0]))

        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        color = [0, 0, 0]
        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color)

        return new_im

    def __getitem__(self, idx):
        if self.in_memory:
            return self.data[idx]
        else:
            # read cache
            if os.path.exists(os.path.join(self.data_root, "2021", idx+".npy")):
                with open(os.path.join(self.data_root, "2021", idx+".npy"), "rb") as f:
                    output_np = np.load(f)
                    img_root_path = os.path.join(self.data_root, "2021", self.date_list[0], idx)
                    image_name = "*_47PQS_{}_{}.tif".format(self.date_list[0], self.feature_list[0])
                    img_path = glob(os.path.join(img_root_path, image_name))[0]
                    label = img_path.split("/")[-1][0]
                    return output_np, label

            output_np = np.zeros((len(self.feature_list), len(self.date_list), 100, 100))
            label = 0
            for j in range(len(self.date_list)):
                for k in range(len(self.feature_list)):
                    date = self.date_list[j]
                    feature = self.feature_list[k]
                    img_root_path = os.path.join(self.data_root, "2021", date, idx)
                    image_name = "*_47PQS_{}_{}.tif".format(date, feature)
                    img_path = glob(os.path.join(img_root_path, image_name))[0]
                    label = img_path.split("/")[-1][0]
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img = self.image_resize(img, desired_size=self.image_size)
                    output_np[j, k, :, :] = img
            # write cache
            with open(os.path.join(self.data_root, "2021", idx+".npy"), "rb") as f:
                np.save(f, output_np)

            return output_np, self.label_map[label]

def get_train_data(batch_size: int = 32, data_root="", image_size=100, in_memory=True):
    """
    Args:
        batch_size: int

    Return:
        train_dataloader: DataLoader
    """
    train_dataset = CustomDataset(data_root, image_size, in_memory)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    return train_dataloader

def get_val_data(batch_size: int = 32, data_root="", image_size=100, in_memory=True):
    """
    Args:
        batch_size: int

    Return:
        val_dataloader: DataLoader
    """
    val_dataset = CustomDataset(data_root, image_size, in_memory)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    return val_dataloader

def get_test_data(batch_size: int = 32, data_root="", image_size=100, in_memory=True):
    """
    Args:
        batch_size: int

    Return:
        test_dataloader: DataLoader
    """
    test_dataset = CustomDataset(data_root, image_size, in_memory)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return test_dataloader