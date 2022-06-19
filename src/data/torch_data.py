import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    # ["AOT", "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL", "TCI", "WVP"]
    # (batch, channel, sequenc, w, h)
    def __init__(self, data_root="", image_size=100, in_memory=True, feature_list=["B02", "B03", "B04", "B08", "B8A"]):
        self.data_root = data_root
        self.date_list = os.listdir(os.path.join(self.data_root, "2021"))
        self.date_list.sort()
        self.train_list = os.listdir(os.path.join(self.data_root, "2021", self.date_list[0]))
        self.train_size = len(self.train_list)
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
            num_workers = 4
            with ThreadPool(num_workers) as pool:
                tmp_data = list(tqdm(pool.imap(self.get_feature, [i for i in range(self.train_size)]), total=self.train_size))

            for i in tmp_data:
                self.data.append(tmp_data[i][0])
                self.labels.append(tmp_data[i][1])

            # prepare data
            # for i in tqdm(range(self.train_size)):
            #     output_np, label = self.get_feature(i)
            #     self.data.append(output_np)
            #     self.labels.append(label)

    def __len__(self):
        return self.train_size

    def get_feature(self, i):
        # read cache
        i = self.train_list[i]
        if os.path.exists(os.path.join(self.data_root, "npy", str(i)+".npy")):
            with open(os.path.join(self.data_root, "npy", str(i)+".npy"), "rb") as f:
                output_np = np.load(f)
                img_root_path = os.path.join(self.data_root, "2021", self.date_list[0], str(i))
                image_name = "*_47PQS_{}_{}.tif".format(self.date_list[0], self.feature_list[0])
                img_path = glob.glob(os.path.join(img_root_path, image_name))[0]
                label = img_path.split("/")[-1][0]
                return output_np, self.label_map[label]

        output_np = np.zeros((len(self.feature_list), len(self.date_list), self.image_size, self.image_size))
        label = 0
        for j in range(len(self.date_list)):
            for k in range(len(self.feature_list)):
                date = self.date_list[j]
                feature = self.feature_list[k]
                img_root_path = os.path.join(self.data_root, "2021", date, str(i))
                image_name = "*_47PQS_{}_{}.tif".format(date, feature)
                img_path = glob.glob(os.path.join(img_root_path, image_name))[0]
                label = img_path.split("/")[-1][0]
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = self.image_resize(img, desired_size=self.image_size)
                output_np[k, j, :, :] = img

        # write cache
        if not os.path.exists(os.path.join(self.data_root, "npy")):
            os.makedirs(os.path.join(self.data_root, "npy"))
        with open(os.path.join(self.data_root, "npy", str(i)+".npy"), "wb") as f:
            np.save(f, output_np)

        return output_np, self.label_map[label]

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
            output_np, label = self.get_feature(idx)

            return output_np, label

def get_train_data(batch_size: int = 32, data_root="", image_size=64, in_memory=True, feature_list=None):
    """
    Args:
        batch_size: int

    Return:
        train_dataloader: DataLoader
    """
    train_dataset = CustomDataset(data_root, image_size, in_memory, feature_list)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    return train_dataloader

def get_val_data(batch_size: int = 32, data_root="", image_size=64, in_memory=True, feature_list=None):
    """
    Args:
        batch_size: int

    Return:
        val_dataloader: DataLoader
    """
    val_dataset = CustomDataset(data_root, image_size, in_memory, feature_list)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    return val_dataloader

def get_test_data(batch_size: int = 32, data_root="", image_size=64, in_memory=True, feature_list=None):
    """
    Args:
        batch_size: int

    Return:
        test_dataloader: DataLoader
    """
    test_dataset = CustomDataset(data_root, image_size, in_memory, feature_list)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return test_dataloader