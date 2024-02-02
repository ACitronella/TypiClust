from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import pandas as pd
import os

class BlinkDataset(Dataset):
    def __init__(self, train:bool, transform, fold_idx=0, dataset_path="../../../pytorchlm/", input_size=256):
        # if isinstance(df_or_path, str):
        #     self.dataset_info = pd.read_csv(df_or_path)
        # elif isinstance(df_or_path, pd.DataFrame):
        #     self.dataset_info = df_or_path
        # else:
        #     raise Exception("Invalid arg")
        # assert self.dataset_info.columns
        self.transform = transform
        self.dataset_path = dataset_path
        info_df = pd.read_csv(os.path.join(dataset_path, "cvat2_dataset", "data_keypoint_interpolated_10points_faster", "dataset_info.csv"))
        # info_df = pd.read_csv(os.path.join(dataset_path, "cvat2_dataset", "data_keypoint_interpolated_10points_uncompress", "dataset_info.csv"))
        info_df["patient_type"] = info_df["patient_code"].apply(lambda x: x[0])

        test_patient_code = set(['A2211', 'C2204'])
        train_val_patient_code = set(info_df["patient_code"].unique()).difference(test_patient_code)
        train_val_df = info_df[info_df["patient_code"].apply(lambda x : x in train_val_patient_code)]
        # FOLDS = 4
        train_val_df.loc[train_val_df["patient_code"] == "A3104", "fold_idx"] = 0
        train_val_df.loc[train_val_df["patient_code"] == "C2104", "fold_idx"] = 0

        train_val_df.loc[train_val_df["patient_code"] == "A2111", "fold_idx"] = 1
        train_val_df.loc[train_val_df["patient_code"] == "C2205", "fold_idx"] = 1

        train_val_df.loc[train_val_df["patient_code"] == "A3102", "fold_idx"] = 2
        train_val_df.loc[train_val_df["patient_code"] == "C2202", "fold_idx"] = 2

        train_val_df.loc[train_val_df["patient_code"] == "A1115", "fold_idx"] = 3
        train_val_df.loc[train_val_df["patient_code"] == "C2201", "fold_idx"] = 3
        train_val_df["fold_idx"] = train_val_df["fold_idx"].astype(int)
        
        if train:
            self.dataset_info = train_val_df[train_val_df["fold_idx"] != fold_idx]
        else:
            self.dataset_info = train_val_df[train_val_df["fold_idx"] == fold_idx]
        self.dataset_info = self.dataset_info.reset_index(drop=True)
        self.input_size = input_size
        self.n = int(self.dataset_info["frames"].sum())

        self.indices_table = (self.dataset_info["frames"].cumsum() - self.dataset_info["frames"]).values # collect start index of each eye
        self.indices_table = np.concatenate([self.indices_table, [self.n]]) # for last file

        # for cache
        self.last_file_idx = -1
        self.x = None
        self.y = None

    def __len__(self):
        return self.n

    def get_patient_code_and_frame_from_idx(self, idx):
        file_idx = np.argmax(self.indices_table > idx) - 1 # get first file that >= idx
        frame_idx = idx - self.indices_table[file_idx]
        return self.dataset_info.loc[file_idx], frame_idx
        
    def get_all_patient_codes(self):
        return self.dataset_info["patient_code"].unique()

    def __getitem__(self, idx):
        file_idx = np.argmax(self.indices_table > idx) - 1 # get first file that >= idx
        frame_idx = idx - self.indices_table[file_idx]

        if file_idx != self.last_file_idx:
            # if file_idx == -1:
            #     print("file_idx = -1", frame_idx, idx)
            #     print(self.indices_table, self.indices_table > idx)
            xy = np.load(os.path.join(self.dataset_path, self.dataset_info.loc[file_idx, "keypoint_file"]), mmap_mode="r")
            self.last_file_idx = file_idx
            self.x = xy[self.dataset_info.loc[file_idx, "eye_key"]]
            # self.y = xy[self.dataset_info.loc[file_idx, "keypoints_key"]]
        
        x = self.x[frame_idx]
        # y = self.y[frame_idx].astype("float32")
        # (ori_h, ori_w, _) = x.shape
        # y[:, 0] = y[:, 0] / ori_w
        # y[:, 1] = y[:, 1] / ori_h
        # if (y[y.shape[0]//2] <= 0).any(): # middle key point is eye center
        #     mask = np.ones((y.shape[0]), dtype="bool")
        #     mask[y.shape[0]//2] = 0
        #     y[y.shape[0]//2, 0] = np.mean(y[mask, 0])
        #     mask[y.shape[0]//2:] = 0
        #     y[y.shape[0]//2, 1] = np.mean(y[mask, 1])
        img_size = x.shape[:2]
        img = Image.fromarray(x)
        left = (img.width - self.input_size) // 2; top = (img.height - self.input_size) // 2
        img = img.crop((left, top, left+self.input_size, top+self.input_size))
        if self.transform is not None:
            img = self.transform(img)
        # target = y.reshape(-1)
        return {"image": img, "frame_idx": frame_idx, "meta": {"im_size": img_size, "index": idx, "class_name": "test" }}



class BlinkDataset2(Dataset):
    def __init__(self, train:bool, transform, fold_idx=0, dataset_path="../../../pytorchlm/", input_size=256):
        # if isinstance(df_or_path, str):
        #     self.dataset_info = pd.read_csv(df_or_path)
        # elif isinstance(df_or_path, pd.DataFrame):
        #     self.dataset_info = df_or_path
        # else:
        #     raise Exception("Invalid arg")
        # assert self.dataset_info.columns
        self.transform = transform
        self.dataset_path = dataset_path
        info_df = pd.read_csv(os.path.join(dataset_path, "cvat2_dataset", "data_keypoint_interpolated_10points_faster", "dataset_info.csv"))
        # info_df = pd.read_csv(os.path.join(dataset_path, "cvat2_dataset", "data_keypoint_interpolated_10points_uncompress", "dataset_info.csv"))
        info_df["patient_type"] = info_df["patient_code"].apply(lambda x: x[0])

        info_df.loc[info_df["patient_code"] == "A3104", "fold_idx"] = 0
        info_df.loc[info_df["patient_code"] == "C2104", "fold_idx"] = 0

        info_df.loc[info_df["patient_code"] == "A2111", "fold_idx"] = 1
        info_df.loc[info_df["patient_code"] == "C2205", "fold_idx"] = 1

        info_df.loc[info_df["patient_code"] == "A3102", "fold_idx"] = 2
        info_df.loc[info_df["patient_code"] == "C2202", "fold_idx"] = 2

        info_df.loc[info_df["patient_code"] == "A1115", "fold_idx"] = 3
        info_df.loc[info_df["patient_code"] == "C2201", "fold_idx"] = 3

        info_df.loc[info_df["patient_code"] == "A2211", "fold_idx"] = 4
        info_df.loc[info_df["patient_code"] == "C2204", "fold_idx"] = 4
        info_df["fold_idx"] = info_df["fold_idx"].astype(int)
        FOLDS = 5
        if train:
            self.dataset_info = info_df[(info_df["fold_idx"] != fold_idx) & (info_df['fold_idx'] != fold_idx%FOLDS)]
        else: # val, leave (fold_idx+1)%FOLDS as test set
            self.dataset_info = info_df[info_df["fold_idx"] == fold_idx]
        self.dataset_info = self.dataset_info.reset_index(drop=True)
        self.input_size = input_size
        self.n = int(self.dataset_info["frames"].sum())

        self.indices_table = (self.dataset_info["frames"].cumsum() - self.dataset_info["frames"]).values # collect start index of each eye
        self.indices_table = np.concatenate([self.indices_table, [self.n]]) # for last file

        # for cache
        self.last_file_idx = -1
        self.x = None
        self.y = None
        print("using Blink2")

    def __len__(self):
        return self.n

    def get_patient_code_and_frame_from_idx(self, idx):
        file_idx = np.argmax(self.indices_table > idx) - 1 # get first file that >= idx
        frame_idx = idx - self.indices_table[file_idx]
        return self.dataset_info.loc[file_idx], frame_idx
        
    def get_all_patient_codes(self):
        return self.dataset_info["patient_code"].unique()

    def __getitem__(self, idx):
        file_idx = np.argmax(self.indices_table > idx) - 1 # get first file that >= idx
        frame_idx = idx - self.indices_table[file_idx]

        if file_idx != self.last_file_idx:
            # if file_idx == -1:
            #     print("file_idx = -1", frame_idx, idx)
            #     print(self.indices_table, self.indices_table > idx)
            xy = np.load(os.path.join(self.dataset_path, self.dataset_info.loc[file_idx, "keypoint_file"]), mmap_mode="r")
            self.last_file_idx = file_idx
            self.x = xy[self.dataset_info.loc[file_idx, "eye_key"]]
            # self.y = xy[self.dataset_info.loc[file_idx, "keypoints_key"]]
        
        x = self.x[frame_idx]
        # y = self.y[frame_idx].astype("float32")
        # (ori_h, ori_w, _) = x.shape
        # y[:, 0] = y[:, 0] / ori_w
        # y[:, 1] = y[:, 1] / ori_h
        # if (y[y.shape[0]//2] <= 0).any(): # middle key point is eye center
        #     mask = np.ones((y.shape[0]), dtype="bool")
        #     mask[y.shape[0]//2] = 0
        #     y[y.shape[0]//2, 0] = np.mean(y[mask, 0])
        #     mask[y.shape[0]//2:] = 0
        #     y[y.shape[0]//2, 1] = np.mean(y[mask, 1])
        img_size = x.shape[:2]
        img = Image.fromarray(x)
        left = (img.width - self.input_size) // 2; top = (img.height - self.input_size) // 2
        img = img.crop((left, top, left+self.input_size, top+self.input_size))
        if self.transform is not None:
            img = self.transform(img)
        # target = y.reshape(-1)
        return {"image": img, "frame_idx": frame_idx, "meta": {"im_size": img_size, "index": idx, "class_name": "test" }}


