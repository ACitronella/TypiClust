from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import pandas as pd
import os
from .blink_dataset import ImageDataFrameBlinkingWrapper # for other if they want to import?

class BlinkDataset2(Dataset):
    def __init__(self, train:bool, transform, test_transform, fold_idx=0, dataset_path="../../pytorchlm/", input_size=256, only_features=False, use_faster=True):
        # if isinstance(df_or_path, str):
        #     self.dataset_info = pd.read_csv(df_or_path)
        # elif isinstance(df_or_path, pd.DataFrame):
        #     self.dataset_info = df_or_path
        # else:
        #     raise Exception("Invalid arg")
        # assert self.dataset_info.columns
        self.transform = transform
        self.test_transform = test_transform
        self.dataset_path = dataset_path
        self.train = train
        self.use_faster = use_faster

        if self.use_faster:
            info_df = pd.read_csv(os.path.join(dataset_path, "cvat2_dataset", "data_keypoint_interpolated_10points_faster", "dataset_info.csv"))
        else:
            # this will have blinking information
            info_df = pd.read_csv(os.path.join(dataset_path, "cvat2_dataset", "data_keypoint_interpolated_10points_uncompress", "dataset_info.csv"))
        info_df["patient_type"] = info_df["patient_code"].apply(lambda x: x[0])

        FOLDS = 5
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
        
        if train:
            self.dataset_info = info_df[(info_df["fold_idx"] != fold_idx) & (info_df["fold_idx"] != (fold_idx+1)%FOLDS)]
        else:
            self.dataset_info = info_df[info_df["fold_idx"] == fold_idx]
        self.dataset_info = self.dataset_info.reset_index(drop=True)
        self.input_size = input_size
        self.n = int(self.dataset_info["frames"].sum())

        self.indices_table = (self.dataset_info["frames"].cumsum() - self.dataset_info["frames"]).values # collect start index of each eye
        self.indices_table = np.concatenate([self.indices_table, [self.n]]) # for last file

        self.only_features = only_features
        self.no_aug = False

        # for cache
        self.last_file_idx = -1
        self.x = None
        self.y = None
        print("using blink2")

    def __len__(self):
        return self.n

    def get_patient_code_and_frame_from_idx(self, idx):
        file_idx = np.argmax(self.indices_table > idx) - 1 # get first file that >= idx
        frame_idx = idx - self.indices_table[file_idx]
        row = self.dataset_info.loc[file_idx].copy()
        row["frame_idx"] = frame_idx
        row["dataset_idx"] = idx
        return row.to_dict()
        
    def get_all_patient_codes(self):
        return self.dataset_info["patient_code"].unique()

    def __getitem__(self, idx):
        file_idx = np.argmax(self.indices_table > idx) - 1 # get first file that >= idx
        frame_idx = idx - self.indices_table[file_idx]

        if file_idx != self.last_file_idx:
            if file_idx == -1:
                print("file_idx = -1", frame_idx, idx)
                print(self.indices_table, self.indices_table > idx)
            xy = np.load(os.path.join(self.dataset_path, self.dataset_info.loc[file_idx, "keypoint_file"]), "r" if self.train else None)
            self.last_file_idx = file_idx
            self.x = xy[self.dataset_info.loc[file_idx, "eye_key"]]
            self.y = xy[self.dataset_info.loc[file_idx, "keypoints_key"]]
        
        x = self.x[frame_idx]
        y = self.y[frame_idx].astype("float32")
        (ori_h, ori_w, _) = x.shape
        y[:, 0] = y[:, 0] / ori_w
        y[:, 1] = y[:, 1] / ori_h

        if (y[y.shape[0]//2] <= 0).any(): # middle key point is eye center
            mask = np.ones((y.shape[0]), dtype="bool")
            mask[y.shape[0]//2] = 0
            y[y.shape[0]//2, 0] = np.mean(y[mask, 0])
            mask[y.shape[0]//2:] = 0
            y[y.shape[0]//2, 1] = np.mean(y[mask, 1])
        
        sample = Image.fromarray(x)
        target = y.reshape(-1)
        if self.only_features:
            sample = self.features[idx]
        else:
            if self.no_aug:
                if self.test_transform is not None:
                    sample = self.test_transform(sample)
            else:
                if self.transform is not None:
                    sample = self.transform(sample)
        return sample, target

    def is_blinking_idx(self, idx): # return Optional[bool] 
        # none is for code C patient where my blink detection doesnot work
        assert not self.use_faster
        file_idx = np.argmax(self.indices_table > idx) - 1 # get first file that >= idx
        frame_idx = idx - self.indices_table[file_idx]
        if file_idx != self.last_file_idx or self.is_blinking is None:
            xy = np.load(os.path.join(self.dataset_path, self.dataset_info.loc[file_idx, "keypoint_file"]), "r")
            is_blinking = xy["is_blinking"] if "is_blinking" in tuple(xy.keys()) else None
            if is_blinking is None:
                return None
            self.is_blinking = is_blinking
            self.last_file_idx = file_idx
            self.x = xy[self.dataset_info.loc[file_idx, "eye_key"]]
            self.y = xy[self.dataset_info.loc[file_idx, "keypoints_key"]]
            
        return self.is_blinking[frame_idx]


# class ImageDataFrameBlinkingWrapper(Dataset):
#     def __init__(self, dataset, is_blinking=False):
#         self.dataset = dataset
#         self.is_blinking = is_blinking
#         tmp = np.asarray([self.dataset.is_blinking_idx(idx) == is_blinking for idx in range(len(self.dataset))], dtype="int32")
#         self.is_blinking_idx = np.argwhere(tmp).flatten()
#         self.n = self.is_blinking_idx.shape[0]

#     def __len__(self):
#         return self.n

#     def __getitem__(self, idx):
#         return self.dataset[self.is_blinking_idx[idx]]