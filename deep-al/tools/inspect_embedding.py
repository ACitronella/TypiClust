import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import pandas as pd
from scipy.spatial import distance

PATH_TO_SCAN_DIR = "../../scan"
embedding_name = "features_seed133_lr0.04_temp0.01.npy"
embedding_path = os.path.join(PATH_TO_SCAN_DIR, "results", "blink2_fold0", "pretext", embedding_name)
embedding = np.load(embedding_path)
embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
rng = np.random.default_rng(42)

embedding = torch.from_numpy(embedding)

info_df = pd.read_csv("../../../../pytorchlm/cvat2_dataset/data_keypoint_interpolated_10points_faster/dataset_info.csv")

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
fold_idx = 0
train_df = info_df[(info_df["fold_idx"] != fold_idx) & (info_df["fold_idx"] != (fold_idx+1)%FOLDS)]

indices_table = train_df["frames"].cumsum()
indices_table = np.concatenate([[0], indices_table]) # for last file
# plt.figure(figsize=(10, 20))
# sp_idx = 1
# for (idx, row), start_idx, end_idx in zip(train_df.iterrows(), indices_table, indices_table[1:]):
#     an_eye_emb = embedding[start_idx:end_idx]  
#     # cdist = torch.cdist(an_eye_emb, an_eye_emb)
#     cdist = 1 - distance.cdist(an_eye_emb, an_eye_emb, "cosine")
#     cdist = cdist.flatten()
#     if cdist.shape[0] > 1_000_000:
#         dist_idx = rng.choice(np.arange(cdist.shape[0]), 1_000_000, replace=False)
#         cdist_sample = cdist[dist_idx]
#     else:
#         cdist_sample = cdist
#     cdist_sample = cdist_sample
#     plt.subplot(7, 2, sp_idx)
#     plt.hist(cdist_sample, bins=200, alpha=0.5, label=os.path.basename(row["keypoint_file"]))
#     plt.legend()
#     sp_idx += 1
#     # break
# plt.savefig(f"cosine within patient fold{fold_idx} {embedding_name}.png", bbox_inches="tight")




# cdist = 1 - distance.cdist(embedding, embedding, "cosine")
cdist = torch.cdist(embedding, embedding)
cdist = cdist.flatten()

dist_idx = rng.choice(np.arange(cdist.shape[0]), 1_000_000, replace=False)
cdist_sample = cdist[dist_idx]

plt.figure()
plt.hist(cdist_sample, bins=200, alpha=0.5)
plt.savefig(f"cosine all patient fold{fold_idx} {embedding_name}.png")
