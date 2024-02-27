import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.manifold

FOLDS = 5

info_df = pd.read_csv("../../../../pytorchlm/cvat2_dataset/data_keypoint_interpolated_10points_uncompress/dataset_info.csv")
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


def io_of_project():
    vid_kp = np.load("../../../../pytorchlm/cvat2_dataset/data_keypoint_interpolated_10points_uncompress/A3102_42-457_point_l.npz")
    vid = vid_kp["leye_frame"]
    kp = vid_kp["leye_kp"]
    avid = vid[100]
    akp = kp[100]
    nrows = 1
    ncols = 2
    plt.figure(figsize=(12, 6))
    plt.subplot(nrows, ncols, 1)
    plt.imshow(avid)
    plt.axis("off")
    plt.title("Input: a video frame of an eye")

    plt.subplot(nrows, ncols, 2)
    plt.scatter(akp[:, 0], akp[:, 1], s=5)
    plt.ylim(avid.shape[1], 0)
    plt.xlim(0, avid.shape[0])
    plt.axis("off")
    plt.title("Output: keypoints of an eyelid")


    # plt.subplot(1, 3, 3)
    # plt.imshow(avid)
    # plt.scatter(akp[:, 0], akp[:, 1], s=0.5)

    plt.savefig("input_output_example.png", bbox_inches="tight")


def tsne_of_embedding():
    # embedding = np.load("../../scan/results/blink2_fold0/pretext/features_seed133_lr0.04_temp0.01.npy")
    embedding = np.load("../../scan/results/blink2_fold0/pretext/features_seed135_simclr128_blink_fold0 embsize128 lr0.npy")
    print(embedding.shape)
    info_df_foldn = info_df[(info_df["fold_idx"] != 0) & (info_df["fold_idx"] != 1)]
    is_blinking_global = []
    for idx, row in info_df_foldn.iterrows():
        vid_kp = np.load(os.path.join("../../../../pytorchlm/", row["keypoint_file"]))
        if "is_blinking" in vid_kp:
            is_blinking = vid_kp["is_blinking"]
        else:
            is_blinking = [None] * row["frames"]
        is_blinking_global.append(is_blinking)
    is_blinking_global = np.concatenate(is_blinking_global) 
    is_blinking_color_mapping = {
        True: "C0",
        False: "C1",
        None: "C2",
    }
    is_blinking_label = {
        True: "Blinking",
        False: "Not Blinking",
        None: "Unknown",
    }

    emb_wo_none = embedding[is_blinking_global != None]
    is_blinking_global_wo_none = is_blinking_global[is_blinking_global != None]

    tsne = sklearn.manifold.TSNE() 
    tsne_emb = tsne.fit_transform(emb_wo_none)

    plt.figure()
    plt.title("TSNE of embeddings from SimCLR")
    for t in [True, False]:
        plt.scatter(tsne_emb[is_blinking_global_wo_none==t, 0], tsne_emb[is_blinking_global_wo_none==t, 1], s=1, c=is_blinking_color_mapping[t], label=is_blinking_label[t])
    plt.legend() 
    plt.savefig("tsne_plot.png", bbox_inches="tight")


if __name__ == "__main__":
    # io_of_project()
    tsne_of_embedding()
