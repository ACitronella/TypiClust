import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)    
import os
import pandas as pd
import torch
import sys
# local
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(os.path.abspath('..'))

import pycls.datasets.utils as ds_utils
from pycls.datasets.data import Data
from pycls.core.config import cfg

def plot_select_each_iter(prob_dist=False):
    if prob_dist:
        embeddings = ds_utils.load_embededing_from_path(cfg.ACTIVE_LEARNING.EMBEDDING_PATH)
        tb = np.concatenate([[0], dataset_info["frames"].cumsum()])
        kernel_size = 11
        kernel = np.ones(kernel_size)/kernel_size
        l = []
        print("embdedding size", embeddings.shape)
        for idx, row_idx in enumerate(dataset_info.index):
            frames = dataset_info.loc[row_idx, "frames"]
            start_idx = tb[idx]
            end_idx = tb[idx+1]
            a_patient_embedding = embeddings[start_idx:end_idx] # embedding of a patient
            centroid_emb = a_patient_embedding.mean(0)
            mean_square_diff_emb = np.mean(np.square(a_patient_embedding - centroid_emb), axis=1)
            # mean_square_diff_emb = np.mean(np.square(a_patient_embedding - centroid_emb), axis=1)
            # l.append(mean_square_diff_emb)
            smoothed_mean_square_diff_emb = np.zeros_like(mean_square_diff_emb)
            smoothed_mean_square_diff_emb[kernel_size//2:(-kernel_size//2)+1] = np.convolve(mean_square_diff_emb, kernel, mode='valid')
            smoothed_mean_square_diff_emb[:kernel_size//2] = smoothed_mean_square_diff_emb[kernel_size//2]
            smoothed_mean_square_diff_emb[-kernel_size//2+1:] = smoothed_mean_square_diff_emb[-kernel_size//2+1]

            if cfg.ACTIVE_LEARNING.SAMPLING_FN == "embedding_difference_as_probability_density_reduce_high_frame_prob":
                l.append(smoothed_mean_square_diff_emb/frames)
            else:
                l.append(smoothed_mean_square_diff_emb)
        v = np.concatenate(l)
        v = v.astype("float64")
        if cfg.ACTIVE_LEARNING.SAMPLING_FN == "embedding_difference_as_probability_density_with_softmax":
            v = torch.softmax(torch.from_numpy(v)*cfg.ACTIVE_LEARNING.SOFTMAX_TEMPERATURE, dim=0).numpy()
        w = (v - v.min())
        y = w / w.max()
        z = y / y.sum() # prob density
        pd_max = z.max()
        pd_min = z.min()
        # prob_mass = [np.sum(z[start_idx:end_idx]) for start_idx, end_idx in zip(tb, tb[1:])]

    ori_shape = all_sampled_set.shape        
    all_sampled_set_with_pcode_fidx = [train_data.get_patient_code_and_frame_from_idx(idx) for idx in all_sampled_set.flatten()]
    all_sampled_set_with_pcode_fidx= np.reshape(all_sampled_set_with_pcode_fidx, ori_shape)
    patient_wise = {} 
    for episode_idx, selected in enumerate(all_sampled_set_with_pcode_fidx):
        for eye_info in selected:
            selected_frames_of_this_patient = patient_wise.setdefault(eye_info["keypoint_file"], [])
            selected_frames_of_this_patient.append((eye_info["frame_idx"], episode_idx, eye_info["dataset_idx"]))

    plt.figure(figsize=(2000/150, 3000/150), dpi=150)
    plt.subplots_adjust(hspace=0.4)
    for idx, (row_idx, eye_info) in enumerate(dataset_info.iterrows()):
        selected_frame_and_episode = patient_wise.get(eye_info["keypoint_file"], [])
        if selected_frame_and_episode:
            selected_frame_and_episode = np.asarray(selected_frame_and_episode)
            selected_frames = selected_frame_and_episode[:, 0]
            episode_selected = selected_frame_and_episode[:, 1]
        else:
            selected_frames = []
            episode_selected = []
        
        plt.subplot(dataset_info.shape[0]//2, 2, idx + 1)
        plt.title(f"{eye_info['patient_code']} {eye_info['keypoints_key'][0]}")
        plt.scatter(selected_frames, episode_selected, c=[f"C{ep}" for ep in episode_selected])
        plt.xlim(-1, eye_info["frames"])
        plt.ylim(-1, 15)
        plt.xticks(selected_frames, fontsize="xx-small")
        plt.yticks(episode_selected, fontsize="xx-small")
        plt.xlabel("Frame index")
        plt.ylabel("AL iter selected")
        for frame_idx, episode_idx, dataset_idx in selected_frame_and_episode:
            plt.text(frame_idx, episode_idx + 0.5, f"({frame_idx}, {episode_idx})", alpha=0.5)
            img, *_ = train_data[dataset_idx]
            imagebox = OffsetImage(np.array(img), zoom=0.10, alpha=0.6)
            ab = AnnotationBbox(imagebox, (frame_idx, 12), frameon=False)
            plt.gca().add_artist(ab)
        if prob_dist:
            plt.twinx()
            plt.plot(z[tb[idx]:tb[idx+1]], "o", markersize=1, alpha=0.1, )
            plt.ylim(pd_min, pd_max)
            plt.yticks(fontsize="xx-small")
            plt.ylabel("Probability density")
    save_path = os.path.join(exp_dir, "selected_frames_re.png")
    print(save_path)
    plt.savefig(save_path, bbox_inches='tight')

def calculate_selected_blink_and_nonblink_frame():
    is_blinking = np.asarray([train_data.is_blinking_idx(idx) for idx in range(len(train_data))])
    blinking_idx = np.argwhere(is_blinking == True).flatten()
    nonblinking_idx = np.argwhere(is_blinking == False).flatten()
    n_blinking = blinking_idx.shape[0]
    n_nonblinking = nonblinking_idx.shape[0]
    n_unknown = len(train_data) - n_blinking - n_nonblinking
    n_blinking_each_iter = []
    n_nonblinking_each_iter = []
    n_unknown_each_iter = []
    plt.figure()
    for al_iter in range(all_sampled_set.shape[0]) :
        selected_this_iter = all_sampled_set[:al_iter+1].flatten()
        selected_blinking_idx = np.intersect1d(selected_this_iter, blinking_idx)
        selected_nonblinking_idx = np.intersect1d(selected_this_iter, nonblinking_idx)
        n_selected_blinking = selected_blinking_idx.shape[0]
        n_selected_nonblinking = selected_nonblinking_idx.shape[0]
        n_selected_unknown = selected_this_iter.size - n_selected_blinking - n_selected_nonblinking
        n_blinking_each_iter.append(n_selected_blinking)
        n_nonblinking_each_iter.append(n_selected_nonblinking)
        n_unknown_each_iter.append(n_selected_unknown)
        
        plt.text(al_iter, 75, f"{selected_this_iter.size}n", horizontalalignment="center")
        plt.text(al_iter, 70, f"{n_selected_blinking/selected_this_iter.size: .2f}", c="C0", horizontalalignment="center")
        plt.text(al_iter, 65, f"{n_selected_nonblinking/selected_this_iter.size: .2f}", c="C1", horizontalalignment="center")
        plt.text(al_iter, 60, f"{n_selected_unknown/selected_this_iter.size: .2f}", c="C2", horizontalalignment="center")
    plt.text(al_iter//2, 90, "Ratio from 100% dataset", horizontalalignment="center")
    plt.text(al_iter//2-1, 85, f"{n_blinking/len(train_data):.2f}", c="C0", horizontalalignment="center")
    plt.text(al_iter//2, 85, f"{n_nonblinking/len(train_data):.2f}", c="C1", horizontalalignment="center")
    plt.text(al_iter//2+1, 85, f"{n_unknown/len(train_data):.2f}", c="C2", horizontalalignment="center")
    plt.plot(n_blinking_each_iter, "--o", label="Blinking", c="C0")
    plt.plot(n_nonblinking_each_iter, "--o", label="Nonblinking", c="C1")
    plt.plot(n_unknown_each_iter, "--o", label="Unknown", c="C2")
    plt.plot()
    plt.ylim(0, 100)
    plt.legend(loc="upper left")
    plt.xlabel("al iter") 
    plt.ylabel("Number of frame") 
    plt.title(f"{exp_name} fold{fold_idx}")
    plt.savefig(os.path.join(exp_dir, "bar.png"))
    print(os.path.join(exp_dir, "bar.png"))
    

if "__main__" == __name__: 
    MODEL_GRAVEYARD = "../model_graveyard_but_use_mse_as_best_indicator"
    # MODEL_GRAVEYARD = "../model_graveyard_but_use_mse2"
    # MODEL_GRAVEYARD = "../model_graveyard_but_use_mse3"
    FOLDS = 5
    for fold_idx in range(FOLDS):
        # cfg_path = f"../configs/blink3/simclr128-lr0.04-temp0.01_emb-diff-as-prob-diff-batch_size10-fold{fold_idx}.yaml"
        cfg_path = f"../configs/blink3/random-finetune-batch_size10-fold{fold_idx}.yaml"
        cfg.merge_from_file(cfg_path)
        exp_name = cfg.EXP_NAME
        fold_idx = cfg.DATASET.FOLD_IDX
        exp_dir = os.path.join(MODEL_GRAVEYARD, exp_name, f"stepsize10n_fold{fold_idx}")

        lset = np.load(os.path.join(exp_dir, "episode_9", "lSet.npy"), allow_pickle=True)
        all_sampled_set = lset.reshape(10, 10) # 10 budget size per al iter, 10 al iters

        dataset_info = pd.read_csv("../../../../pytorchlm/cvat2_dataset/data_keypoint_interpolated_10points_uncompress/dataset_info.csv")
        dataset_info.loc[dataset_info["patient_code"] == "A3104", "fold_idx"] = 0
        dataset_info.loc[dataset_info["patient_code"] == "C2104", "fold_idx"] = 0

        dataset_info.loc[dataset_info["patient_code"] == "A2111", "fold_idx"] = 1
        dataset_info.loc[dataset_info["patient_code"] == "C2205", "fold_idx"] = 1

        dataset_info.loc[dataset_info["patient_code"] == "A3102", "fold_idx"] = 2
        dataset_info.loc[dataset_info["patient_code"] == "C2202", "fold_idx"] = 2

        dataset_info.loc[dataset_info["patient_code"] == "A1115", "fold_idx"] = 3
        dataset_info.loc[dataset_info["patient_code"] == "C2201", "fold_idx"] = 3

        dataset_info.loc[dataset_info["patient_code"] == "A2211", "fold_idx"] = 4
        dataset_info.loc[dataset_info["patient_code"] == "C2204", "fold_idx"] = 4
        dataset_info["fold_idx"] = dataset_info["fold_idx"].astype(int)
        dataset_info = dataset_info[(dataset_info["fold_idx"] != (fold_idx % FOLDS)) & (dataset_info["fold_idx"] != (fold_idx + 1) % FOLDS)].reset_index()

        cfg.DATASET.ROOT_DIR = os.path.join(os.path.abspath('../..'), cfg.DATASET.ROOT_DIR)
        data_obj = Data(cfg)
        train_data, train_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=True, isDownload=True, fold_idx=cfg.DATASET.FOLD_IDX, is_blinking=cfg.DATASET.IS_BLINKING, use_faster=False)

        print("training set size:", train_size)
        # calculate_selected_blink_and_nonblink_frame()
        plot_select_each_iter(prob_dist=False)
        