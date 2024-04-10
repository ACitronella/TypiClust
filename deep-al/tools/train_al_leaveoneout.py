import os
import pickle
import sys
from datetime import datetime
import argparse
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional
import torch.nn as nn
# import torch.optim as optim
import torch.profiler
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)    
from utils import CustomImageDataset4, imshow_kp, batch_process_pip_out, MetricsStorage, cal_and_process_meanface, imshow_kp_from_dl

import functools
from copy import deepcopy

# local
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(os.path.abspath('..'))

from pycls.al.ActiveLearning import ActiveLearning
import pycls.core.builders as model_builder
from pycls.core.config import cfg, dump_cfg
# import pycls.core.losses as losses
import pycls.core.optimizer as optim
from pycls.datasets.data import Data
# import pycls.utils.checkpoint as cu
import pycls.utils.logging as lu
# import pycls.utils.metrics as mu
# import pycls.utils.net as nu
# from pycls.utils.meters import TestMeter, TrainMeter, ValMeter
from pycls.datasets.blink_dataset import BlinkDataset, ImageDataFrameBlinkingWrapper
from pycls.datasets.blink_dataset_with_no_test_set import BlinkDataset2
from pycls.datasets.blink_dataset_all import BlinkDatasetAll
import pycls.datasets.utils as ds_utils

from utils import eval_on_pipnet

logger = lu.get_logger(__name__)
# model params
input_size = 256
num_nb = 10
num_lms = 21
net_stride = 32

def compute_loss_pip(outputs_map, 
                     outputs_local_x, outputs_local_y, 
                     outputs_nb_x, outputs_nb_y, 
                     labels_map, 
                     labels_local_x, labels_local_y, 
                     labels_nb_x, labels_nb_y,  
                     criterion_cls, criterion_reg, num_nb):
    tmp_batch, tmp_channel, tmp_height, tmp_width = outputs_map.size()
    batch_times_channel = tmp_batch*tmp_channel
    batch_times_channel_times_nb = batch_times_channel * num_nb
    labels_map = labels_map.view(batch_times_channel, -1)
    labels_max_ids = torch.argmax(labels_map, 1)
    labels_max_ids = labels_max_ids.view(-1, 1)
    labels_max_ids_nb = labels_max_ids.repeat(1, num_nb).view(-1, 1)

    outputs_local_x = outputs_local_x.view(batch_times_channel, -1)
    outputs_local_x_select = torch.gather(outputs_local_x, 1, labels_max_ids)
    outputs_local_y = outputs_local_y.view(batch_times_channel, -1)
    outputs_local_y_select = torch.gather(outputs_local_y, 1, labels_max_ids)
    outputs_nb_x = outputs_nb_x.view(batch_times_channel_times_nb, -1)
    outputs_nb_x_select = torch.gather(outputs_nb_x, 1, labels_max_ids_nb)
    outputs_nb_y = outputs_nb_y.view(batch_times_channel_times_nb, -1)
    outputs_nb_y_select = torch.gather(outputs_nb_y, 1, labels_max_ids_nb)

    labels_local_x = labels_local_x.view(batch_times_channel, -1)
    labels_local_x_select = torch.gather(labels_local_x, 1, labels_max_ids)
    labels_local_y = labels_local_y.view(batch_times_channel, -1)
    labels_local_y_select = torch.gather(labels_local_y, 1, labels_max_ids)
    labels_nb_x = labels_nb_x.view(batch_times_channel_times_nb, -1)
    labels_nb_x_select = torch.gather(labels_nb_x, 1, labels_max_ids_nb)
    labels_nb_y = labels_nb_y.view(batch_times_channel_times_nb, -1)
    labels_nb_y_select = torch.gather(labels_nb_y, 1, labels_max_ids_nb)

    labels_map = labels_map.view(tmp_batch, tmp_channel, tmp_height, tmp_width)
    loss_map = criterion_cls(outputs_map, labels_map)
    loss_x = criterion_reg(outputs_local_x_select, labels_local_x_select)
    loss_y = criterion_reg(outputs_local_y_select, labels_local_y_select)
    loss_nb_x = criterion_reg(outputs_nb_x_select, labels_nb_x_select)
    loss_nb_y = criterion_reg(outputs_nb_y_select, labels_nb_y_select)
    return loss_map, loss_x, loss_y, loss_nb_x, loss_nb_y

def compute_nme(lms_pred, lms_gt, norm):
    lms_pred = lms_pred.reshape((-1, 2))
    lms_gt = lms_gt.reshape((-1, 2))
    nme = np.mean(np.linalg.norm(lms_pred - lms_gt, axis=1)) / norm 
    return nme

def train_model_pip(net, train_loader, criterion_cls, criterion_reg, cls_loss_weight, reg_loss_weight, 
                num_nb, reverse_index1, reverse_index2, max_len, optimizer, num_epochs, scheduler, model_save_path, 
                use_patient_and_save_best, device, val_loader, metrics: MetricsStorage, indicate_best_by:str, how_best:str="min"):
    assert how_best in ["min", "max"]
    epoch = len(metrics.training_losses)
    cur_best_metric = np.inf if how_best == "min" else -np.inf
    count_current_epochs = 0
    while count_current_epochs < num_epochs:
        count_current_epochs += 1
        epoch += 1
        net.train()
        epoch_loss = epoch_mse = epoch_mse_nb = 0.0 # epoch_nme = epoch_nme_nb =
        count_img = len(train_loader.dataset)
        start_train_time = perf_counter()
        # prof.start()
        for idx, (inputs, labels_map, labels_x, labels_y, labels_nb_x, labels_nb_y) in enumerate(train_loader):
            # prof.step()
            # if idx >= 1 + 1 + 3 + 1:
            #     break
            inputs = inputs.to(device); labels_map = labels_map.to(device);
            labels_x = labels_x.to(device); labels_y = labels_y.to(device);
            labels_nb_x = labels_nb_x.to(device); labels_nb_y = labels_nb_y.to(device)
            outputs_map, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y = net(inputs)
            loss_map, loss_x, loss_y, loss_nb_x, loss_nb_y = compute_loss_pip(outputs_map, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y, labels_map, labels_x, labels_y, labels_nb_x, labels_nb_y, criterion_cls, criterion_reg, num_nb)
            loss = cls_loss_weight*loss_map + reg_loss_weight*(loss_x + loss_y + loss_nb_x + loss_nb_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                lms_gts, _ = batch_process_pip_out(labels_map, labels_x, labels_y, labels_nb_x, labels_nb_y, reverse_index1, reverse_index2, max_len)
                lms_preds, lms_pred_merges = batch_process_pip_out(outputs_map, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y, reverse_index1, reverse_index2, max_len)
                epoch_mse += torch.nn.functional.mse_loss(lms_gts, lms_preds, reduction="none").mean(dim=1).sum()
                epoch_mse_nb += torch.nn.functional.mse_loss(lms_gts, lms_pred_merges, reduction='none').mean(dim=1).sum()
                epoch_loss += loss
        # prof.stop()
        # exit(0)
        epoch_loss /= len(train_loader)
        epoch_mse /= count_img
        epoch_mse_nb /= count_img
        training_usage = perf_counter() - start_train_time
        
        val_epoch_loss = val_epoch_mse = val_epoch_mse_nb = 0.0 # val_epoch_nme = val_epoch_nme_nb =
        count_img = len(val_loader.dataset)
        net.eval()
        start_val_time = perf_counter()
        # prof.start()
        with torch.no_grad():
            for idx, (inputs, labels_map, labels_x, labels_y, labels_nb_x, labels_nb_y) in enumerate(val_loader):
                # prof.step()
                # if idx >= 1 + 1 + 3: 
                #     break
                inputs = inputs.to(device, non_blocking=True); labels_map = labels_map.to(device, non_blocking=True);
                labels_x = labels_x.to(device, non_blocking=True); labels_y = labels_y.to(device, non_blocking=True);
                labels_nb_x = labels_nb_x.to(device, non_blocking=True); labels_nb_y = labels_nb_y.to(device, non_blocking=True)
                outputs_map, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y = net(inputs)
                loss_map, loss_x, loss_y, loss_nb_x, loss_nb_y = compute_loss_pip(outputs_map, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y, labels_map, labels_x, labels_y, labels_nb_x, labels_nb_y, criterion_cls, criterion_reg, num_nb)
                loss = cls_loss_weight*loss_map + reg_loss_weight*(loss_x + loss_y + loss_nb_x + loss_nb_y)

                lms_gts, _ = batch_process_pip_out(labels_map, labels_x, labels_y, labels_nb_x, labels_nb_y, reverse_index1, reverse_index2, max_len)
                lms_preds, lms_pred_merges = batch_process_pip_out(outputs_map, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y, reverse_index1, reverse_index2, max_len)

                val_epoch_mse += torch.nn.functional.mse_loss(lms_gts, lms_preds, reduction="none").mean(dim=1).sum()
                val_epoch_mse_nb += torch.nn.functional.mse_loss(lms_gts, lms_pred_merges, reduction='none').mean(dim=1).sum()
                val_epoch_loss += loss
        # prof.stop() 
        # exit(0)
                
        val_loader.num_workers = 4 # because of cache in the Custom dataset class
        val_epoch_loss /= len(val_loader)
        val_epoch_mse /= count_img
        val_epoch_mse_nb /= count_img
        validation_usage = perf_counter() - start_val_time
        metrics.push_metrics(trn_loss=epoch_loss, trn_mse=epoch_mse, trn_mse_nb=epoch_mse_nb, 
                             val_loss=val_epoch_loss, val_mse=val_epoch_mse, val_mse_nb=val_epoch_mse_nb)
        
        cur_metric = metrics.get_the_latest_value_of(indicate_best_by)
        if ((how_best == "min" and cur_metric < cur_best_metric) or (how_best == "max" and cur_metric > cur_best_metric)):
            msg = "found better %s %.6f (formerly %.6f), selecting new best model at epoch %d, at %s" % (indicate_best_by, metrics.get_the_latest_value_of(indicate_best_by), cur_best_metric, epoch, model_save_path)
            print(msg);logger.info(msg)
            torch.save(net.state_dict(), model_save_path)
            cur_best_metric = cur_metric
            if use_patient_and_save_best:
                count_current_epochs = 0
        msg = 'Epoch {:d}: trn elapsed: {:.2f}s val elapsed: {:.2f}s trn loss: {:.6f} val loss: {:.6f} trn mse: {:.6f} val mse: {:.6f} trn mse nb: {:.6f} val mse nb: {:.6f} best {:s}: {:.6f}'.format(
            epoch, training_usage, validation_usage, epoch_loss, val_epoch_loss, epoch_mse, val_epoch_mse, epoch_mse_nb, val_epoch_mse_nb, indicate_best_by, cur_best_metric)
        print(msg);logger.info(msg)
        if scheduler is not None:
            scheduler.step()  

    # convert metrics to python float
    metrics.cpu()
    return net, metrics
def plot_tsne_selected(train_data, train_data_feat_embbeded, clusters, activeSet):
    plt.title("TSNE of blink dataset")
    plt.scatter(train_data_feat_embbeded[:, 0], train_data_feat_embbeded[:, 1], c=clusters, s=0.5)
    plt.xlabel("TSNE dim 0")
    plt.ylabel("TSNE dim 1")
    for idx in activeSet:
        eye_info = train_data.get_patient_code_and_frame_from_idx(idx)
        frame_idx = eye_info["frame_idx"]
        plt.scatter(train_data_feat_embbeded[idx, 0], train_data_feat_embbeded[idx, 1], c="black", alpha=0.5)
        img, *_ = train_data[idx]
        imagebox = OffsetImage(np.array(img), zoom=0.15)
        ab = AnnotationBbox(imagebox, (train_data_feat_embbeded[idx, 0]-10, train_data_feat_embbeded[idx, 1]), frameon=False)
        plt.gca().add_artist(ab)
        plt.text(train_data_feat_embbeded[idx, 0]-10, train_data_feat_embbeded[idx, 1]+10, f"{eye_info['patient_code']} {frame_idx}/{eye_info['frames']}", horizontalalignment='center')
def plot_metrics(metrics: dict, title:str="loss", show=True, save_path=None):
    # `losses` and `ious` are dict that has key as label of the plot, and values as list of loss/iou values
    n_epoch = min([len(l) for l in metrics.values()])
    epochs = range(1, n_epoch+1)
    plt.title(title)
    for legend, metric_values in metrics.items():
        plt.plot(epochs, metric_values, "-o", label=legend, alpha=0.4)
    plt.legend()
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
        plt.close("all")
def plot_metrics_al(losses: dict, update_dataset_at: list, title:str="loss", show=False, save_path=None):
    plot_metrics(losses, title, show=False)
    fig = plt.gcf()
    for ax in fig.get_axes():
        for at in update_dataset_at:
            y_bot, y_top = ax.get_ylim()
            y_size_5p = (y_top - y_bot)*0.05
            ax.plot([at, at], [y_bot + y_size_5p, y_top - y_size_5p], "r", alpha=0.5)
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
        plt.close("all")



def plot_all_metrics(n_epochs_for_ac_iter, training_losses, validation_losses, training_mses, validation_mses, training_mses_nb, validation_mses_nb, save_path):
    plt.subplot(3, 1, 1)
    plot_metrics_al({"training loss": training_losses, "validation loss": validation_losses}, n_epochs_for_ac_iter, title="loss", show=False)
    # plt.yscale("log")
    plt.subplot(3, 1, 2)
    plot_metrics_al({"training mses": training_mses, "validation mses": validation_mses}, n_epochs_for_ac_iter, title="mse", show=False)
    plt.yscale("log")
    plt.subplot(3, 1, 3)
    plot_metrics_al({"training mses nb": training_mses_nb, "validation mses nb": validation_mses_nb}, n_epochs_for_ac_iter, title="mse nb", show=False)
    plt.yscale("log")
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

def get_frame_selected_patient_wise(all_sampled_set, train_data):
    frame_selected_patient_wise = {} 
    ori_shape = all_sampled_set.shape
    all_sampled_set = [train_data.get_patient_code_and_frame_from_idx(idx) for idx in all_sampled_set.flatten()]
    all_sampled_set = np.reshape(all_sampled_set, ori_shape)
    for episode_idx, selected in enumerate(all_sampled_set):
        for eye_info in selected:
            selected_frames_of_this_patient = frame_selected_patient_wise.setdefault(eye_info["keypoint_file"], [])
            selected_frames_of_this_patient.append((eye_info["frame_idx"], episode_idx, eye_info["dataset_idx"]))

    return frame_selected_patient_wise

def calculate_param_for_plot_embedding(embedding_path, dataset_info):
    embeddings = ds_utils.load_embededing_from_path(embedding_path)
    tb = np.concatenate([[0], dataset_info["frames"].cumsum()])
    kernel_size = 11
    kernel = np.ones(kernel_size)/kernel_size
    emb_list = []
    for row_idx in set(dataset_info.index):
        frames = dataset_info.loc[row_idx, "frames"]
        start_idx = tb[row_idx]
        end_idx = tb[row_idx+1]
        a_patient_embedding = embeddings[start_idx:end_idx] # embedding of a patient
        if np.prod(a_patient_embedding.shape) == 0:
                continue
        centroid_emb = a_patient_embedding.mean(0)
        mean_square_diff_emb = np.mean(np.square(a_patient_embedding - centroid_emb), axis=1)
        smoothed_mean_square_diff_emb = np.zeros_like(mean_square_diff_emb)
        smoothed_mean_square_diff_emb[kernel_size//2:(-kernel_size//2)+1] = np.convolve(mean_square_diff_emb, kernel, mode='valid')
        smoothed_mean_square_diff_emb[:kernel_size//2] = smoothed_mean_square_diff_emb[kernel_size//2]
        smoothed_mean_square_diff_emb[-kernel_size//2+1:] = smoothed_mean_square_diff_emb[-kernel_size//2+1]

        if cfg.ACTIVE_LEARNING.SAMPLING_FN == "embedding_difference_as_probability_density_reduce_high_frame_prob":
            emb_list.append(smoothed_mean_square_diff_emb/frames)
        else:
            emb_list.append(smoothed_mean_square_diff_emb)
    v = np.concatenate(emb_list)
    if cfg.ACTIVE_LEARNING.SAMPLING_FN == "embedding_difference_as_probability_density_with_softmax":
        v = torch.softmax(torch.from_numpy(v)*cfg.ACTIVE_LEARNING.SOFTMAX_TEMPERATURE, dim=0).numpy()
    w = (v - v.min())
    y = w / w.max()
    z = y / y.sum() # prob density
    pd_max = z.max()
    pd_min = z.min()
    return z, tb, pd_min, pd_max, 

def plot_selected_frames(exp_dir, dataset_info, patient_wise, train_data, use_embedding=False, z=None, tb=None, pd_min=None, pd_max=None):
    print(tb)
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
    
        plt.subplot(dataset_info.shape[0]//2, 2, row_idx + 1)
        plt.title(eye_info["patient_code"] + " " + eye_info["keypoints_key"][0])
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
        
        if use_embedding:
            plt.twinx()
            plt.plot(z[tb[idx]:tb[idx+1]], "o", markersize=1, alpha=0.5, )
            plt.ylim(pd_min, pd_max)
            plt.yticks(fontsize="xx-small")
            plt.ylabel("Probability density")
    plt.savefig(os.path.join(exp_dir, "selected_frames.png"), bbox_inches='tight')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def is_eval_epoch(cur_epoch):
    """Determines if the model should be evaluated at the current epoch."""
    return (
        (cur_epoch + 1) % cfg.TRAIN.EVAL_PERIOD == 0 or
        (cur_epoch + 1) == cfg.OPTIM.MAX_EPOCH
    )

def sanity_check_dataset(train_dataset, exp_dir, reverse_index1, reverse_index2, max_len):
    xy = train_dataset[0]
    img = xy[0]
    kp, kp_merge = batch_process_pip_out(xy[1].unsqueeze(0), xy[2].unsqueeze(0), xy[3].unsqueeze(0), xy[4].unsqueeze(0), xy[5].unsqueeze(0), reverse_index1, reverse_index2, max_len)
    plt.figure()
    imshow_kp(img, lms_pred=kp[0])
    plt.savefig(os.path.join(exp_dir, "assert-train.png"), bbox_inches='tight')
    plt.close("all")
    return np.isclose(kp, kp_merge).all()


def main(cfg):
    torch.backends.cudnn.benchmark = True
    # Setting up GPU args
    use_cuda = (cfg.NUM_GPUS > 0) and torch.cuda.is_available()
    device = torch.device(f"cuda:{cfg.GPU_ID}" if use_cuda else "cpu")
    # kwargs = {'num_workers': cfg.DATA_LOADER.NUM_WORKERS, 'pin_memory': cfg.DATA_LOADER.PIN_MEMORY} if use_cuda else {}

    # Auto assign a RNG_SEED when not supplied a value
    if cfg.RNG_SEED is None:
        # cfg.RNG_SEED = np.random.randint(100)
        raise NotImplementedError('seed must be provided')
    print("RNG_SEED:", cfg.RNG_SEED)
    
    # Using specific GPU
    # os.environ['NVIDIA_VISIBLE_DEVICES'] = str(cfg.GPU_ID)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # print("Using GPU : {}.\n".format(cfg.GPU_ID))

    # Getting the output directory ready (default is "/output")
    cfg.OUT_DIR = os.path.join(os.path.abspath('../..'), cfg.OUT_DIR)
    if not os.path.exists(cfg.OUT_DIR):
        os.mkdir(cfg.OUT_DIR)
    # Create "DATASET/MODEL TYPE" specific directory
    dataset_out_dir = os.path.join(cfg.OUT_DIR, cfg.DATASET.NAME, cfg.MODEL.TYPE)
    if not os.path.exists(dataset_out_dir):
        os.makedirs(dataset_out_dir)
    # Creating the experiment directory inside the dataset specific directory 
    # all logs, labeled, unlabeled, validation sets are stroed here 
    # E.g., output/CIFAR10/resnet18/{timestamp or cfg.EXP_NAME based on arguments passed}
    if cfg.EXP_NAME == 'auto':
        assert False
        now = datetime.now()
        exp_dir = f'{now.year}_{now.month}_{now.day}_{now.hour:02}{now.minute:02}{now.second:02}_{now.microsecond}'
    else:
        exp_name = cfg.EXP_NAME

    exp_dir = os.path.join(cfg.MODEL_GRAVEYARD, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    print("Directory is {}.\n".format(exp_dir))
    # if not os.path.exists(exp_dir):
    #     os.makedirs(exp_dir)
    #     print("Directory is {}.\n".format(exp_dir))
    # else:
    #     assert False, f"please explicitly delete the experiment result by yourself at {exp_dir}"
    #     print("Experiment Directory Already Exists: {}. Reusing it may lead to loss of old logs in the directory.\n".format(exp_dir))
    cfg.EXP_DIR = exp_dir

    # Save the config file in EXP_DIR
    dump_cfg(cfg)

    # Setup Logger
    lu.setup_logging(cfg)

    # Dataset preparing steps
    print("\n======== PREPARING DATA AND MODEL ========\n")
    cfg.DATASET.ROOT_DIR = os.path.join(os.path.abspath('../..'), cfg.DATASET.ROOT_DIR)
    data_obj = Data(cfg)
    all_data, train_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=True, isDownload=True, fold_idx=cfg.DATASET.FOLD_IDX, is_blinking=cfg.DATASET.IS_BLINKING)
    if isinstance(all_data, (BlinkDataset, BlinkDataset2, BlinkDatasetAll)):
        dataset_info = all_data.dataset_info
    elif isinstance(all_data, ImageDataFrameBlinkingWrapper):
        dataset_info = all_data.dataset.dataset_info
    else:
        dataset_info = None
        print("expect blink dataset")
        exit(0)
    all_patient_code = dataset_info["patient_code"].unique()
    msg = "Dataset {} Loaded Sucessfully. Total Train Size: {}\n".format(cfg.DATASET.NAME, train_size,)
    print(msg);logger.info(msg)
    print(all_patient_code)
    # exit(0)
    for patient_code in all_patient_code:
        print("training set size before spliting", len(all_data))
        exp_patient_dir = os.path.join(cfg.EXP_DIR, patient_code)
        os.makedirs(exp_patient_dir, exist_ok=True)
        train_patient_code = np.setdiff1d(all_patient_code, patient_code)
        lSet_path, uSet_path = data_obj.makeLUSetsByPatients(train_patient_code, all_data, exp_patient_dir)
        cfg.ACTIVE_LEARNING.LSET_PATH = lSet_path
        cfg.ACTIVE_LEARNING.USET_PATH = uSet_path
        # cfg.ACTIVE_LEARNING.VALSET_PATH = valSet_path

        lSet, uSet = data_obj.loadLUPartitions(lSetPath=lSet_path, uSetPath=uSet_path)
        model = model_builder.get_keypoint_model(num_nb, num_lms, input_size, net_stride)
        init_model_state_dict = torch.load(cfg.MODEL_INIT_WEIGHTS, map_location=device)
        model.load_state_dict(init_model_state_dict, strict=False); model.train()
        # model = model.to(device)
        model = torch.nn.parallel.DataParallel(model, [device])
        
        # # Construct the optimizer
        optimizer = optim.construct_optimizer(cfg, model)
        # opt_init_state = deepcopy(optimizer.state_dict())
        cfg_for_finetune = cfg.clone()
        cfg_for_finetune.OPTIM.BASE_LR = cfg_for_finetune.OPTIM.BASE_LR * 1e-2
        # model_init_state = deepcopy(model.state_dict().copy())

        print("optimizer: {}\n".format(optimizer))
        logger.info("optimizer: {}\n".format(optimizer))

        print("AL Query Method: {}\nMax AL Episodes: {}\n".format(cfg.ACTIVE_LEARNING.SAMPLING_FN, cfg.ACTIVE_LEARNING.MAX_ITER))
        logger.info("AL Query Method: {}\nMax AL Episodes: {}\n".format(cfg.ACTIVE_LEARNING.SAMPLING_FN, cfg.ACTIVE_LEARNING.MAX_ITER))

        use_patient_and_save_best = False
        n_epochs = cfg.TRAIN.NUM_EPOCHS
        cls_loss_weight = 10
        reg_loss_weight = 1
        pil_augment = True
        criterion_cls = nn.MSELoss()
        criterion_reg = nn.L1Loss()
        scheduler = None
        
        indicate_best_by = cfg.TRAIN.INDICATE_BEST_BY
        how_best="min"
        skip_training = False

        # calculate meanface here?
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        base_transforms = transforms.Compose([transforms.ToTensor(), normalize])
        
        vSet = np.random.choice(lSet, 1024, replace=False)
        lSet = np.setdiff1d(lSet, vSet)
        meanface, meanface_indices, reverse_index1, reverse_index2, max_len = cal_and_process_meanface([all_data[idx] for idx in lSet], num_lms, num_nb)
        reverse_index1 = torch.tensor(reverse_index1, dtype=torch.long).to(device)
        reverse_index2 = torch.tensor(reverse_index2, dtype=torch.long).to(device)
        train_dataset = CustomImageDataset4(all_data, input_size, num_lms, net_stride, meanface_indices, 
                                            base_transforms, pil_augment=pil_augment, allow_idxs=lSet)
        val_dataset = CustomImageDataset4(all_data, input_size, num_lms, net_stride, meanface_indices, 
                                          base_transforms, pil_augment=False, allow_idxs=vSet)

        # plot assertion, if any keypoint are not align, check immediately
        assert sanity_check_dataset(train_dataset, exp_dir, reverse_index1, reverse_index2, max_len)

        train_dataloader_wrapper = functools.partial(DataLoader, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.DATA_LOADER.NUM_WORKERS, shuffle=True, pin_memory=True)
        val_dataloader_wrapper = functools.partial(DataLoader, batch_size=cfg.TEST.BATCH_SIZE, num_workers=0, shuffle=False, pin_memory=True)
        lSet_loader = train_dataloader_wrapper(train_dataset)
        valSet_loader = val_dataloader_wrapper(val_dataset)
        
        msg = "Leaving {} as unlabeled set, Labeled Set: {}, Unlabeled Set: {}\n".format(patient_code, len(lSet), len(uSet))
        print(msg); logger.info(msg)

        last_pretrained_model_save_path = os.path.join(exp_patient_dir, f"init_last_{len(train_dataset)}n.pt")
        best_pretrained_model_save_path = cfg.PRETRAINED_WEIGHTS_EACH_PATIENT.format(patient_code=patient_code, n_train=len(train_dataset))
        if not os.path.exists(best_pretrained_model_save_path):
            best_pretrained_model_save_path = os.path.join(exp_patient_dir, f"init_best_{len(train_dataset)}n.pt")

        metrics = MetricsStorage()
        # Train init model
        msg = "======== TRAINING INIT ========"
        print(msg);logger.info(msg)
        if not skip_training and not os.path.exists(best_pretrained_model_save_path):
            model, metrics = train_model_pip(model, lSet_loader, criterion_cls, criterion_reg, cls_loss_weight, reg_loss_weight, 
                                            num_nb, reverse_index1, reverse_index2, max_len, optimizer, n_epochs, scheduler, 
                                            best_pretrained_model_save_path, use_patient_and_save_best, device, 
                                            valSet_loader, metrics, indicate_best_by, how_best)
            pretrained_model_state_dict = model.state_dict().copy()
            torch.save(pretrained_model_state_dict, last_pretrained_model_save_path)
            with open(os.path.join(exp_patient_dir, "training_metrics.pkl"), 'wb') as f:
                pickle.dump(metrics.to_dict(), f)
        if not skip_training or os.path.exists(best_pretrained_model_save_path):
            print(f"load existing baseline model from : {best_pretrained_model_save_path}")
            pretrained_model_state_dict = torch.load(best_pretrained_model_save_path)
            model.load_state_dict(pretrained_model_state_dict)
        if metrics.training_losses is not None and not os.path.exists(os.path.join(exp_patient_dir, "init_losses.png")):
            plt.figure(figsize=(1000/100, 1000/100), dpi=100)
            plot_all_metrics(metrics.n_epochs_for_ac_iter, metrics.training_losses, metrics.validation_losses, metrics.training_mses, metrics.validation_mses, metrics.training_mses_nb, metrics.validation_mses_nb, os.path.join(exp_patient_dir, "init_losses.png"))
            plt.close("all")
        
        # model_mse, model_mse_nb, model_mae, model_mae_nb = eval_on_pipnet(model, lSet_loader, reverse_index1, reverse_index2, max_len, device)
        # print("train: ", model_mse, model_mse_nb, model_mae, model_mae_nb)
        # model_mse, model_mse_nb, model_mae, model_mae_nb = eval_on_pipnet(model, valSet_loader, reverse_index1, reverse_index2, max_len, device)
        # print("val:", model_mse, model_mse_nb, model_mae, model_mae_nb)
        # imshow_kp_from_dl(model, lSet_loader, valSet_loader, device, reverse_index1, reverse_index2, max_len, "train_train_val.png")
        
        # exit(0)
        lSet = np.asarray([])
        all_sampled_set = []
        msg = "drop all patient, to be added from uSet"
        print(msg);logger.info(msg)
        metrics = MetricsStorage()
        for cur_episode in range(0, cfg.ACTIVE_LEARNING.MAX_ITER):
            print("======== EPISODE {} BEGINS ========\n".format(cur_episode))
            logger.info("======== EPISODE {} BEGINS ========\n".format(cur_episode))

            # Creating output directory for the episode
            episode_dir = os.path.join(exp_patient_dir, f'episode_{cur_episode}')
            os.makedirs(episode_dir, exist_ok=True)
            cfg.EPISODE_DIR = episode_dir

             # Active Sample 
            print("======== ACTIVE SAMPLING ========\n")
            logger.info("======== ACTIVE SAMPLING ========\n")
            al_obj = ActiveLearning(data_obj, cfg, cur_episode+1, dataset_info)
            if cfg.ACTIVE_LEARNING.SAMPLING_FN == "typiclust_rp":
                activeSet, new_uSet, clusters = al_obj.sample_from_uSet(model, lSet, uSet, all_data, dataset_info=dataset_info if "blink" in cfg.DATASET.NAME else None, patient_code=patient_code)
            else:
                activeSet, new_uSet = al_obj.sample_from_uSet(model, lSet, uSet, all_data, dataset_info=dataset_info if "blink" in cfg.DATASET.NAME else None, patient_code=patient_code)

            all_sampled_set.append(activeSet)
            # Save current lSet, new_uSet and activeSet in the episode directory
            data_obj.saveSets(lSet, uSet, activeSet, episode_dir)

            # Add activeSet to lSet, save new_uSet as uSet and update dataloader for the next episode
            lSet = np.append(lSet, activeSet).astype(int)
            uSet = new_uSet
            msg = "Active Sampling Complete. After Episode {}:\nNew Labeled Set: {}, New Unlabeled Set: {}, Active Set: {}\n".format(cur_episode, len(lSet), len(uSet), len(activeSet))
            print(msg);logger.info(msg)
            
            # # use meanface from all previous patients
            train_dataset = CustomImageDataset4(all_data, input_size, num_lms, net_stride, meanface_indices, base_transforms,
                                                pil_augment=pil_augment, allow_idxs=lSet)
            # val_dataset = CustomImageDataset4(train_data, input_size, num_lms, net_stride, meanface_indices, train_transforms,
            #                                   pil_augment=False, allow_idxs=vSet)
            
            lSet_loader = train_dataloader_wrapper(train_dataset)
            # valSet_loader = val_dataloader_wrapper(val_dataset) # use the same validation set from 
            print("================================\n\n")
            logger.info("================================\n\n")

            if not skip_training and not cfg.ACTIVE_LEARNING.FINE_TUNE: 
                # load model from the previous al iter
                # init_model_state_dict = torch.load(best_init_model_save_path)
                print("load existing pretrained model")
                model.load_state_dict(pretrained_model_state_dict)
                
            # optimizer.load_state_dict(opt_init_state)
            optimizer = optim.construct_optimizer(cfg_for_finetune, model)
            # print("optimizer for finetuning", optimizer)
            
            if not skip_training:
                # Train model
                print("======== TRAINING ========")
                logger.info("======== TRAINING ========")
                best_model_save_path = os.path.join(exp_patient_dir,  f"best_{len(train_dataset)}n.pt")
                last_model_save_path = os.path.join(exp_patient_dir, f"last_{len(train_dataset)}n.pt")
                model, metrics = train_model_pip(model, lSet_loader, criterion_cls, criterion_reg, cls_loss_weight, reg_loss_weight, 
                                                num_nb, reverse_index1, reverse_index2, max_len, optimizer, n_epochs, scheduler, 
                                                best_model_save_path, use_patient_and_save_best, device, 
                                                valSet_loader, metrics, indicate_best_by, how_best)
                metrics.next_al_iter()
                torch.save(model.state_dict(), last_model_save_path)

                # Test best model checkpoint
                print("======== TESTING ========\n")
                logger.info("======== TESTING ========\n")
                model_mse, model_mse_nb, model_mae, model_mae_nb = eval_on_pipnet(model, lSet_loader, reverse_index1, reverse_index2, max_len, device)
                print("train: ", model_mse, model_mse_nb, model_mae, model_mae_nb)
                model_mse, model_mse_nb, model_mae, model_mae_nb = eval_on_pipnet(model, valSet_loader, reverse_index1, reverse_index2, max_len, device)
                print("val:", model_mse, model_mse_nb, model_mae, model_mae_nb)

            data_obj.saveSet(lSet, 'lSet', episode_dir)
            data_obj.saveSet(uSet, 'uSet', episode_dir)
            if len(uSet) == 0:
                print("Get out of training loop since out of unlabeled set")
                break

            if metrics.training_losses is not None:
                plt.figure(figsize=(1000/100, 1000/100), dpi=100)
                plot_all_metrics(metrics.n_epochs_for_ac_iter, metrics.training_losses, metrics.validation_losses, metrics.training_mses, metrics.validation_mses, metrics.training_mses_nb, metrics.validation_mses_nb, os.path.join(exp_patient_dir, "al_losses.png"))
                plt.close("all")
            # os.remove(checkpoint_file)
            # break
        with open(os.path.join(exp_patient_dir, "training_metrics.pkl"), 'wb') as f:
            pickle.dump(metrics, f)

        if metrics.training_losses is not None:
            plt.figure(figsize=(1000/100, 1000/100), dpi=100)
            plot_all_metrics(metrics.n_epochs_for_ac_iter, metrics.training_losses, metrics.validation_losses, metrics.training_mses, metrics.validation_mses, metrics.training_mses_nb, metrics.validation_mses_nb, os.path.join(exp_patient_dir, "al_losses.png"))
            plt.close("all")
        if isinstance(all_data, ImageDataFrameBlinkingWrapper):
            all_data = all_data.dataset
        all_sampled_set = np.asarray(all_sampled_set)
        if all_sampled_set.size > 0:
            frame_selected_patient_wise = get_frame_selected_patient_wise(all_sampled_set, all_data)
            use_embedding = any([cfg.ACTIVE_LEARNING.SAMPLING_FN in s for s in ["probcover", "typiclust_rp", "embedding_difference_as_probability_density", ]])
            z = tb = pd_max = pd_min = None
            if use_embedding:
                z, tb, pd_min, pd_max = calculate_param_for_plot_embedding(cfg.ACTIVE_LEARNING.EMBEDDING_PATH.format(patient_code=patient_code), dataset_info)
            plot_selected_frames(exp_patient_dir, dataset_info, frame_selected_patient_wise, all_data, use_embedding, z, tb, pd_min, pd_max)

        # break # for debug purpose

def argparser():
    parser = argparse.ArgumentParser(description='Active Learning - Image Classification')
    parser.add_argument('--cfg', dest='cfg_file', help='Config file', required=True, type=str)
    # parser.add_argument('--exp-name', help='Experiment Name', required=True, type=str)
    # parser.add_argument('--al', help='AL Method', required=True, type=str)
    # parser.add_argument('--budget', help='Budget Per Round', required=True, type=int)
    # parser.add_argument('--initial_size', help='Size of the initial random labeled set', default="0", type=str)
    # parser.add_argument('--seed', help='Random seed', default=1, type=int)
    # parser.add_argument('--finetune', help='Whether to continue with existing model between rounds', type=str2bool, default=False)
    # parser.add_argument('--linear_from_features', help='Whether to use a linear layer from self-supervised features', action='store_true')
    # parser.add_argument('--delta', help='Relevant only for ProbCover', default=0.6, type=float)
    # parser.add_argument('--frame_diff_factor', default=0.5, type=float)
    parser.add_argument('--gpu_id', default='0', type=str)
    parser.add_argument('--model_graveyard', default='../model_graveyard', type=str)
    return parser

if __name__ == "__main__":
    args = argparser().parse_args()
    cfg.merge_from_file(args.cfg_file)

    cfg.GPU_ID = args.gpu_id
    cfg.MODEL_GRAVEYARD = args.model_graveyard
    print(cfg)

    main(cfg)
