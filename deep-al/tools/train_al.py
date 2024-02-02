import os
import pickle
import sys
from datetime import datetime
import argparse
from time import perf_counter
import random
from math import floor, ceil
import copy

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional
import torch.nn as nn
import torch.optim as optim
import torch.profiler
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)    

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
# import pycls.core.optimizer as optim
from pycls.datasets.data import Data
import pycls.utils.checkpoint as cu
import pycls.utils.logging as lu
import pycls.utils.metrics as mu
import pycls.utils.net as nu
from pycls.utils.meters import TestMeter, TrainMeter, ValMeter
from pycls.datasets.blink_dataset import BlinkDataset, ImageDataFrameBlinkingWrapper
from pycls.datasets.blink_dataset_with_no_test_set import BlinkDataset2

logger = lu.get_logger(__name__)
# model params
input_size = 256
num_nb = 10
num_lms = 21
net_stride = 32
MODEL_GRAVEYARD = "../model_graveyard"

plot_episode_xvalues = []
plot_episode_yvalues = []

plot_epoch_xvalues = []
plot_epoch_yvalues = []

plot_it_x_values = []
plot_it_y_values = []

def process_meanface(meanface, num_nb):
    meanface = np.array(meanface).reshape(-1, 2)
    # each landmark predicts num_nb neighbors
    meanface_indices = []
    for i in range(meanface.shape[0]):
        pt = meanface[i,:]
        dists = np.sum(np.power(pt-meanface, 2), axis=1)
        indices = np.argsort(dists)
        meanface_indices.append(indices[1:1+num_nb])
    
    # each landmark predicted by X neighbors, X varies
    meanface_indices_reversed = {}
    for i in range(meanface.shape[0]):
        meanface_indices_reversed[i] = [[],[]]
    for i in range(meanface.shape[0]):
        for j in range(num_nb):
            meanface_indices_reversed[meanface_indices[i][j]][0].append(i)
            meanface_indices_reversed[meanface_indices[i][j]][1].append(j)
    
    max_len = 0
    for i in range(meanface.shape[0]):
        tmp_len = len(meanface_indices_reversed[i][0])
        if tmp_len > max_len:
            max_len = tmp_len
    
    # tricks, make them have equal length for efficient computation
    for i in range(meanface.shape[0]):
        tmp_len = len(meanface_indices_reversed[i][0])
        meanface_indices_reversed[i][0] += meanface_indices_reversed[i][0]*10
        meanface_indices_reversed[i][1] += meanface_indices_reversed[i][1]*10
        meanface_indices_reversed[i][0] = meanface_indices_reversed[i][0][:max_len]
        meanface_indices_reversed[i][1] = meanface_indices_reversed[i][1][:max_len]

    # make the indices 1-dim
    reverse_index1 = []
    reverse_index2 = []
    for i in range(meanface.shape[0]):
        reverse_index1 += meanface_indices_reversed[i][0]
        reverse_index2 += meanface_indices_reversed[i][1]
    return meanface_indices, reverse_index1, reverse_index2, max_len

def cal_and_process_meanface(train_data, num_lms=num_lms, num_nb=num_nb):
    meanface = np.zeros((num_lms*2, ))
    for idx in range(len(train_data)):
        img, lms = train_data[idx]
        meanface += lms
    meanface /= len(train_data)
    meanface_indices, reverse_index1, reverse_index2, max_len = process_meanface(meanface, num_nb)
    return meanface, meanface_indices, reverse_index1, reverse_index2, max_len

def compute_loss_pip(outputs_map, 
                     outputs_local_x, outputs_local_y, 
                     outputs_nb_x, outputs_nb_y, 
                     labels_map, 
                     labels_local_x, labels_local_y, 
                     labels_nb_x, labels_nb_y,  
                     criterion_cls, criterion_reg, num_nb):
    tmp_batch, tmp_channel, tmp_height, tmp_width = outputs_map.size()
    labels_map = labels_map.view(tmp_batch*tmp_channel, -1)
    labels_max_ids = torch.argmax(labels_map, 1)
    labels_max_ids = labels_max_ids.view(-1, 1)
    labels_max_ids_nb = labels_max_ids.repeat(1, num_nb).view(-1, 1)

    outputs_local_x = outputs_local_x.view(tmp_batch*tmp_channel, -1)
    outputs_local_x_select = torch.gather(outputs_local_x, 1, labels_max_ids)
    outputs_local_y = outputs_local_y.view(tmp_batch*tmp_channel, -1)
    outputs_local_y_select = torch.gather(outputs_local_y, 1, labels_max_ids)
    outputs_nb_x = outputs_nb_x.view(tmp_batch*num_nb*tmp_channel, -1)
    outputs_nb_x_select = torch.gather(outputs_nb_x, 1, labels_max_ids_nb)
    outputs_nb_y = outputs_nb_y.view(tmp_batch*num_nb*tmp_channel, -1)
    outputs_nb_y_select = torch.gather(outputs_nb_y, 1, labels_max_ids_nb)

    labels_local_x = labels_local_x.view(tmp_batch*tmp_channel, -1)
    labels_local_x_select = torch.gather(labels_local_x, 1, labels_max_ids)
    labels_local_y = labels_local_y.view(tmp_batch*tmp_channel, -1)
    labels_local_y_select = torch.gather(labels_local_y, 1, labels_max_ids)
    labels_nb_x = labels_nb_x.view(tmp_batch*num_nb*tmp_channel, -1)
    labels_nb_x_select = torch.gather(labels_nb_x, 1, labels_max_ids_nb)
    labels_nb_y = labels_nb_y.view(tmp_batch*num_nb*tmp_channel, -1)
    labels_nb_y_select = torch.gather(labels_nb_y, 1, labels_max_ids_nb)

    labels_map = labels_map.view(tmp_batch, tmp_channel, tmp_height, tmp_width)
    loss_map = criterion_cls(outputs_map, labels_map)
    loss_x = criterion_reg(outputs_local_x_select, labels_local_x_select)
    loss_y = criterion_reg(outputs_local_y_select, labels_local_y_select)
    loss_nb_x = criterion_reg(outputs_nb_x_select, labels_nb_x_select)
    loss_nb_y = criterion_reg(outputs_nb_y_select, labels_nb_y_select)
    return loss_map, loss_x, loss_y, loss_nb_x, loss_nb_y

def process_pip_out(outputs_cls, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y, reverse_index1, reverse_index2, max_len):
    tmp_batch, tmp_channel, tmp_height, tmp_width = outputs_cls.size()
    assert tmp_batch == 1, str(outputs_cls.size())

    outputs_cls = outputs_cls.view(tmp_batch*tmp_channel, -1)
    max_ids = torch.argmax(outputs_cls, 1)
    # max_cls = torch.max(outputs_cls, 1)[0]
    max_ids = max_ids.view(-1, 1)
    max_ids_nb = max_ids.repeat(1, num_nb).view(-1, 1)

    outputs_x = outputs_x.view(tmp_batch*tmp_channel, -1)
    outputs_x_select = torch.gather(outputs_x, 1, max_ids)
    outputs_x_select = outputs_x_select.squeeze(1)
    outputs_y = outputs_y.view(tmp_batch*tmp_channel, -1)
    outputs_y_select = torch.gather(outputs_y, 1, max_ids)
    outputs_y_select = outputs_y_select.squeeze(1)

    outputs_nb_x = outputs_nb_x.view(tmp_batch*num_nb*tmp_channel, -1)
    outputs_nb_x_select = torch.gather(outputs_nb_x, 1, max_ids_nb)
    outputs_nb_x_select = outputs_nb_x_select.squeeze(1).view(-1, num_nb)
    outputs_nb_y = outputs_nb_y.view(tmp_batch*num_nb*tmp_channel, -1)
    outputs_nb_y_select = torch.gather(outputs_nb_y, 1, max_ids_nb)
    outputs_nb_y_select = outputs_nb_y_select.squeeze(1).view(-1, num_nb)

    lms_pred_x = (max_ids%tmp_width).view(-1,1).float()+outputs_x_select.view(-1,1)
    lms_pred_y = (max_ids//tmp_width).view(-1,1).float()+outputs_y_select.view(-1,1)
    lms_pred_x /= 1.0 * input_size / net_stride
    lms_pred_y /= 1.0 * input_size / net_stride
    lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten()

    tmp_nb_x = (max_ids%tmp_width).view(-1,1).float()+outputs_nb_x_select
    tmp_nb_y = (max_ids//tmp_width).view(-1,1).float()+outputs_nb_y_select
    tmp_nb_x = tmp_nb_x.view(-1, num_nb)
    tmp_nb_y = tmp_nb_y.view(-1, num_nb)
    tmp_nb_x /= 1.0 * input_size / net_stride
    tmp_nb_y /= 1.0 * input_size / net_stride
    tmp_nb_x = tmp_nb_x[reverse_index1, reverse_index2].view(num_lms, max_len)
    tmp_nb_y = tmp_nb_y[reverse_index1, reverse_index2].view(num_lms, max_len)
    tmp_nb_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1,1)
    tmp_nb_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1,1)
    lms_pred_merge = torch.cat((tmp_nb_x, tmp_nb_y), dim=1).flatten()
    
    return lms_pred, lms_pred_merge

def batch_process_pip_out(outputs_cls, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y, reverse_index1, reverse_index2, max_len):
    assert outputs_cls.ndim == 4
    tmp_batch, tmp_channel, tmp_height, tmp_width = outputs_cls.size()
    outputs_cls = outputs_cls.view(tmp_batch, tmp_channel, -1)
    max_ids = torch.argmax(outputs_cls, 2)
    # max_cls = torch.max(outputs_cls, 1)[0]
    max_ids = max_ids.view(tmp_batch, -1, 1)
    max_ids_nb = max_ids.repeat(1, 1, num_nb).view(tmp_batch, -1, 1)
    # assert (max_ids_nb[1].flatten() == max_ids_nb1.flatten()).all()
    
    outputs_x = outputs_x.view(tmp_batch, tmp_channel, -1)
    outputs_x_select = torch.gather(outputs_x, 2, max_ids)
    outputs_x_select = outputs_x_select.squeeze(2)
    outputs_y = outputs_y.view(tmp_batch, tmp_channel, -1)
    outputs_y_select = torch.gather(outputs_y, 2, max_ids)
    outputs_y_select = outputs_y_select.squeeze(2)
    # assert (outputs_x[0].flatten() == outputs_x1.flatten()).all() and (outputs_y_select[0].flatten() == outputs_y_select1.flatten()).all()
    
    outputs_nb_x = outputs_nb_x.view(tmp_batch, num_nb*tmp_channel, -1)
    outputs_nb_x_select = torch.gather(outputs_nb_x, 2, max_ids_nb)
    outputs_nb_x_select = outputs_nb_x_select.squeeze(2).view(tmp_batch, -1, num_nb)
    outputs_nb_y = outputs_nb_y.view(tmp_batch, num_nb*tmp_channel, -1)
    outputs_nb_y_select = torch.gather(outputs_nb_y, 2, max_ids_nb)
    outputs_nb_y_select = outputs_nb_y_select.squeeze(2).view(tmp_batch, -1, num_nb)
    # assert (outputs_nb_x[0].flatten() == outputs_nb_x1.flatten()).all() and (outputs_nb_y_select[0].flatten() == outputs_nb_y_select1.flatten()).all()
    
    lms_pred_x = (max_ids%tmp_width).view(tmp_batch, -1, 1).float()+outputs_x_select.view(tmp_batch, -1, 1)
    lms_pred_y = (max_ids//tmp_width).view(tmp_batch, -1, 1).float()+outputs_y_select.view(tmp_batch, -1, 1)
    lms_pred_x /= 1.0 * input_size / net_stride
    lms_pred_y /= 1.0 * input_size / net_stride
    lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=2).view(tmp_batch, -1)
    # assert (lms_pred_y[0].flatten() == lms_pred_y1.flatten()).all() and (lms_pred[0].flatten() == lms_pred1.flatten()).all() 
    
    tmp_nb_x = (max_ids%tmp_width).view(tmp_batch, -1, 1).float()+outputs_nb_x_select
    tmp_nb_y = (max_ids//tmp_width).view(tmp_batch, -1, 1).float()+outputs_nb_y_select
    tmp_nb_x = tmp_nb_x.view(tmp_batch, -1, num_nb)
    tmp_nb_y = tmp_nb_y.view(tmp_batch, -1, num_nb)
    tmp_nb_x /= 1.0 * input_size / net_stride
    tmp_nb_y /= 1.0 * input_size / net_stride
    tmp_nb_x = tmp_nb_x[:, reverse_index1, reverse_index2].view(tmp_batch, num_lms, max_len)
    tmp_nb_y = tmp_nb_y[:, reverse_index1, reverse_index2].view(tmp_batch, num_lms, max_len)
    tmp_nb_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=2), dim=2).view(tmp_batch, -1, 1)
    tmp_nb_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=2), dim=2).view(tmp_batch, -1, 1)
    lms_pred_merge = torch.cat((tmp_nb_x, tmp_nb_y), dim=2).view(tmp_batch, -1)
    # assert (lms_pred_merge[0].flatten() == lms_pred_merge1.flatten()).all()
    return lms_pred, lms_pred_merge

def compute_nme(lms_pred, lms_gt, norm):
    lms_pred = lms_pred.reshape((-1, 2))
    lms_gt = lms_gt.reshape((-1, 2))
    nme = np.mean(np.linalg.norm(lms_pred - lms_gt, axis=1)) / norm 
    return nme

def train_model_pip(net, train_loader, criterion_cls, criterion_reg, cls_loss_weight, reg_loss_weight, 
                num_nb, reverse_index1, reverse_index2, max_len, optimizer, num_epochs, scheduler, model_save_path, 
                use_patient_and_save_best, device, val_loader, metrics, indicate_best_by:str, how_best:str="min"):
    training_losses = metrics.setdefault("training_losses", [])
    # training_nmes = metrics.setdefault("training_nmes", [])
    # training_nmes_nb = metrics.setdefault("training_nmes_nb", [])
    training_mses = metrics.setdefault("training_mses", [])
    training_mses_nb = metrics.setdefault("training_mses_nb", [])

    validation_losses = metrics.setdefault("validation_losses", [])
    # validation_nmes = metrics.setdefault("validation_nmes", [])
    # validation_nmes_nb = metrics.setdefault("validation_nmes_nb", [])
    validation_mses = metrics.setdefault("validation_mses", [])
    validation_mses_nb = metrics.setdefault("validation_mses_nb", [])
    assert indicate_best_by in metrics and how_best in ["min", "max"]
     
    # norm_factor = 1.0
    epoch = len(training_losses)
    cur_best_metric = np.inf if how_best == "min" else -np.inf
    count_current_epochs = 0
    # prof = torch.profiler.profile(
    #         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    #         on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18'),
    #         record_shapes=True,
    #         with_stack=True, )

    while count_current_epochs < num_epochs:
        count_current_epochs += 1
        epoch += 1
        net.train()
        epoch_loss = epoch_mse = epoch_mse_nb = 0.0 # epoch_nme = epoch_nme_nb =
        count_img = 0
        start_train_time = perf_counter()
        for inputs, labels_map, labels_x, labels_y, labels_nb_x, labels_nb_y in train_loader:
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
                count_img += lms_gts.shape[0]
                epoch_loss += loss
        epoch_loss /= len(train_loader)
        # epoch_nme /= count_img
        # epoch_nme_nb /= count_img
        epoch_mse /= count_img
        epoch_mse_nb /= count_img
        training_losses.append(epoch_loss)
        # training_nmes.append(epoch_nme)
        # training_nmes_nb.append(epoch_nme_nb)
        training_mses.append(epoch_mse)
        training_mses_nb.append(epoch_mse_nb)
        training_usage = perf_counter() - start_train_time
        
        val_epoch_loss = val_epoch_mse = val_epoch_mse_nb = 0.0 # val_epoch_nme = val_epoch_nme_nb =
        count_img = 0
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
                count_img += lms_gts.shape[0]
                val_epoch_loss += loss
        # prof.stop() 
        # exit(0)
                
        val_loader.num_workers = 8 # because of cache in the Custom dataset class
        val_epoch_loss /= len(val_loader)
        # val_epoch_nme /= count_img
        # val_epoch_nme_nb /= count_img
        val_epoch_mse /= count_img
        val_epoch_mse_nb /= count_img
        validation_losses.append(val_epoch_loss)
        # validation_nmes.append(val_epoch_nme)
        # validation_nmes_nb.append(val_epoch_nme_nb)
        validation_mses.append(val_epoch_mse)
        validation_mses_nb.append(val_epoch_mse_nb)
        validation_usage = perf_counter() - start_val_time
        
        cur_metric = metrics[indicate_best_by][-1]
        if use_patient_and_save_best and ((how_best == "min" and cur_metric < cur_best_metric) or (how_best == "max" and cur_metric > cur_best_metric)):
            msg = "found better %s %.6f (formerly %.6f), selecting new best model at epoch %d, at %s" % (indicate_best_by, metrics[indicate_best_by][-1], cur_best_metric, len(training_losses), model_save_path)
            print(msg)
            logger.info(msg)
            torch.save(net.state_dict(), model_save_path)
            cur_best_metric = cur_metric
            count_current_epochs = 0
        msg = 'Epoch {:d}: trn elapsed: {:.2f}s val elapsed: {:.2f}s trn loss: {:.6f} val loss: {:.6f} trn mse: {:.6f} val mse: {:.6f} trn mse nb: {:.6f} val mse nb: {:.6f} best {:s}: {:.6f}'.format(
            epoch, training_usage, validation_usage, epoch_loss, val_epoch_loss, epoch_mse, val_epoch_mse, epoch_mse_nb, val_epoch_mse_nb, indicate_best_by, cur_best_metric)
        print(msg)
        logger.info(msg)
        if scheduler is not None:
            scheduler.step()  

    # convert metrics to python float
    for k in metrics:
        for idx in range(len(metrics[k])):
            if isinstance(metrics[k][idx], torch.Tensor):
                metrics[k][idx] = metrics[k][idx].item()
    return net, metrics

def random_translate(image, target):
    if random.random() > 0.5:
        image_height, image_width = image.size
        a = 1
        b = 0
        #c = 30 #left/right (i.e. 5/-5)
        c = int((random.random()-0.5) * 60)
        d = 0
        e = 1
        #f = 30 #up/down (i.e. 5/-5)
        f = int((random.random()-0.5) * 60)
        image = image.transform(image.size, Image.Transform.AFFINE, (a, b, c, d, e, f))
        target_translate = target.copy()
        target_translate = target_translate.reshape(-1, 2)
        target_translate[:, 0] -= 1.*c/image_width
        target_translate[:, 1] -= 1.*f/image_height
        target_translate = target_translate.flatten()
        target_translate[target_translate < 0] = 0
        target_translate[target_translate > 1] = 1
        return image, target_translate
    else:
        return image, target

def random_blur(image):
    if random.random() > 0.7:
        image = image.filter(ImageFilter.GaussianBlur(random.random()*5))
    return image

def random_occlusion(image):
    if random.random() > 0.5:
        image_np = np.array(image).astype(np.uint8)
        image_np = image_np[:,:,::-1]
        image_height, image_width, _ = image_np.shape
        occ_height = int(image_height*0.4*random.random())
        occ_width = int(image_width*0.4*random.random())
        occ_xmin = int((image_width - occ_width - 10) * random.random())
        occ_ymin = int((image_height - occ_height - 10) * random.random())
        image_np[occ_ymin:occ_ymin+occ_height, occ_xmin:occ_xmin+occ_width, 0] = int(random.random() * 255)
        image_np[occ_ymin:occ_ymin+occ_height, occ_xmin:occ_xmin+occ_width, 1] = int(random.random() * 255)
        image_np[occ_ymin:occ_ymin+occ_height, occ_xmin:occ_xmin+occ_width, 2] = int(random.random() * 255)
        image_pil = Image.fromarray(image_np[:,:,::-1].astype('uint8'), 'RGB')
        return image_pil
    else:
        return image

def random_flip(image, target, points_flip):
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        target = np.array(target).reshape(-1, 2)
        target = target[points_flip, :]
        target[:,0] = 1-target[:,0]
        target = target.flatten()
        return image, target
    else:
        return image, target

def random_rotate(image, target, angle_max):
    if random.random() > 0.5:
        center_x = 0.5
        center_y = 0.5
        landmark_num= int(len(target) / 2)
        target_center = np.array(target) - np.array([center_x, center_y]*landmark_num)
        target_center = target_center.reshape(landmark_num, 2)
        theta_max = np.radians(angle_max)
        theta = random.uniform(-theta_max, theta_max)
        angle = np.degrees(theta)
        image = image.rotate(angle)

        c, s = np.cos(theta), np.sin(theta)
        rot = np.array(((c,-s), (s, c)))
        target_center_rot = np.matmul(target_center, rot)
        target_rot = target_center_rot.reshape(landmark_num*2) + np.array([center_x, center_y]*landmark_num)
        return image, target_rot
    else:
        return image, target

def random_crop(image, target, left_most, right_least, up_most, bot_least):
    # left_most, right_least, up_most, bot_least = (0.1, 0.89375, 0.19791667, 0.8020833)
    original_size = (image.width, image.height)
    left = random.uniform(0, left_most)
    right = random.uniform(right_least, 1)
    up = random.uniform(0, up_most)
    bot = random.uniform(bot_least, 1)
    image = image.crop((left * original_size[0], up * original_size[1], right * original_size[0], bot * original_size[1]))
    image = image.resize(original_size)
    target[::2] = (target[::2] - left) / (right - left)
    target[1::2] = (target[1::2] - up) / (bot - up) 
    assert (target >= 0).all() and (target <= 1).all()
    return image, target

def random_brightness(image, min_range, max_range):
    enhancer = ImageEnhance.Brightness(image)
    factor = random.uniform(min_range, max_range)
    image = enhancer.enhance(factor)
    return image

def random_crop_from_bigger_image(image, target, after_crop_size):
    ori_width = image.width; ori_height = image.height
    left_most = ori_width - after_crop_size
    up_most = ori_height - after_crop_size
    left = random.randrange(0, left_most)
    up = random.randrange(0, up_most)
    image = image.crop((left, up, left+after_crop_size, up+after_crop_size))
    target[::2] = (target[::2]*ori_width - left) / after_crop_size
    target[1::2] = (target[1::2]*ori_height - up) / after_crop_size
    assert ((target >= 0) & (target <= 1)).all()
    return image, target

def crop_center(img, target, input_size):
    width, height = img.size  
    left = (width - input_size)/2
    top = (height - input_size)/2
    right = (width + input_size)/2
    bottom = (height + input_size)/2
    img = img.crop((left, top, right, bottom))
    target[::2] = (target[::2] * width - left) / input_size
    target[1::2] = (target[1::2] * height - top) / input_size
    return img, target

def gen_target_pip(target, meanface_indices, target_map, target_local_x, target_local_y, target_nb_x, target_nb_y):
    num_nb = len(meanface_indices[0])
    map_channel, map_height, map_width = target_map.shape
    target = target.reshape(-1, 2)
    assert map_channel == target.shape[0]

    for i in range(map_channel):
        mu_x = int(floor(target[i][0] * map_width))
        mu_y = int(floor(target[i][1] * map_height))
        mu_x = max(0, mu_x)
        mu_y = max(0, mu_y)
        mu_x = min(mu_x, map_width-1)
        mu_y = min(mu_y, map_height-1)
        target_map[i, mu_y, mu_x] = 1
        shift_x = target[i][0] * map_width - mu_x
        shift_y = target[i][1] * map_height - mu_y
        target_local_x[i, mu_y, mu_x] = shift_x
        target_local_y[i, mu_y, mu_x] = shift_y

        for j in range(num_nb):
            nb_x = target[meanface_indices[i][j]][0] * map_width - mu_x
            nb_y = target[meanface_indices[i][j]][1] * map_height - mu_y
            target_nb_x[num_nb*i+j, mu_y, mu_x] = nb_x
            target_nb_y[num_nb*i+j, mu_y, mu_x] = nb_y

    return target_map, target_local_x, target_local_y, target_nb_x, target_nb_y

class CustomImageDataset4(Dataset):
    def __init__(self, dataset, input_size, num_lms, net_stride, meanface_indices, transform=None, target_transform=None, pil_augment=True, allow_idxs=None):
        super().__init__()
        self.num_lms = num_lms
        self.net_stride = net_stride
        # self.points_flip = points_flip
        self.meanface_indices = meanface_indices
        self.num_nb = len(meanface_indices[0])
        self.transform = transform
        self.target_transform = target_transform
        self.input_size = input_size
        self.pil_augment = pil_augment
        
        self.dataset = dataset 
        if allow_idxs is None:
            self.allow_idxs = np.arange(len(dataset), dtype="uint32")
        else:
            self.allow_idxs = allow_idxs
        assert len(self.allow_idxs) <= len(self.dataset)

        # cache for small dataset
        self.cache_xy = None
        if len(self.allow_idxs) < 5000:
            self.cache_xy = [None] * len(self.allow_idxs)
        
    def __len__(self):
        return len(self.allow_idxs)
    
    def __getitem__(self, idx):
        mapped_idx = self.allow_idxs[idx] # N -> N
        
        if self.cache_xy is not None:
            if self.cache_xy[idx] is None:
                self.cache_xy[idx] = self.dataset[mapped_idx]
            img, target = self.cache_xy[idx]
            img, target = copy.deepcopy(img), target.copy()
        else:
            img, target = self.dataset[mapped_idx]

        if self.pil_augment:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            # img, target = random_crop(img, target, 0.1 - 0.01, 0.89375 + 0.01, 0.19791667 - 0.01, 0.8020833 + 0.01)
            # img, target = random_translate(img, target)
            # img = random_occlusion(img)
            # img, target = random_flip(img, target, self.points_flip)

            img, target = random_crop_from_bigger_image(img, target, self.input_size)
            img, target = random_rotate(img, target, 30)
            img = random_blur(img)
            img = random_brightness(img, 0.9, 1.3)
        else:
            img, target = crop_center(img, target, self.input_size)

        target_map = np.zeros((self.num_lms, int(self.input_size/self.net_stride), int(self.input_size/self.net_stride)))
        target_local_x = np.zeros((self.num_lms, int(self.input_size/self.net_stride), int(self.input_size/self.net_stride)))
        target_local_y = np.zeros((self.num_lms, int(self.input_size/self.net_stride), int(self.input_size/self.net_stride)))
        target_nb_x = np.zeros((self.num_nb*self.num_lms, int(self.input_size/self.net_stride), int(self.input_size/self.net_stride)))
        target_nb_y = np.zeros((self.num_nb*self.num_lms, int(self.input_size/self.net_stride), int(self.input_size/self.net_stride)))
        target_map, target_local_x, target_local_y, target_nb_x, target_nb_y = gen_target_pip(target, self.meanface_indices, target_map, target_local_x, target_local_y, target_nb_x, target_nb_y)
        
        target_map = torch.from_numpy(target_map).float()
        target_local_x = torch.from_numpy(target_local_x).float()
        target_local_y = torch.from_numpy(target_local_y).float()
        target_nb_x = torch.from_numpy(target_nb_x).float()
        target_nb_y = torch.from_numpy(target_nb_y).float()

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target_map = self.target_transform(target_map)
            target_local_x = self.target_transform(target_local_x)
            target_local_y = self.target_transform(target_local_y)
            target_nb_x = self.target_transform(target_nb_x)
            target_nb_y = self.target_transform(target_nb_y)

        return img, target_map, target_local_x, target_local_y, target_nb_x, target_nb_y

def imshow_kp(img, lms_gt=None, lms_pred=None, lms_pred_merge=None, *, img_channel_first=True, img_normalized=True, show_kp_idx=False):
    if img_normalized:
        img = img*0.22+0.45 # good enough approximation of unnormlized
    if img_channel_first:
        img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)
    input_size = img.shape[1]
    if lms_gt is not None:
        plt.scatter(lms_gt[::2]*input_size, lms_gt[1::2]*input_size, alpha=0.4, s=5, label="gt", c="C0")
        if show_kp_idx:
            for x_idx, y_idx in zip(range(0, num_lms*2, 2), range(1, num_lms*2, 2)):
                plt.text(lms_gt[x_idx], lms_gt[y_idx], f"({x_idx}, {y_idx})", fontsize=6, c="C0", horizontalalignment='right')
    if lms_pred is not None:
        plt.scatter(lms_pred[::2]*input_size, lms_pred[1::2]*input_size, alpha=0.4, s=5, label="pred", c="C1")
        if show_kp_idx:
            for x_idx, y_idx in zip(range(0, num_lms*2, 2), range(1, num_lms*2, 2)):
                plt.text(lms_pred[x_idx], lms_pred[y_idx], f"({x_idx}, {y_idx})", fontsize=6, c="C1", horizontalalignment='right')
    if lms_pred_merge is not None:
        plt.scatter(lms_pred_merge[::2]*input_size, lms_pred_merge[1::2]*input_size, alpha=0.4, s=5, label="pred nb", c="C2")
        if show_kp_idx:
            for x_idx, y_idx in zip(range(0, num_lms*2, 2), range(1, num_lms*2, 2)):
                plt.text(lms_pred_merge[x_idx], lms_pred_merge[y_idx], f"({x_idx}, {y_idx})", fontsize=6, c="C2", horizontalalignment='right')
    # plt.xticks([])
    # plt.yticks([])
    plt.legend()

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

    exp_dir = os.path.join(cfg.MODEL_GRAVEYARD, exp_name, f"stepsize10n_fold{cfg.DATASET.FOLD_IDX}")
    print(exp_dir)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        print("Experiment Directory is {}.\n".format(exp_dir))
    else:
        assert False, "please explicitly delete the experiment result by yourself"
        print("Experiment Directory Already Exists: {}. Reusing it may lead to loss of old logs in the directory.\n".format(exp_dir))
    cfg.EXP_DIR = exp_dir

    # Save the config file in EXP_DIR
    dump_cfg(cfg)

    # Setup Logger
    lu.setup_logging(cfg)

    # Dataset preparing steps
    print("\n======== PREPARING DATA AND MODEL ========\n")
    cfg.DATASET.ROOT_DIR = os.path.join(os.path.abspath('../..'), cfg.DATASET.ROOT_DIR)
    data_obj = Data(cfg)
    train_data, train_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=True, isDownload=True, fold_idx=cfg.DATASET.FOLD_IDX, is_blinking=cfg.DATASET.IS_BLINKING)
    val_data, val_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=False, isDownload=True, fold_idx=cfg.DATASET.FOLD_IDX, is_blinking=None)
    # if args.initial_size == "all":
    #     cfg.ACTIVE_LEARNING.INIT_L_RATIO = 1.0
    # else:
    #     cfg.ACTIVE_LEARNING.INIT_L_RATIO = int(args.initial_size) / train_size
    print("\nDataset {} Loaded Sucessfully.\nTotal Train Size: {} and Total Test Size: {}\n".format(cfg.DATASET.NAME, train_size, val_size))
    logger.info("Dataset {} Loaded Sucessfully. Total Train Size: {} and Total Test Size: {}\n".format(cfg.DATASET.NAME, train_size, val_size))

    lSet_path, uSet_path, valSet_path = data_obj.makeLUVSets(train_split_ratio=cfg.ACTIVE_LEARNING.INIT_L_RATIO,
        val_split_ratio=0., data=train_data, seed_id=cfg.RNG_SEED, save_dir=cfg.EXP_DIR) # cfg.DATASET.VAL_RATIO

    cfg.ACTIVE_LEARNING.LSET_PATH = lSet_path
    cfg.ACTIVE_LEARNING.USET_PATH = uSet_path
    cfg.ACTIVE_LEARNING.VALSET_PATH = valSet_path

    lSet, uSet, valSet = data_obj.loadPartitions(lSetPath=cfg.ACTIVE_LEARNING.LSET_PATH, \
            uSetPath=cfg.ACTIVE_LEARNING.USET_PATH, valSetPath = cfg.ACTIVE_LEARNING.VALSET_PATH)
    model = model_builder.get_keypoint_model(num_nb, num_lms, input_size, net_stride)
    init_model_state_dict = torch.load(cfg.MODEL_INIT_WEIGHTS, map_location=device)
    model.load_state_dict(init_model_state_dict); model.train()
    model = model.to(device)
    # model = model_builder.build_model(cfg).cuda()
    
    all_sampled_set = []
    if isinstance(train_data, (BlinkDataset, BlinkDataset2)):
        dataset_info = train_data.dataset_info
    elif isinstance(train_data, ImageDataFrameBlinkingWrapper):
        dataset_info = train_data.dataset.dataset_info
    else:
        dataset_info = None
    if len(lSet) == 0:
        print('Labeled Set is Empty - Sampling an Initial Pool')
        al_obj = ActiveLearning(data_obj, cfg)
        if cfg.ACTIVE_LEARNING.SAMPLING_FN == "typiclust_rp":
            activeSet, new_uSet, clusters = al_obj.sample_from_uSet(model, lSet, uSet, train_data, dataset_info=dataset_info)
            # train_data_feat_embbeded = TSNE(2, init="pca").fit_transform(train_data.features)
            # plot_tsne_selected(train_data, train_data_feat_embbeded, clusters, activeSet)  
        else:
            activeSet, new_uSet = al_obj.sample_from_uSet(model, lSet, uSet, train_data, dataset_info=dataset_info)
        print(f'Initial Pool is {activeSet}')
        all_sampled_set.append(activeSet)
        # Save current lSet, new_uSet and activeSet in the episode directory
        # data_obj.saveSets(lSet, uSet, activeSet, cfg.EPISODE_DIR)
        # Add activeSet to lSet, save new_uSet as uSet and update dataloader for the next episode
        lSet = np.append(lSet, activeSet)
        uSet = new_uSet
        
    print("Data Partitioning Complete. \nLabeled Set: {}, Unlabeled Set: {}, Validation Set: {}\n".format(len(lSet), len(uSet), len(valSet)))
    logger.info("Labeled Set: {}, Unlabeled Set: {}, Validation Set: {}\n".format(len(lSet), len(uSet), len(valSet)))

    # Preparing dataloaders for initial training
    # lSet_loader = data_obj.getIndexesDataLoader(indexes=lSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
    # valSet_loader = data_obj.getIndexesDataLoader(indexes=valSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
    # test_loader = data_obj.getTestLoader(data=val_data, test_batch_size=cfg.TRAIN.BATCH_SIZE, seed_id=cfg.RNG_SEED)

    # # Initialize the model.  
    # model = model_builder.build_model(cfg)
    # print("model: {}\n".format(cfg.MODEL.TYPE))
    # logger.info("model: {}\n".format(cfg.MODEL.TYPE))

    # # Construct the optimizer
    # optimizer = optim.construct_optimizer(cfg, model)
    # opt_init_state = deepcopy(optimizer.state_dict())
    # model_init_state = deepcopy(model.state_dict().copy())

    # # print("optimizer: {}\n".format(optimizer))
    # logger.info("optimizer: {}\n".format(optimizer))

    print("AL Query Method: {}\nMax AL Episodes: {}\n".format(cfg.ACTIVE_LEARNING.SAMPLING_FN, cfg.ACTIVE_LEARNING.MAX_ITER))
    logger.info("AL Query Method: {}\nMax AL Episodes: {}\n".format(cfg.ACTIVE_LEARNING.SAMPLING_FN, cfg.ACTIVE_LEARNING.MAX_ITER))

    use_patient_and_save_best = True
    n_epochs = cfg.TRAIN.NUM_EPOCHS
    cls_loss_weight = 10
    reg_loss_weight = 1
    init_lr = 0.001
    lr_decay_steps = [30, 50]
    lr_decay_factor = 0.1
    pil_augment = True
    # BATCH_SIZE = batch_size = 32
    criterion_cls = nn.MSELoss()
    criterion_reg = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_steps, gamma=lr_decay_factor)
    # scheduler = None
    metrics = {}
    training_losses = metrics.setdefault("training_losses", [])
    validation_losses = metrics.setdefault("validation_losses", [])
    training_mses = metrics.setdefault("training_mses", [])
    validation_mses = metrics.setdefault("validation_mses", [])
    training_mses_nb = metrics.setdefault("training_mses_nb", [])
    validation_mses_nb = metrics.setdefault("validation_mses_nb", [])
    n_epochs_for_ac_iter = metrics.setdefault("n_epochs_for_ac_iter", [])
    indicate_best_by = cfg.TRAIN.INDICATE_BEST_BY
    how_best="min"
    skip_training = False

    # calculate meanface here?
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transforms = transforms.Compose([transforms.ToTensor(), normalize])
    val_transforms = train_transforms
    meanface, meanface_indices, reverse_index1, reverse_index2, max_len = cal_and_process_meanface([train_data[idx] for idx in lSet], num_lms, num_nb)
    reverse_index1 = torch.tensor(reverse_index1, dtype=torch.long).to(device)
    reverse_index2 = torch.tensor(reverse_index2, dtype=torch.long).to(device)
    train_dataset = CustomImageDataset4(train_data, input_size, num_lms, net_stride, meanface_indices, 
                                        train_transforms, 
                                        pil_augment=pil_augment, allow_idxs=lSet)
    val_dataset = CustomImageDataset4(val_data, input_size, num_lms, net_stride, meanface_indices, 
                                      val_transforms, 
                                      pil_augment=False)
    
    # plot assertion, if any keypoint are not align, check immediately
    xy = train_dataset[0]
    img = xy[0]
    plt.figure()
    kp, kp_merge = process_pip_out(xy[1].unsqueeze(0), xy[2].unsqueeze(0), xy[3].unsqueeze(0), xy[4].unsqueeze(0), xy[5].unsqueeze(0), reverse_index1, reverse_index2, max_len)
    assert np.isclose(kp, kp_merge).all()
    imshow_kp(img, lms_pred=kp)
    plt.savefig(os.path.join(MODEL_GRAVEYARD, "assert-train.png"), bbox_inches='tight')
    
    xy = val_dataset[0]
    img = xy[0]
    plt.figure()
    kp, kp_merge = process_pip_out(xy[1].unsqueeze(0), xy[2].unsqueeze(0), xy[3].unsqueeze(0), xy[4].unsqueeze(0), xy[5].unsqueeze(0), reverse_index1, reverse_index2, max_len)
    assert np.isclose(kp, kp_merge).all()
    imshow_kp(img, lms_pred=kp)
    plt.savefig(os.path.join(MODEL_GRAVEYARD, "assert-val.png"), bbox_inches='tight')

    train_dataloader_wrapper = functools.partial(DataLoader, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=0, shuffle=True, pin_memory=False)
    val_dataloader_wrapper = functools.partial(DataLoader, batch_size=512, num_workers=0, shuffle=False, pin_memory=False)
    lSet_loader = train_dataloader_wrapper(train_dataset)
    valSet_loader = val_dataloader_wrapper(val_dataset)

    for cur_episode in range(0, cfg.ACTIVE_LEARNING.MAX_ITER):

        print("======== EPISODE {} BEGINS ========\n".format(cur_episode))
        logger.info("======== EPISODE {} BEGINS ========\n".format(cur_episode))

        # Creating output directory for the episode
        episode_dir = os.path.join(cfg.EXP_DIR, f'episode_{cur_episode}')
        if not os.path.exists(episode_dir):
            os.mkdir(episode_dir)
        cfg.EPISODE_DIR = episode_dir

        if not skip_training:
            # Train model
            print("======== TRAINING ========")
            logger.info("======== TRAINING ========")
            best_model_save_path = os.path.join(exp_dir, f"best_{len(train_dataset)}n.pt")
            last_model_save_path = os.path.join(exp_dir, f"last_{len(train_dataset)}n.pt")
            model, metrics = train_model_pip(model, lSet_loader, criterion_cls, criterion_reg, cls_loss_weight, reg_loss_weight, 
                                            num_nb, reverse_index1, reverse_index2, max_len, optimizer, n_epochs, scheduler, 
                                            best_model_save_path, use_patient_and_save_best, device, valSet_loader, metrics, 
                                            indicate_best_by, how_best)
            torch.save(model.state_dict(), last_model_save_path)
            n_epochs_for_ac_iter.append(len(metrics[indicate_best_by]))
            # best_val_acc, best_val_epoch, checkpoint_file = train_model(lSet_loader, valSet_loader, model, optimizer, cfg)

            # print("Best Validation Accuracy: {}\nBest Epoch: {}\n".format(round(best_val_acc, 4), best_val_epoch))
            # logger.info("EPISODE {} Best Validation Accuracy: {}\tBest Epoch: {}\n".format(cur_episode, round(best_val_acc, 4), best_val_epoch))

            # Test best model checkpoint
            print("======== TESTING ========\n")
            logger.info("======== TESTING ========\n")
            # test_acc = test_model(test_loader, checkpoint_file, cfg, cur_episode)
            # print("Test Accuracy: {}.\n".format(round(test_acc, 4)))
            # logger.info("EPISODE {} Test Accuracy {}.\n".format(cur_episode, test_acc))

        # No need to perform active sampling in the last episode iteration
        if cur_episode+1 == cfg.ACTIVE_LEARNING.MAX_ITER:
            # Save current lSet, uSet in the final episode directory
            data_obj.saveSet(lSet, 'lSet', cfg.EPISODE_DIR)
            data_obj.saveSet(uSet, 'uSet', cfg.EPISODE_DIR)
            break
        if len(uSet) == 0:
            print("Get out of training loop since out of unlabeled set")
            break
        # Active Sample 
        print("======== ACTIVE SAMPLING ========\n")
        logger.info("======== ACTIVE SAMPLING ========\n")
        al_obj = ActiveLearning(data_obj, cfg)
        # clf_model = model_builder.build_model(cfg)
        # clf_model = cu.load_checkpoint(checkpoint_file, clf_model)
        # activeSet, new_uSet, clusters = al_obj.sample_from_uSet(clf_model, lSet, uSet, train_data)
        if cfg.ACTIVE_LEARNING.SAMPLING_FN == "typiclust_rp":
            activeSet, new_uSet, clusters = al_obj.sample_from_uSet(model, lSet, uSet, train_data, dataset_info=dataset_info if "blink" in cfg.DATASET.NAME else None)
            # plt.figure()
            # plot_tsne_selected(train_data, train_data_feat_embbeded, clusters, activeSet)
            # plt.savefig(os.path.join(MODEL_GRAVEYARD, f"selected al iter{cur_episode}.png"))
            # plt.clf()
        else:
            activeSet, new_uSet = al_obj.sample_from_uSet(model, lSet, uSet, train_data, dataset_info=dataset_info if "blink" in cfg.DATASET.NAME else None)

        all_sampled_set.append(activeSet)
        # Save current lSet, new_uSet and activeSet in the episode directory
        data_obj.saveSets(lSet, uSet, activeSet, cfg.EPISODE_DIR)

        # Add activeSet to lSet, save new_uSet as uSet and update dataloader for the next episode
        lSet = np.append(lSet, activeSet)
        uSet = new_uSet

        if not skip_training:
            meanface, meanface_indices, reverse_index1, reverse_index2, max_len = cal_and_process_meanface([train_data[idx] for idx in lSet], num_lms, num_nb)
        train_dataset = CustomImageDataset4(train_data, input_size, num_lms, net_stride, meanface_indices, train_transforms,
                                            pil_augment=pil_augment, allow_idxs=lSet)
        val_dataset = CustomImageDataset4(val_data, input_size, num_lms, net_stride, meanface_indices, val_transforms,
                                          pil_augment=False)
        
        lSet_loader = train_dataloader_wrapper(train_dataset)
        valSet_loader = val_dataloader_wrapper(val_dataset)

        # lSet_loader = data_obj.getIndexesDataLoader(indexes=lSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
        # valSet_loader = data_obj.getIndexesDataLoader(indexes=valSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
        # uSet_loader = data_obj.getSequentialDataLoader(indexes=uSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)

        print("Active Sampling Complete. After Episode {}:\nNew Labeled Set: {}, New Unlabeled Set: {}, Active Set: {}\n".format(cur_episode, len(lSet), len(uSet), len(activeSet)))
        logger.info("Active Sampling Complete. After Episode {}:\nNew Labeled Set: {}, New Unlabeled Set: {}, Active Set: {}\n".format(cur_episode, len(lSet), len(uSet), len(activeSet)))
        print("================================\n\n")
        logger.info("================================\n\n")

        if not cfg.ACTIVE_LEARNING.FINE_TUNE:
            # start model from scratch
            print('Starting model from scratch - ignoring existing weights.')
            # model = model_builder.build_model(cfg)
            # # Construct the optimizer
            # optimizer = optim.construct_optimizer(cfg, model)
            # print(model.load_state_dict(model_init_state))
            # print(optimizer.load_state_dict(opt_init_state))
            # model = model_builder.get_keypoint_model(num_nb, num_lms, input_size, net_stride).to(device)
            model.load_state_dict(init_model_state_dict); model.train()
            model = model.to(device)

            optimizer = optim.Adam(model.parameters(), lr=init_lr)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_steps, gamma=lr_decay_factor)
        if training_losses is not None:
            plt.figure(figsize=(1000/100, 800/100), dpi=100)
            plt.subplot(3, 1, 1)
            plot_metrics_al({"training loss": training_losses, "validation loss": validation_losses}, n_epochs_for_ac_iter, title="loss", show=False)
            plt.yscale("log")
            plt.subplot(3, 1, 2)
            plot_metrics_al({"training mses": training_mses, "validation mses": validation_mses}, n_epochs_for_ac_iter, title="mse", show=False)
            plt.yscale("log")
            plt.subplot(3, 1, 3)
            plot_metrics_al({"training mses nb": training_mses_nb, "validation mses nb": validation_mses_nb}, n_epochs_for_ac_iter, title="mse nb", show=False)
            plt.yscale("log")
            plt.savefig(os.path.join(exp_dir, "al_losses.png"), bbox_inches="tight")
        # os.remove(checkpoint_file)
    with open(os.path.join(exp_dir, "training_metrics.pkl"), 'wb') as f:
        pickle.dump(metrics, f)

    if training_losses is not None:
        plt.figure(figsize=(1000/100, 800/100), dpi=100)
        plt.subplot(3, 1, 1)
        plot_metrics_al({"training loss": training_losses, "validation loss": validation_losses}, n_epochs_for_ac_iter, title="loss", show=False)
        plt.yscale("log")
        plt.subplot(3, 1, 2)
        plot_metrics_al({"training mses": training_mses, "validation mses": validation_mses}, n_epochs_for_ac_iter, title="mse", show=False)
        plt.yscale("log")
        plt.subplot(3, 1, 3)
        plot_metrics_al({"training mses nb": training_mses_nb, "validation mses nb": validation_mses_nb}, n_epochs_for_ac_iter, title="mse nb", show=False)
        plt.yscale("log")
        plt.savefig(os.path.join(exp_dir, "al_losses.png"), bbox_inches="tight")
    if isinstance(train_data, ImageDataFrameBlinkingWrapper):
        train_data = train_data.dataset
    all_sampled_set = np.asarray(all_sampled_set)
    if all_sampled_set.size > 0:
        ori_shape = all_sampled_set.shape
        all_sampled_set = [train_data.get_patient_code_and_frame_from_idx(idx) for idx in all_sampled_set.flatten()]
        all_sampled_set = np.reshape(all_sampled_set, ori_shape)
        patient_wise = {} 
        for episode_idx, selected in enumerate(all_sampled_set):
            for eye_info in selected:
                selected_frames_of_this_patient = patient_wise.setdefault(eye_info["keypoint_file"], [])
                selected_frames_of_this_patient.append((eye_info["frame_idx"], episode_idx, eye_info["dataset_idx"]))
        plt.figure(figsize=(2000/150, 3000/150), dpi=150)
        plt.subplots_adjust(hspace=0.4)
        for row_idx, eye_info in dataset_info.iterrows():
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
        plt.savefig(os.path.join(exp_dir, "selected_frames.png"), bbox_inches='tight')

def train_model(train_loader, val_loader, model, optimizer, cfg):
    global plot_episode_xvalues
    global plot_episode_yvalues

    global plot_epoch_xvalues
    global plot_epoch_yvalues

    global plot_it_x_values
    global plot_it_y_values

    start_epoch = 0
    loss_fun = losses.get_loss_fun()

    # Create meters
    train_meter = TrainMeter(len(train_loader))
    val_meter = ValMeter(len(val_loader))

    # Perform the training loop
    # print("Len(train_loader):{}".format(len(train_loader)))
    logger.info('Start epoch: {}'.format(start_epoch + 1))
    val_set_acc = 0.

    temp_best_val_acc = 0.
    temp_best_val_epoch = 0

    # Best checkpoint model and optimizer states
    best_model_state = None
    best_opt_state = None

    val_acc_epochs_x = []
    val_acc_epochs_y = []

    clf_train_iterations = cfg.OPTIM.MAX_EPOCH * int(len(train_loader)/cfg.TRAIN.BATCH_SIZE)
    clf_change_lr_iter = clf_train_iterations // 25
    clf_iter_count = 0

    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):

        # Train for one epoch
        train_loss, clf_iter_count = train_epoch(train_loader, model, loss_fun, optimizer, train_meter, \
                                        cur_epoch, cfg, clf_iter_count, clf_change_lr_iter, clf_train_iterations)

        # Compute precise BN stats
        if cfg.BN.USE_PRECISE_STATS:
            nu.compute_precise_bn_stats(model, train_loader)


        # Model evaluation
        if is_eval_epoch(cur_epoch):
            # Original code[PYCLS] passes on testLoader but we want to compute on val Set
            val_loader.dataset.no_aug = True
            val_set_err = test_epoch(val_loader, model, val_meter, cur_epoch)
            val_set_acc = 100. - val_set_err
            val_loader.dataset.no_aug = False
            if temp_best_val_acc < val_set_acc:
                temp_best_val_acc = val_set_acc
                temp_best_val_epoch = cur_epoch + 1

                # Save best model and optimizer state for checkpointing
                model.eval()

                best_model_state = model.module.state_dict() if cfg.NUM_GPUS > 1 else model.state_dict()
                best_opt_state = optimizer.state_dict()

                model.train()

            # Since we start from 0 epoch
            val_acc_epochs_x.append(cur_epoch+1)
            val_acc_epochs_y.append(val_set_acc)

        plot_epoch_xvalues.append(cur_epoch+1)
        plot_epoch_yvalues.append(train_loss)

        # save_plot_values([plot_epoch_xvalues, plot_epoch_yvalues, plot_it_x_values, plot_it_y_values, val_acc_epochs_x, val_acc_epochs_y],\
        #     ["plot_epoch_xvalues", "plot_epoch_yvalues", "plot_it_x_values", "plot_it_y_values","val_acc_epochs_x","val_acc_epochs_y"], out_dir=cfg.EPISODE_DIR, isDebug=False)
        logger.info("Successfully logged numpy arrays!!")

        # Plot arrays
        # plot_arrays(x_vals=plot_epoch_xvalues, y_vals=plot_epoch_yvalues, \
        # x_name="Epochs", y_name="Loss", dataset_name=cfg.DATASET.NAME, out_dir=cfg.EPISODE_DIR)
        #
        # plot_arrays(x_vals=val_acc_epochs_x, y_vals=val_acc_epochs_y, \
        # x_name="Epochs", y_name="Validation Accuracy", dataset_name=cfg.DATASET.NAME, out_dir=cfg.EPISODE_DIR)

        # save_plot_values([plot_epoch_xvalues, plot_epoch_yvalues, plot_it_x_values, plot_it_y_values, val_acc_epochs_x, val_acc_epochs_y], \
        #         ["plot_epoch_xvalues", "plot_epoch_yvalues", "plot_it_x_values", "plot_it_y_values","val_acc_epochs_x","val_acc_epochs_y"], out_dir=cfg.EPISODE_DIR)

        print('Training Epoch: {}/{}\tTrain Loss: {}\tVal Accuracy: {}'.format(cur_epoch+1, cfg.OPTIM.MAX_EPOCH, round(train_loss, 4), round(val_set_acc, 4)))

    # Save the best model checkpoint (Episode level)
    checkpoint_file = cu.save_checkpoint(info="vlBest_acc_"+str(int(temp_best_val_acc)), \
        model_state=best_model_state, optimizer_state=best_opt_state, epoch=temp_best_val_epoch, cfg=cfg)

    print('\nWrote Best Model Checkpoint to: {}\n'.format(checkpoint_file.split('/')[-1]))
    logger.info('Wrote Best Model Checkpoint to: {}\n'.format(checkpoint_file))

    # plot_arrays(x_vals=plot_epoch_xvalues, y_vals=plot_epoch_yvalues, \
    #     x_name="Epochs", y_name="Loss", dataset_name=cfg.DATASET.NAME, out_dir=cfg.EPISODE_DIR)
    #
    # plot_arrays(x_vals=plot_it_x_values, y_vals=plot_it_y_values, \
    #     x_name="Iterations", y_name="Loss", dataset_name=cfg.DATASET.NAME, out_dir=cfg.EPISODE_DIR)
    #
    # plot_arrays(x_vals=val_acc_epochs_x, y_vals=val_acc_epochs_y, \
    #     x_name="Epochs", y_name="Validation Accuracy", dataset_name=cfg.DATASET.NAME, out_dir=cfg.EPISODE_DIR)

    plot_epoch_xvalues = []
    plot_epoch_yvalues = []
    plot_it_x_values = []
    plot_it_y_values = []

    best_val_acc = temp_best_val_acc
    best_val_epoch = temp_best_val_epoch

    return best_val_acc, best_val_epoch, checkpoint_file


def test_model(test_loader, checkpoint_file, cfg, cur_episode):

    global plot_episode_xvalues
    global plot_episode_yvalues

    global plot_epoch_xvalues
    global plot_epoch_yvalues

    global plot_it_x_values
    global plot_it_y_values

    test_meter = TestMeter(len(test_loader))

    model = model_builder.build_model(cfg)
    model = cu.load_checkpoint(checkpoint_file, model)

    test_err = test_epoch(test_loader, model, test_meter, cur_episode)
    test_acc = 100. - test_err

    plot_episode_xvalues.append(cur_episode)
    plot_episode_yvalues.append(test_acc)

    # plot_arrays(x_vals=plot_episode_xvalues, y_vals=plot_episode_yvalues, \
    #     x_name="Episodes", y_name="Test Accuracy", dataset_name=cfg.DATASET.NAME, out_dir=cfg.EXP_DIR)
    #
    # save_plot_values([plot_episode_xvalues, plot_episode_yvalues], \
    #     ["plot_episode_xvalues", "plot_episode_yvalues"], out_dir=cfg.EXP_DIR)

    return test_acc


def train_epoch(train_loader, model, loss_fun, optimizer, train_meter, cur_epoch, cfg, clf_iter_count, clf_change_lr_iter, clf_max_iter):
    """Performs one epoch of training."""
    global plot_episode_xvalues
    global plot_episode_yvalues

    global plot_epoch_xvalues
    global plot_epoch_yvalues

    global plot_it_x_values
    global plot_it_y_values

    # Shuffle the data
    #loader.shuffle(train_loader, cur_epoch)
    if cfg.NUM_GPUS>1:  train_loader.sampler.set_epoch(cur_epoch)

    # Update the learning rate
    # Currently we only support LR schedules for only 'SGD' optimizer
    lr = optim.get_epoch_lr(cfg, cur_epoch)
    if cfg.OPTIM.TYPE == "sgd":
        optim.set_lr(optimizer, lr)

    if torch.cuda.is_available():
        model.cuda()

    # Enable training mode
    model.train()
    train_meter.iter_tic() #This basically notes the start time in timer class defined in utils/timer.py

    len_train_loader = len(train_loader)
    for cur_iter, (inputs, labels) in enumerate(train_loader):
        #ensuring that inputs are floatTensor as model weights are
        inputs = inputs.type(torch.cuda.FloatTensor)
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Perform the forward pass
        preds = model(inputs)
        # Compute the loss
        loss = loss_fun(preds, labels)
        # Perform the backward pass
        optimizer.zero_grad()
        loss.backward()
        # Update the parametersSWA
        optimizer.step()
        # Compute the errors
        top1_err, top5_err = mu.topk_errors(preds, labels, [1, 5])
        # Combine the stats across the GPUs
        # if cfg.NUM_GPUS > 1:
        #     #Average error and losses across GPUs
        #     #Also this this calls wait method on reductions so we are ensured
        #     #to obtain synchronized results
        #     loss, top1_err = du.scaled_all_reduce(
        #         [loss, top1_err]
        #     )
        # Copy the stats from GPU to CPU (sync point)
        loss, top1_err = loss.item(), top1_err.item()
        # #Only master process writes the logs which are used for plotting
        # if du.is_master_proc():
        if cur_iter != 0 and cur_iter%19 == 0:
            #because cur_epoch starts with 0
            plot_it_x_values.append((cur_epoch)*len_train_loader + cur_iter)
            plot_it_y_values.append(loss)
            # save_plot_values([plot_it_x_values, plot_it_y_values],["plot_it_x_values", "plot_it_y_values"], out_dir=cfg.EPISODE_DIR, isDebug=False)
            # print(plot_it_x_values)
            # print(plot_it_y_values)
            #Plot loss graphs
            # plot_arrays(x_vals=plot_it_x_values, y_vals=plot_it_y_values, x_name="Iterations", y_name="Loss", dataset_name=cfg.DATASET.NAME, out_dir=cfg.EPISODE_DIR,)
            print('Training Epoch: {}/{}\tIter: {}/{}'.format(cur_epoch+1, cfg.OPTIM.MAX_EPOCH, cur_iter, len(train_loader)))

        #Compute the difference in time now from start time initialized just before this for loop.
        train_meter.iter_toc()
        train_meter.update_stats(top1_err=top1_err, loss=loss, \
            lr=lr, mb_size=inputs.size(0) * cfg.NUM_GPUS)
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
    # Log epoch stats
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()
    return loss, clf_iter_count


@torch.no_grad()
def test_epoch(test_loader, model, test_meter, cur_epoch):
    """Evaluates the model on the test set."""

    global plot_episode_xvalues
    global plot_episode_yvalues

    global plot_epoch_xvalues
    global plot_epoch_yvalues

    global plot_it_x_values
    global plot_it_y_values

    if torch.cuda.is_available():
        model.cuda()

    # Enable eval mode
    model.eval()
    test_meter.iter_tic()

    misclassifications = 0.
    totalSamples = 0.

    for cur_iter, (inputs, labels) in enumerate(test_loader):
        with torch.no_grad():
            # Transfer the data to the current GPU device
            inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
            inputs = inputs.type(torch.cuda.FloatTensor)
            # Compute the predictions
            preds = model(inputs)
            # Compute the errors
            top1_err, top5_err = mu.topk_errors(preds, labels, [1, 5])
            # Combine the errors across the GPUs
            # if cfg.NUM_GPUS > 1:
            #     top1_err = du.scaled_all_reduce([top1_err])
            #     #as above returns a list
            #     top1_err = top1_err[0]
            # Copy the errors from GPU to CPU (sync point)
            top1_err = top1_err.item()
            # Multiply by Number of GPU's as top1_err is scaled by 1/Num_GPUs
            misclassifications += top1_err * inputs.size(0) * cfg.NUM_GPUS
            totalSamples += inputs.size(0)*cfg.NUM_GPUS
            test_meter.iter_toc()
            # Update and log stats
            test_meter.update_stats(
                top1_err=top1_err, mb_size=inputs.size(0) * cfg.NUM_GPUS
            )
            test_meter.log_iter_stats(cur_epoch, cur_iter)
            test_meter.iter_tic()
    # Log epoch stats
    test_meter.log_epoch_stats(cur_epoch)
    test_meter.reset()

    return misclassifications/totalSamples



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
