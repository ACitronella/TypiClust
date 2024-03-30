# from train_al_leaveoneout import CustomImageDataset4, cal_and_process_meanface, batch_process_pip_out, imshow_kp, imshow_kp_from_dl

import os
import sys
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional
# import torch.optim as optim
import torch.profiler
import torchvision.transforms as transforms

import functools
from utils import CustomImageDataset4, cal_and_process_meanface, eval_on_pipnet, imshow_kp, input_size, num_lms, num_nb, net_stride, imshow_kp_from_dl

# local
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(os.path.abspath('..'))

import pycls.core.builders as model_builder
from pycls.core.config import cfg, dump_cfg
from pycls.datasets.data import Data
from pycls.datasets.blink_dataset import BlinkDataset, ImageDataFrameBlinkingWrapper
from pycls.datasets.blink_dataset_with_no_test_set import BlinkDataset2
from pycls.datasets.blink_dataset_all import BlinkDatasetAll

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == "__main__":
    cfg.merge_from_file("../configs/blinkleaveoneout/test.yaml") 
    exp_name = cfg.EXP_NAME
    exp_dir = os.path.join(cfg.MODEL_GRAVEYARD, exp_name)
    cfg.DATASET.ROOT_DIR = os.path.join(os.path.abspath('../..'), cfg.DATASET.ROOT_DIR)
    data_obj = Data(cfg)
    train_data, train_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=True, isDownload=True, fold_idx=cfg.DATASET.FOLD_IDX, is_blinking=cfg.DATASET.IS_BLINKING)
    if isinstance(train_data, (BlinkDataset, BlinkDataset2, BlinkDatasetAll)):
        dataset_info = train_data.dataset_info
    elif isinstance(train_data, ImageDataFrameBlinkingWrapper):
        dataset_info = train_data.dataset.dataset_info
    else:
        dataset_info = None
        exit(0)
    all_patient_code = dataset_info["patient_code"].unique()
    msg = "Dataset {} Loaded Sucessfully. Total Train Size: {}\n".format(cfg.DATASET.NAME, train_size,)
    print(msg)

    for patient_code in all_patient_code:
        print("training set size before spliting", len(train_data))
        exp_patient_dir = os.path.join(cfg.EXP_DIR, patient_code)
        os.makedirs(exp_patient_dir, exist_ok=True)
        train_patient_code = np.setdiff1d(all_patient_code, patient_code)
        lSet_path, uSet_path = data_obj.makeLUSetsByPatients(train_patient_code, train_data, exp_patient_dir)
        lSet, uSet = data_obj.loadLUPartitions(lSetPath=lSet_path, uSetPath=uSet_path)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
        train_transforms = transforms.Compose([transforms.ToTensor(), normalize])
        # val_transforms = train_transforms
    
        vSet = np.random.choice(lSet, 1024, replace=False)
        lSet = np.setdiff1d(lSet, vSet)
        meanface, meanface_indices, reverse_index1, reverse_index2, max_len = cal_and_process_meanface([train_data[idx] for idx in lSet], num_lms, num_nb)
        reverse_index1 = torch.tensor(reverse_index1, dtype=torch.long).to(device)
        reverse_index2 = torch.tensor(reverse_index2, dtype=torch.long).to(device)
        train_dataset = CustomImageDataset4(train_data, input_size, num_lms, net_stride, meanface_indices, 
                                            train_transforms, pil_augment=False, allow_idxs=lSet)
        val_dataset = CustomImageDataset4(train_data, input_size, num_lms, net_stride, meanface_indices, 
                                            train_transforms, pil_augment=False, allow_idxs=vSet)

        model = model_builder.get_keypoint_model(num_nb , num_lms, input_size, net_stride)
        model = torch.nn.parallel.DataParallel(model, [1, 0])
        # model_state_dict = torch.load(cfg.MODEL_INIT_WEIGHTS, map_location=device)
        model_state_dict = torch.load("../model_graveyard_but_leaveoneout/test/C2205/init_best_17118n.pt", map_location=device)
        model.load_state_dict(model_state_dict, strict=False); model.eval()
        model = model.to(device)
    
        # train_dataloader_wrapper = functools.partial(DataLoader, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.DATA_LOADER.NUM_WORKERS, shuffle=False, pin_memory=True)
        val_dataloader_wrapper = functools.partial(DataLoader, batch_size=cfg.TEST.BATCH_SIZE, num_workers=cfg.DATA_LOADER.NUM_WORKERS, shuffle=False, pin_memory=True)
        lSet_loader = val_dataloader_wrapper(train_dataset)
        valSet_loader = val_dataloader_wrapper(val_dataset)
        model_mse, model_mse_nb, model_mae, model_mae_nb = eval_on_pipnet(model, lSet_loader, reverse_index1, reverse_index2, max_len, device)
        print("train:", model_mse, model_mse_nb, model_mae, model_mae_nb)
        model_mse, model_mse_nb, model_mae, model_mae_nb = eval_on_pipnet(model, valSet_loader, reverse_index1, reverse_index2, max_len, device)
        print("val:", model_mse, model_mse_nb, model_mae, model_mae_nb)
        imshow_kp_from_dl(model, lSet_loader, valSet_loader, device, reverse_index1, reverse_index2, max_len, "eval_train_val.png")
        # print(lSet_loader.dataset[0][0])
        # plt.figure()
        # imshow_kp(lSet_loader.dataset[0][0])
        # plt.savefig("test.png"`)
        break


