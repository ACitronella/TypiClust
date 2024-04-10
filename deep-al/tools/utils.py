import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import copy
from typing import Union
import random
from math import floor

input_size = 256
num_nb = 10
num_lms = 21
net_stride = 32


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
    # plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.legend()

def imshow_kp_from_dl(model, lSet_loader, valSet_loader, device, reverse_index1, reverse_index2, max_len, file_path):
    img, *t = next(iter(lSet_loader))
    with torch.no_grad():
        o = model(img.to(device))
    target_kp, _ = batch_process_pip_out(*t, reverse_index1, reverse_index2, max_len)
    kp, kp_nb = batch_process_pip_out(*o, reverse_index1, reverse_index2, max_len)
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    imshow_kp(img[0], target_kp[0], kp.cpu()[0], kp_nb.cpu()[0])

    img, *t = next(iter(valSet_loader))
    with torch.no_grad():
        o = model(img.to(device))
    target_kp, _ = batch_process_pip_out(*t, reverse_index1, reverse_index2, max_len)
    kp, kp_nb = batch_process_pip_out(*o, reverse_index1, reverse_index2, max_len)
    plt.subplot(1, 2, 2)
    imshow_kp(img[0], target_kp[0], kp.cpu()[0], kp_nb.cpu()[0])
    plt.savefig(file_path, bbox_inches='tight')
    

class MetricsStorage():
    def __init__(self) -> None:
        self.metrics = {}
        self.training_losses = self.metrics.setdefault("training_losses", [])
        self.validation_losses = self.metrics.setdefault("validation_losses", [])
        self.training_mses = self.metrics.setdefault("training_mses", [])
        self.validation_mses = self.metrics.setdefault("validation_mses", [])
        self.training_mses_nb = self.metrics.setdefault("training_mses_nb", [])
        self.validation_mses_nb = self.metrics.setdefault("validation_mses_nb", [])
        self.n_epochs_for_ac_iter = self.metrics.setdefault("n_epochs_for_ac_iter", [])

    def next_al_iter(self):
        self.n_epochs_for_ac_iter.append(len(self.training_losses))

    def push_metrics(self, trn_loss=None, val_loss=None, trn_mse=None, val_mse=None, trn_mse_nb=None, val_mse_nb=None):
        if trn_loss:
            self.training_losses.append(trn_loss)
        if val_loss:
            self.validation_losses.append(val_loss)
        if trn_mse:
            self.training_mses.append(trn_mse)
        if val_mse:
            self.validation_mses.append(val_mse)
        if trn_mse_nb:
            self.training_mses_nb.append(trn_mse_nb)
        if val_mse_nb:
            self.validation_mses_nb.append(val_mse_nb)
    
    def to_dict(self):
        return self.metrics

    def get_the_latest_value_of(self, metrics_name: str):
        return self.metrics[metrics_name][-1]

    def cpu(self):
        for metrics in [self.training_losses, self.validation_losses, self.training_mses, self.validation_mses, self.training_mses_nb, self.validation_mses_nb]:
            for idx, l in enumerate(metrics):
                if isinstance(l, torch.Tensor):
                    metrics[idx] = l.item()


def batch_process_pip_out(outputs_cls, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y, reverse_index1, reverse_index2, max_len):
    assert outputs_cls.ndim == 4
    input_size_and_net_stride_ratio = 1.0 * input_size / net_stride
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
    
    tmp_x = (max_ids%tmp_width).view(tmp_batch, -1, 1).float()
    tmp_y = torch.div(max_ids, tmp_width, rounding_mode="floor").view(tmp_batch, -1, 1).float()
    lms_pred_x = tmp_x + outputs_x_select.view(tmp_batch, -1, 1)
    lms_pred_y = tmp_y + outputs_y_select.view(tmp_batch, -1, 1)
    lms_pred_x /= input_size_and_net_stride_ratio
    lms_pred_y /= input_size_and_net_stride_ratio
    lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=2).view(tmp_batch, -1)
    # assert (lms_pred_y[0].flatten() == lms_pred_y1.flatten()).all() and (lms_pred[0].flatten() == lms_pred1.flatten()).all() 
    
    tmp_nb_x = (tmp_x + outputs_nb_x_select).view(tmp_batch, -1, num_nb)
    tmp_nb_y = (tmp_y + outputs_nb_y_select).view(tmp_batch, -1, num_nb)
    tmp_nb_x /= input_size_and_net_stride_ratio
    tmp_nb_y /= input_size_and_net_stride_ratio
    tmp_nb_x = tmp_nb_x[:, reverse_index1, reverse_index2].view(tmp_batch, num_lms, max_len)
    tmp_nb_y = tmp_nb_y[:, reverse_index1, reverse_index2].view(tmp_batch, num_lms, max_len)
    tmp_nb_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=2), dim=2).view(tmp_batch, -1, 1)
    tmp_nb_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=2), dim=2).view(tmp_batch, -1, 1)
    lms_pred_merge = torch.cat((tmp_nb_x, tmp_nb_y), dim=2).view(tmp_batch, -1)
    # assert (lms_pred_merge[0].flatten() == lms_pred_merge1.flatten()).all()
    return lms_pred, lms_pred_merge


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


@torch.no_grad()
def eval_on_pipnet(model, dl, reverse_index1, reverse_index2, max_len, DEVICE:Union[torch.device, str]='cpu'):
    mse_fn = torch.nn.MSELoss(reduction="none")
    mae_fn = torch.nn.L1Loss(reduction="none")
    
    model_mse = model_mse_nb = model_mae = model_mae_nb = 0
    count_img = len(dl.dataset)
    for imgs, labels_map, labels_x, labels_y, labels_nb_x, labels_nb_y in dl:
        imgs = imgs.to(DEVICE); labels_map = labels_map.to(DEVICE); labels_x = labels_x.to(DEVICE)
        labels_y = labels_y.to(DEVICE); labels_nb_x = labels_nb_x.to(DEVICE); labels_nb_y = labels_nb_y.to(DEVICE)
        outputs_map, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y = model(imgs)
        
        lms_preds, lms_preds_merge = batch_process_pip_out(outputs_map, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y, reverse_index1, reverse_index2, max_len)
        lms_gts, _ = batch_process_pip_out(labels_map, labels_x, labels_y, labels_nb_x, labels_nb_y, reverse_index1, reverse_index2, max_len)
        # for img, lms_gt, lms_pred, lms_pred_merge in zip(imgs, lms_gts, lms_preds, lms_preds_merge):
        mse_num = mse_fn(lms_gts, lms_preds);mse_nb_num = mse_fn(lms_gts, lms_preds_merge)
        mae_num = mae_fn(lms_gts, lms_preds);mae_nb_num = mae_fn(lms_gts, lms_preds_merge)

        # model_mse += torch.nn.functional.mse_loss(lms_gts, lms_preds, reduction="none").mean(dim=1).sum()
        # model_mse_nb += torch.nn.functional.mse_loss(lms_gts, lms_preds_merge, reduction="none").mean(dim=1).sum()
        model_mse += mse_num.mean(axis=1).sum(); model_mse_nb += mse_nb_num.mean(axis=1).sum()
        model_mae += mae_num.mean(axis=1).sum(); model_mae_nb += mae_nb_num.mean(axis=1).sum()

    return model_mse/count_img, model_mse_nb/count_img, model_mae/count_img, model_mae_nb/count_img
