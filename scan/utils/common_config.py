"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import math
import numpy as np
import torch
import torchvision.transforms as transforms
from data.augment import Augment, Cutout
from utils.collate import collate_custom
import sys
# local
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path(os.path.abspath('../deep-al'))
from pycls.datasets.blink_dataset_all import BlinkDatasetAll
from pycls.datasets.data import Data

def get_criterion(p):
    if p['criterion'] == 'simclr':
        from losses.losses import SimCLRLoss
        criterion = SimCLRLoss(**p['criterion_kwargs'])

    elif p['criterion'] == 'scan':
        from losses.losses import SCANLoss
        criterion = SCANLoss(**p['criterion_kwargs'])

    elif p['criterion'] == 'confidence-cross-entropy':
        from losses.losses import ConfidenceBasedCE
        criterion = ConfidenceBasedCE(p['confidence_threshold'], p['criterion_kwargs']['apply_class_balancing'])

    else:
        raise ValueError('Invalid criterion {}'.format(p['criterion']))

    return criterion


def get_feature_dimensions_backbone(p):
    if p['backbone'] == 'resnet18':
        return 512
    elif p['backbone'] == 'efficientnet-b0':
        return 512
    elif p['backbone'] == 'resnet50':
        return 2048

    else:
        raise NotImplementedError


def get_model(p, pretrain_path=None):
    # Get backbone
    assert 'blink' in p['train_db_name'] or (p['model_kwargs']['pre_lasts_dim'] == 512) or (p['model_kwargs']['pre_lasts_dim'] == 2048) or (p['model_kwargs']['pre_lasts_dim'] == 1280)

    if p['backbone'] == 'resnet18':
        if p['train_db_name'] in ['cifar-10', 'cifar-100']:
            from models.resnet_cifar import resnet18
            backbone = resnet18()

        elif p['train_db_name'] == 'stl-10':
            from models.resnet_stl import resnet18
            backbone = resnet18()
        elif p['train_db_name'] == 'tiny-imagenet':
            from models.resnet_tinyimagenet import resnet18
            backbone = resnet18()
        elif 'blink' in p['train_db_name']:
            # from models.resnet_tinyimagenet_pe import resnet18
            from models.resnet_tinyimagenet import resnet18
            backbone = resnet18(last_dim=p['model_kwargs']['pre_lasts_dim'])
        else:
            raise NotImplementedError

    elif p['backbone'] == 'resnet50':
        if 'imagenet' in p['train_db_name']:
            from models.resnet import resnet50
            backbone = resnet50()  
        elif 'blink' in p['train_db_name'] and p['setup'] == 'byol':
            from torchvision.models import resnet50
            backbone = {"backbone": resnet50(pretrained=False), "dim": 2048}
        else:
            raise NotImplementedError 
    elif p['backbone'] == "efficientnet-b0":
        from torchvision.models import efficientnet_b0
        backbone = {"backbone": efficientnet_b0(pretrained=False), "dim": 1280}

    else:
        raise ValueError('Invalid backbone {}'.format(p['backbone']))

    # Setup
    if p['setup'] in ['simclr', 'moco']:
        from models.models import ContrastiveModel
        model = ContrastiveModel(backbone, **p['model_kwargs'])

    elif p['setup'] == "simclr_pe":
        from models.models import ContrastiveModelPE
        model = ContrastiveModelPE(backbone, **p['model_kwargs'])

    elif p['setup'] in ['scan', 'selflabel']:
        from models.models import ClusteringModel
        if p['setup'] == 'selflabel':
            assert(p['num_heads'] == 1)
        model = ClusteringModel(backbone, p['num_classes'], p['num_heads'])
    elif p['setup'] == 'byol':
        from byol_pytorch import BYOL 
        model = BYOL(backbone["backbone"], image_size=p["transformation_kwargs"]["crop_size"], hidden_layer='avgpool') # default setting
    else:
        raise ValueError('Invalid setup {}.'.format(p['setup']))

    # Load pretrained weights
    if pretrain_path is not None and os.path.exists(pretrain_path):
        state = torch.load(pretrain_path, map_location='cpu')
        
        if p['setup'] == 'scan': # Weights are supposed to be transfered from contrastive training
            missing = model.load_state_dict(state, strict=False)
            assert(set(missing[1]) == {
                'contrastive_head.0.weight', 'contrastive_head.0.bias', 
                'contrastive_head.2.weight', 'contrastive_head.2.bias'}
                or set(missing[1]) == {
                'contrastive_head.weight', 'contrastive_head.bias'})

        elif p['setup'] == 'selflabel': # Weights are supposed to be transfered from scan 
            # We only continue with the best head (pop all heads first, then copy back the best head)
            model_state = state['model']
            all_heads = [k for k in model_state.keys() if 'cluster_head' in k]
            best_head_weight = model_state['cluster_head.%d.weight' %(state['head'])]
            best_head_bias = model_state['cluster_head.%d.bias' %(state['head'])]
            for k in all_heads:
                model_state.pop(k)

            model_state['cluster_head.0.weight'] = best_head_weight
            model_state['cluster_head.0.bias'] = best_head_bias
            missing = model.load_state_dict(model_state, strict=True)

        else:
            raise NotImplementedError

    elif pretrain_path is not None and not os.path.exists(pretrain_path):
        raise ValueError('Path with pre-trained weights does not exist {}'.format(pretrain_path))

    else:
        pass

    return model

class DSWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        return {"image": self.dataset[idx][0]}


def get_train_dataset(p, transform, to_augmented_dataset=False,
                        to_neighbors_dataset=False, split=None):
    # Base dataset
    if p['train_db_name'] == 'cifar-10':
        from data.cifar import CIFAR10
        dataset = CIFAR10(train=True, transform=transform, download=True)

    elif p['train_db_name'] == 'cifar-100':
        from data.cifar import CIFAR100
        dataset = CIFAR100(train=True, transform=transform, download=True)

    elif p['train_db_name'] == 'stl-10':
        from data.stl import STL10
        dataset = STL10(split=split, transform=transform, download=True)

    elif p['train_db_name'] == 'tiny-imagenet':
        from data.tinyimagenet import TinyImageNet
        dataset = TinyImageNet(root='', split='train', transform=transform)

    elif p['train_db_name'] == 'imagenet':
        from data.imagenet import ImageNet
        dataset = ImageNet(split='train', transform=transform)

    elif p['train_db_name'] in ['imagenet_50', 'imagenet_100', 'imagenet_200']:
        from data.imagenet import ImageNetSubset
        subset_file = './data/imagenet_subsets/%s.txt' %(p['train_db_name'])
        dataset = ImageNetSubset(subset_file=subset_file, split='train', transform=transform)

    elif "blink_all" in p['train_db_name']:
        dataset = BlinkDatasetAll(train=True, transform=None, test_transform=None, dataset_path="../../../pytorchlm")
        pcode_to_exclude = p["fold_idx"]
        patient_code_all = dataset.dataset_info["patient_code"].unique()
        train_pcode = np.setdiff1d(patient_code_all, [pcode_to_exclude])
        train_idx, _ = Data.makeLUSetsByPatientsNotSave(train_pcode, dataset)
        dataset = torch.utils.data.Subset(dataset, train_idx)
        dataset = DSWrapper(dataset, transform)

    elif "blink2" in p['train_db_name']:
        from data.blink_dataset import BlinkDataset2
        dataset = BlinkDataset2(train=True, transform=transform, fold_idx=p["fold_idx"])
        

    elif "blink" in p['train_db_name']:
        from data.blink_dataset import BlinkDataset
        dataset = BlinkDataset(train=True, transform=transform, fold_idx=p["fold_idx"])
        
    else:
        raise ValueError('Invalid train dataset {}'.format(p['train_db_name']))
    
    # Wrap into other dataset (__getitem__ changes)
    if to_augmented_dataset: # Dataset returns an image and an augmentation of that image.
        from data.custom_dataset import AugmentedDataset
        dataset = AugmentedDataset(dataset)

    if to_neighbors_dataset: # Dataset returns an image and one of its nearest neighbors.
        from data.custom_dataset import NeighborsDataset
        indices = np.load(p['topk_neighbors_train_path'])
        dataset = NeighborsDataset(dataset, indices, p['num_neighbors'])

    # if "blink" in p['train_db_name']:
    #     blink_indices_table = dataset.indices_table
    #     return dataset, BlinkSampler(blink_indices_table)
    return dataset


def get_val_dataset(p, transform=None, to_neighbors_dataset=False):
    # Base dataset
    if p['val_db_name'] == 'cifar-10':
        from data.cifar import CIFAR10
        dataset = CIFAR10(train=False, transform=transform, download=True)
    
    elif p['val_db_name'] == 'cifar-100':
        from data.cifar import CIFAR100
        dataset = CIFAR100(train=False, transform=transform, download=True)

    elif p['val_db_name'] == 'stl-10':
        from data.stl import STL10
        dataset = STL10(split='test', transform=transform, download=True)

    elif p['train_db_name'] == 'tiny-imagenet':
        from data.tinyimagenet import TinyImageNet
        dataset = TinyImageNet(root='', split='val', transform=transform)

    elif p['val_db_name'] == 'imagenet':
        from data.imagenet import ImageNet
        dataset = ImageNet(split='val', transform=transform)
    
    elif p['val_db_name'] in ['imagenet_50', 'imagenet_100', 'imagenet_200']:
        from data.imagenet import ImageNetSubset
        subset_file = './data/imagenet_subsets/%s.txt' %(p['val_db_name'])
        dataset = ImageNetSubset(subset_file=subset_file, split='val', transform=transform)
    elif "blink_all" in p["train_db_name"]:
        dataset = BlinkDatasetAll(train=False, transform=transform, test_transform=None, dataset_path="../../../pytorchlm")
        # pcode_to_include = p["fold_idx"]
        # train_idx, _ = Data.makeLUSetsByPatientsNotSave([pcode_to_include], dataset)
        # dataset = torch.utils.data.Subset(dataset, train_idx)
        dataset = DSWrapper(dataset, transform)
    elif "blink2" in p['train_db_name']:
        from data.blink_dataset import BlinkDataset2
        dataset = BlinkDataset2(train=False, transform=transform, fold_idx=p["fold_idx"])
    elif "blink" in p['train_db_name']:
        from data.blink_dataset import BlinkDataset
        dataset = BlinkDataset(train=False, transform=transform)
    else:
        raise ValueError('Invalid validation dataset {}'.format(p['val_db_name']))
    
    # Wrap into other dataset (__getitem__ changes) 
    if to_neighbors_dataset: # Dataset returns an image and one of its nearest neighbors.
        from data.custom_dataset import NeighborsDataset
        indices = np.load(p['topk_neighbors_val_path'])
        dataset = NeighborsDataset(dataset, indices, 5) # Only use 5

    return dataset
import copy
class BlinkSampler(torch.utils.data.Sampler):
    def __init__(self, indices_table):
        indices_table = np.asarray(indices_table)
        self.start_end_idx = np.asarray((indices_table[:-1], indices_table[1:])).T
        self.max_frames = np.amax(self.start_end_idx[:, 1] - self.start_end_idx[:, 0])
        self.rng = np.random.default_rng()
        self.n = len(self.start_end_idx) * self.max_frames
        self.available_patient_idx = []
        for start_idx, end_idx in self.start_end_idx:
            available_idx = np.concatenate((
                np.arange(start_idx, end_idx), 
                self.rng.integers(start_idx, end_idx, size=self.max_frames - (end_idx - start_idx))) ) # oversample
            self.available_patient_idx.append(available_idx)


        # tmp = np.concatenate(self.available_patient_idx)
        # assert len(tmp) == self.n and len(set(tmp)) == indices_table[-1]
        
        # print(len(self.available_patient_idx))
        # print(indices_table)

    def __iter__(self):
        available_patient_idx = copy.deepcopy(self.available_patient_idx)
        n_available_patient = len(available_patient_idx)
        for idx in range(len(available_patient_idx)):
            self.rng.shuffle(available_patient_idx[idx])
        for frame_idx in range(self.max_frames):
            for p_idx in range(n_available_patient):
                # print(available_patient_idx[p_idx][frame_idx], end=" ")
                yield available_patient_idx[p_idx][frame_idx]
            # print()
    def __len__(self):
        return self.n

def get_train_dataloader(p, dataset, sampler=None):
    if p['batch_size'] == "npatient":
        batch_size = dataset.dataset_info.shape[0]
        return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'], 
                batch_size=batch_size, pin_memory=True, collate_fn=collate_custom,
                drop_last=True, shuffle=sampler is None, sampler=sampler)
    
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'], 
            batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
            drop_last=True, shuffle=sampler is None, sampler=sampler)
    # return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'], 
    #         batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
    #         drop_last=True, shuffle=False)


def get_val_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'],
            batch_size=p['batch_size'] if isinstance(p['batch_size'], int) else 64, pin_memory=True, collate_fn=collate_custom,
            drop_last=False, shuffle=False)


def get_train_transformations(p):
    if p['augmentation_strategy'] == 'standard':
        # Standard augmentation strategy
        return transforms.Compose([
            transforms.RandomResizedCrop(**p['augmentation_kwargs']['random_resized_crop']),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**p['augmentation_kwargs']['normalize'])
        ])
    
    elif p['augmentation_strategy'] == 'simclr':
        # Augmentation strategy from the SimCLR paper
        return transforms.Compose([
            transforms.RandomResizedCrop(**p['augmentation_kwargs']['random_resized_crop']),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(**p['augmentation_kwargs']['color_jitter'])
            ], p=p['augmentation_kwargs']['color_jitter_random_apply']['p']),
            transforms.RandomGrayscale(**p['augmentation_kwargs']['random_grayscale']),
            transforms.ToTensor(),
            transforms.Normalize(**p['augmentation_kwargs']['normalize'])
        ])
    elif p['augmentation_strategy'] == 'reduced_simclr':
        return transforms.Compose([
            transforms.RandomResizedCrop(**p['augmentation_kwargs']['random_resized_crop']),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomApply([
            #     transforms.ColorJitter(**p['augmentation_kwargs']['color_jitter'])
            # ], p=p['augmentation_kwargs']['color_jitter_random_apply']['p']),
            # transforms.RandomGrayscale(**p['augmentation_kwargs']['random_grayscale']),
            transforms.ToTensor(),
            transforms.Normalize(**p['augmentation_kwargs']['normalize'])
        ])
    elif p['augmentation_strategy'] == 'simclr_wo_flip':
        return transforms.Compose([
            transforms.RandomResizedCrop(**p['augmentation_kwargs']['random_resized_crop']),
            # transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(**p['augmentation_kwargs']['color_jitter'])
            ], p=p['augmentation_kwargs']['color_jitter_random_apply']['p']),
            transforms.RandomGrayscale(**p['augmentation_kwargs']['random_grayscale']),
            transforms.ToTensor(),
            transforms.Normalize(**p['augmentation_kwargs']['normalize'])
        ])


    elif p['augmentation_strategy'] == 'ours':
        # Augmentation strategy from our paper 
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(p['augmentation_kwargs']['crop_size']),
            Augment(p['augmentation_kwargs']['num_strong_augs']),
            transforms.ToTensor(),
            transforms.Normalize(**p['augmentation_kwargs']['normalize']),
            Cutout(
                n_holes = p['augmentation_kwargs']['cutout_kwargs']['n_holes'],
                length = p['augmentation_kwargs']['cutout_kwargs']['length'],
                random = p['augmentation_kwargs']['cutout_kwargs']['random'])])
    elif p['augmentation_strategy'] == 'same_as_val':
        return get_val_transformations(p)
    else:
        raise ValueError('Invalid augmentation strategy {}'.format(p['augmentation_strategy']))


def get_val_transformations(p):
    if p["transformation_kwargs"].get("normalize", None):
        return transforms.Compose([
                transforms.CenterCrop(p['transformation_kwargs']['crop_size']),
                transforms.ToTensor(),  
                transforms.Normalize(**p['transformation_kwargs']['normalize'])])
    else:
        return transforms.Compose([
            transforms.CenterCrop(p['transformation_kwargs']['crop_size']),
            transforms.ToTensor()
        ])

def get_optimizer(p, model, cluster_head_only=False):
    if cluster_head_only: # Only weights in the cluster head will be updated 
        for name, param in model.named_parameters():
                if 'cluster_head' in name:
                    param.requires_grad = True 
                else:
                    param.requires_grad = False 
        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        assert(len(params) == 2 * p['num_heads'])

    else:
        params = model.parameters()
                

    if p['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params, **p['optimizer_kwargs'])

    elif p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params, **p['optimizer_kwargs'])
    
    else:
        raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    return optimizer


def adjust_learning_rate(p, optimizer, epoch):
    lr = p['optimizer_kwargs']['lr']
    
    if p['scheduler'] == 'cosine':
        eta_min = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / p['epochs'])) / 2
         
    elif p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['scheduler_kwargs']['lr_decay_epochs']))
        if steps > 0:
            lr = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** steps)

    elif p['scheduler'] == 'constant':
        lr = lr

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
