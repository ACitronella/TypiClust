import numpy as np
import pandas as pd
import torch
import pycls.datasets.utils as ds_utils

class RandomOnBlinkingPeriodWithProportion:
    def __init__(self, cfg, lSet, uSet, budgetSize, embedding_path, blinking_to_full_ratio, dataset_info):
        self.cfg = cfg
        self.seed = self.cfg['RNG_SEED']
        # self.embedding_path = embedding_path
        # self.all_features = ds_utils.load_embededing_from_path(embedding_path)
        self.lSet = lSet
        self.uSet = uSet
        self.budgetSize = budgetSize
        self.blinking_to_full_ratio = blinking_to_full_ratio 
        self.dataset_info = dataset_info
        tmp = []
        for kp_file in self.dataset_info.keypoint_file:
            vid_kp_b = np.load(kp_file, mmap_mode="r")
            tmp.append(vid_kp_b["is_blinking"])
        self.is_blinking = np.concatenate(tmp)
        self.relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(int)
        # self.rel_features = self.all_features[self.relevant_indices] 
        self.rel_is_blinking = self.is_blinking[self.relevant_indices]
        self.rel_is_blinking_idx = self.rel_is_blinking_idx.argwhere().flatten()
        self.rng = np.random.default_rng(self.seed)
        self.blinking_nsample = self.budgetSize * self.blinking_to_full_ratio
        self.non_blinking_nsample = self.budgetSize * (1-self.blinking_to_full_ratio)

    def select_samples(self):
        
        self.rng.choice()
        self.rel_is_blinking 
        pass
        # return activeSet, remainSet
