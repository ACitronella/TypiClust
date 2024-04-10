import numpy as np
import pandas as pd
import torch
import pycls.datasets.utils as ds_utils

class EmbeddingDifferenceAsProbabilityDensity:
    def __init__(self, cfg, lSet, uSet, budgetSize, embedding_path, dataset_info, kernel_size=11):
        self.cfg = cfg
        self.embedding_path = embedding_path
        self.all_features = ds_utils.load_embededing_from_path(embedding_path)
        self.lSet = np.asarray(lSet).astype(int)
        self.uSet = np.asarray(uSet).astype(int)
        self.budgetSize = budgetSize
        self.dataset_info = dataset_info
        self.kernel_size = kernel_size
        kernel = np.ones(kernel_size)/kernel_size
        self.tb = self.dataset_info["frames"].cumsum()
        
        self.relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(int)
        self.rel_features = self.all_features[self.relevant_indices]
        self.rng = np.random.default_rng()
        
        v = []
        for row_idx in self.dataset_info.index:
            # frames = self.dataset_info.loc[row_idx, "frames"]
            start_idx = self.tb[row_idx-1] if row_idx-1 >= 0 else 0
            end_idx = self.tb[row_idx]
            a_patient_embedding = self.all_features[start_idx:end_idx]
            if np.prod(a_patient_embedding.shape) == 0:
                continue
            centroid = a_patient_embedding.mean(0)
            mean_square_diff_em = np.mean(np.square(a_patient_embedding - centroid), axis=1)
            smoothed_mean_square_diff_em = np.zeros_like(mean_square_diff_em)            
            smoothed_mean_square_diff_em[kernel_size//2:(-kernel_size//2)+1] = np.convolve(mean_square_diff_em, kernel, mode='valid')
            smoothed_mean_square_diff_em[:kernel_size//2] = smoothed_mean_square_diff_em[kernel_size//2]
            smoothed_mean_square_diff_em[-kernel_size//2+1:] = smoothed_mean_square_diff_em[-kernel_size//2+1]
            v.append(smoothed_mean_square_diff_em)
        v = np.concatenate(v)[self.uSet]
        w = (v - v.min())
        y = w / w.max()
        self.prob_density = y / y.sum()
        
    def select_samples(self):
        idx = np.arange(0, self.uSet.shape[0]) 
        activeSet = self.rng.choice(idx, self.budgetSize, replace=False, p=self.prob_density)

        remain_idx = np.setdiff1d(idx, activeSet)
        remainSet = self.uSet[remain_idx]
        activeSet = self.uSet[activeSet]
        assert len(np.setdiff1d(self.uSet, np.union1d(remainSet, activeSet))) == 0
        return activeSet, remainSet

        
