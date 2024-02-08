import os, sys
from sklearn.cluster import KMeans 
# local
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
add_path(os.path.abspath('..'))
from pycls.datasets import utils as ds_utils

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

class ProbCoverFindDelta:
    def __init__(self, all_features, delta, km_predicts):
        self.all_features = all_features
        self.lSet = []
        self.uSet = np.arange(all_features.shape[0])
        self.budgetSize = all_features.shape[0]
        self.delta = delta
        self.km_predicts = km_predicts
        self.relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(int)
        self.rel_features = self.all_features[self.relevant_indices]
        self.graph_df = self.construct_graph()

    def construct_graph(self, batch_size=500):
        """
        creates a directed graph where:
        x->y iff l2(x,y) < delta.

        represented by a list of edges (a sparse matrix).
        stored in a dataframe
        """
        xs, ys, ds = [], [], []
        # print(f'Start constructing graph using delta={self.delta}')
        # distance computations are done in GPU
        cuda_feats = torch.tensor(self.rel_features).cuda()
        for i in range(len(self.rel_features) // batch_size):
            # distance comparisons are done in batches to reduce memory consumption
            cur_feats = cuda_feats[i * batch_size: (i + 1) * batch_size]
            dist = torch.cdist(cur_feats, cuda_feats)
            mask = dist < self.delta
            # saving edges using indices list - saves memory.
            x, y = mask.nonzero().T
            xs.append(x.cpu() + batch_size * i)
            ys.append(y.cpu())
            ds.append(dist[mask].cpu())

        xs = torch.cat(xs).numpy()
        ys = torch.cat(ys).numpy()
        ds = torch.cat(ds).numpy()

        df = pd.DataFrame({'x': xs, 'y': ys, 'd': ds})
        # print(f'Finished constructing graph using delta={self.delta}')
        # print(f'Graph contains {len(df)} edges.')
        return df

    def select_samples(self):
        """
        selecting samples using the greedy algorithm.
        iteratively:
        - removes incoming edges to all covered samples
        - selects the sample high the highest out degree (covers most new samples)

        """
        # print(f'Start selecting {self.budgetSize} samples.')
        selected = []
        # removing incoming edges to all covered samples from the existing labeled set
        edge_from_seen = np.isin(self.graph_df.x, np.arange(len(self.lSet)))
        covered_samples = self.graph_df.y[edge_from_seen].unique()
        cur_df = self.graph_df[(~np.isin(self.graph_df.y, covered_samples))]
        correct = 0
        from_all = 0
        for i in range(self.budgetSize):
            # coverage = len(covered_samples) / len(self.relevant_indices)
            # selecting the sample with the highest degree
            degrees = np.bincount(cur_df.x, minlength=len(self.relevant_indices))
            # print(f'Iteration is {i}.\tGraph has {len(cur_df)} edges.\tMax degree is {degrees.max()}.\tCoverage is {coverage:.3f}')
            cur = degrees.argmax()
            # cur = np.random.choice(degrees.argsort()[::-1][:5]) # the paper randomizes selection

            # removing incoming edges to newly covered samples
            new_covered_samples = cur_df.y[(cur_df.x == cur)].values 
            assert len(np.intersect1d(covered_samples, new_covered_samples)) == 0, 'all samples should be new'
            cur_df = cur_df[(~np.isin(cur_df.y, new_covered_samples))]

            covered_samples = np.concatenate([covered_samples, new_covered_samples])
            selected.append(cur)
            correct += np.sum(self.km_predicts[cur] == self.km_predicts[new_covered_samples])
            from_all += len(new_covered_samples)
        # print(correct, from_all, "purity: ", correct/from_all)   
        assert len(selected) == self.budgetSize, 'added a different number of samples'
        # activeSet = self.relevant_indices[selected]
        # remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))))

        # print(f'Finished the selection of {len(activeSet)} samples.')
        # print(f'Active set is {activeSet}')
        # return activeSet, remainSet
        return correct/from_all

info_df = pd.read_csv(os.path.join("../../../../pytorchlm", "cvat2_dataset", "data_keypoint_interpolated_10points_faster", "dataset_info.csv"))
info_df["patient_type"] = info_df["patient_code"].apply(lambda x: x[0])

test_patient_code = set(['A2211', 'C2204'])
train_val_patient_code = set(info_df["patient_code"].unique()).difference(test_patient_code)
train_val_df = info_df[info_df["patient_code"].apply(lambda x : x in train_val_patient_code)]
# FOLDS = 4
fold_idx = np.zeros((train_val_df.shape[0]), dtype="int32")
fold_idx[train_val_df["patient_code"] == "A3104"] = 0
fold_idx[train_val_df["patient_code"] == "C2104"] = 0

fold_idx[train_val_df["patient_code"] == "A2111"] = 1
fold_idx[train_val_df["patient_code"] == "C2205"] = 1

fold_idx[train_val_df["patient_code"] == "A3102"] = 2
fold_idx[train_val_df["patient_code"] == "C2202"] = 2

fold_idx[train_val_df["patient_code"] == "A1115"] = 3
fold_idx[train_val_df["patient_code"] == "C2201"] = 3
train_val_df.loc[:, "fold_idx"] = fold_idx

delta_list = np.linspace(0.05, 1, 20, endpoint=True)
print("delta to be tested", delta_list)
plt.figure(figsize=(10, 10))
for fold_idx in range(4):
    print(f"fold {fold_idx}")
    npatient = (train_val_df["fold_idx"] != fold_idx).sum()
    nclasses = 5
    # features = ds_utils.load_features(f"blink2_fold{fold_idx}", seed=132, train=True)
    features = np.load(f"../../scan/results/blink2_fold{fold_idx}/pretext/features_seed132.npy")
    km = KMeans(nclasses)
    predict = km.fit_predict(features)
    purity_each_delta = []
    for delta in delta_list:
        pc = ProbCoverFindDelta(features, delta=delta, km_predicts=predict)
        purity = pc.select_samples()
        print(f"delta={delta:.2f}, purity = {purity:.2f}")
        purity_each_delta.append(purity)
    first_drop_below_thres = np.argmax(np.asarray(purity_each_delta) < 0.95)
    plt.subplot(2, 2, fold_idx+1)
    plt.title(f"blink fold{fold_idx} kmean with {nclasses} classes")
    plt.plot(delta_list, purity_each_delta, "-o")
    plt.plot((delta_list[first_drop_below_thres], delta_list[first_drop_below_thres]), (0, 1), "--")
    plt.xlabel("delta")
    plt.ylabel("purity")
plt.savefig(f"blink_delta.png")
plt.close("all")