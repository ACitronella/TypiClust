import numpy as np
import pandas as pd
import torch
import pycls.datasets.utils as ds_utils

class ProbCoverIdxStandardDist:
    def __init__(self, cfg, lSet, uSet, budgetSize, delta, embedding_path, dataset_info, frame_diff_factor):
        self.cfg = cfg
        self.ds_name = self.cfg['DATASET']['NAME']
        self.seed = self.cfg['RNG_SEED']
        # self.all_features = ds_utils.load_features(self.ds_name, self.seed)
        self.embedding_path = embedding_path
        self.all_features = ds_utils.load_embededing_from_path(embedding_path)
        self.lSet = lSet
        self.uSet = uSet
        self.budgetSize = budgetSize
        self.delta = delta
        self.relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(int)
        self.rel_features = self.all_features[self.relevant_indices]
        self.dataset_info = dataset_info

        # since our all_feature is train, we use all of them
        # self.frame_idx = torch.cat([torch.arange(frames, dtype=torch.float) for frames in dataset_info["frames"]]).unsqueeze(1)
        self.frame_idx = [torch.arange(frames, dtype=torch.float) for frames in dataset_info["frames"]]
        self.frame_idx = torch.cat([(f - f.mean()) / f.std() for f in self.frame_idx]).unsqueeze(1)
        self.rel_frame_idx = self.frame_idx[self.relevant_indices]
        self.frame_diff_factor = frame_diff_factor
        assert len(self.frame_idx) == len(self.all_features) and len(self.rel_features) == len(self.rel_frame_idx)

        # this must be the last thing to run
        self.graph_df = self.construct_graph()

    def construct_graph(self, batch_size=500):
        """
        creates a directed graph where:
        x->y iff l2(x,y) < delta.

        represented by a list of edges (a sparse matrix).
        stored in a dataframe
        """
        xs, ys, ds = [], [], []
        print(f'Start constructing graph using delta={self.delta}')
        # distance computations are done in GPU
        cuda_feats = torch.tensor(self.rel_features).cuda()
        cuda_frame_idx = self.rel_frame_idx.cuda()
        for i in range(len(self.rel_features) // batch_size):
            # distance comparisons are done in batches to reduce memory consumption
            cur_feats = cuda_feats[i * batch_size: (i + 1) * batch_size]
            cur_frame_idx = cuda_frame_idx[i * batch_size: (i + 1) * batch_size]
            dist = torch.cdist(cur_feats, cuda_feats)
            frame_diff = self.frame_diff_factor*torch.cdist(cur_frame_idx, cuda_frame_idx, p=1)
            print(torch.std_mean(dist), torch.std_mean(frame_diff))
            dist = dist + frame_diff
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
        print(f'Finished constructing graph using delta={self.delta}')
        print(f'Graph contains {len(df)} edges.')
        return df

    def select_samples(self):
        """
        selecting samples using the greedy algorithm.
        iteratively:
        - removes incoming edges to all covered samples
        - selects the sample high the highest out degree (covers most new samples)

        """
        print(f'Start selecting {self.budgetSize} samples.')
        selected = []
        # removing incoming edges to all covered samples from the existing labeled set
        edge_from_seen = np.isin(self.graph_df.x, np.arange(len(self.lSet)))
        covered_samples = self.graph_df.y[edge_from_seen].unique()
        cur_df = self.graph_df[(~np.isin(self.graph_df.y, covered_samples))]
        for i in range(self.budgetSize):
            coverage = len(covered_samples) / len(self.relevant_indices)
            # selecting the sample with the highest degree
            degrees = np.bincount(cur_df.x, minlength=len(self.relevant_indices))
            print(f'Iteration is {i}.\tGraph has {len(cur_df)} edges.\tMax degree is {degrees.max()}.\tCoverage is {coverage:.3f}')
            cur = degrees.argmax()
            # cur = np.random.choice(degrees.argsort()[::-1][:5]) # the paper randomizes selection

            # removing incoming edges to newly covered samples
            new_covered_samples = cur_df.y[(cur_df.x == cur)].values
            assert len(np.intersect1d(covered_samples, new_covered_samples)) == 0, 'all samples should be new'
            cur_df = cur_df[(~np.isin(cur_df.y, new_covered_samples))]

            covered_samples = np.concatenate([covered_samples, new_covered_samples])
            selected.append(cur)

        assert len(selected) == self.budgetSize, 'added a different number of samples'
        activeSet = self.relevant_indices[selected]
        remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))))

        print(f'Finished the selection of {len(activeSet)} samples.')
        print(f'Active set is {activeSet}')
        return activeSet, remainSet
