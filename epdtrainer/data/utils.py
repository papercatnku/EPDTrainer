import torch
import cv2
from random import shuffle
import numpy as np
from tqdm import tqdm


class JointDataset(torch.utils.data.Dataset):
    def __init__(self, ds_ls=[]):
        super().__init__()
        self.dataset_ls = []
        self.dataset_idx = [0]
        total_num = 0
        for _ds in ds_ls:
            self.dataset_ls.append(_ds)
            total_num += len(_ds)
            self.dataset_idx.append(total_num)
        return

    def __len__(self):
        return self.dataset_idx[-1]

    def __getitem__(self, idx):
        for i, _idstart in enumerate(self.dataset_idx[:-1]):
            if (idx < self.dataset_idx[i + 1]):
                return self.dataset_ls[i][idx - _idstart]


class TsfDataset(torch.utils.data.Dataset):
    def __init__(self, ds, tsf):
        super().__init__()
        self.ds = ds
        self.tsf = tsf

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        out_dict = self.tsf(self.ds[idx])
        return out_dict


class SeqTsfDataset(torch.torch.utils.data.Dataset):
    def __init__(self, ds, tsf_seq):
        self.ds = ds
        self.tsf_seq = tsf_seq
        return

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        out_data = self.ds[idx]
        for tsf in self.tsf_seq:
            out_data = tsf(out_data)
        return out_data


def DatasetSplitTrainEval(dataset, eval_ratio=0.1, seed=42):
    total_num = len(dataset)
    eval_num = int(round(total_num * eval_ratio))
    train_num = total_num - eval_num

    train_set, eval_set = torch.utils.data.random_split(
        dataset, [train_num, eval_num], generator=torch.Generator().manual_seed(seed))

    return train_set, eval_set


def getRandomSubsetDataset(dataset, max_num, if_shuffle=True):
    total_num = min(max_num, len(dataset))
    ran_id = [i for i in range(len(dataset))]
    shuffle(ran_id)

    class _dataset(torch.utils.data.Dataset):
        def __init__(self, id_ls):
            super().__init__()
            self._ds = dataset
            self.id_ls = id_ls
            return

        def __len__(self,):
            return total_num

        def __getitem__(self, idx):
            return self._ds[idx]

    return _dataset(ran_id[:total_num])
