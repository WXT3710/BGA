
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def load_attribute_matrix(config):
    dataset = config["dataset"]
    if dataset == "AWA":
        path = "dataset/AWA/AWA2/JPEGImages/predicate-matrix-binary.txt"
        A = np.loadtxt(path, dtype=np.float32)
    elif dataset == "CUB":
        path = "dataset/CUB/CUB/images/class_attribute_labels_continuous.txt"
        A = np.loadtxt(path, dtype=np.float32)
    elif dataset == "SUN":
        path = "dataset/SUN_Attribute/filetxt_500_217/sun_attr_continuous.npy"
        A = np.load(path).astype(np.float32)
    else:
        raise ValueError(f"attribute matrix not defined for dataset {dataset}")

    A_cont = A.astype(np.float32)
    A = (A > 0).astype(np.float32)

    assert A.shape[0] == config["n_class"], \
        f"attr matrix classes {A.shape[0]} != config n_class {config['n_class']}"

    A_t = torch.from_numpy(A)
    A_norm = F.normalize(A_t, dim=1)
    S_attr = A_norm @ A_norm.T

    Ac = torch.from_numpy(A_cont)
    S_attr_cont = F.normalize(Ac, dim=1) @ F.normalize(Ac, dim=1).T

    if dataset == "SUN":
        seen_idx = np.load(
            "dataset/SUN_Attribute/filetxt_500_217/sun_seen_idx.npy")
        unseen_idx = np.load(
            "dataset/SUN_Attribute/filetxt_500_217/sun_unseen_idx.npy")
    else:
        num_seen = config["num_seen"]
        seen_idx = np.arange(num_seen, dtype=np.int64)
        unseen_idx = np.arange(num_seen, config["n_class"], dtype=np.int64)

    return A_t, torch.from_numpy(A_cont), S_attr, seen_idx, unseen_idx, S_attr_cont

class AttributeTransferLoss(nn.Module):

    def __init__(self, config, S_attr, seen_idx, unseen_idx):
        super().__init__()
        self.register_buffer("S_attr", S_attr)
        self.register_buffer("seen_idx", torch.as_tensor(seen_idx, dtype=torch.long))
        self.register_buffer("unseen_idx", torch.as_tensor(unseen_idx, dtype=torch.long))
        self.temperature = config.get("attr_transfer_temp", 0.1)

    def _weights(self):
        logits = self.S_attr[self.unseen_idx][:, self.seen_idx] / self.temperature
        return F.softmax(logits, dim=1)

    def init_centres(self, centres):
        W = self._weights()
        transferred = W @ centres.data[self.seen_idx]
        centres.data[self.unseen_idx] = transferred
        return centres

    def forward(self, centres):
        W = self._weights()
        target = W @ centres[self.seen_idx].detach()
        return F.mse_loss(centres[self.unseen_idx], target)

class AttributeCenterLoss(nn.Module):

    def __init__(self, config, S_attr, mode="full", topk=16,
                 seen_idx=None, unseen_idx=None):
        super().__init__()
        self.register_buffer("S_target", S_attr)
        self.mode = mode
        self.topk = topk
        if seen_idx is not None:
            self.register_buffer("seen_idx", torch.as_tensor(seen_idx, dtype=torch.long))
        if unseen_idx is not None:
            self.register_buffer("unseen_idx", torch.as_tensor(unseen_idx, dtype=torch.long))

    def forward(self, centres):
        c_norm = F.normalize(centres, dim=1)
        centre_sim = c_norm @ c_norm.T
        S = self.S_target
        if self.mode == "seen_unseen":
            return F.mse_loss(centre_sim[self.unseen_idx][:, self.seen_idx],
                              S[self.unseen_idx][:, self.seen_idx])
        if self.mode == "topk":
            topk = min(self.topk, S.size(1) - 1)
            _, idx = S.topk(topk + 1, dim=-1)
            mask = torch.zeros_like(S)
            mask.scatter_(1, idx, 1.0)
            mask.fill_diagonal_(0.0)
            n = mask.sum()
            if n == 0:
                return torch.tensor(0.0, device=centres.device)
            return ((centre_sim - S) ** 2 * mask).sum() / n
        return F.mse_loss(centre_sim, S)