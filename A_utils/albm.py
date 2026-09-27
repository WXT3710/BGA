"""ALBM-inspired attribute + patch transfer for SST.

Maps ALBM's "transfer class information from the attribute / patch angles"
onto the SST hashing baseline.  The class "centre" in `Center_Loss1` is the
carrier of class information in Hamming space, so all three mechanisms
operate on it (or on the patch features that feed it):

  (A) AttributeTransferLoss  -- unseen-class centre = softmax-weighted
                                 combination of seen-class centres, weighted
                                 by attribute similarity  (ALBM Eq.5 in hash
                                 space, the "attribute angle").
  (B) patch -> attribute head -- patches predict the class attribute vector,
                                 forcing the encoder to encode attribute
                                 semantics (the "patch angle").
  (C) AttributeCenterLoss     -- align the geometry of the hash centres with
                                 the geometry of the attribute space.

None of these are baked into `Center_Loss1`; they are separate modules added
to the total loss with their own weight hyper-parameters.

The seen/unseen split is passed in as explicit index lists rather than a
contiguous range, so AWA/CUB (contiguous) and SUN (interleaved) both work.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_attribute_matrix(config):
    """Load the class-attribute matrix and the seen/unseen class split.

    Returns:
        A          : torch.FloatTensor [n_class, n_attr]  (0/1, binarised)
        A_cont     : torch.FloatTensor [n_class, n_attr]  (raw values, for the patch loss)
        S_attr     : torch.FloatTensor [n_class, n_class]  cosine similarity of A
        seen_idx   : np.int64 [num_seen]
        unseen_idx : np.int64 [num_unseen]
        S_attr_cont: torch.FloatTensor [n_class, n_class]  cosine similarity of A_cont
    """
    dataset = config["dataset"]
    if dataset == "AWA":
        path = "dataset/AWA/AWA2/JPEGImages/predicate-matrix-binary.txt"
        A = np.loadtxt(path, dtype=np.float32)            # 50 x 85, already 0/1
    elif dataset == "CUB":
        path = "dataset/CUB/CUB/images/class_attribute_labels_continuous.txt"
        A = np.loadtxt(path, dtype=np.float32)            # 200 x 312, continuous
    elif dataset == "SUN":
        # Precomputed class-level mean attribute matrix (717 x 102, 0..1),
        # aggregated per label index from SUNAttributeDB image-level labels.
        path = "dataset/SUN_Attribute/filetxt_500_217/sun_attr_continuous.npy"
        A = np.load(path).astype(np.float32)
    else:
        raise ValueError(f"attribute matrix not defined for dataset {dataset}")

    A_cont = A.astype(np.float32)
    A = (A > 0).astype(np.float32)                        # binarise

    assert A.shape[0] == config["n_class"], \
        f"attr matrix classes {A.shape[0]} != config n_class {config['n_class']}"

    A_t = torch.from_numpy(A)
    A_norm = F.normalize(A_t, dim=1)                      # row-normalise
    S_attr = A_norm @ A_norm.T                            # cosine similarity

    # The same similarity computed from the CONTINUOUS attributes instead of the
    # thresholded presence vectors. Kept separate so the geometry loss can be run with
    # either target: a non-degenerate presence distribution does not by itself show that
    # thresholding is the better choice, only a matched comparison does.
    Ac = torch.from_numpy(A_cont)
    S_attr_cont = F.normalize(Ac, dim=1) @ F.normalize(Ac, dim=1).T

    # seen/unseen class indices
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
    """(A) Semantic centre transfer for unseen classes.

    Initialises unseen-class centres as a softmax-weighted combination of the
    seen-class centres (weighted by attribute similarity), and keeps them
    anchored there during training.
    """

    def __init__(self, config, S_attr, seen_idx, unseen_idx):
        super().__init__()
        self.register_buffer("S_attr", S_attr)            # [n_class, n_class]
        self.register_buffer("seen_idx", torch.as_tensor(seen_idx, dtype=torch.long))
        self.register_buffer("unseen_idx", torch.as_tensor(unseen_idx, dtype=torch.long))
        self.temperature = config.get("attr_transfer_temp", 0.1)

    def _weights(self):
        """W[u, s] = softmax_s(S_attr[u, s] / tau) over seen classes."""
        logits = self.S_attr[self.unseen_idx][:, self.seen_idx] / self.temperature
        return F.softmax(logits, dim=1)                   # [n_unseen, n_seen]

    def init_centres(self, centres):
        """Overwrite unseen centres with the semantic prior (call once)."""
        W = self._weights()
        transferred = W @ centres.data[self.seen_idx]     # [n_unseen, bit]
        centres.data[self.unseen_idx] = transferred
        return centres

    def forward(self, centres):
        W = self._weights()
        target = W @ centres[self.seen_idx].detach()      # anchor (seen evolve freely)
        return F.mse_loss(centres[self.unseen_idx], target)


class AttributeCenterLoss(nn.Module):
    """(C) Align centre geometry with attribute geometry.

    mode='full'       : full class-pair MSE (original form)
    mode='seen_unseen': constrain only unseen rows x seen columns (transfer-relevant structure)
    mode='topk'       : constrain only top-k attribute-similar class pairs per row (sparse strong relationships)
    """

    def __init__(self, config, S_attr, mode="full", topk=16,
                 seen_idx=None, unseen_idx=None):
        super().__init__()
        self.register_buffer("S_target", S_attr)          # [n_class, n_class]
        self.mode = mode
        self.topk = topk
        if seen_idx is not None:
            self.register_buffer("seen_idx", torch.as_tensor(seen_idx, dtype=torch.long))
        if unseen_idx is not None:
            self.register_buffer("unseen_idx", torch.as_tensor(unseen_idx, dtype=torch.long))

    def forward(self, centres):
        c_norm = F.normalize(centres, dim=1)
        centre_sim = c_norm @ c_norm.T                    # [n_class, n_class]
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
