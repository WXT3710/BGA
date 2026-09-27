# Compatibility monkey-patch for transformers >= 5.x
# Fix: all_tied_weights_keys must exist on PreTrainedModel subclasses
import transformers
_orig_init = transformers.PreTrainedModel.__init__
def _patched_init(self, config, *args, **kwargs):
    _orig_init(self, config, *args, **kwargs)
    if not hasattr(self, 'all_tied_weights_keys'):
        self.all_tied_weights_keys = {}
transformers.PreTrainedModel.__init__ = _patched_init

from A_utils.tools_with_category_name import *
from A_utils.albm import load_attribute_matrix, AttributeTransferLoss, AttributeCenterLoss

import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
blip_main_models_path = os.path.join(current_path, 'BLIP_main')
sys.path.insert(0, blip_main_models_path)

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import random
from loguru import logger
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy('file_system')

import argparse

from BLIP_main.models import blip_itm
from torch.optim.lr_scheduler import StepLR


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(42)


def parse_args():
    parser = argparse.ArgumentParser(description="BLIP Hash Model Training")

    parser.add_argument('--dataset', type=str, choices=['AWA', 'CUB', 'SUN'], default="AWA", help="Dataset to use")
    parser.add_argument('--caption', type=int, default=1)
    parser.add_argument('--TGI', type=int, default=1)

    parser.add_argument('--freeze_img', type=int, default=1)
    parser.add_argument('--freeze_txt', type=int, default=1)

    parser.add_argument('--blip_loss', type=str, default="-logsoftmax")
    parser.add_argument('--hash_loss', type=str, default="center_loss")
    parser.add_argument('--inform', type=str, default="SST")

    parser.add_argument('--save_path', type=str)

    parser.add_argument('--train_file_override', type=str, default=None,
                        help="")

    parser.add_argument('--dump_attr', type=int, default=0,
                        help="save the patch-branch attribute embedding of the best epoch "
                             "(query and database) for the patch-representation analysis")
    parser.add_argument('--seed', type=int, default=42,
                        help="global RNG seed; drives data order, augmentation, dropout, "
                             "and the Bernoulli/Gaussian centre initialisation, which all "
                             "draw from the global torch RNG")
    parser.add_argument('--gpu', type=int, default=0,
                        help="GPU device id to use (default: 0)")
    parser.add_argument('--bit', type=int, default=64,
                        choices=[24, 48, 64, 128],
                        help="Hash code length in bits (default: 64)")
    parser.add_argument('--init', type=str, default='bernoulli',
                        choices=['randn', 'bernoulli'],
                        help="Centre initialisation: randn (N(0,1)) or bernoulli ({-1,+1})")
    parser.add_argument('--epoch', type=int, default=50,
                        help="Number of training epochs (paper: 50)")
    parser.add_argument('--hash_weight', type=float, default=None,
                        help="alpha: weight of Lhash vs LCL (Eq.12). None=dataset default (AWA 0.5 / CUB 0.1)")
    parser.add_argument('--proxy_beta', type=float, default=None,
                        help="beta: weight of Lproxy (Eq.11). None=dataset default (AWA 1.0 / CUB 0.7)")

    # ---- ALBM-inspired: attribute + patch transfer (separate λ-weighted losses) ----
    parser.add_argument('--attr_transfer', type=int, default=0,
                        help="(A) unseen-centre attribute transfer (0=off, 1=on)")
    parser.add_argument('--attr_transfer_weight', type=float, default=1.0)
    parser.add_argument('--attr_transfer_temp', type=float, default=0.1)
    parser.add_argument('--attr_center', type=int, default=0,
                        help="(C) centre-attribute structure alignment (0=off, 1=on)")
    parser.add_argument('--attr_center_weight', type=float, default=0.1)
    parser.add_argument('--attr_patch', type=int, default=0,
                        help="(B) patch attribute prediction head (0=off, 1=on)")
    parser.add_argument('--attr_patch_weight', type=float, default=0.1)
    parser.add_argument('--attr_patch_mode', type=str, default='bce',
                        choices=['bce', 'bce_topk', 'pair', 'pair_topk'],
                        help="")
    parser.add_argument('--attr_patch_topk', type=int, default=32)
    parser.add_argument('--attr_center_mode', type=str, default='full',
                        choices=['full', 'seen_unseen', 'topk'],
                        help="")
    parser.add_argument('--attr_center_topk', type=int, default=16)
    parser.add_argument('--attr_center_target', type=str, default='presence',
                        choices=['presence', 'continuous'],
                        help="geometry target: presence vectors 1[a>0] (paper) or the "
                             "raw continuous attributes (control)")

    parser.add_argument('--batch_size', type=int, default=None,
                        help="")
    parser.add_argument('--aug', type=int, default=0,
                        help="")
    parser.add_argument('--aug_strong', type=int, default=0,
                        help="")
    parser.add_argument('--tta', type=int, default=0,
                        help="")
    parser.add_argument('--consistency_weight', type=float, default=None,
                        help="")
    parser.add_argument('--tau', type=float, default=None,
                        help="")
    parser.add_argument('--attr_patch_pool', type=str, default='mean',
                        choices=['mean', 'attention', 'clspatch', 'cls', 'queries'],
                        help="")

    return parser.parse_args()


def get_config(args):
    config = {
        'inform': args.inform,
        'seed': args.seed,
        'dump_attr': bool(args.dump_attr),
        'freeze_img_encoder': args.freeze_img,
        'freeze_txt_encoder': args.freeze_txt,

        "TGI": args.TGI,
        "blip_loss": args.blip_loss,
        "hash_loss": args.hash_loss,

        "net": Blip_Hash_NET,

        "dataset": args.dataset,
        "caption": bool(args.caption),

        'without_BN': 1,

        'lambda': 0.0001,
        'lambda1': 1,
        'lambda2': 0.0001,
        'beta': 1,
        'epoch_change': 10,
        'mome': 0.9,

        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "info": "[Blip_Hash_768_3]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": args.batch_size if args.batch_size else 32,

        "epoch": args.epoch,
        "test_map": 2,
        "save_path": args.save_path,
        "device": torch.device(f"cuda:{args.gpu}"),
        "bit_list": [args.bit],

        "blip_pretrained_pth": 'BLIP_main/models/BLIP_base.pth',
        "blip_med_config": 'BLIP_main/configs/med_config.json',
        "blip_vit_mode": 'base',

        "centre_init": args.init,     # randn | bernoulli

        # ALBM-inspired: attribute + patch transfer (separate λ-weighted losses)
        "attr_transfer": bool(args.attr_transfer),
        "attr_transfer_weight": args.attr_transfer_weight,
        "attr_transfer_temp": args.attr_transfer_temp,
        "attr_center": bool(args.attr_center),
        "attr_center_weight": args.attr_center_weight,
        "attr_patch": bool(args.attr_patch),
        "attr_patch_weight": args.attr_patch_weight,
        "attr_patch_mode": args.attr_patch_mode,
        "attr_patch_topk": args.attr_patch_topk,
        "attr_center_mode": args.attr_center_mode,
        "attr_center_topk": args.attr_center_topk,
        "attr_center_target": args.attr_center_target,

        "aug": bool(args.aug),
        "aug_strong": bool(args.aug_strong),
        "tta": bool(args.tta),
        "consistency_weight": args.consistency_weight if args.consistency_weight is not None else 0.1,
        "tau": args.tau if args.tau is not None else 0.07,
        "attr_patch_pool": args.attr_patch_pool,
    }
    if config["dataset"] == "AWA":
        config["txt2img_n"] = 100
        config["n_class"] = 50
        config["num_train"] = 5000
        config["num_seen"] = 40
        if args.hash_weight is None:
            config["hash_weight"] = 0.5
        if args.proxy_beta is None:
            config["proxy_beta"] = 1.0
    elif config["dataset"] == "CUB":
        config["txt2img_n"] = 40
        config["n_class"] = 200
        config["num_train"] = 6500
        config["num_seen"] = 150
        if args.hash_weight is None:
            config["hash_weight"] = 0.1
        if args.proxy_beta is None:
            config["proxy_beta"] = 0.7
    elif config["dataset"] == "SUN":
        config["txt2img_n"] = 0
        config["n_class"] = 717
        config["num_train"] = 12557  # 500 seen x 16 real + 217 unseen x 21 pseudo
        config["num_seen"] = 500
        if args.hash_weight is None:
            config["hash_weight"] = 0.5
        if args.proxy_beta is None:
            config["proxy_beta"] = 1.0

    # allow CLI to override the dataset defaults
    if args.hash_weight is not None:
        config["hash_weight"] = args.hash_weight
    if args.proxy_beta is not None:
        config["proxy_beta"] = args.proxy_beta

    config = config_dataset(config)
    if args.train_file_override:
        config["data"]["train_set"]["list_path"] = args.train_file_override
    return config


class Blip_Hash(nn.Module):
    def __init__(self, config, bit):
        super().__init__()
        self.hash_bit = bit

        self.blip = blip_itm.blip_itm(
            pretrained=config["blip_pretrained_pth"],
            med_config=config["blip_med_config"],
            image_size=224,
            vit=config["blip_vit_mode"],
        )
        self.visual_encoder = self.blip.visual_encoder
        self.vision_proj = self.blip.vision_proj

        self.hash_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, self.hash_bit)
        )

        # (B) patch -> attribute head (the "patch angle"): patches predict the
        # class attribute vector, forcing the encoder to encode attribute semantics.
        self.attr_head = None
        self.attr_pool = config.get("attr_patch_pool", "mean")
        self.patch_attn = None
        self.attr_queries = None
        if config.get("attr_patch", False):
            n_attr = config["n_attr"]
            if self.attr_pool == "queries":
                # Fine-grained variant: one learnable query per attribute dimension.
                # Each query cross-attends over the 196 patch tokens, so the attention
                # map [n_attr, 196] IS the patch<->attribute alignment -- every attribute
                # explicitly selects the patches it reads from. The output stays
                # [B, n_attr], so the downstream relational loss is unchanged.
                self.attr_queries = nn.Parameter(torch.randn(n_attr, 768) * 0.02)
                self.attr_head = nn.Linear(768, 1)
            else:
                in_dim = 1536 if self.attr_pool == "clspatch" else 768
                self.attr_head = nn.Sequential(
                    nn.Linear(in_dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, n_attr)
                )
            if self.attr_pool == "attention":
                self.patch_attn = nn.Linear(768, 1)

    def forward(self, x):
        vision_embeds = self.visual_encoder(x)
        cls_feat = vision_embeds[:, 0, :]

        proj_feat = F.normalize(cls_feat, dim=-1)

        hash_feat = self.hash_head(cls_feat)
        hash_feat = torch.tanh(hash_feat)

        attr_logits = None
        if self.attr_head is not None:
            patch_feats = vision_embeds[:, 1:, :]       # [B, 196, 768]
            if self.attr_pool == "attention":
                w = torch.softmax(self.patch_attn(patch_feats), dim=1)  # [B,196,1]
                patch_pool = (patch_feats * w).sum(dim=1)               # [B,768]
            elif self.attr_pool == "clspatch":
                patch_pool = torch.cat([cls_feat, patch_feats.mean(dim=1)], dim=-1)  # [B,1536]
            elif self.attr_pool == "cls":
                # Control: same MLP shape (768 -> 512 -> n_attr) and the same relational
                # loss, but fed the CLS token instead of the pooled patch tokens. Isolates
                # whether the gain comes from the patch features or merely from attaching
                # a second attribute branch to the shared encoder.
                patch_pool = cls_feat                                   # [B,768]
            elif self.attr_pool == "queries":
                # Fine-grained: per-attribute queries attend over the patches.
                att = torch.einsum('ad,bpd->bap', self.attr_queries, patch_feats)
                att = (att / (patch_feats.shape[-1] ** 0.5)).softmax(dim=-1)  # [B,n_attr,196]
                pooled = att @ patch_feats                              # [B, n_attr, 768]
                attr_logits = self.attr_head(pooled).squeeze(-1)        # [B, n_attr]
            else:
                patch_pool = patch_feats.mean(dim=1)                    # [B,768]
            if self.attr_pool != "queries":
                attr_logits = self.attr_head(patch_pool)    # [B, n_attr]
        return proj_feat, hash_feat, attr_logits


class Blip_Hash_NET(nn.Module):
    def __init__(self, config, hash_bit):
        super(Blip_Hash_NET, self).__init__()
        self.m = config['mome']
        self.encoder_q = Blip_Hash(config, hash_bit)
        self.encoder_k = Blip_Hash(config, hash_bit)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, x):
        blip_f1, encode_x, attr_logits = self.encoder_q(x)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            blip_f2, encode_x2, _ = self.encoder_k(x)
        return blip_f1, encode_x, encode_x2, attr_logits


class LogSoftmaxContrastiveLoss_all_positive_samples(nn.Module):
    def __init__(self, temperature=0.07, exclude_self=True):
        super().__init__()
        self.temperature = temperature
        self.exclude_self = exclude_self
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, pred_feats, target_feats, labels):
        pred_feats = F.normalize(pred_feats, dim=-1)
        target_feats = F.normalize(target_feats, dim=-1)

        logits = torch.matmul(pred_feats, target_feats.T) / self.temperature
        log_probs = self.log_softmax(logits)  # [B, B]

        with torch.no_grad():
            pos_mask = (labels @ labels.T)
            pos_mask = (pos_mask > 0).float()

            if self.exclude_self:
                pos_mask.fill_diagonal_(0.0)

            pos_count = pos_mask.sum(dim=1, keepdim=True)
            pos_mask = pos_mask / (pos_count + 1e-6)

        loss = - (pos_mask * log_probs).sum(dim=1).mean()

        return loss


class Center_Loss1(nn.Module):
    """
    
    """

    def __init__(self, config, bit, class_list):
        super(Center_Loss1, self).__init__()
        self.bit = bit
        self.n_class = len(class_list)
        self.device = config["device"]

        init_mode = config.get("centre_init", "bernoulli")
        if init_mode == "bernoulli":
            centres_init = torch.bernoulli(
                torch.full((self.n_class, bit), 0.5)) * 2 - 1
        else:
            centres_init = torch.randn(self.n_class, bit)
        self.centres = nn.Parameter(centres_init)
        self.consistency_weight = config.get("consistency_weight", 0.1)

    def forward(self, pred_hash, pred_hash2, label_onehot, ind, epoch):
        if self.centres.device != pred_hash.device:
            self.centres.data = self.centres.data.to(pred_hash.device)

        labels = torch.argmax(label_onehot, dim=1)                # [B]
        batch_centres = self.centres[labels]                      # [B, bit]

        loss_centre = F.mse_loss(pred_hash, batch_centres)

        pred_q_norm = F.normalize(pred_hash, p=2, dim=1)
        pred_k_norm = F.normalize(pred_hash2, p=2, dim=1)
        loss_consistency = F.mse_loss(pred_q_norm, pred_k_norm.detach())

        return loss_centre + self.consistency_weight * loss_consistency


class SSTOfficialHashLoss(nn.Module):
    """

    """
    def __init__(self, config, bit):
        super().__init__()
        self.bit = bit
        self.n_class = config["n_class"]
        self.lambda_quant = 1e-4
        self.proxy_warmup = 10
        self.beta = config.get("proxy_beta", 1.0)
        self.register_buffer("hash_center",
                             self._generate_hash_centers(bit, self.n_class),
                             persistent=False)
        self.register_buffer("label_center", torch.eye(self.n_class), persistent=False)
        self.register_buffer("hash_memory",
                             torch.randn(config["num_train"], bit), persistent=False)
        self.register_buffer("label_memory",
                             torch.zeros(config["num_train"], self.n_class), persistent=False)
        
        self.centres = self.hash_center

    def _generate_hash_centers(self, bit, n_class, seed=42):
        rng = np.random.default_rng(seed)
        centers = np.zeros((n_class, bit), dtype=np.int8)
        for _ in range(30):
            for class_idx in range(n_class):
                center = np.ones(bit, dtype=np.int8)
                center[rng.choice(bit, size=bit // 2, replace=False)] = -1
                centers[class_idx] = center
            distances = [np.sum(centers[i] != centers[j])
                         for i in range(n_class) for j in range(i + 1, n_class)]
            distances = np.asarray(distances)
            if distances.min() > bit // 4 and distances.mean() >= bit / 2:
                break
        return torch.tensor(centers, dtype=torch.float32)

    def forward(self, query_hash, momentum_hash, labels, indices, epoch):
        self.hash_memory[indices, :] = momentum_hash.detach()
        self.label_memory[indices, :] = labels.detach()

        center_loss = self._center_loss(query_hash, labels)
        quantization_loss = (query_hash.abs() - 1).pow(2).mean()
        proxy_loss = 0.0 if epoch < self.proxy_warmup else self._proxy_loss(query_hash, labels)
        return center_loss + self.lambda_quant * quantization_loss + self.beta * proxy_loss

    def _center_loss(self, hash_codes, labels):
        logits = torch.matmul(F.normalize(hash_codes), F.normalize(self.hash_center).t())
        logits = (self.bit ** 0.5) * logits
        targets = (labels @ self.label_center.t()).float()
        probs = torch.softmax(logits, dim=1).clamp(min=1e-6, max=1 - 1e-6)
        loss = targets * torch.log(probs) + (1 - targets) * torch.log(1 - probs)
        return -loss.mean()

    def _proxy_loss(self, hash_codes, labels):
        hash_codes = F.normalize(hash_codes)
        memory_hash = F.normalize(self.hash_memory)
        positive_mask = (labels @ self.label_memory.t() > 0).float()
        similarity = hash_codes @ memory_hash.t()
        loss = positive_mask * torch.log1p(torch.exp(0.5 * (1 - similarity)))
        return loss.sum() / (positive_mask.sum() + 1e-6)


class ProxyHashLoss(nn.Module):
    """
    SST hashing loss (paper Eq. 8-11):  Lhash = Lcos + Lquant + beta*Lproxy

      - Lcos:    cosine-softmax classification of each hash code against the
                 class-wise semantic centres (Bernoulli {−1,+1}, selected from
                 multiple trials to maximise pairwise Hamming distance), scaled
                 by alpha = sqrt(K).
      - Lquant:  binarisation loss pushing |bi| towards 1.
      - Lproxy:  proxy contrastive loss over same-class pairs, encouraging
                 intra-class compactness, weighted by beta.

    This replaces the earlier MSE-to-centre "Center_Loss1", which did NOT match
    the paper's formulation.
    """

    def __init__(self, config, bit, class_list):
        super(ProxyHashLoss, self).__init__()
        self.bit = bit
        self.n_class = len(class_list)
        self.cos_scale = float(bit) ** 0.5          # alpha in Eq.8 (sqrt(K))
        self.beta = config.get("proxy_beta", 1.0)   # beta in Eq.11

        init_mode = config.get("centre_init", "bernoulli")
        if init_mode == "bernoulli":
            centres_init = self._bernoulli_centres(self.n_class, bit)
        else:
            centres_init = torch.randn(self.n_class, bit)
        self.centres = nn.Parameter(centres_init)

    def _bernoulli_centres(self, n_class, bit, n_trials=10):
        """Bernoulli {−1,+1} centres; keep the trial with the largest average
        pairwise Hamming distance (paper Eq.7)."""
        best, best_ham = None, -1.0
        for _ in range(n_trials):
            c = torch.bernoulli(torch.full((n_class, bit), 0.5)) * 2 - 1
            ham = (bit - c @ c.T) / 2.0                 # [n_class, n_class]
            off = ~torch.eye(n_class, dtype=torch.bool)
            avg_ham = ham[off].mean().item()
            if avg_ham > best_ham:
                best_ham, best = avg_ham, c
        return best

    def forward(self, bi, label_onehot):
        """
        Args:
            bi:           continuous hash codes from query encoder [B, bit],
                          tanh ∈ (-1,+1)
            label_onehot: one-hot labels [B, n_class]

        Returns:
            Lcos + Lquant + beta * Lproxy
        """
        if self.centres.device != bi.device:
            self.centres.data = self.centres.data.to(bi.device)

        labels = torch.argmax(label_onehot, dim=1)        # [B]

        bi_n = F.normalize(bi, dim=1)
        c_n = F.normalize(self.centres, dim=1)

        # --- Lcos: cosine-softmax classification (Eq.8) ---
        logits = self.cos_scale * (bi_n @ c_n.T)          # [B, n_class]
        Lcos = F.cross_entropy(logits, labels)

        # --- Lquant: binarisation (Eq.9) ---
        Lquant = ((bi.abs() - 1.0) ** 2).mean()

        # --- Lproxy: proxy contrastive over positive pairs (Eq.10) ---
        cos_pair = bi_n @ bi_n.T                          # [B, B]
        pos = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        pos.fill_diagonal_(0.0)
        n_pos = pos.sum()
        if n_pos > 0:
            pair_loss = F.softplus((1.0 - cos_pair) / 2.0)
            Lproxy = (pair_loss * pos).sum() / n_pos
        else:
            Lproxy = torch.tensor(0.0, device=bi.device)

        return Lcos + Lquant + self.beta * Lproxy


def compute_result_plain(dataloader, net, device):
    """"""
    bs, clses = [], []
    net.eval()
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Computing hash codes"):
            clses.append(labels)
            images = images.to(device)
            _, hash_codes, _, _ = net(images)
            bs.append(hash_codes.data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)


def train_val(config, bit):

    device = config["device"]

    # Load attribute semantics once if any ALBM-style mechanism is enabled
    attr_matrix = None
    transfer_loss_fn = None
    center_attr_loss_fn = None
    use_attr = (config.get("attr_transfer", False)
                or config.get("attr_center", False)
                or config.get("attr_patch", False))
    if use_attr:
        attr_matrix, attr_cont, S_attr, seen_idx, unseen_idx, S_attr_cont = \
            load_attribute_matrix(config)
        # The geometry loss can be aligned to either target. Default keeps the presence
        # vectors, matching the paper; 'continuous' is the control that asks whether the
        # thresholding is actually doing work.
        if config.get("attr_center_target", "presence") == "continuous":
            S_attr = S_attr_cont
            logger.info("(C) geometry target = CONTINUOUS attributes (control)")
        config["n_attr"] = attr_matrix.shape[1]
        attr_matrix = attr_matrix.to(device)
        attr_cont = attr_cont.to(device)
        
        attr_topk_mask = None
        if config.get("attr_patch_mode", "bce") == "bce_topk":
            freq = attr_matrix.mean(dim=0)                     # [n_attr]
            score = attr_matrix * (1.0 - freq)                 
            k = min(config.get("attr_patch_topk", 32), attr_matrix.shape[1])
            _, idx = score.topk(k, dim=1)
            attr_topk_mask = torch.zeros_like(attr_matrix)
            attr_topk_mask.scatter_(1, idx, 1.0)
            attr_topk_mask = attr_topk_mask.to(device)

    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train  

    net = config["net"](config, bit).to(device)

    optimizer = config["optimizer"]["type"](net.parameters(), **config["optimizer"]["optim_params"])
    scheduler = StepLR(optimizer, step_size=15, gamma=0.5)

    cosine_loss_fn = LogSoftmaxContrastiveLoss_all_positive_samples(
        temperature=config.get("tau", 0.07))

    l = list(range(config['n_class']))


    if config["hash_loss"] == "center_loss":
        hash_criterion = Center_Loss1(config, bit, l).to(device)
    elif config["hash_loss"] == "official":
        hash_criterion = SSTOfficialHashLoss(config, bit).to(device)
    else:
        hash_criterion = ProxyHashLoss(config, bit, l).to(device)

    optimizer.add_param_group(
        {"params": hash_criterion.parameters(), "lr": 2e-4})

    # ---- ALBM-inspired: attribute + patch transfer (separate λ-weighted losses) ----
    if config.get("attr_transfer", False):
        transfer_loss_fn = AttributeTransferLoss(config, S_attr, seen_idx, unseen_idx).to(device)
        transfer_loss_fn.init_centres(hash_criterion.centres)   # semantic init of unseen centres
        logger.info("(A) Attribute centre transfer enabled")

    if config.get("attr_center", False):
        center_attr_loss_fn = AttributeCenterLoss(
            config, S_attr,
            mode=config.get("attr_center_mode", "full"),
            topk=config.get("attr_center_topk", 16),
            seen_idx=seen_idx, unseen_idx=unseen_idx).to(device)
        logger.info(f"(C) Centre-attribute structure alignment enabled (mode={config.get('attr_center_mode', 'full')})")

    np.save(
        os.path.join(config["save_path"], f"{config['dataset']}_init_centres_bit{bit}.npy"),
        hash_criterion.centres.detach().cpu().numpy()
    )

    tta_test_loader = tta_db_loader = None
    if config.get("tta", False):
        from A_utils.tools_with_category_name import get_data_tta, compute_result_tta
        tta_test_loader, tta_db_loader = get_data_tta(config)

    best_mAP = 0

    for epoch in range(config["epoch"]):
        net.train()
        train_loss = 0
        align_loss_sum = 0
        hash_loss_sum2 = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epoch']} Training", ncols=150)

        for images, label_onehot, BLIP_target, ind in pbar:
            images = images.to(device)
            label_onehot = label_onehot.to(device).float()
            BLIP_target = BLIP_target.to(device)

            optimizer.zero_grad()

            pred_features, pred_hash, pred_hash2, attr_logits = net(images)

            align_loss = cosine_loss_fn(pred_features, BLIP_target, label_onehot)

            if config["hash_loss"] == "center_loss":
                # SST-main baseline: loss = align + 1.0 * L_center
                h_loss = hash_criterion(pred_hash, pred_hash2, label_onehot, ind, epoch)
                loss = align_loss + h_loss
                hash_term = h_loss.item()
            elif config["hash_loss"] == "official":
                h_loss = hash_criterion(pred_hash, pred_hash2, label_onehot, ind, epoch)
                loss = align_loss + config["hash_weight"] * h_loss
                hash_term = config["hash_weight"] * h_loss.item()
            else:
                h_loss = hash_criterion(pred_hash, label_onehot)
                loss = align_loss + config["hash_weight"] * h_loss
                hash_term = config["hash_weight"] * h_loss.item()

            # (A) unseen-centre attribute transfer (anchor)
            if transfer_loss_fn is not None:
                loss = loss + config["attr_transfer_weight"] * transfer_loss_fn(hash_criterion.centres)

            # (C) centre-attribute structure alignment
            if center_attr_loss_fn is not None:
                loss = loss + config["attr_center_weight"] * center_attr_loss_fn(hash_criterion.centres)

            # (B) patch -> attribute supervision
            attr_loss = torch.tensor(0.0, device=device)
            if attr_logits is not None:
                labels = torch.argmax(label_onehot, dim=1)
                attr_target = attr_matrix[labels]
                mode = config.get("attr_patch_mode", "bce")
                if mode == "bce":
                    attr_loss = F.binary_cross_entropy_with_logits(attr_logits, attr_target)
                elif mode == "bce_topk":
                    m = attr_topk_mask[labels]
                    attr_loss = F.binary_cross_entropy_with_logits(
                        attr_logits, attr_target, reduction='none') * m
                    attr_loss = attr_loss.sum() / (m.sum() + 1e-6)
                elif mode == "pair":
                    attr_emb = F.normalize(attr_logits, dim=-1)
                    sim = attr_emb @ attr_emb.T
                    tgt = attr_cont[labels]
                    tgt_sim = F.normalize(tgt, dim=-1) @ F.normalize(tgt, dim=-1).T
                    attr_loss = F.mse_loss(sim, tgt_sim)
                elif mode == "pair_topk":
                    attr_emb = F.normalize(attr_logits, dim=-1)
                    sim = attr_emb @ attr_emb.T
                    tgt = attr_cont[labels]
                    tgt_sim = F.normalize(tgt, dim=-1) @ F.normalize(tgt, dim=-1).T
                    k = min(config.get("attr_patch_topk", 16), tgt_sim.size(1) - 1)
                    _, idx = tgt_sim.topk(k + 1, dim=-1)
                    mask = torch.zeros_like(tgt_sim)
                    mask.scatter_(1, idx, 1.0)
                    mask = mask * (1.0 - torch.eye(mask.size(0), device=mask.device))
                    n = mask.sum()
                    attr_loss = ((sim - tgt_sim) ** 2 * mask).sum() / (n + 1e-6)
                loss = loss + config["attr_patch_weight"] * attr_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            align_loss_sum += align_loss.item()
            hash_loss_sum2 += hash_term

            postfix = {
                'loss': f'{loss.item():.4f}',
                'align': f'{align_loss.item():.4f}',
                'hash': f'{hash_term:.4f}',
            }
            if attr_logits is not None:
                postfix['attr'] = f'{attr_loss.item():.4f}'
            pbar.set_postfix(postfix)

        scheduler.step()
        train_loss = train_loss / len(train_loader)
        align_loss_sum = align_loss_sum / len(train_loader)
        hash_loss_sum2 = hash_loss_sum2 / len(train_loader)

        print(f"[Epoch {epoch+1}] Align={align_loss_sum:.4f} | Hash={hash_loss_sum2:.4f}")
        logger.info(f"[Epoch {epoch+1}] Align={align_loss_sum:.4f} | Hash={hash_loss_sum2:.4f}")

        if (epoch + 1) % config["test_map"] == 0:
            with torch.no_grad():
                if config.get("tta", False):
                    from A_utils.tools_with_category_name import compute_result_tta
                    tst_binary, tst_label = compute_result_tta(tta_test_loader, net, device)
                    trn_binary, trn_label = compute_result_tta(tta_db_loader, net, device)
                elif config["hash_loss"] == "official":
                    tst_binary, tst_label = compute_result_plain(test_loader, net, device)
                    trn_binary, trn_label = compute_result_plain(dataset_loader, net, device)
                else:
                    tst_binary, tst_label = compute_result_BlipHash1(test_loader, net, device)
                    trn_binary, trn_label = compute_result_BlipHash1(dataset_loader, net, device)

            mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(), config["topK"])
            print(f"[Eval] Epoch {epoch+1} mAP: {mAP:.4f} Best mAP: {best_mAP:.4f}")
            logger.info(f"[Eval] Epoch {epoch+1} mAP: {mAP:.4f} Best mAP: {best_mAP:.4f}")

            if mAP > best_mAP:
                best_mAP = mAP

                np.save(
                    os.path.join(config["save_path"], f"{config['dataset']}_best_centres_bit{bit}.npy"),
                    hash_criterion.centres.detach().cpu().numpy()
                )
                np.save(
                    os.path.join(config["save_path"], f"{config['dataset']}_best_tst_binary_bit{bit}.npy"),
                    tst_binary.numpy()
                )
                np.save(
                    os.path.join(config["save_path"], f"{config['dataset']}_best_tst_label_bit{bit}.npy"),
                    tst_label.numpy()
                )
                np.save(
                    os.path.join(config["save_path"], f"{config['dataset']}_best_trn_binary_bit{bit}.npy"),
                    trn_binary.numpy()
                )
                np.save(
                    os.path.join(config["save_path"], f"{config['dataset']}_best_trn_label_bit{bit}.npy"),
                    trn_label.numpy()
                )

                logger.info(f"✅ Best model saved with mAP: {mAP:.4f}")

                # Patch-branch capture. The run never saves model weights, so the only
                # chance to record what the attribute branch represents is here, at the
                # epoch the reported number comes from.
                # Guard on attr_patch: the ablated variant has no attribute branch, so
                # attr_logits is None and there is nothing to capture.
                if config.get("dump_attr", False) and config.get("attr_patch", False):
                    # Save the labels from the same pass, so the embedding-to-class
                    # pairing does not depend on the two loaders agreeing on order.
                    _, tst_lab_a, tst_attr = compute_result_BlipHash_attr(test_loader, net, device)
                    _, trn_lab_a, trn_attr = compute_result_BlipHash_attr(dataset_loader, net, device)
                    for tag, arr in (("tst", tst_attr), ("trn", trn_attr),
                                     ("tst_label", tst_lab_a), ("trn_label", trn_lab_a)):
                        np.save(os.path.join(config["save_path"],
                                             f"{config['dataset']}_best_attr_{tag}_bit{bit}.npy"),
                                arr.numpy())
                    logger.info("   patch-branch attribute embeddings saved")


if __name__ == "__main__":
    args = parse_args()
    # Re-seed from the flag. The module-level setup_seed(42) above still runs at import
    # time and would otherwise pin every experiment to seed 42.
    setup_seed(args.seed)
    config = get_config(args)

    dataset_name = f"{config['dataset']}"
    logger.add(f"{config['save_path']}/" + dataset_name + config['blip_loss'] + "--" + config['hash_loss'] + "--" + config["inform"] + f'{time.ctime()}.log')
    import json
    logger.info("config：\n" + json.dumps(config, indent=4, ensure_ascii=False, default=str))

    for bit in config["bit_list"]:

        logger.info(f"{config['inform']}")
        logger.info(f"{config['dataset']}")
        logger.info(f"{bit}")
        logger.info(f"blip_loss：{config['blip_loss']}，  hash_loss：{config['hash_loss']}")

        train_val(config, bit)

    #  pip install transformers==4.36.2
