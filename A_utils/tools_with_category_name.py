import numpy as np
import torch.utils.data as util_data
from torchvision import transforms
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.datasets as dsets
from torchvision.transforms.functional import InterpolationMode
from torch.cuda.amp import autocast
import os

def config_dataset(config):
    if config["dataset"] in ["CUB", "CUB_add"]:
        config["topK"] = 1000
        config["n_class"] = 200
    elif config["dataset"] == "AWA":
        config["topK"] = 4000
        config["n_class"] = 50
    elif config["dataset"] == "SUN":
        config["topK"] = 4000
        config["n_class"] = 717

    if config["dataset"] == "CUB":
          config["data_path"] = "dataset/CUB/CUB-last50_is_txt2img/images/"
    if config["dataset"] == "AWA":
        config["data_path"] = "dataset/AWA/JPEGImages/"
    if config["dataset"] == "SUN":
        config["data_path"] = "dataset/SUN_Attribute/images/"

    if config["dataset"] == "CUB" :
        if config["TGI"] == 1:
            config["data"] = {
            "train_set" : {"list_path": f"dataset/CUB/CUB-last50_is_txt2img/images/train_40_with_caption_catgoryname_blip768F.txt", "batch_size": config["batch_size"]},

            "database": {"list_path": f"dataset/CUB/CUB-last50_is_txt2img/images/database1.txt", "batch_size": config["batch_size"]},
            "test": {"list_path": f"dataset/CUB/CUB-last50_is_txt2img/images/test1.txt", "batch_size": config["batch_size"]}
            }
        else:
            config["data"] = {
            "train_set" : {"list_path": f"dataset/CUB/CUB-last50_is_txt2img/images/train_40_with_caption_catgoryname_blip768F_NOTGI.txt", "batch_size": config["batch_size"]},

            "database": {"list_path": f"dataset/CUB/CUB-last50_is_txt2img/images/database1.txt", "batch_size": config["batch_size"]},
            "test": {"list_path": f"dataset/CUB/CUB-last50_is_txt2img/images/test1.txt", "batch_size": config["batch_size"]}
            }
    elif config["dataset"] == "SUN":
        if config["TGI"] == 1:
            config["data"] = {
                "train_set" : {"list_path": f"dataset/SUN_Attribute/filetxt_500_217/train_seen_blip.txt", "batch_size": config["batch_size"]},

                "database": {"list_path": f"dataset/SUN_Attribute/filetxt_500_217/database.txt", "batch_size": config["batch_size"]},
                "test": {"list_path": f"dataset/SUN_Attribute/filetxt_500_217/test_full.txt", "batch_size": config["batch_size"]}
                }
        else:
            config["data"] = {
                "train_set" : {"list_path": f"dataset/SUN_Attribute/filetxt_500_217/train_seen_real_blip.txt", "batch_size": config["batch_size"]},

                "database": {"list_path": f"dataset/SUN_Attribute/filetxt_500_217/database.txt", "batch_size": config["batch_size"]},
                "test": {"list_path": f"dataset/SUN_Attribute/filetxt_500_217/test_full.txt", "batch_size": config["batch_size"]}
                }
    else:
        if config["TGI"] == 1:
            config["data"] = {
                "train_set" : {"list_path": f"dataset/AWA/JPEGImages/train_100_with_caption_catgoryname_AttrVoc_mskimg_SDimg_blip768F.txt", "batch_size": config["batch_size"]},

                "database": {"list_path": f"dataset/AWA/filetxt/database.txt", "batch_size": config["batch_size"]},
                "test": {"list_path": f"dataset/AWA/filetxt/test.txt", "batch_size": config["batch_size"]}
                }
        else:
            config["data"] = {
                "train_set" : {"list_path": f"dataset/AWA/JPEGImages/train_100_with_caption_catgoryname_AttrVoc_mskimg_SDimg_blip768F_NOTGI.txt", "batch_size": config["batch_size"]},

                "database": {"list_path": f"dataset/AWA/filetxt/database.txt", "batch_size": config["batch_size"]},
                "test": {"list_path": f"dataset/AWA/filetxt/test.txt", "batch_size": config["batch_size"]}
                }
    return config

draw_range = [1, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500,
              9000, 9500, 10000]

def pr_curve(rF, qF, rL, qL, draw_range=draw_range):
    n_query = qF.shape[0]
    Gnd = (np.dot(qL, rL.transpose()) > 0).astype(np.float32)
    Rank = np.argsort(CalcHammingDist(qF, rF))
    P, R = [], []
    for k in tqdm(draw_range):
        p = np.zeros(n_query)
        r = np.zeros(n_query)
        for it in range(n_query):
            gnd = Gnd[it]
            gnd_all = np.sum(gnd)
            if gnd_all == 0:
                continue
            asc_id = Rank[it][:k]
            gnd = gnd[asc_id]
            gnd_r = np.sum(gnd)
            p[it] = gnd_r / k
            r[it] = gnd_r / gnd_all
        P.append(np.mean(p))
        R.append(np.mean(r))
    return P, R

class ImageList_for_train(object):
    def __init__(self, data_path, image_list, transform_for_vae, transform_for_clip, aug=False, strong=False):
        self.imgs = [
            (
                data_path + val.split('\t')[0],
                np.array([int(la) for la in val.split('\t')[1].split()]),

                val.split('\t')[-1],
            )
            for val in image_list
        ]
        self.transform_for_blip = image_transform_for_blip(aug, strong)

    def __getitem__(self, index):
        path, label_onehot, BLIP_target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img_for_blip = self.transform_for_blip(img)

        BLIP_target = torch.tensor([float(x) for x in BLIP_target.strip().split()], dtype=torch.float)

        return img_for_blip, label_onehot, BLIP_target, index

    def __len__(self):
        return len(self.imgs)

class ImageList(object):
    def __init__(self, data_path, image_list, transform_for_vae=None, transform_for_clip=None):
        self.imgs = [
            (
                data_path + val.split('\t')[0],
                np.array([int(la) for la in val.split('\t')[1].split()]),
            )
            for val in image_list
        ]
        self.transform_for_blip = image_transform_for_blip()

    def __getitem__(self, index):
        path, label_onehot = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img_for_blip = self.transform_for_blip(img)
        return img_for_blip, label_onehot, index

    def __len__(self):
        return len(self.imgs)
    
def image_transform_for_vae():
    return transforms.Compose([
       transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ])

def image_transform_for_clip():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])
def image_transform_for_blip(aug=False, strong=False):
    t = []
    if aug:
        scale = (0.5, 1.0) if strong else (0.8, 1.0)
        t.append(transforms.RandomResizedCrop(224, scale=scale,
                                              interpolation=InterpolationMode.BICUBIC))
        t.append(transforms.RandomHorizontalFlip(p=0.5))
        if strong:
            t.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)))
    else:
        t.append(transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    ))
    return transforms.Compose(t)

class MyCIFAR10(dsets.CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        target = np.eye(10, dtype=np.int8)[np.array(target)]
        return img, target, index

def cifar_dataset(config):
    batch_size = config["batch_size"]

    train_size = 500
    test_size = 100

    if config["dataset"] == "cifar10-2":
        train_size = 5000
        test_size = 1000

    transform = transforms.Compose([
        transforms.Resize(config["crop_size"]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    cifar_dataset_root = 'dataset/cifar/'
    train_dataset = MyCIFAR10(root=cifar_dataset_root,
                              train=True,
                              transform=transform,
                              download=True)

    test_dataset = MyCIFAR10(root=cifar_dataset_root,
                             train=False,
                             transform=transform)

    database_dataset = MyCIFAR10(root=cifar_dataset_root,
                                 train=False,
                                 transform=transform)

    X = np.concatenate((train_dataset.data, test_dataset.data))
    L = np.concatenate((np.array(train_dataset.targets), np.array(test_dataset.targets)))

    first = True
    for label in range(10):
        index = np.where(L == label)[0]

        N = index.shape[0]
        perm = np.random.permutation(N)
        index = index[perm]

        if first:
            test_index = index[:test_size]
            train_index = index[test_size: train_size + test_size]
            database_index = index[train_size + test_size:]
        else:
            test_index = np.concatenate((test_index, index[:test_size]))
            train_index = np.concatenate((train_index, index[test_size: train_size + test_size]))
            database_index = np.concatenate((database_index, index[train_size + test_size:]))
        first = False

    if config["dataset"] == "cifar10":
        pass
    elif config["dataset"] == "cifar10-1":
        database_index = np.concatenate((train_index, database_index))
    elif config["dataset"] == "cifar10-2":
        database_index = train_index

    train_dataset.data = X[train_index]
    train_dataset.targets = L[train_index]
    test_dataset.data = X[test_index]
    test_dataset.targets = L[test_index]
    database_dataset.data = X[database_index]
    database_dataset.targets = L[database_index]

    print("train_dataset", train_dataset.data.shape[0])
    print("test_dataset", test_dataset.data.shape[0])
    print("database_dataset", database_dataset.data.shape[0])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4)

    database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=4)

    return train_loader, test_loader, database_loader, \
           train_index.shape[0], test_index.shape[0], database_index.shape[0]

def get_data(config):
    if "cifar" in config["dataset"]:
        return cifar_dataset(config)

    dsets = {}
    dset_loaders = {}
    data_config = config["data"]

    for data_set in ["train_set"]:
        dsets[data_set] = ImageList_for_train(config["data_path"],
                                    open(data_config[data_set]["list_path"]).readlines(),
                                    transform_for_vae=image_transform_for_vae(),
                                    transform_for_clip=image_transform_for_clip(),
                                    aug=config.get("aug", False),
                                    strong=config.get("aug_strong", False))
        print(data_set, len(dsets[data_set]))
        dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                    batch_size=data_config[data_set]["batch_size"],
                                                    shuffle=True, num_workers=4)
        
    for data_set in ["test", "database"]:
        dsets[data_set] = ImageList(config["data_path"],
                                    open(data_config[data_set]["list_path"]).readlines(),
                                    transform_for_vae=image_transform_for_vae(),
                                    transform_for_clip=image_transform_for_clip())
        print(data_set, len(dsets[data_set]))
        dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                    batch_size=data_config[data_set]["batch_size"],
                                                    shuffle=False, num_workers=4)
    return dset_loaders["train_set"], dset_loaders["test"], dset_loaders["database"], \
        len(dsets["train_set"]), len(dsets["test"]), len(dsets["database"])

class ImageList_for_train_CLip(object):
    def __init__(self, data_path, image_list, transform_for_clip):
        self.imgs = [
            (
                os.path.join(data_path, val.split('\t')[0]),
                np.array([int(la) for la in val.split('\t')[1].split()]),
                val.split('\t')[4]
            )
            for val in image_list
        ]
        self.transform_for_clip = transform_for_clip

    def __getitem__(self, index):
        path, label_onehot, clip_target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img_for_clip = self.transform_for_clip(img)
        clip_target = torch.tensor([float(x) for x in clip_target.strip().split()], dtype=torch.float)
        return img_for_clip, label_onehot, clip_target, index

    def __len__(self):
        return len(self.imgs)

class ImageList_for_Clip(object):
    def __init__(self, data_path, image_list, transform_for_clip):
        self.imgs = [
            (
                os.path.join(data_path, val.split('\t')[0]),
                np.array([int(la) for la in val.split('\t')[1].split()])
            )
            for val in image_list
        ]
        self.transform_for_clip = transform_for_clip

    def __getitem__(self, index):
        path, label_onehot = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img_for_clip = self.transform_for_clip(img)
        return img_for_clip, label_onehot, index

    def __len__(self):
        return len(self.imgs)

def get_data_for_CLIP(config):
    if "cifar" in config["dataset"]:
        return cifar_dataset(config)

    dsets = {}
    dset_loaders = {}
    data_config = config["data"]

    for data_set in ["train_set"]:
        dsets[data_set] = ImageList_for_train_CLip(config["data_path"],
                                    open(data_config[data_set]["list_path"]).readlines(),
                                    transform_for_clip=image_transform_for_clip())
        print(data_set, len(dsets[data_set]))
        dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                    batch_size=data_config[data_set]["batch_size"],
                                                    shuffle=True, num_workers=4)
        
    for data_set in ["test", "database"]:
        dsets[data_set] = ImageList_for_Clip(config["data_path"],
                                    open(data_config[data_set]["list_path"]).readlines(),
                                    transform_for_clip=image_transform_for_clip())
        print(data_set, len(dsets[data_set]))
        dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                    batch_size=data_config[data_set]["batch_size"],
                                                    shuffle=False, num_workers=4)
    return dset_loaders["train_set"], dset_loaders["test"], dset_loaders["database"], \
        len(dsets["train_set"]), len(dsets["test"]), len(dsets["database"])

def compute_result(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    for image_for_vae, image_for_clip, cls, ind in tqdm(dataloader):
        clses.append(cls)

        hash_codes, _ = net(image_for_vae.to(device), image_for_clip.to(device)) 
        bs.append(hash_codes.data.cpu())    
    return torch.cat(bs).sign(), torch.cat(clses)

def compute_result_BlipHash1(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Computing hash codes"):
            clses.append(labels)
            images = images.to(device)

            with autocast():
                _, hash_codes, _, _ = net(images)
            bs.append(hash_codes.data.cpu())

    return torch.cat(bs).sign(), torch.cat(clses)

def compute_result_BlipHash_attr(dataloader, net, device):
    bs, clses, attrs = [], [], []
    net.eval()
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Computing hash codes + attr"):
            clses.append(labels)
            images = images.to(device)
            with autocast():
                _, hash_codes, _, attr_logits = net(images)
            bs.append(hash_codes.data.cpu())
            if attr_logits is None:
                raise RuntimeError("--dump_attr requires --attr_patch 1")
            attrs.append(attr_logits.data.float().cpu())
    return torch.cat(bs).sign(), torch.cat(clses), torch.cat(attrs)

def compute_result_BlipHash_4_5(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Computing hash codes"):
            clses.append(labels)
            images = images.to(device)

            with autocast():
                _, hash_codes, _, _ = net(images)
            bs.append(hash_codes.data.cpu())

    return torch.cat(bs).sign(), torch.cat(clses)

def compute_result_BlipHash2(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Computing hash codes"):
            clses.append(labels)
            images = images.to(device)

            with autocast():
                _, hash_codes, _ = net(images)
            bs.append(hash_codes.data.cpu())

    return torch.cat(bs).sign(), torch.cat(clses)

def compute_result_with_caption(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)

        hash_codes, _ = net(img.to(device))
        bs.append(hash_codes.data.cpu())    
    return torch.cat(bs).sign(), torch.cat(clses)

def compute_result_with_caption_imgandtxtloss(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)
        
        hash_codes, _, _ = net(img.to(device)) 
        bs.append(hash_codes.data.cpu())    
    return torch.cat(bs).sign(), torch.cat(clses)

def compute_result_with_caption_txtangimg_all_in_hash(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)
        
        hash_codes, _, _, _, _ = net(img.to(device)) 
        bs.append(hash_codes.data.cpu())    
    return torch.cat(bs).sign(), torch.cat(clses)

def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

def _tta_crops(img):
    img = transforms.Resize(256, interpolation=InterpolationMode.BICUBIC)(img)
    crops = []
    for i, j in [(0, 0), (0, 32), (32, 0), (32, 32), (16, 16)]:
        c = img.crop((j, i, j + 224, i + 224))
        crops.append(c)
        crops.append(transforms.functional.hflip(c))
    return crops

class TTAImageList(object):

    def __init__(self, data_path, image_list):
        self.imgs = [
            (data_path + val.split('\t')[0],
             np.array([int(la) for la in val.split('\t')[1].split()]))
            for val in image_list
        ]
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711)),
        ])

    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = Image.open(path).convert('RGB')
        views = torch.stack([self.to_tensor(c) for c in _tta_crops(img)])
        return views, label, index

    def __len__(self):
        return len(self.imgs)

def get_data_tta(config):
    data_config = config["data"]
    test_set = TTAImageList(config["data_path"],
                            open(data_config["test"]["list_path"]).readlines())
    db_set = TTAImageList(config["data_path"],
                          open(data_config["database"]["list_path"]).readlines())
    test_loader = util_data.DataLoader(test_set, batch_size=data_config["test"]["batch_size"],
                                       shuffle=False, num_workers=4)
    db_loader = util_data.DataLoader(db_set, batch_size=data_config["database"]["batch_size"],
                                     shuffle=False, num_workers=4)
    return test_loader, db_loader

def compute_result_tta(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    with torch.no_grad():
        for views, labels, _ in tqdm(dataloader, desc="TTA hash codes"):
            clses.append(labels)
            v = views.view(-1, 3, 224, 224).to(device)
            _, h, _, _ = net(v)
            h = h.view(views.size(0), 10, -1).tanh().mean(dim=1).sign()
            bs.append(h.data.cpu())
    return torch.cat(bs), torch.cat(clses)

def CalcTopMap(rB, qB, retrievalL, queryL, topk):
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap