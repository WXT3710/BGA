import numpy as np
import torch.utils.data as util_data
from torchvision import transforms
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.datasets as dsets


class ImageList(object):
    def __init__(self, data_path, image_list, transform):
        self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)
    
class ImageList_with_caption(object):
    def __init__(self, data_path, image_list, transform):
        self.imgs = [
            (
                data_path + val.split('\t')[0],
                np.array([int(la) for la in val.split('\t')[1].split()]),
                val.split('\t')[2],
            )
            for val in image_list
        ]
        self.transform = transform

    def __getitem__(self, index):
        path, target, description = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, description, index

    def __len__(self):
        return len(self.imgs)
       
def image_transform(resize_size, crop_size, data_set):
    if data_set == "train_set":
        step = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(crop_size)]
    else:
        step = [transforms.CenterCrop(crop_size)]
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(crop_size),
    ] + step + [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711])
    ])


def get_data(config):
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]

    if config['caption'] == 1:
        dsets["train_set"] = ImageList_with_caption(config["data_path"],
                                        open(data_config["train_set"]["list_path"]).readlines(),
                                        transform=image_transform(config["resize_size"], config["crop_size"], "train_set"))
        print("train_set", len(dsets["train_set"]))
        dset_loaders["train_set"] = util_data.DataLoader(dsets["train_set"],
                                                        batch_size=data_config["train_set"]["batch_size"],
                                                        shuffle=True, num_workers=4)
        for data_set in ["test", "database"]:
            dsets[data_set] = ImageList(config["data_path"],
                                        open(data_config[data_set]["list_path"]).readlines(),
                                        transform=image_transform(config["resize_size"], config["crop_size"], data_set))
            print(data_set, len(dsets[data_set]))
            dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                        batch_size=data_config[data_set]["batch_size"],
                                                        shuffle=True, num_workers=4)

        return dset_loaders["train_set"], dset_loaders["test"], dset_loaders["database"], \
            len(dsets["train_set"]), len(dsets["test"]), len(dsets["database"])
    else:
        for data_set in ["train_set", "test", "database"]:
            dsets[data_set] = ImageList(config["data_path"],
                                        open(data_config[data_set]["list_path"]).readlines(),
                                        transform=image_transform(config["resize_size"], config["crop_size"], data_set))
            print(data_set, len(dsets[data_set]))
            dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                        batch_size=data_config[data_set]["batch_size"],
                                                        shuffle=True, num_workers=4)

        return dset_loaders["train_set"], dset_loaders["test"], dset_loaders["database"], \
            len(dsets["train_set"]), len(dsets["test"]), len(dsets["database"])
    

def compute_result(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)

        # output = net(img.to(device))    #  ``````````````````
        hash_codes, _ = net(img.to(device)) # ``````````````````
        bs.append(hash_codes.data.cpu())    # ``````````````````
        # bs.append((net(img.to(device))).data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)


def compute_result_with_caption(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)

        hash_codes, _ = net(img.to(device)) # ``````````````````
        bs.append(hash_codes.data.cpu())    # ``````````````````
        # bs.append((net(img.to(device))).data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)

def compute_result_with_caption_imgandtxtloss(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)
        
        hash_codes, _, _ = net(img.to(device)) # ``````````````````
        bs.append(hash_codes.data.cpu())    # ``````````````````
        # bs.append((net(img.to(device))).data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)

def compute_result_with_caption_txtangimg_all_in_hash(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)
        
        hash_codes, _, _, _, _ = net(img.to(device)) # ``````````````````
        bs.append(hash_codes.data.cpu())    # ``````````````````
        # bs.append((net(img.to(device))).data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)

def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def CalcTopMap(rB, qB, retrievalL, queryL, topk):  # topk = -1
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

