import numpy as np
import torch.utils.data as util_data
from torchvision import transforms
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.datasets as dsets


class ImageList(object):
    # ImageList 类的新实例时，会调用这个函数。它接受三个参数：
    # data_path：包含图像文件的目录路径。
    # image_list：一个列表，其中包含图像文件的路径和对应的标签。每个元素是一个字符串，空格分隔，第一个元素是图像的相对路径，剩下的是标签值。
    # transform：一个函数或转换对象，用于对图像进行预处理（如缩放、归一化等）。
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
    
class ImageList_with_caption(object):  # 新写一个，用来读带有文本描述的数据集。
    # ImageList 类的新实例时，会调用这个函数。它接受三个参数： 
    # data_path：包含图像文件的目录路径。
    # image_list：一个列表，其中包含图像文件的路径和对应的标签。每个元素是一个字符串，空格分隔，第一个元素是图像的相对路径，剩下的是标签值。
    # transform：一个函数或转换对象，用于对图像进行预处理（如缩放、归一化等）。
    def __init__(self, data_path, image_list, transform):
        self.imgs = [
            (
                data_path + val.split('\t')[0],  # 图像路径
                np.array([int(la) for la in val.split('\t')[1].split()]),  # 标签向量
                val.split('\t')[2],  # 文本描述
            )
            for val in image_list
        ]
        self.transform = transform

    def __getitem__(self, index):
        path, target, description = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, description, index  # 返回图像、标签、描述和索引

    def __len__(self):
        return len(self.imgs)
       
def image_transform(resize_size, crop_size, data_set):
    if data_set == "train_set":
        step = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(crop_size)]
    else:
        step = [transforms.CenterCrop(crop_size)]
    # 确保图像大小符合 CLIP 的输入要求
    return transforms.Compose([
        transforms.Resize(resize_size),  # 调整图像大小
        transforms.CenterCrop(crop_size),  # 将图像裁剪到指定大小
    ] + step + [
        transforms.ToTensor(),
        # 使用 CLIP 的预处理归一化参数
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711])
    ])


def get_data(config):
    dsets = {}
    dset_loaders = {}
    data_config = config["data"] # 图片训练测试检索三种图片集合的位置

    if config['caption'] == 1:  # 读取带有文本描述的数据集
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
            len(dsets["train_set"]), len(dsets["test"]), len(dsets["database"]) # 这里返回的就是训练，测试，检索集的分别图片数量
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
            len(dsets["train_set"]), len(dsets["test"]), len(dsets["database"]) # 这里返回的就是训练，测试，检索集的分别图片数量
    

def compute_result(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)

        # output = net(img.to(device))    #  ``````````````````
        # print(output)  # 打印输出内容，检查返回的值``````````````````
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
        # print(f"gnd.shape = {gnd.shape}------------------------") 这里打印出来就是检索集总图片数量  CUB就是5788
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap  # 返回 topkmap

