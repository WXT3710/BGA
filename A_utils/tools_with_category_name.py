import numpy as np
import torch.utils.data as util_data
from torchvision import transforms
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.datasets as dsets
from torchvision.transforms.functional import InterpolationMode
from torch.cuda.amp import autocast  # 自动混合精度
import os

def config_dataset(config):
    if config["dataset"] in ["CUB", "CUB_add"]:
        config["topK"] = 1000  # 1000
        config["n_class"] = 200
    elif config["dataset"] == "AWA":
        config["topK"] = 4000  # 4000
        config["n_class"] = 50
    elif config["dataset"] == "SUN":
        config["topK"] = 4000  # 4000
        config["n_class"] = 717

    if config["dataset"] == "CUB":  # CUB_整合版，其实图片库就用加了文生图的就行，只是txt要变，所以把原版的train.txt也放到文生图目录里，这样就整合了
          config["data_path"] = "dataset/CUB/CUB-last50_is_txt2img/images/"
    if config["dataset"] == "AWA":  # AWA
        config["data_path"] = "dataset/AWA/JPEGImages/"
    if config["dataset"] == "SUN":  # SUN (SUNAttributeDB 真实图像，无文生图)
        config["data_path"] = "dataset/SUN_Attribute/images/"

    if config["dataset"] == "CUB" :
        if config["TGI"] == 1:
            config["data"] = {
            # CUB_整合版(其实只有训练集不同，其他都相同的):
            "train_set" : {"list_path": f"dataset/CUB/CUB-last50_is_txt2img/images/train_40_with_caption_catgoryname_blip768F.txt", "batch_size": config["batch_size"]},

            
            "database": {"list_path": f"dataset/CUB/CUB-last50_is_txt2img/images/database1.txt", "batch_size": config["batch_size"]},
            "test": {"list_path": f"dataset/CUB/CUB-last50_is_txt2img/images/test1.txt", "batch_size": config["batch_size"]}
            }
        else:
            config["data"] = {
            # CUB_整合版(其实只有训练集不同，其他都相同的):
            "train_set" : {"list_path": f"dataset/CUB/CUB-last50_is_txt2img/images/train_40_with_caption_catgoryname_blip768F_NOTGI.txt", "batch_size": config["batch_size"]},

            "database": {"list_path": f"dataset/CUB/CUB-last50_is_txt2img/images/database1.txt", "batch_size": config["batch_size"]},
            "test": {"list_path": f"dataset/CUB/CUB-last50_is_txt2img/images/test1.txt", "batch_size": config["batch_size"]}
            }
    elif config["dataset"] == "SUN":
        if config["TGI"] == 1:  # TGI=1: 含 unseen 伪图 (USI, 论文正确设置)
            config["data"] = {  # SUN 的
                "train_set" : {"list_path": f"dataset/SUN_Attribute/filetxt_500_217/train_seen_blip.txt", "batch_size": config["batch_size"]},

                "database": {"list_path": f"dataset/SUN_Attribute/filetxt_500_217/database.txt", "batch_size": config["batch_size"]},
                "test": {"list_path": f"dataset/SUN_Attribute/filetxt_500_217/test_full.txt", "batch_size": config["batch_size"]}
                }
        else:  # TGI=0: 仅 seen 真实图像 (纯 inductive)
            config["data"] = {
                "train_set" : {"list_path": f"dataset/SUN_Attribute/filetxt_500_217/train_seen_real_blip.txt", "batch_size": config["batch_size"]},

                "database": {"list_path": f"dataset/SUN_Attribute/filetxt_500_217/database.txt", "batch_size": config["batch_size"]},
                "test": {"list_path": f"dataset/SUN_Attribute/filetxt_500_217/test_full.txt", "batch_size": config["batch_size"]}
                }
    else:  # AWA
        if config["TGI"] == 1:
            config["data"] = {  # AWA的
                "train_set" : {"list_path": f"dataset/AWA/JPEGImages/train_100_with_caption_catgoryname_AttrVoc_mskimg_SDimg_blip768F.txt", "batch_size": config["batch_size"]},

                "database": {"list_path": f"dataset/AWA/filetxt/database.txt", "batch_size": config["batch_size"]},
                "test": {"list_path": f"dataset/AWA/filetxt/test.txt", "batch_size": config["batch_size"]}
                }
        else:
            config["data"] = {  # AWA的
                "train_set" : {"list_path": f"dataset/AWA/JPEGImages/train_100_with_caption_catgoryname_AttrVoc_mskimg_SDimg_blip768F_NOTGI.txt", "batch_size": config["batch_size"]},

                "database": {"list_path": f"dataset/AWA/filetxt/database.txt", "batch_size": config["batch_size"]},
                "test": {"list_path": f"dataset/AWA/filetxt/test.txt", "batch_size": config["batch_size"]}
                }
    return config



draw_range = [1, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500,
              9000, 9500, 10000]

def pr_curve(rF, qF, rL, qL, draw_range=draw_range):
    #  https://blog.csdn.net/HackerTom/article/details/89425729
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

class ImageList_for_train(object):  # 新写一个，用来读带有文本描述的数据集。
    # ImageList_for_train 类的新实例时，会调用这个函数。它接受三个参数： 
    # data_path：包含图像文件的目录路径。
    # image_list：一个列表，其中包含图像文件的路径和对应的标签。每个元素是一个字符串，空格分隔，第一个元素是图像的相对路径，剩下的是标签值。
    # transform：一个函数或转换对象，用于对图像进行预处理（如缩放、归一化等）。
    def __init__(self, data_path, image_list, transform_for_vae, transform_for_clip, aug=False, strong=False):
        self.imgs = [
            (
                data_path + val.split('\t')[0],  # 图像路径
                np.array([int(la) for la in val.split('\t')[1].split()]),  # 标签向量

                # val.split('\t')[7],  # BLIP特征  AWA用这个
                val.split('\t')[-1],  # BLIP特征  CUB用这个
                # val.split('\t')[2],  # 文本描述
                # val.split('\t')[3]  # 类别名
            )
            for val in image_list
        ]
        # self.transform_for_vae = transform_for_vae()
        # self.transform_for_clip = transform_for_clip()
        self.transform_for_blip = image_transform_for_blip(aug, strong)

    def __getitem__(self, index):
        path, label_onehot, BLIP_target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        # img_for_vae = self.transform_for_vae(img)
        # img_for_clip = self.transform_for_clip(img)
        img_for_blip = self.transform_for_blip(img)

        # 💥 这里加！！把BLIP_target从字符串变成Tensor
        BLIP_target = torch.tensor([float(x) for x in BLIP_target.strip().split()], dtype=torch.float)

        return img_for_blip, label_onehot, BLIP_target, index  # 返回图像、标签、描述和索引

    def __len__(self):
        return len(self.imgs)

class ImageList(object):  # 新写一个，用来读带有文本描述的数据集。
    # ImageList 类的新实例时，会调用这个函数。它接受三个参数： 
    # data_path：包含图像文件的目录路径。
    # image_list：一个列表，其中包含图像文件的路径和对应的标签。每个元素是一个字符串，空格分隔，第一个元素是图像的相对路径，剩下的是标签值。
    # transform：一个函数或转换对象，用于对图像进行预处理（如缩放、归一化等）。
    def __init__(self, data_path, image_list, transform_for_vae=None, transform_for_clip=None):
        self.imgs = [
            (
                data_path + val.split('\t')[0],  # 图像路径
                np.array([int(la) for la in val.split('\t')[1].split()]),  # 标签向量
                # val.split('\t')[7],  # BLIP特征
                # val.split('\t')[2],  # 文本描述
                # val.split('\t')[3]  # 类别名
            )
            for val in image_list
        ]
        # self.transform_for_vae = transform_for_vae()
        # self.transform_for_clip = transform_for_clip()
        self.transform_for_blip = image_transform_for_blip()

    def __getitem__(self, index):
        path, label_onehot = self.imgs[index]
        img = Image.open(path).convert('RGB')
        # img_for_vae = self.transform_for_vae(img)
        # img_for_clip = self.transform_for_clip(img)
        img_for_blip = self.transform_for_blip(img)
        return img_for_blip, label_onehot, index  # 返回图像、标签、描述和索引

    def __len__(self):
        return len(self.imgs)
    
def image_transform_for_vae():
    # 确保图像大小符合 vae 的输入要求
    return transforms.Compose([
       transforms.Resize((512, 512)),  # 调整图像大小到512x512
        transforms.ToTensor(),          # 将 PIL Image 转为张量，范围 [0,1]
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)  # 使用均值0.5和标准差0.5, 将[0,1]范围变换到[-1,1]
        )
    ])

def image_transform_for_clip():
    # 确保图像大小符合 CLIP 的输入要求
    return transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小到224x224
        transforms.ToTensor(),          # 将 PIL Image 转为张量，范围 [0,1]
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
    # Dataset
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
        # test:1000, train:5000, database:54000
        pass
    elif config["dataset"] == "cifar10-1":
        # test:1000, train:5000, database:59000
        database_index = np.concatenate((train_index, database_index))
    elif config["dataset"] == "cifar10-2":
        # test:10000, train:50000, database:50000
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
    data_config = config["data"] # 图片训练测试检索三种图片集合的位置

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
        len(dsets["train_set"]), len(dsets["test"]), len(dsets["database"]) # 这里返回的就是训练，测试，检索集的分别图片数量



# ==================================== get_data_for_CLIP ====================================
class ImageList_for_train_CLip(object):
    def __init__(self, data_path, image_list, transform_for_clip):
        self.imgs = [
            (
                os.path.join(data_path, val.split('\t')[0]),  # 图像路径
                np.array([int(la) for la in val.split('\t')[1].split()]),  # 标签 one-hot
                val.split('\t')[4]  # 第5列为保存的CLIP特征字符串
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
    data_config = config["data"] # 图片训练测试检索三种图片集合的位置

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
        len(dsets["train_set"]), len(dsets["test"]), len(dsets["database"]) # 这里返回的就是训练，测试，检索集的分别图片数量




# ==================================== 计算精度 ====================================
def compute_result(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    for image_for_vae, image_for_clip, cls, ind in tqdm(dataloader):
        clses.append(cls)

        # output = net(img.to(device))   
        # print(output)  # 打印输出内容，检查返回的值
        hash_codes, _ = net(image_for_vae.to(device), image_for_clip.to(device)) 
        bs.append(hash_codes.data.cpu())    
        # bs.append((net(img.to(device))).data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)

def compute_result_BlipHash1(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Computing hash codes"):
            clses.append(labels)
            images = images.to(device)

            with autocast():  # 💥 开启混合精度加速推理，加速程序运行效率
                _, hash_codes, _, _ = net(images)
            # _, hash_codes = net(images)  # ✅只输入图像，取出哈希层输出
            bs.append(hash_codes.data.cpu())

    return torch.cat(bs).sign(), torch.cat(clses)

def compute_result_BlipHash_attr(dataloader, net, device):
    """As compute_result_BlipHash1, but also returns the attribute-branch embedding.

    Only used when --dump_attr is set, to capture the patch-branch output for the
    analyses of what that branch actually learns. net(x) already returns attr_logits,
    so this costs no extra forward pass.
    """
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

            with autocast():  # 💥 开启混合精度加速推理，加速程序运行效率
                _, hash_codes, _, _ = net(images)
            # _, hash_codes = net(images)  # ✅只输入图像，取出哈希层输出
            bs.append(hash_codes.data.cpu())

    return torch.cat(bs).sign(), torch.cat(clses)

def compute_result_BlipHash2(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Computing hash codes"):
            clses.append(labels)
            images = images.to(device)

            with autocast():  # 💥 开启混合精度加速推理，加速程序运行效率
                _, hash_codes, _ = net(images)
            # _, hash_codes = net(images)  # ✅只输入图像，取出哈希层输出
            bs.append(hash_codes.data.cpu())

    return torch.cat(bs).sign(), torch.cat(clses)

def compute_result_with_caption(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)

        hash_codes, _ = net(img.to(device))
        bs.append(hash_codes.data.cpu())    
        # bs.append((net(img.to(device))).data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)

def compute_result_with_caption_imgandtxtloss(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)
        
        hash_codes, _, _ = net(img.to(device)) 
        bs.append(hash_codes.data.cpu())    
        # bs.append((net(img.to(device))).data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)

def compute_result_with_caption_txtangimg_all_in_hash(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)
        
        hash_codes, _, _, _, _ = net(img.to(device)) 
        bs.append(hash_codes.data.cpu())    
        # bs.append((net(img.to(device))).data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)

def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


# ==================================== TTA（测试时增强） ====================================
def _tta_crops(img):
    """resize 256 -> 5 裁剪 (224) x hflip -> 10 views。"""
    img = transforms.Resize(256, interpolation=InterpolationMode.BICUBIC)(img)
    crops = []
    for i, j in [(0, 0), (0, 32), (32, 0), (32, 32), (16, 16)]:
        c = img.crop((j, i, j + 224, i + 224))
        crops.append(c)
        crops.append(transforms.functional.hflip(c))
    return crops


class TTAImageList(object):
    """返回每张图的 10 个视角 [10,3,224,224]。"""

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
    """10 视角平均后的 sign hash。"""
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


def CalcTopMap(rB, qB, retrievalL, queryL, topk):  # topk = -1
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        # print(f"gnd.shape = {gnd.shape}------------------------") 打印出来就是检索集总图片数量  CUB就是5788
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap  # 返回 topkmap

