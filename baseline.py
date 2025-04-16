import pandas as pd
import os
import requests
import zipfile
import shutil
from glob import glob
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import torch as t
from PIL import Image
import torch.nn as nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts,CosineAnnealingLR, MultiStepLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import torch.nn.functional as F
import json
from torchvision.models.resnet import resnet50
from torchsummary import summary
import random
from torch.optim import Adam
import torchvision.models as models
import timm
# 导入wandb进行实验跟踪
import wandb
from datetime import datetime
from utils import create_timestamped_folder
from transformers import ViTFeatureExtractor, ViTModel, ViTConfig, AutoConfig

# 下载数据集
if __name__ == "__main__":
    links = pd.read_csv("./mchar_data_list_0515.csv")  # 修改成你电脑对应的路径
    dataset_path = "./dataset"
    print(f"数据集目录：{dataset_path}")
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    for i, link in enumerate(links["link"]):
        file_name = links["file"][i]
        print(file_name, "\t", link)
        file_name = os.path.join(dataset_path, file_name)
        if not os.path.exists(file_name):
            response = requests.get(link, stream=True)
            with open(file_name, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
    zip_list = ["mchar_train", "mchar_test_a", "mchar_val"]
    for little_zip in zip_list:
        zip_name = os.path.join(dataset_path, little_zip)
        if not os.path.exists(zip_name):
            zip_file = zipfile.ZipFile(
                os.path.join(dataset_path, f"{little_zip}.zip"), "r"
            )
            zip_file.extractall(path=dataset_path)
    # if os.path.exists(os.path.join(dataset_path,'__MACOSX')):
    #     shutil.rmtree(os.path.join(dataset_path,'__MACOSX'))

    # 构建数据集路径索引
dataset_path = "./dataset"
data_dir = {
    "train_data": f"{dataset_path}/svhn_train/",
    "val_data": f"{dataset_path}/mchar_val/",
    "test_data": f"{dataset_path}/mchar_test_a/",
    "train_label": f"{dataset_path}/svhn_train.json",
    "val_label": f"{dataset_path}/mchar_val.json",
    "extra_data": f"{dataset_path}/mchar_extra/",
    "extra_label": f"{dataset_path}/mchar_extra.json",
    "submit_file": f"{dataset_path}/mchar_sample_submit_A.csv",
    
}

# train_list = glob(data_dir["train_data"] + "*.png")
# test_list = glob(data_dir["test_data"] + "*.png")
# val_list = glob(data_dir["val_data"] + "*.png")
# print("train image counts: %d" % len(train_list))
# print("val image counts: %d" % len(val_list))
# print("test image counts: %d" % len(test_list))


# 看train数据集第一张的信息，长宽高等
def look_train_json():
    with open(data_dir["train_label"], "r", encoding="utf-8") as f:
        content = f.read()

    content = json.loads(content)
    print(content["000000.png"])


# look_train_json()


# 看需要输出文件的信息
def look_submit():
    df = pd.read_csv(data_dir["submit_file"], sep=",")
    print(df.head(5))


# look_submit()


# 统计图片的大小
def img_size_summary():
    sizes = []

    for img in glob(data_dir["train_data"] + "*.png"):
        img = Image.open(img)

        sizes.append(img.size)

    sizes = np.array(sizes)

    plt.figure(figsize=(10, 8))
    plt.scatter(sizes[:, 0], sizes[:, 1])
    plt.xlabel("Width")
    plt.ylabel("Height")

    plt.title("image width-height summary")
    # 保存图像到文件
    plt.savefig("image_size_summary.png")
    plt.close()

    # 记录到wandb
    if wandb.run is not None:
        wandb.log({"image_size_summary": wandb.Image("image_size_summary.png")})


# img_size_summary()


# 统计bbox的大小
def bbox_summary():
    marks = json.loads(open(data_dir["train_label"], "r").read())
    bboxes = []

    for img, mark in marks.items():
        for i in range(len(mark["label"])):
            bboxes.append(
                [mark["left"][i], mark["top"][i], mark["width"][i], mark["height"][i]]
            )

    bboxes = np.array(bboxes)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(bboxes[:, 2], bboxes[:, 3])
    ax.set_title("bbox width-height summary")
    ax.set_xlabel("width")
    ax.set_ylabel("height")
    # 保存图像到文件
    plt.savefig("bbox_summary.png")
    plt.close()

    # 记录到wandb
    if wandb.run is not None:
        wandb.log({"bbox_summary": wandb.Image("bbox_summary.png")})


# bbox_summary()


# 统计图片分别含有数字的个数
def label_summary():
    marks = json.load(open(data_dir["train_label"], "r"))
    dicts = {}
    for img, mark in marks.items():
        if len(mark["label"]) not in dicts:
            dicts[len(mark["label"])] = 0
        dicts[len(mark["label"])] += 1

    dicts = sorted(dicts.items(), key=lambda x: x[0])

    # 创建可视化
    labels = [f"{k}个数字" for k, _ in dicts]
    values = [v for _, v in dicts]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values)
    plt.title("Number of Digits Distribution")
    plt.ylabel("Count")
    plt.savefig("label_distribution.png")
    plt.close()

    # 记录到wandb
    if wandb.run is not None:
        wandb.log({"label_distribution": wandb.Image("label_distribution.png")})

    for k, v in dicts:
        print("%d个数字的图片数目: %d" % (k, v))


# label_summary()


# 超参数设定
class Config:
    batch_size = 64
    lr = 1e-3
    momentum = 0.9
    weights_decay = 1e-4
    class_num = 11
    eval_interval = 1
    checkpoint_interval = 5
    print_interval = 50
    checkpoints = "./checkpoints"  # 自己创建一个文件夹用来储存权重
    pretrained = None
    start_epoch = 0
    epoches = 20
    smooth = 0.1
    erase_prob = 0.5
    model = "resnet101"
    freeze_layer_num = 6
    scheduler = "CosineAnnealingWarmRestarts"
    scheduler_T0 = 12
    scheduler_Tmult = 2
    scheduler_eta_min = 0


config = Config()


class DigitsDataset(Dataset):
    """
    DigitsDataset
    Params:
      data_dir(string): data directory
      label_path(string): label path
      aug(bool): wheather do image augmentation, default: True
    """
    
    def __init__(self, mode="train", size=(128, 256),imgs = None, aug=True):
        super(DigitsDataset, self).__init__()
        self.aug = aug
        self.size = size
        self.mode = mode
        self.width = 224
        self.batch_count = 0
        if "vit" in config.model.lower():
            self.trans_transform = ViTFeatureExtractor.from_pretrained(
                "google/vit-large-patch16-224"
            )
        if imgs is None:
            if mode == "test":
                self.imgs = glob(data_dir["test_data"] + "*.png")
                self.labels = None
            elif mode == "val":
                labels = json.load(open(data_dir["%s_label" % mode], "r"))
                imgs = glob(data_dir["%s_data" % mode] + "*.png") 
                self.imgs = [
                    (img, labels[os.path.split(img)[-1]])
                    for img in imgs
                    if os.path.split(img)[-1] in labels
                ]
            elif mode == "train":
                labels_train = json.load(open(data_dir["%s_label" % mode], "r"))
                train_imgs = glob(data_dir["train_data" ] + "*.png") 
                self.imgs = [
                    (img, labels_train[os.path.split(img)[-1]])
                    for img in train_imgs
                    if os.path.split(img)[-1] in labels_train
                ]
            elif mode == "extra":
                labels_extra = json.load(open(data_dir["extra_label"], "r"))
                extra_imgs = glob(data_dir["extra_data"] + "*.png")
                self.imgs = [
                    (img, labels_extra[os.path.split(img)[-1]])
                    for img in extra_imgs
                    if os.path.split(img)[-1] in labels_extra
                ]
        else:
            self.imgs = imgs  
            
    def __add__(self, other:'DigitsDataset'):
        
        return DigitsDataset(imgs=self.imgs + other.imgs,mode=self.mode,size=self.size,aug=self.aug)
    def __matmul__(self, num:int):
        if(num>len(self)):
            raise ValueError("num should be less than the length of dataset")
        random.shuffle(self.imgs)
        return DigitsDataset(imgs=self.imgs[:num],mode=self.mode,size=self.size,aug=self.aug)
    def split(self,train_ratio=0.8):
        random.shuffle(self.imgs)
        train_num = int(len(self.imgs)*train_ratio)
        return DigitsDataset(imgs=self.imgs[:train_num],mode=self.mode,size=self.size,aug=self.aug),DigitsDataset(imgs=self.imgs[train_num:],mode=self.mode,size=self.size,aug=self.aug)
    def __getitem__(self, idx):
        if self.mode != "test":
            img, label = self.imgs[idx]
        else:
            img = self.imgs[idx]
            label = None
        img = Image.open(img)
        if "vit" in config.model.lower():
            img = self.trans_transform(np.array(img), return_tensors="pt")[
                "pixel_values"
            ].squeeze(0)
            if self.mode != "test":
                return (
                    img,
                    t.tensor(
                        label["label"][:4] + (4 - len(label["label"])) * [10]
                    ).long(),
                )
            else:
                # trans1.append(transforms.RandomErasing(scale=(0.02, 0.1)))
                return img, self.imgs[idx]
        else:
            trans0 = [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
            min_size = (
                self.size[0]
                if (img.size[1] / self.size[0]) < ((img.size[0] / self.size[1]))
                else self.size[1]
            )
            trans1 = [transforms.Resize(128), transforms.CenterCrop((128, self.width))]
            if self.aug:
                trans1.extend(
                    [
                        transforms.ColorJitter(0.1, 0.1, 0.1),
                        transforms.RandomGrayscale(0.1),
                        transforms.RandomAffine(15, translate=(0.05, 0.1), shear=5),
                    ]
                )
            trans1.extend(trans0)
            if self.mode != "test":
                return (
                    transforms.Compose(trans1)(img),
                    t.tensor(
                        label["label"][:4] + (4 - len(label["label"])) * [10]
                    ).long(),
                )
            else:
                # trans1.append(transforms.RandomErasing(scale=(0.02, 0.1)))
                return transforms.Compose(trans1)(img), self.imgs[idx]

    def __len__(self):
        return len(self.imgs)

    def collect_fn(self, batch):
        imgs, labels = zip(*batch)
        if self.mode == "train":
            if self.batch_count > 0 and self.batch_count % 10 == 0:
                self.width = random.choice(range(224, 256, 16))

        self.batch_count += 1
        return t.stack(imgs).float(), t.stack(labels)


class DigitsResnet101(nn.Module):
    def __init__(self, class_num=11):
        super(DigitsResnet101, self).__init__()
        self.net = models.resnet101(weights="IMAGENET1K_V2")
        self.net = nn.Sequential(*list(self.net.children())[:-1])
        ct = 0
        for child in self.net.children():
            ct += 1
            if ct < config.freeze_layer_num:
                for param in child.parameters():
                    param.requires_grad = False
        self.cnn = self.net
        self.fc1 = nn.Linear(2048, class_num)
        self.fc2 = nn.Linear(2048, class_num)
        self.fc3 = nn.Linear(2048, class_num)
        self.fc4 = nn.Linear(2048, class_num)

    def forward(self, img):
        feat = self.cnn(img)
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        return c1, c2, c3, c4

# class DigitsInceptionV3(nn.Module):
#     def __init__(self, class_num=11):
#         super(DigitsInceptionV3, self).__init__()
#         self.net = models.inception_v3(weights="IMAGENET1K_V1")
#         # 禁用辅助分类器
#         self.net.aux_logits = False
#         # 移除最后的分类层
#         self.net.fc = nn.Identity()
        
#         ct = 0
#         for child in self.net.children():
#             ct += 1
#             if ct < config.freeze_layer_num:
#                 for param in child.parameters():
#                     param.requires_grad = False
        
#         self.cnn = self.net
#         # InceptionV3特征维度是2048
#         self.fc1 = nn.Linear(2048, class_num)
#         self.fc2 = nn.Linear(2048, class_num)
#         self.fc3 = nn.Linear(2048, class_num)
#         self.fc4 = nn.Linear(2048, class_num)

#     def forward(self, img):
#         feat = self.cnn(img)
#         feat = feat.view(feat.shape[0], -1)
#         c1 = self.fc1(feat)
#         c2 = self.fc2(feat)
#         c3 = self.fc3(feat)
#         c4 = self.fc4(feat)
#         return c1, c2, c3, c4
class DigitsInceptionResNetV2(nn.Module):
    def __init__(self, class_num=11):
        super(DigitsInceptionResNetV2, self).__init__()
        # 使用timm库加载InceptionResNetV2
        self.net = timm.create_model('inception_resnet_v2', pretrained=True)
        # 移除分类层
        # self.net.global_pool = nn.Identity()
        self.net.classif = nn.Identity()
        
        ct = 0
        for name, child in self.net.named_children():
            ct += 1
            if ct < config.freeze_layer_num:
                for param in child.parameters():
                    param.requires_grad = False
        print(ct) # 18
        self.cnn = self.net
        # InceptionResNetV2特征维度是1536
        self.fc1 = nn.Linear(1536, class_num)
        self.fc2 = nn.Linear(1536, class_num)
        self.fc3 = nn.Linear(1536, class_num)
        self.fc4 = nn.Linear(1536, class_num)

    def forward(self, img):
        feat = self.cnn(img)
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        return c1, c2, c3, c4
class DigitsMobileNet(nn.Module):
    def __init__(self, class_num=11, model_name='mobilenetv3_large_100'):
        super(DigitsMobileNet, self).__init__()
        # 使用timm库加载MobileNet
        self.net = timm.create_model(model_name, pretrained=True)
        
        # 移除分类层，保留特征提取部分
        if hasattr(self.net, 'classifier'):
            self.net.classifier = nn.Identity()
        elif hasattr(self.net, 'head'):
            self.net.head = nn.Identity()
        
        # 冻结指定层数
        ct = 0
        for name, child in self.net.named_children():
            ct += 1
            if ct < config.freeze_layer_num:
                for param in child.parameters():
                    param.requires_grad = False
        print("total_layers:", ct)
        # 确定特征维度
        feature_dim = 1280  # MobileNetV2/V3 Large的特征维度
        if 'mobilenetv3_small' in model_name:
            feature_dim = 576  # MobileNetV3 Small的特征维度
        
        # 四个数字的分类层
        self.cnn = self.net
        self.fc1 = nn.Linear(feature_dim, class_num)
        self.fc2 = nn.Linear(feature_dim, class_num)
        self.fc3 = nn.Linear(feature_dim, class_num)
        self.fc4 = nn.Linear(feature_dim, class_num)

    def forward(self, img):
        feat = self.cnn(img)  # 已经经过全局池化，得到特征向量
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        return c1, c2, c3, c4
# class DigitsViT(nn.Module):
#     def __init__(self, class_num=11, model_name='vit_huge_patch16_224'):
#         super(DigitsViT, self).__init__()
#         # 使用timm库加载ViT
#         self.net = timm.create_model(model_name, pretrained=True)
        
#         # 移除分类头，只保留特征提取部分
#         self.net.head = nn.Identity()
        
#         # 冻结指定层数
#         ct = 0
#         for name, child in self.net.named_children():
#             ct += 1
#             if ct < config.freeze_layer_num:
#                 for param in child.parameters():
#                     param.requires_grad = False
        
#         # 确定特征维度 - 根据ViT变体不同而异
#         if 'base' in model_name:
#             feature_dim = 768
#         elif 'large' in model_name:
#             feature_dim = 1024
#         elif 'huge' in model_name:
#             feature_dim = 1280
#         else:
#             feature_dim = 768  # 默认值
        
#         # 四个数字的分类层
#         self.transformer = self.net
#         self.fc1 = nn.Linear(feature_dim, class_num)
#         self.fc2 = nn.Linear(feature_dim, class_num)
#         self.fc3 = nn.Linear(feature_dim, class_num)
#         self.fc4 = nn.Linear(feature_dim, class_num)

#     def forward(self, img):
#         feat = self.transformer(img)  # 获取CLS token特征
#         c1 = self.fc1(feat)
#         c2 = self.fc2(feat)
#         c3 = self.fc3(feat)
#         c4 = self.fc4(feat)
#         return c1, c2, c3, c4

class DigitsXception(nn.Module):
    def __init__(self, class_num=11):
        super(DigitsXception, self).__init__()
        # 使用timm库加载Xception
        self.net = timm.create_model('xception', pretrained=True)
        # 移除分类层
        # self.net.global_pool = nn.Identity()
        self.net.fc = nn.Identity()
        
        ct = 0
        for name, child in self.net.named_children():
            ct += 1
            if ct < config.freeze_layer_num:
                for param in child.parameters():
                    param.requires_grad = False
        print(ct) # 26
        self.cnn = self.net
        # Xception特征维度是2048
        self.fc1 = nn.Linear(2048, class_num)
        self.fc2 = nn.Linear(2048, class_num)
        self.fc3 = nn.Linear(2048, class_num)
        self.fc4 = nn.Linear(2048, class_num)

    def forward(self, img):
        feat = self.cnn(img)
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        return c1, c2, c3, c4


class DigitsInceptionV4(nn.Module):
    def __init__(self, class_num=11):
        super(DigitsInceptionV4, self).__init__()
        # 加载预训练的InceptionV4模型
        self.net = timm.create_model('inception_v4', pretrained=True)
        
        # 移除最后的分类层，只保留特征提取部分
        self.net.global_pool = nn.Identity()
        self.net.last_linear = nn.Identity()
        
        # 冻结指定层数的参数
        ct = 0
        for name, child in self.net.named_children():
            ct += 1
            if ct < config.freeze_layer_num:
                for param in child.parameters():
                    param.requires_grad = False
        print(ct)
        self.cnn = self.net
        # InceptionV4的特征输出是1536维
        self.fc1 = nn.Linear(1536, class_num)
        self.fc2 = nn.Linear(1536, class_num)
        self.fc3 = nn.Linear(1536, class_num)
        self.fc4 = nn.Linear(1536, class_num)

    def forward(self, img):
        # 提取特征
        feat = self.cnn(img)
        # 将特征展平
        feat = feat.view(feat.shape[0], -1)
        # 通过四个不同的全连接层得到四个分类结果
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        return c1, c2, c3, c4

    
class DigitsViT(nn.Module):
    def __init__(self, class_num=11, dp_rate=0):
        raise NotImplementedError
        super(DigitsViT, self).__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

        count = 0
        for child in self.vit.children():
            count += 1
            if count < 5:
                for param in child.parameters():
                    param.requires_grad = False
        # self.model_trans_top = nn.Sequential(*list(vit.children())[:-2])
        # self.trans_layer_norm = list(vit.children())[2]
        # self.trans_flatten = nn.Flatten()
        # self.trans_linear = nn.Linear(150528, 2048)
        # self.dropout = nn.Dropout(dp_rate)
        # self.fc1 = nn.Linear(2048, class_num)
        # self.fc2 = nn.Linear(2048, class_num)
        # self.fc3 = nn.Linear(2048, class_num)
        # self.fc4 = nn.Linear(2048, class_num)
        self.fc1 = nn.Linear(self.vit.config.hidden_size, class_num)
        
        self.fc2 = nn.Linear(self.vit.config.hidden_size, class_num)
        
        self.fc3 = nn.Linear(self.vit.config.hidden_size, class_num)
        
        self.fc4 = nn.Linear(self.vit.config.hidden_size, class_num)
    def forward(self, img):
        feat = self.vit(img)    
        # patch_state = result_trans.last_hidden_state[
        #     :, 1:, :
        # ]  # Remove the classification token and get the last hidden state of all patchs
        # result_trans = self.trans_layer_norm(patch_state)
        # result_trans = self.trans_flatten(result_trans)
        # result_trans = self.dropout(result_trans)
        # result_trans = self.trans_linear(result_trans)
        # feat = self.dropout(result_trans)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        return c1, c2, c3, c4


class DigitsResnet50(nn.Module):
    def __init__(self, class_num=11):
        super(DigitsResnet50, self).__init__()
        self.net = resnet50(pretrained=True)
        self.net = nn.Sequential(*list(self.net.children())[:-1])
        ct = 0
        for child in self.net.children():
            ct += 1
            if ct < 6:
                for param in child.parameters():
                    param.requires_grad = False
        self.cnn = self.net
        self.fc1 = nn.Linear(2048, class_num)
        self.fc2 = nn.Linear(2048, class_num)
        self.fc3 = nn.Linear(2048, class_num)
        self.fc4 = nn.Linear(2048, class_num)

    def forward(self, img):
        feat = self.cnn(img)
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        return c1, c2, c3, c4


class LabelSmoothEntropy(nn.Module):
    def __init__(self, smooth=0.1, class_weights=None, size_average="mean"):
        super(LabelSmoothEntropy, self).__init__()
        self.size_average = size_average
        self.smooth = smooth
        self.class_weights = class_weights

    def forward(self, preds, targets):
        lb_pos, lb_neg = 1 - self.smooth, self.smooth / (preds.shape[0] - 1)
        smoothed_lb = (
            t.zeros_like(preds).fill_(lb_neg).scatter_(1, targets[:, None], lb_pos)
        )
        log_soft = F.log_softmax(preds, dim=1)
        if self.class_weights is not None:
            loss = -log_soft * smoothed_lb * self.class_weights[None, :]
        else:
            loss = -log_soft * smoothed_lb
        loss = loss.sum(1)
        if self.size_average == "mean":
            return loss.mean()
        elif self.size_average == "sum":
            return loss.sum()
        else:
            raise NotImplementedError


class Trainer:
    def __init__(self, val=True):
        self.device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
        
        
        if val:
            # all_set = DigitsDataset(mode="train")#  + DigitsDataset(mode="val")  # hook for dataset wandb 
            # self.train_set, self.val_set = all_set.split(train_ratio=0.9)
            self.train_set = DigitsDataset(mode="train")+DigitsDataset(mode="extra") 
            # print(len(self.train_set))
            self.val_set = DigitsDataset(mode="val") # hook for dataset wandb
            # self.val_set.aug = False 
            
            self.train_loader = DataLoader(
                self.train_set,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=8,
                pin_memory=True,
                persistent_workers=True,
                drop_last=True,
                collate_fn=self.train_set.collect_fn,
            )
            self.val_loader = DataLoader(
                self.val_set,
                batch_size=config.batch_size,
                num_workers=8,
                shuffle=False,
                pin_memory=True,
                drop_last=False,
                persistent_workers=True,
            )
        else:
            raise NotImplementedError
            # all_set = DigitsDataset(mode="train") + DigitsDataset(mode="val")
            # self.train_loader = DataLoader(
            #     self.train_set,
            #     batch_size=config.batch_size,
            #     shuffle=True,
            #     num_workers=8,
            #     pin_memory=True,
            #     persistent_workers=True,
            #     drop_last=True,
            #     collate_fn=self.train_set.collect_fn,
            # )
            # self.val_loader = None
            
        if config.model == "resnet101":
            self.model = DigitsResnet101(config.class_num).to(self.device)
        elif config.model == "resnet50":
            self.model = DigitsResnet50(config.class_num).to(self.device)
        elif config.model == "vit":
            self.model = DigitsViT(config.class_num).to(self.device)
        elif config.model == "xception":
            self.model = DigitsXception(config.class_num).to(self.device)
        elif config.model == "inception_v4":
            self.model = DigitsInceptionV4(config.class_num).to(self.device)
        elif config.model == "inception_resnet_v2":
            self.model = DigitsInceptionResNetV2(config.class_num).to(self.device)
        elif config.model == "mobilenet_v2":
            self.model = DigitsMobileNet(config.class_num,model_name='mobilenetv2_100').to(self.device)
        elif config.model == "mobilenet_v3":
            self.model = DigitsMobileNet(config.class_num,model_name='mobilenetv3_large_100').to(self.device)
        else:
            RuntimeError("No such model")
        self.criterion = LabelSmoothEntropy().to(self.device)
        self.optimizer = Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            amsgrad=False,
        )
        if config.scheduler == "CosineAnnealingWarmRestarts":
            self.lr_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.scheduler_T0,
            T_mult=config.scheduler_Tmult,
            eta_min=config.scheduler_eta_min,
        )
        elif config.scheduler =="CosineAnnealingLR":
            self.lr_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.scheduler_T0,
            eta_min=config.scheduler_eta_min,
        )
        else:
            raise NotImplementedError
        self.best_acc = 0
        self.best_checkpoint_path = ""
        self.best_acc = 0
        self.best_checkpoint_path = ""
        # 是否载入预训练模型
        if config.pretrained is not None:
            self.load_model(config.pretrained)
            # print('Load model from %s'%config.pretrained)
            if self.val_loader is not None:
                acc = self.eval_tts(6)
            # self.best_acc = acc
            print("Load model from %s, Eval Acc: %.2f" % (config.pretrained, acc * 100))
            # 记录预训练模型的验证准确率
            if wandb.run is not None:
                wandb.log({"pretrained_val_acc": acc * 100})

    def train(self):
        for epoch in range(config.start_epoch, config.epoches):
            # 训练一个epoch并获取训练准确率
            train_acc = self.train_epoch(epoch)

            # 使用wandb记录训练准确率
            if wandb.run is not None:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_acc": train_acc,
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                    }
                )

            if (epoch + 1) % config.eval_interval == 0:
                print("Start Evaluation")
                if self.val_loader is not None:
                    acc = self.eval()

                    # 使用wandb记录验证准确率
                    if wandb.run is not None:
                        wandb.log({"val_acc": acc * 100, "epoch": epoch})

                    # 保存最优模型
                    # if acc > self.best_acc:
                    os.makedirs(config.checkpoints, exist_ok=True)
                    save_path = os.path.join(
                        config.checkpoints,
                        "%s-epoch-%d-acc-%.2f.pth" % (config.model, epoch + 1, acc * 100),
                    )
                    self.save_model(save_path)
                    print("%s saved successfully..." % save_path)
                    self.best_acc = acc
                    self.best_checkpoint_path = save_path

    def train_epoch(self, epoch):
        total_loss = 0
        corrects = 0
        tbar = tqdm(self.train_loader)
        self.model.train()
        for i, (img, label) in enumerate(tbar):
            img = img.to(self.device)
            label = label.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(img)
            loss = (
                self.criterion(pred[0], label[:, 0])
                + self.criterion(pred[1], label[:, 1])
                + self.criterion(pred[2], label[:, 2])
                + self.criterion(pred[3], label[:, 3])
            )

            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            # 计算准确率
            temp = t.stack(
                [
                    pred[0].argmax(1) == label[:, 0],
                    pred[1].argmax(1) == label[:, 1],
                    pred[2].argmax(1) == label[:, 2],
                    pred[3].argmax(1) == label[:, 3],
                ],
                dim=1,
            )
            corrects += t.all(temp, dim=1).sum().item()

            batch_acc = corrects * 100 / ((i + 1) * config.batch_size)
            tbar.set_description("loss: %.6f, acc: %.3f" % (total_loss / (i + 1), batch_acc))

            # 定期记录batch训练信息到wandb
            if (i + 1) % config.print_interval == 0:
                self.lr_scheduler.step()

            # 记录学习率、损失和准确率
            if wandb.run is not None:
                wandb.log(
                    {
                        "batch": i + epoch * len(self.train_loader),
                        "epoch": epoch + i / len(self.train_loader),
                        "batch_loss": loss.item(),
                        "batch_acc": batch_acc,
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                    }
                )

        # 返回整个epoch的训练准确率
        return corrects * 100 / ((i + 1) * config.batch_size)

    def eval_tts(self,cnt=6):
        self.model.eval()


        with t.no_grad():
            
            batch_pred = {}
            for index in range(cnt):
                corrects = 0
                total_samples = 0
                tbar = tqdm(self.val_loader)
                for i, (img, label) in enumerate(tbar):
                    img = img.to(self.device)
                    label = label.to(self.device)
                    pred = self.model(img)
                    if index==0:
                        batch_pred[i] =list(pred)+[label]#{batch_cnt:(tensor(64,11),tensor(64,11),tensor(64,11),tensor(64,11))}
                    else:         
                        for digit in range(len(batch_pred[i])-1):
                            batch_pred[i][digit] += pred[digit]
                    # 计算准确率
                    temp = t.stack(
                        [
                            pred[0].argmax(1) == label[:, 0],
                            pred[1].argmax(1) == label[:, 1],
                            pred[2].argmax(1) == label[:, 2],
                            pred[3].argmax(1) == label[:, 3],
                        ],
                        dim=1,
                    )
                    batch_corrects = t.all(temp, dim=1).sum().item()
                    corrects += batch_corrects
                    total_samples += label.shape[0]
                    tbar.set_description("Val Acc: %.2f" % (corrects * 100 / total_samples))
            corrects = 0
            total_samples = 0
            tbar = tqdm(batch_pred.items())
            for i,batch_result in tbar:
                ch1, ch2, ch3, ch4, label = batch_result
                ch1, ch2, ch3, ch4 = ch1.argmax(1), ch2.argmax(1), ch3.argmax(1), ch4.argmax(1)
                temp = t.stack(
                    [
                        ch1 == label[:, 0],
                        ch2 == label[:, 1],
                        ch3 == label[:, 2],
                        ch4 == label[:, 3],
                    ],
                    dim=1,
                )
                batch_corrects = t.all(temp, dim=1).sum().item()
                corrects += batch_corrects
                total_samples += label.shape[0]
                tbar.set_description("Val Acc: %.2f" % (corrects * 100 / total_samples))
        self.model.train()
        return corrects / total_samples
    
    def eval(self):
        self.model.eval()
        corrects = 0
        total_samples = 0

        with t.no_grad():
            tbar = tqdm(self.val_loader)
            for i, (img, label) in enumerate(tbar):
                img = img.to(self.device)
                label = label.to(self.device)
                pred = self.model(img)
                
                # 计算准确率
                temp = t.stack(
                    [
                        pred[0].argmax(1) == label[:, 0],
                        pred[1].argmax(1) == label[:, 1],
                        pred[2].argmax(1) == label[:, 2],
                        pred[3].argmax(1) == label[:, 3],
                    ],
                    dim=1,
                )
                batch_corrects = t.all(temp, dim=1).sum().item()
                corrects += batch_corrects
                total_samples += img.size(0)

                # 更新进度条信息
                tbar.set_description("Val Acc: %.2f" % (corrects * 100 / total_samples))

        self.model.train()
        return corrects / total_samples

    def save_model(self, save_path, save_opt=False, save_config=False):
        dicts = {}
        dicts["model"] = self.model.state_dict()
        if save_opt:
            dicts["opt"] = self.optimizer.state_dict()
        if save_config:
            dicts["config"] = {
                s: config.__getattribute__(s)
                for s in dir(config)
                if not s.startswith("_")
            }
        t.save(dicts, save_path)

        # 将模型保存到wandb
        # if wandb.run is not None:
        #     wandb.save(save_path)

    def load_model(self, load_path, changed=False, save_opt=False, save_config=False):
        dicts = t.load(load_path)
        if not changed:
            self.model.load_state_dict(dicts["model"])
        # else:
        #     dicts = t.load(load_path)['model']
        #     keys = list(net.state_dict().keys())
        #     values = list(dicts.values())
        #     new_dicts = {k: v for k, v in zip(keys, values)}
        #     self.model.load_state_dict(new_dicts)

        if save_opt:
            self.optimizer.load_state_dict(dicts["opt"])

        if save_config:
            for k, v in dicts["config"].items():
                config.__setattr__(k, v)


def parse2class(prediction):
    """
    Params:
    prediction(tuple of tensor):
    """
    ch1, ch2, ch3, ch4 = prediction
    char_list = [str(i) for i in range(10)]
    char_list.append("")
    ch1, ch2, ch3, ch4 = ch1.argmax(1), ch2.argmax(1), ch3.argmax(1), ch4.argmax(1)
    ch1, ch2, ch3, ch4 = (
        [char_list[i.item()] for i in ch1],
        [char_list[i.item()] for i in ch2],
        [char_list[i.item()] for i in ch3],
        [char_list[i.item()] for i in ch4],
    )
    res = [c1 + c2 + c3 + c4 for c1, c2, c3, c4 in zip(ch1, ch2, ch3, ch4)]
    return res


def write2csv(results, csv_path):
    """
    results(list):
    """
    # 定义输出文件
    df = pd.DataFrame(results, columns=["file_name", "file_code"])
    df["file_name"] = df["file_name"].apply(lambda x: x.split("/")[-1])
    save_name = csv_path
    df.to_csv(save_name, sep=",", index=None)
    print("Results.saved to %s" % save_name)

def predicts_tts(model_path, csv_path,tts_times=10):
    csv_path = os.path.join(os.path.split(os.path.split(model_path)[0])[0], csv_path)
    print(csv_path)
    # 初始化wandb以记录推理过程（如果还没有初始化）
    if wandb.run is None:
        wandb.init(
            project="digits-recognition-inference",
            name=f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={"model_path": model_path},
        )

    test_loader = DataLoader(
        DigitsDataset(mode="test", aug=True if tts_times>1 else False),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )
    results = []
    res_path = model_path

    # 根据模型路径判断使用哪个模型架构
    if "resnet101" in model_path:
        res_net = DigitsResnet101().cuda()
    elif "resnet50" in model_path:
        res_net = DigitsResnet50().cuda()
    elif "vit" in model_path:
        res_net = DigitsViT().cuda()
    elif "inception_v4" in model_path:
        res_net = DigitsInceptionV4().cuda()
    elif "inception_resnet_v2" in model_path:
        res_net = DigitsInceptionResNetV2().cuda()
    elif"xception" in model_path:
        res_net = DigitsXception().cuda()
    elif "mobilenet_v2" in model_path:
        res_net = DigitsMobileNet(model_name='mobilenetv2_100').cuda()
    elif "mobilenet_v3" in model_path:
        res_net = DigitsMobileNet(model_name='mobilenetv3_large_100').cuda()
    else:
        raise NotImplementedError
    res_net.load_state_dict(t.load(res_path)["model"])
    print("Load model from %s successfully" % model_path)

    # 记录模型结构到wandb
    if wandb.run is not None:
        wandb.watch(res_net, log="all", log_freq=100)

    
    res_net.eval()
    batch_pred = {}
    for index in range(tts_times):
        tbar = tqdm(test_loader)
        with t.no_grad():
            for i, (img, img_names) in enumerate(tbar):
                img = img.to(t.device("cuda"))
                pred = res_net(img)
                if index==0:
                    batch_pred[i] =list(pred)+[img_names]#{batch_cnt:(tensor(64,11),tensor(64,11),tensor(64,11),tensor(64,11))}
                else:         
                    for digit in range(len(batch_pred[i])-1):
                        batch_pred[i][digit] += pred[digit]
    results = []
    result = []
    for k,v in batch_pred.items():
        result  = parse2class(v[:-1])
        results += [[name, code] for name, code in zip(v[-1], result)]

    results = sorted(results, key=lambda x: x[0])
    write2csv(results, csv_path)
    return results
def predicts(model_path, csv_path):
    # 初始化wandb以记录推理过程（如果还没有初始化）

    if wandb.run is None:
        wandb.init(
            project="digits-recognition-inference",
            name=f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={"model_path": model_path},
        )

    test_loader = DataLoader(
        DigitsDataset(mode="test", aug=False),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )
    results = []
    res_path = model_path

    # 根据模型路径判断使用哪个模型架构
    if "resnet101" in model_path:
        res_net = DigitsResnet101().cuda()
    elif "resnet50" in model_path:
        res_net = DigitsResnet50().cuda()
    elif "vit" in model_path:
        res_net = DigitsViT().cuda()
    elif "inception_v4" in model_path:
        res_net = DigitsInceptionV4().cuda()
    elif "inception_resnet_v2" in model_path:
        res_net = DigitsInceptionResNetV2().cuda()
    elif"xception" in model_path:
        res_net = DigitsXception().cuda()
    elif "mobilenet_v2" in model_path:
        res_net = DigitsMobileNet(model_name='mobilenetv2_100').cuda()
    elif "mobilenet_v3" in model_path:
        res_net = DigitsMobileNet(model_name='mobilenetv3_large_100').cuda()
    else:
        raise NotImplementedError
    res_net.load_state_dict(t.load(res_path)["model"])
    print("Load model from %s successfully" % model_path)

    # 记录模型结构到wandb
    if wandb.run is not None:
        wandb.watch(res_net, log="all", log_freq=100)

    tbar = tqdm(test_loader)
    res_net.eval()
    with t.no_grad():
        for i, (img, img_names) in enumerate(tbar):
            img = img.to(t.device("cuda"))
            pred = res_net(img)

            # 获取预测结果
            batch_results = parse2class(pred)
            results += [[name, code] for name, code in zip(img_names, batch_results)]

            # 记录一些预测样本到wandb

            # 将固定间隔改为概率抽样
            if random.random() < 1 / 50 and wandb.run is not None:
                sample_imgs = img[:].cpu()
                sample_preds = batch_results[:]

                # 显示样本和预测结果
                wandb.log(
                    {
                        f"test_samples_{i}": [
                            wandb.Image(img_tensor, caption=f"Pred: {pred}")
                            for img_tensor, pred in zip(sample_imgs, sample_preds[:])
                        ]
                    }
                )

    # 排序并保存结果
    results = sorted(results, key=lambda x: x[0])
    write2csv(results, csv_path)

    # 记录结果文件到wandb
    if wandb.run is not None:
        wandb.save(csv_path)
        # wandb.finish()  # 完成推理后结束wandb会话

    return results
def find_line_with_content(content):
    """查找脚本中包含特定内容的行"""
    lines=[]
    with open(__file__, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file, 1):
            if content in line:
                lines.append(line.strip())

# if __name__ == "__main__":
#     # 创建带时间戳的文件夹
#     config.model = "resnet101"
#     config.epoches = 100
#     config.scheduler_T0 = 60
#     config.lr =0.0005
#     config.scheduler_eta_min = 0.0001
#     config.scheduler = "CosineAnnealingWarmRestarts"
#     config.pretrained = "result/2025-04-14_16-51-52_freeze_0_resnet101/checkpoints/epoch-resnet50-14-acc-78.50.pth"
#     # for freeze_layer_num in [4,8,12,16,2,6,10,14,]:
#     config.freeze_layer_num = 0
#     save_dir = create_timestamped_folder(f"{config.model}_freeze{config.freeze_layer_num}", base_dir="result")
#     config.checkpoints = os.path.join(save_dir, "checkpoints")
#     # 初始化wandb
#     config_dict = {}
#     for attr in dir(config):
#         if not attr.startswith('__'):
#             value = getattr(config, attr)
#             if not callable(value):
#                 config_dict[attr] = value
#     config_dict["dataset"]=find_line_with_content("hook for dataset wandb")
#     wandb.init(
#         project="digits-recognition",
#         name=f"{config.model}_train+extra_val_freeze{config.freeze_layer_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
#         config=config_dict
#     )

#     # 训练模型
#     trainer = Trainer()
#     trainer.train()

#     result_csv = os.path.join(save_dir, "result.csv")
#     predicts(trainer.best_checkpoint_path, result_csv)

#     # 结束wandb会话
#     wandb.finish()

predicts_tts("result/2025-04-14_14-39-18_freeze_0_resnet101/checkpoints/epoch-resnet50-13-acc-79.06.pth","result.csv",1)
