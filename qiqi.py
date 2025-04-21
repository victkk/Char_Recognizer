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
import torch as t
import torch.nn as nn
from tqdm.auto import tqdm
from torchvision import transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset
import random
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights
import torch.nn.functional as F
import timm

# 下载数据集
links = pd.read_csv("./mchar_data_list_0515.csv")
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

# 解压数据集
zip_list = ["mchar_train", "mchar_test_a", "mchar_val"]
for little_zip in zip_list:
    zip_name = os.path.join(dataset_path, little_zip)
    if not os.path.exists(zip_name):
        zip_file = zipfile.ZipFile(os.path.join(dataset_path, f"{little_zip}.zip"), "r")
        zip_file.extractall(path=dataset_path)

# 构建数据集路径索引
data_dir = {
    "train_data": f"{dataset_path}/mchar_train/",
    "val_data": f"{dataset_path}/mchar_val/",
    "test_data": f"{dataset_path}/mchar_test_a/",
    "train_label": f"{dataset_path}/mchar_train.json",
    "val_label": f"{dataset_path}/mchar_val.json",
    "submit_file": f"{dataset_path}/mchar_sample_submit_A.csv",
}


# 改进后的超参数设定
class Config:
    batch_size = 64
    lr = 1e-3
    momentum = 0.9
    weights_decay = 1e-4
    class_num = 11
    eval_interval = 1
    checkpoint_interval = 5
    checkpoints = "./checkpoints"
    print_interval = 50
    checkpoints = "./checkpoints"  # 自己创建一个文件夹用来储存权重
    pretrained = None
    start_epoch = 0
    epoches = 20
    smooth = 0.1
    model = "resnet50"
    scheduler = "CosineAnnealingWarmRestarts"
    scheduler_T0 = 2500
    scheduler_Tmult = 2
    scheduler_eta_min = 0


config = Config()


# 改进后的数据集类：增强数据增强
class DigitsDataset(Dataset):
    def __init__(self, mode="train", size=(128, 256), aug=True):
        super(DigitsDataset, self).__init__()
        self.aug = aug
        self.size = size
        self.mode = mode
        self.width = 224
        self.batch_count = 0

        if mode == "test":
            self.imgs = glob(data_dir["test_data"] + "*.png")
            self.labels = None
        else:
            labels = json.load(open(data_dir["%s_label" % mode], "r"))
            imgs = glob(data_dir["%s_data" % mode] + "*.png")
            self.imgs = [
                (img, labels[os.path.split(img)[-1]])
                for img in imgs
                if os.path.split(img)[-1] in labels
            ]

    def __getitem__(self, idx):
        if self.mode != "test":
            img, label = self.imgs[idx]
        else:
            img = self.imgs[idx]
            label = None

        img = Image.open(img)

        # 基础转换
        trans0 = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        # 数据增强
        trans1 = [transforms.Resize(128), transforms.CenterCrop((128, self.width))]
        if self.aug:
            trans1.extend(
                [
                    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),  # 增强颜色抖动
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomAffine(
                        degrees=15, translate=(0.1, 0.1)
                    ),  # 增强仿射变换
                    transforms.RandomPerspective(
                        distortion_scale=0.2, p=0.2
                    ),  # 添加透视变换
                    transforms.RandomRotation(10),  # 添加随机旋转
                ]
            )

        trans1.extend(trans0)

        if self.mode != "test":
            return (
                transforms.Compose(trans1)(img),
                t.tensor(label["label"][:4] + (4 - len(label["label"])) * [10]).long(),
            )
        else:
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


# 改进后的Resnet50模型：增加中间层和dropout
class EnhancedDigitsResnet50(nn.Module):
    def __init__(self, class_num=11):
        super(EnhancedDigitsResnet50, self).__init__()
        from torchvision.models.resnet import ResNet50_Weights

        # 加载预训练的ResNet50
        self.net = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.net = nn.Sequential(*list(self.net.children())[:-1])
        self.cnn = self.net

        # 增加中间层
        self.mid_layer = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # 定义四个分类头
        self.fc1 = nn.Linear(512, class_num)
        self.fc2 = nn.Linear(512, class_num)
        self.fc3 = nn.Linear(512, class_num)
        self.fc4 = nn.Linear(512, class_num)

    def forward(self, img):
        feat = self.cnn(img)
        feat = feat.view(feat.shape[0], -1)
        feat = self.mid_layer(feat)

        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        return c1, c2, c3, c4

class DigitsInceptionResNetV2(nn.Module):
    def __init__(self, class_num=11):
        super(DigitsInceptionResNetV2, self).__init__()
        # 使用timm库加载InceptionResNetV2
        self.net = timm.create_model('inception_resnet_v2', pretrained=True)
        # 移除分类层
        # self.net.global_pool = nn.Identity()
        self.net.classif = nn.Identity()
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
class DigitsXception(nn.Module):
    def __init__(self, class_num=11):
        super(DigitsXception, self).__init__()

        self.net = timm.create_model("xception", pretrained=True)
        # self.net = nn.Sequential(*list(self.net.children())[:-1])
        self.net.fc = nn.Identity()
        self.cnn = self.net

        # 定义四个分类头
        self.fc1 = nn.Linear(2048, class_num)
        self.fc2 = nn.Linear(2048, class_num)
        self.fc3 = nn.Linear(2048, class_num)
        self.fc4 = nn.Linear(2048, class_num)

    def forward(self, img):
        feat = self.cnn(img)
        # feat = feat.view(feat.shape[0], -1)
        # feat = self.mid_layer(feat)  # 通过中间层

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
        # 标签平滑处理
        lb_pos, lb_neg = 1.0 - self.smooth, self.smooth / (preds.size(1) - 1)
        smoothed_lb = (
            t.zeros_like(preds).fill_(lb_neg).scatter_(1, targets.unsqueeze(1), lb_pos)
        )

        # 计算log softmax
        log_soft = F.log_softmax(preds, dim=1)

        # 计算损失
        if self.class_weights is not None:
            loss = -log_soft * smoothed_lb * self.class_weights[None, :]
        else:
            loss = -log_soft * smoothed_lb

        loss = loss.sum(1)

        # 返回平均损失或总和
        if self.size_average == "mean":
            return loss.mean()
        elif self.size_average == "sum":
            return loss.sum()
        else:
            raise NotImplementedError


# 改进后的训练器：增加模型融合
class EnhancedTrainer:
    def __init__(self, val=True):
        self.device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

        # 数据加载
        self.train_set = DigitsDataset(mode="train")
        if os.name == "nt":
            self.train_loader = DataLoader(
                self.train_set,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=0,  # 数据加载线程
                pin_memory=True,
                drop_last=True,
                collate_fn=self.train_set.collect_fn,
            )

            if val:
                self.val_loader = DataLoader(
                    DigitsDataset(mode="val", aug=False),
                    batch_size=config.batch_size,
                    num_workers=0,
                    pin_memory=True,
                    drop_last=False,
                )
            else:
                self.val_loader = None
        elif os.name == "posix":
            self.train_loader = DataLoader(
                self.train_set,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=8,  # 数据加载线程
                pin_memory=True,
                drop_last=True,
                persistent_workers=True,
                collate_fn=self.train_set.collect_fn,
            )

            if val:
                self.val_loader = DataLoader(
                    DigitsDataset(mode="val", aug=False),
                    batch_size=config.batch_size,
                    num_workers=8,
                    persistent_workers=True,
                    pin_memory=True,
                    drop_last=False,
                )
            else:
                self.val_loader = None
        # 模型和优化器
        if config.model == "resnet50":
            self.model = EnhancedDigitsResnet50(config.class_num).to(self.device)
        elif config.model == "xception":
            self.model = DigitsXception(config.class_num).to(self.device)
        elif config.model == "inception_resnet_v2":
            self.model = DigitsInceptionResNetV2(config.class_num).to(self.device)
        else:
            raise NotImplementedError
        if t.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        # 损失函数
        self.criterion = LabelSmoothEntropy(smooth=config.smooth).to(self.device)

        # 优化器
        self.optimizer = Adam(
            self.model.parameters(), lr=config.lr, weight_decay=config.weights_decay
        )

        # 学习率调度器
        if config.scheduler == "CosineAnnealingWarmRestarts":
            self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.scheduler_T0,
            T_mult=config.scheduler_Tmult,
            eta_min=config.scheduler_eta_min,
        )
        elif config.scheduler == "reduced":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=0.5,
                patience=config.patience,
                verbose=True,
            )

        else:
            raise NotImplementedError
        
        
        self.best_acc = 0
        self.best_checkpoint_path = ""

        if config.pretrained is not None:
            self.load_model(config.pretrained)
            if self.val_loader is not None:
                acc = self.eval()
            self.best_acc = acc
            print(f"Loaded model from {config.pretrained}, Eval Acc: {acc*100:.2f}%")

    def train(self):
        for epoch in range(config.start_epoch, config.epoches):
            # 训练一个epoch
            train_loss, train_acc = self.train_epoch(epoch)

            # 验证
            if (epoch + 1) % config.eval_interval == 0 and self.val_loader is not None:
                val_acc, val_loss = self.eval()
                if config.scheduler == "reduced":
                    self.scheduler.step(val_acc)
                # 打印epoch总结
                print(f"\nEpoch {epoch+1}/{config.epoches} Summary:")
                print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
                print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

                # 保存最佳模型
                if val_acc > self.best_acc:
                    self.best_acc = val_acc
                    os.makedirs(config.checkpoints, exist_ok=True)
                    save_path = os.path.join(
                        config.checkpoints, f"epoch-{epoch+1}-acc-{val_acc:.2f}.pth"
                    )
                    self.save_model(save_path)
                    self.best_checkpoint_path = save_path
                    print(f"New best model saved to {save_path}")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        corrects = 0
        total_samples = 0

        tbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{config.epoches}")
        for i, (img, label) in enumerate(tbar):
            img = img.to(self.device)
            label = label.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            pred = self.model(img)

            # 计算损失
            loss = (
                self.criterion(pred[0], label[:, 0])
                + self.criterion(pred[1], label[:, 1])
                + self.criterion(pred[2], label[:, 2])
                + self.criterion(pred[3], label[:, 3])
            )

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 统计信息
            total_loss += loss.item()
            temp = t.stack(
                [
                    pred[0].argmax(1) == label[:, 0],
                    pred[1].argmax(1) == label[:, 1],
                    pred[2].argmax(1) == label[:, 2],
                    pred[3].argmax(1) == label[:, 3],
                ],
                dim=1,
            )

            batch_correct = t.all(temp, dim=1).sum().item()
            corrects += batch_correct
            total_samples += label.size(0)
            if config.scheduler == "CosineAnnealingWarmRestarts":
                self.scheduler.step()

            # 更新进度条
            tbar.set_postfix(
                {
                    "loss": f"{total_loss/(i+1):.3f}",
                    "acc": f"{corrects*100/total_samples:.2f}%",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                }
            )

        avg_loss = total_loss / len(self.train_loader)
        avg_acc = corrects * 100 / total_samples
        return avg_loss, avg_acc

    def eval(self):
        self.model.eval()
        total_loss = 0
        corrects = 0
        total_samples = 0

        with t.no_grad():
            tbar = tqdm(self.val_loader, desc="Validating")
            for i, (img, label) in enumerate(tbar):
                img = img.to(self.device)
                label = label.to(self.device)

                pred = self.model(img)

                # 计算损失
                loss = (
                    self.criterion(pred[0], label[:, 0])
                    + self.criterion(pred[1], label[:, 1])
                    + self.criterion(pred[2], label[:, 2])
                    + self.criterion(pred[3], label[:, 3])
                )
                total_loss += loss.item()

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

                batch_correct = t.all(temp, dim=1).sum().item()
                corrects += batch_correct
                total_samples += label.size(0)

                tbar.set_postfix(
                    {
                        "val_loss": f"{total_loss/(i+1):.3f}",
                        "val_acc": f"{corrects*100/total_samples:.2f}%",
                    }
                )

        avg_loss = total_loss / len(self.val_loader)
        avg_acc = corrects * 100 / total_samples
        return avg_acc, avg_loss

    def save_model(self, save_path, save_opt=False, save_config=False):
        dicts = {
            "model": self.model.state_dict(),
            "model_type": config.model,  # 保存模型类型
        }
        if save_opt:
            dicts["opt"] = self.optimizer.state_dict()
        if save_config:
            dicts["config"] = {
                s: config.__getattribute__(s)
                for s in dir(config)
                if not s.startswith("_")
            }
        t.save(dicts, save_path)

    def load_model(self, load_path, changed=False, save_opt=False, save_config=False):
        dicts = t.load(load_path)
        if not changed:
            self.model.load_state_dict(dicts["model"])
        if save_opt and "opt" in dicts:
            self.optimizer.load_state_dict(dicts["opt"])
        if save_config and "config" in dicts:
            for k, v in dicts["config"].items():
                config.__setattr__(k, v)

    # 模型融合评估函数
    def ensemble_eval(
        self, model_paths, weights=None
    ):  # weights表示各模型权重，None为平均
        if weights is None:
            weights = [1.0 / len(model_paths)] * len(model_paths)
        models = []
        for path in model_paths:
            if "resnet50" in path.lower():
                model = EnhancedDigitsResnet50(config.class_num).to(self.device)
            elif "xception" in path.lower():
                model = DigitsXception(config.class_num).to(self.device)
            elif "inception_resnet_v2" in path.lower():
                model = DigitsInceptionResNetV2(config.class_num).to(self.device)
            else:
                raise ValueError(f"Unknown model type in path: {path}")

            state_dict = t.load(path)["model"]
            if isinstance(model, nn.DataParallel):
                model = model.module
            model.load_state_dict(state_dict)
            model.eval()
            models.append(model)

        corrects = 0
        total_samples = 0

        with t.no_grad():
            tbar = tqdm(self.val_loader, desc="Ensemble Validating")
            for i, (img, label) in enumerate(tbar):
                img = img.to(self.device)
                label = label.to(self.device)

                # 初始化预测结果
                preds = [None] * 4
                for j in range(4):
                    preds[j] = t.zeros((img.size(0), config.class_num)).to(self.device)

                # 收集所有模型的预测
                for model, weight in zip(models, weights):
                    outputs = model(img)
                    for j in range(4):
                        preds[j] += weight * outputs[j]

                # 计算准确率
                temp = t.stack(
                    [
                        preds[0].argmax(1) == label[:, 0],
                        preds[1].argmax(1) == label[:, 1],
                        preds[2].argmax(1) == label[:, 2],
                        preds[3].argmax(1) == label[:, 3],
                    ],
                    dim=1,
                )

                batch_correct = t.all(temp, dim=1).sum().item()
                corrects += batch_correct
                total_samples += label.size(0)

                tbar.set_postfix({"ens_acc": f"{corrects*100/total_samples:.2f}%"})

        avg_acc = corrects * 100 / total_samples
        print(f"Ensemble Validation Accuracy: {avg_acc:.2f}%")
        return avg_acc


# 训练过程



# 预测函数 (保持不变)
def parse2class(prediction):
    char_list = [str(i) for i in range(10)]
    char_list.append("")
    ch1, ch2, ch3, ch4 = prediction
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
    df = pd.DataFrame(results, columns=["file_name", "file_code"])
    df["file_name"] = df["file_name"].apply(lambda x: x.split("/")[-1])
    df.to_csv(csv_path, sep=",", index=None)
    print(f"Results saved to {csv_path}")


def predicts(model_path, csv_path):
    if os.name == "nt":
        test_loader = DataLoader(
            DigitsDataset(mode="test", aug=False),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
    elif os.name == "posix":
        test_loader = DataLoader(
            DigitsDataset(mode="test", aug=False),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
        )
    results = []

    # 从权重文件读取模型类型
    checkpoint = t.load(model_path)
    model_type = checkpoint.get("model_type", "resnet50")  # 默认resnet50

    # 动态选择模型
    print(model_type)
    if model_type == "resnet50":
        model = EnhancedDigitsResnet50(config.class_num).cuda()
    elif model_type == "xception":
        model = DigitsXception(config.class_num).cuda()
    elif model_type == "inception_resnet_v2":
        model = DigitsInceptionResNetV2(config.class_num).cuda()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # 加载模型权重
    model.load_state_dict(checkpoint["model"])
    print(f"Loaded {model_type} model from {model_path}")

    model.eval()
    with t.no_grad():
        tbar = tqdm(test_loader, desc="Predicting")
        for i, (img, img_names) in enumerate(tbar):
            img = img.cuda()
            pred = model(img)
            results += [
                [name, code] for name, code in zip(img_names, parse2class(pred))
            ]

    results = sorted(results, key=lambda x: x[0])
    write2csv(results, csv_path)
    return results


# 模型融合预测函数
def ensemble_predict(model_paths, csv_path, weights=None):
    if weights is None:
        weights = [1.0 / len(model_paths)] * len(model_paths)
    if os.name =='nt':
        test_loader = DataLoader(
            DigitsDataset(mode="test", aug=False),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
    elif os.name == 'posix':
        test_loader = DataLoader(
            DigitsDataset(mode="test", aug=False),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
        )

    # 加载所有模型
    models = []
    for path in model_paths:
        checkpoint = t.load(path)
        model_type = checkpoint.get("model_type", "resnet50")

        if model_type == "resnet50":
            model = EnhancedDigitsResnet50(config.class_num).cuda()
        elif model_type == "xception":
            model = DigitsXception(config.class_num).cuda()
        elif model_type == "inception_resnet_v2":
            model = DigitsInceptionResNetV2(config.class_num).cuda()  
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        model.load_state_dict(checkpoint["model"])
        models.append(model.eval())
        print(f"Loaded {model_type} model from {path}")

    results = []

    with t.no_grad():
        tbar = tqdm(test_loader, desc="Ensemble Predicting")
        for i, (img, img_names) in enumerate(tbar):
            img = img.cuda()

            # 初始化预测结果
            preds = [None] * 4
            for j in range(4):
                preds[j] = t.zeros((img.size(0), config.class_num)).cuda()

            # 收集所有模型的预测
            for model, weight in zip(models, weights):
                outputs = model(img)
                for j in range(4):
                    preds[j] += weight * outputs[j]

            # 解析预测结果
            results += [
                [name, code] for name, code in zip(img_names, parse2class(preds))
            ]

    results = sorted(results, key=lambda x: x[0])
    write2csv(results, csv_path)
    return results

# config.model = "inception_resnet_v2"    
# trainer = EnhancedTrainer()
# trainer.train()

model_paths = glob(os.path.join(config.checkpoints, "*.pth"))
model_paths =[
    "./checkpoints/epoch-15-acc-78.93.pth",
    "./checkpoints/epoch-16-acc-79.00.pth",
    "./checkpoints/epoch-15-acc-78.95.pth"
]
ensemble_predict(model_paths, "ensemble_result.csv")
# trainer.ensemble_eval(model_paths[:2])你加载的权重是这两个，看起来不像是训的比较好的