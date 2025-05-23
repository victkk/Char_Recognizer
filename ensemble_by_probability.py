dataset_path = "./dataset"
data_dir = {
    "train_data": f"{dataset_path}/mchar_train/",
    "val_data": f"{dataset_path}/mchar_val/",
    "test_data": f"{dataset_path}/mchar_test_a/",
    "train_label": f"{dataset_path}/mchar_train.json",
    "val_label": f"{dataset_path}/mchar_val.json",
    "extra_data": f"{dataset_path}/mchar_extra/",
    "extra_label": f"{dataset_path}/mchar_extra.json",
    "submit_file": f"{dataset_path}/mchar_sample_submit_A.csv",
    
}
from baseline import DigitsResnet50,DigitsResnet101,DigitsDataset,DataLoader,DigitsInceptionResNetV2,DigitsInceptionV4,DigitsXception,DigitsViT,DigitsMobileNet,parse2class,write2csv
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
model_paths = [
     "/data/zhangzicheng/workspace/study/Char_Recognizer/result/2025-04-14_18-11-33_freeze_3_resnet101/checkpoints/epoch-resnet50-16-acc-79.08.pth",
    "/data/zhangzicheng/workspace/study/Char_Recognizer/result/2025-04-15_22-23-20_xception_freeze3/checkpoints/xception-epoch-16-acc-77.07.pth"
    ]


test_loader = DataLoader(
        DigitsDataset(mode="test", aug=False),
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )
batch_pred = {}
for index,model_path in enumerate(model_paths):
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

    res_net.load_state_dict(torch.load(model_path)["model"])
    res_net.eval()
    tbar = tqdm(test_loader)
    with torch.no_grad():
        for i, (img, img_names) in enumerate(tbar):
            img = img.to(torch.device("cuda"))
            pred = res_net(img)
            if index==0:
                batch_pred[i] =list(pred)+[img_names]#{batch_cnt:(tensor(64,11),tensor(64,11),tensor(64,11),tensor(64,11))}
            else:         
                for digit in range(len(batch_pred[i])-1):
                    batch_pred[i][digit] += pred[digit]*0.6
results = []
result = []
for k,v in batch_pred.items():
    result  = parse2class(v[:-1])
    results += [[name, code] for name, code in zip(v[-1], result)]

results = sorted(results, key=lambda x: x[0])
csv_path = f"./result6x10r.csv"
write2csv(results, csv_path)

            