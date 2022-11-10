import os
import json

import torch
from PIL import Image
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

from model import mobile_vit_xx_small as create_model


@torch.no_grad()
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tp = 0  # A：真阳性-——labels=1,pred=1
    tn = 0  # D:真阴性——-labels=0,pred=0
    fp = 0  # B：假阳性——-labels=0,pred=1
    fn = 0  # C：假阴性——-labels=1,pred=0
    img_size = 224
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    imgs_root = "D:/SLN/all_dataset/zaoying/val"
    assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
    dataset = datasets.ImageFolder(root=imgs_root, transform=data_transform)
    test_num = len(dataset)
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    batch_size = 1
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    test_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=nw)

    # create model
    model = create_model(num_classes=2).to(device)
    # load model weights
    model_weight_path = "./weights/best_model_zy.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    for step, data in enumerate(test_loader):
        images, labels = data
        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        if labels.sum().item() == pred_classes.sum().item():
            if pred_classes.sum().item() == 1:
                tp += 1
            else:
                tn += 1
        else:
            if pred_classes.sum().item() == 1:
                fp += 1
            else:
                fn += 1

    print("--------------------------------------")
    A = tp
    B = fp
    C = fn
    D = tn
    print("模型判断结果：")
    print("A真阳性(true positive,TP=11）:", A)
    print("B假阳性(false positive,FP=01）:", B)
    print("C假阴性(false negative,FN=10）:", C)
    print("D真阴性(true negative,TP=00）:", D)
    print("模型预测正确数: {}/{}".format(A + D, A + C + B + D))
    print("模型预测正确率(accuracy,Ac): {:.2f}%".format(100 * (A + D) / (A + C + B + D)))
    print("\t敏感度(sensitivity,Sn) = {:.2f}%".format(100 * A / (A + C)))
    print("\t特异度(specificity,Sp) = {:.2f}%".format(100 * D / (B + D)))
    print("\t阳性预测值 = {:.2f}%".format(100 * A / (A + B)))
    print("\t阴性预测值 = {:.2f}%".format(100 * D / (C + D)))
    print("\t假阳性率 = {:.2f}%".format(100 * B / (B + D)))
    print("\t假阴性率 = {:.2f}%".format(100 * C / (A + C)))


if __name__ == "__main__":
    main()
