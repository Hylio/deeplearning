import json
import time

import torch
import torch.optim as optim
from torchvision import transforms, datasets, utils
import torch.nn as nn
from model import AlexNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
data_transform = {
    "train": transforms.Compose([
        # 随机裁剪并缩放成224*224
        transforms.RandomResizedCrop(224),
        # 随机水平反转
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}

# 导入和加载训练集
image_path = "D:/code/deep-learning-for-image-processing-master/data_set/flower_data"

train_dataset = datasets.ImageFolder(root=image_path + "/train",
                                     transform=data_transform["train"])
train_num = len(train_dataset)

# 分batch
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=32,
                                           shuffle=True,
                                           num_workers=0)

# 导入和加载测试集
validate_dataset = datasets.ImageFolder(root=image_path + "/val",
                                     transform=data_transform["val"])
val_num = len(validate_dataset)

# 分batch
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                           batch_size=32,
                                           shuffle=True,
                                           num_workers=0)

# 为了方便读取信息，将标签和索引的映射表存入到JSON文件中
# "daisy:0, dandelion:1, rose:2, sunflower:3, tulips:4"
flower_list = train_dataset.class_to_idx
class_dict = dict((val, key)for key, val in flower_list.items())

json_str = json.dumps(class_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

net = AlexNet(num_class=5, init_weights=True)
net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0002)

save_path = './Alex.pth'
best_acc = 0
train_steps = len(train_loader)
for epoch in range(10):
    net.train()
    running_loss = 0
    start_time = time.perf_counter()

    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        outputs = net(images.to(device))
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        # 优化器更新参数
        optimizer.step()
        running_loss += loss.item()

        rate = (step+1) / len(train_loader)
        a = "*" * int(rate*50)
        b = "." * int((1-rate)*50)
        print("\r train loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate*100), a, b, loss), end="")
    print()
    print("%f s'" % (time.perf_counter()-start_time))

    net.eval()
    acc = 0
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == val_labels.to(device)).sum().item()
        val_accurate = acc / val_num

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

        print("[epoch %d] train_loss: %.3f test_accuracy: %.3f \n" %
              (epoch+1, running_loss / train_steps, val_accurate))

print("finish training, best accuracy:%.3f" % best_acc)
