import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from model import RepVGG
from torch.utils.tensorboard import SummaryWriter


def main():
    device = torch.device("cuda:0")
    print("using {} device".format(device))

    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    }

    data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    image_path = os.path.join(data_root, "dataset", "PetImages")
    assert os.path.exists(image_path), "{} does not exist.".format(image_path)

    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"), transform=data_transform["train"])
    train_num = len(train_dataset)

    class_list = train_dataset.class_to_idx
    class_dict = dict((val, key) for key, val in class_list.items())
    json_str = json.dumps(class_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"), transform=data_transform["val"])
    validata_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)
    val_num = len(validate_dataset)
    print("using {} images for training and {} images for validation.".format(train_num, val_num))

    net = RepVGG(num_blocks=[2, 4, 14, 1], num_classes=2, width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=False)
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.0001)

    best_acc = 0
    epochs = 50
    save_path = "RepVGG.pth"

    writer = SummaryWriter()

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            # writer.add_graph(net, images.to(device))
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss :{:3f}".format(epoch+1, epochs, loss)

        writer.add_scalar("loss_train", scalar_value=running_loss, global_step=epoch)

        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validata_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)

        val_accurate = acc/val_num
        print("[epoch {}] train loss:{:3f} val_accurate: {:3f}".format(epoch+1, running_loss/len(train_loader), val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

        writer.add_scalar("accurate_val", scalar_value=val_accurate, global_step=epoch)
    writer.close()
    print("finish training.")


if __name__ == '__main__':
    main()









