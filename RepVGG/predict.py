import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model import RepVGG, repvgg_model_convert


device = torch.device("cuda:0")
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

image_path = "mycat.jpg"
assert os.path.exists(image_path), "file : {} does not exists.".format(image_path)
image = Image.open(image_path)
plt.imshow(image)
image = data_transform(image)
image = torch.unsqueeze(image, dim=0)

json_path = "class_indices.json"
assert os.path.exists(json_path), "file : {} does not exists.".format(json_path)
with open(json_path, 'r') as f:
    class_dict = json.load(f)

model = RepVGG(num_blocks=[2, 4, 14, 1], num_classes=2, width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=False)
model.to(device)
weights_path = "RepVGG.pth"
assert os.path.exists(weights_path), "file : {} does not exists.".format(weights_path)
model.load_state_dict(torch.load(weights_path))
deploy_model = repvgg_model_convert(model, save_path="RepVGG_deploy.pth", do_copy=True)

model.eval()
with torch.no_grad():
    output = torch.squeeze(model(image.to(device))).cpu()
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()

title = "class: {} prob: {:.3}".format(class_dict[str(predict_cla)], predict[predict_cla].numpy())
plt.title(title)
for i in range(len(predict)):
    print("class :{} prob: {:.3f}".format(class_dict[str(i)], predict[i].numpy()))
plt.show()
