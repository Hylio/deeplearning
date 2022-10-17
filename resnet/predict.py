import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model import resnet34

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image_path = "mycat.jpg"
assert os.path.exists(image_path), "file: '{}' does not exist.".format(image_path)
image = Image.open(image_path)
plt.imshow(image)
image = data_transform(image)
image = torch.unsqueeze(image, dim=0)

json_path = 'class_indices.json'
assert os.path.exists(json_path), "file: {} does not exist".format(json_path)
with open(json_path, "r") as f:
    class_indict = json.load(f)

model = resnet34(num_classes=2).to(device)
weights_path = "resNet34_1016.pth"
assert os.path.exists(weights_path), "file: {} does not exist".format(weights_path)
model.load_state_dict(torch.load(weights_path, map_location=device))

model.eval()
with torch.no_grad():
    output = torch.squeeze(model(image.to(device))).cpu()
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()

print_res = "class: {} prob: {:.3}".format(class_indict[str(predict_cla)], predict[predict_cla].numpy())
plt.title(print_res)
for i in range(len(predict)):
    print("class: {:10} prob:{:.3}".format(class_indict[str(i)], predict[i].numpy()))
plt.show()
