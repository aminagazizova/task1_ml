import torch
from torchvision import models, transforms
from PIL import Image
import requests
import urllib.request

url_img = "https://pytorch.org/assets/images/deeplab1.png"
urllib.request.urlretrieve(url_img, "test.jpg")
image_path = "test.jpg"

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

#image_path = "kittens.jpg"
try:
    img = Image.open(image_path).convert("RGB")
except FileNotFoundError:
    print(f" Файл {image_path} не найден.")
    exit()

input_tensor = preprocess(img).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)

url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = requests.get(url).text.split("\n")

probabilities = torch.nn.functional.softmax(output[0], dim=0)
top5_prob, top5_catid = torch.topk(probabilities, 5)

print(f"\n Результаты классификации для {image_path}:\n")
for i in range(top5_prob.size(0)):
    print(f"{labels[top5_catid[i]]:25s}: {top5_prob[i].item():.3f}")
