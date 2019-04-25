import torchvision.transforms as transforms
import torchvision.models as models
import torch
import torch.nn as nn
import sys
from PIL import Image
import numpy as np
import os
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients

eps = 2.0 / 255.0 / 0.229

model = models.resnet50(pretrained=True)
model.eval()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def clip(A, min_value, max_value):
    A = torch.max(A, min_value * torch.ones(A.shape))
    A = torch.min(A, max_value * torch.ones(A.shape))
    return A

preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
                ])

postprocess = transforms.Compose([
                transforms.Normalize(mean=[0, 0, 0], std=[1 / x for x in std]),
                transforms.Normalize(mean=[-x for x in mean], std=[1, 1, 1]),
                transforms.Lambda(lambda x : clip(x, 0.0, 1.0)),
                transforms.ToPILImage()
                ])

is_cuda = torch.cuda.is_available()

if is_cuda:
    print("Using GPU")
    model = model.cuda()
else:
    print("Using CPU")

#img_names = os.listdir(sys.argv[1])
#img_names.sort(key=lambda x : int(x[:3]))
img_names = [str(i).zfill(3) + '.png' for i in range(200)]

criterion = nn.CrossEntropyLoss()

for img_name in img_names:
    print('processing', img_name)
    img_origin = Image.open(os.path.join(sys.argv[1], img_name))
    img = preprocess(img_origin)

    #upperBound = img + eps
    #lowerBound = img - eps

    #fgsm
    if is_cuda:
        img = img.cuda()
        #upperBound = upperBound.cuda()
        #lowerBound = lowerBound.cuda()

    img = img.unsqueeze(0)
    img.requires_grad = True

    zero_gradients(img)

    img_preds = model(img)
    label_origin = img_preds.argmax().unsqueeze(0)

    loss = criterion(img_preds, label_origin)
    loss.backward()

    img = img + eps * img.grad.sign_()

    #img = torch.min(torch.max(img, lowerBound), upperBound)

    img_adv = postprocess(img.cpu()[0])
    img_adv.save(os.path.join(sys.argv[2], img_name))
    #image.save_img(os.path.join(sys.argv[2], img_name), img_adv)
