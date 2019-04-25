import torchvision.transforms as transforms
import torchvision.models as models
import torch
import sys
from PIL import Image
import numpy as np
import os
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients

eps = 2.0 / 255.0 / 0.229
max_iter = 50
overshoot = 0.05
subsample = 10

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

for img_name in img_names:
    print('processing', img_name)
    img_origin = Image.open(os.path.join(sys.argv[1], img_name))
    img = preprocess(img_origin)

    upperBound = img + eps
    lowerBound = img - eps

    #deep fool
    if is_cuda:
        img = img.cuda()
        upperBound = upperBound.cuda()
        lowerBound = lowerBound.cuda()

    img = img.unsqueeze(0)
    img.requires_grad = True

    img_preds = model(img).detach().cpu().numpy().ravel()
    labels = img_preds.argsort()[::-1]

    labels = labels[0:subsample]
    label_origin = labels[0]

    x = img.detach()
    x.requires_grad = True

    x_preds = model(x)
    k_i = label_origin

    for iter_num in range(max_iter):
        if k_i != label_origin:
            if np.linalg.norm((x - img).detach().cpu().numpy().ravel(), np.inf) < eps / 2:
                r2 = r / np.max(r) * eps / 2

                if is_cuda:
                    x = x.detach() + torch.from_numpy(r2).cuda()
                else:
                    x = x.detach() + torch.from_numpy(r2)

                x = torch.min(torch.max(x, lowerBound), upperBound)

                x.requires_grad = True
                x_preds = model(x)
                k_i = np.argmax(x_preds.detach().cpu().numpy().ravel())

                continue

            else:
                break
        score = np.inf

        x_preds[0, label_origin].backward(retain_graph=True)
        origin_grad = x.grad.detach().cpu().numpy()

        for k in range(1, subsample):
            zero_gradients(x)

            x_preds[0, labels[k]].backward(retain_graph=True)
            cur_grad = x.grad.detach().cpu().numpy()

            w_k = cur_grad - origin_grad
            f_k = (x_preds[0, labels[k]] - x_preds[0, label_origin]).detach().cpu().numpy()

            score_k = np.abs(f_k) / (np.linalg.norm(w_k.ravel(), 1) + 1e-8)

            if score_k < score:
                score = score_k
                w = w_k

        r = (score + 1e-4) * np.sign(w)

        if is_cuda:
            x = x.detach() + (1 + overshoot) * torch.from_numpy(r).cuda()
        else:
            x = x.detach() + (1 + overshoot) * torch.from_numpy(r)

        x = torch.min(torch.max(x, lowerBound), upperBound)

        x.requires_grad = True
        x_preds = model(x)
        k_i = np.argmax(x_preds.detach().cpu().numpy().ravel())

    x = torch.min(torch.max(x, lowerBound), upperBound)


    print(k_i, label_origin)

    img_adv = postprocess(x.cpu()[0])
    img_adv.save(os.path.join(sys.argv[2], img_name))
    #image.save_img(os.path.join(sys.argv[2], img_name), img_adv)
