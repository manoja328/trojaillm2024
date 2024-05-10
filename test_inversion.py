import logging
import os
import numpy as np
import torch.nn.utils.prune as prune
from torch.utils.data import Dataset, DataLoader
import torch, torch.nn as nn
import torch.nn.functional as F
import torch
import itertools
import random
from sklearn.metrics import accuracy_score
from types import SimpleNamespace
import matplotlib.pyplot as plt
from cnn_trojan import *

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# Function to save the output of each block
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook


## https://github.com/VinAIResearch/Warping-based_Backdoor_Attack-release/blob/94453080f241053ac7c8cc4717da20806ee17e5c/defenses/neural_cleanse/detecting.py
# class SimpleTrigger(nn.Module):
#     def __init__(self, init_mask):
#         super().__init__()
#         self._EPSILON = 1e-8
#         if init_mask is not None:
#             self.mask_tanh = nn.Parameter(torch.tensor(init_mask))
#             self.pertubation = nn.Parameter(torch.zeros_like(self.mask_tanh))

#     def forward(self, x):
#         mask = nn.Tanh()(self.mask_tanh)
#         bounded = mask / (2 + self._EPSILON) + 0.5
#         features = bounded * self.pertubation + (1-bounded) * x
#         return features

## https://github.com/frkl/trojai-fuzzing-vision/blob/main/arch/fixed_polygon.py
## https://github.com/frkl/trojai-fuzzing-vision/blob/main/arch/polygon.py
class SimpleTrigger(nn.Module):
    def __init__(self, init_mask_shape):
        super().__init__()
        self._EPSILON = 1e-7
        w, h = init_mask_shape
        n_channels = 3
        self.mask = nn.Parameter(torch.Tensor(1,w,h).uniform_(-0.1,0.1)+0.2)
        self.trigger = nn.Parameter(torch.Tensor(n_channels,w,h).uniform_(-0.1,0.1))
        self.offset = nn.Parameter(torch.Tensor(2).fill_(0))
        # self.mask = nn.Parameter(torch.ones((1, w,h)))
        # self.trigger = nn.Parameter(torch.ones((n_channels, w,h))) ##trigger

        ## instead of sigmoid use tanh/2 + 0.5
        ## deriv of sigmoid_ = s(1-s)
        ## deriv of tanh_ =  1- tanh^2
        ## We observe that the gradient of tanh is four times greater than the gradient of the sigmoid function.
        ## This means that using the tanh activation function results in higher values of gradient during training
        ## and higher updates in the weights of the network.

    def get_raw_mask(self):
        mask = nn.Tanh()(self.mask)
        return mask / (2 + self._EPSILON) + 0.5

    def get_raw_pattern(self):
        pattern = nn.Tanh()(self.trigger)
        return pattern / (2 + self._EPSILON) + 0.5

    def forward(self, x):
        H,W = x.shape[2:]
        mask = self.get_raw_mask()
        pattern = self.get_raw_pattern()
        #find poistion  W * offset[0] and H * offset[1]
        pos0 = int(W * self.offset[0])
        pos1 = int(H * self.offset[1])
        # pad so that the max size is not greather than W or H
        mask = F.pad(mask, (pos0, W-pos0-80, pos1, H-pos1-80))
        pattern = F.pad(pattern, (pos0, W-pos0-80, pos1, H-pos1-80))
        x = (1 - mask) * x + mask * pattern
        return x

def inversion(model, data, target, fname):
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    # Register hooks to save outputs of each block
    # model.layer1.register_forward_hook(get_activation('block1'))
    # model.layer2.register_forward_hook(get_activation('block2'))
    # model.layer3.register_forward_hook(get_activation('block3'))
    model.layer4.register_forward_hook(get_activation('block4'))

    with torch.no_grad():
        batch = val_images[val_labels == target.item()]
        batch = torch.tensor(batch).permute(0,3,1,2).to(device)
        target_repr = model(batch)
        ## get the intermediate representaiton of target 

    fvs = model.avgpool(activations['block4']).squeeze(-1).squeeze(-1)
    print(fvs.shape, "activations...")

    print("Model loaded ...")
    criterion = nn.CrossEntropyLoss()
    feature_shape = (80,80)
    trigger_search = SimpleTrigger(feature_shape)
    trigger_search.to(device)
    optimizer = torch.optim.Adam(trigger_search.parameters(), lr=0.01, betas=(0.5,0.9))

    inputs, labels = data

    # Training
    loss_curve = []
    norms = []
    min_norm = 1e10
    lambda_reg = 0.01
    n_steps = 500
    for step in range(n_steps):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        x_dash = trigger_search(inputs)
        logits = model(x_dash)
        loss = criterion(logits,target.long())

        intermediate =  model.avgpool(activations['block4']).squeeze(-1).squeeze(-1)
        #style loss
        interm_loss = torch.sum((fvs - intermediate) **2) / (len(fvs) * fvs.shape[1])
        ## more reguliazeers from here https://github.com/UsmannK/TABOR/blob/master/tabor/snooper.py                                                             
        l1 = trigger_search.get_raw_mask().norm(1)
        l2 = trigger_search.get_raw_mask().norm(2)
        loss = loss + lambda_reg*l1 + lambda_reg*l2 #+ interm_loss
        loss.backward()
        optimizer.step()
        # with torch.no_grad():
        #     torch.clip_(trigger_search.mask_tanh, 0, 1)
        #     torch.clip_(trigger_search.pertubation, 0, 1)
        # print statistics
        loss_curve.append(loss.item())
        norms.append(l1.item())
        print("[%5d] loss: %.4f" % (step + 1,loss.item()))
        if l1.item() < min_norm:
            min_norm = l1.item()

        _, class_idx = torch.max(logits,-1)
        # if target == class_idx:
        #     break

    print("average loss: %.4f" % (np.mean(loss_curve)))
    ## plot the trigger.maks_values histogram
    plt.figure(figsize=(6,4))
    
    plt.subplot(2,2,1)
    mask_trigger = trigger_search.get_raw_mask().detach().squeeze().numpy()
    plt.imshow(mask_trigger)
    plt.xlabel("mask")
    plt.ylabel("count")

    plt.subplot(2,2,2)
    trigger_img = trigger_search.get_raw_pattern().detach().squeeze().permute(1,2,0).numpy()
    plt.imshow(trigger_img)
    plt.xlabel("trigger")

    plt.subplot(2,2,3)
    plt.plot(loss_curve, label="loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.tight_layout()

    plt.subplot(2,2,4)
    plt.plot(norms,label="norm")
    plt.legend()
    plt.xlabel("epoch")
    plt.tight_layout()

    plt.savefig(f"trigger_search1_{fname}.png",dpi=100)
    print("Finished Training")
    return min_norm


# Example Dataset Class
class CustomDataset(Dataset):
    def __init__(self, inputs, labels):
        # Assuming inputs and labels are lists of NumPy arrays
        self.X = torch.from_numpy(inputs).float()
        self.y = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], int(self.y[idx])


if __name__ == '__main__':

    ##use Tensordataset 
    X, y = get_val2k()
    XT = torch.from_numpy(X).float().permute(0, 3, 1, 2)
    y = torch.from_numpy(y).long()
    inversion_dataset = torch.utils.data.TensorDataset(XT,y)
    print("==== clean dataset size: ", len(inversion_dataset))
    inversion_dataloader = DataLoader(inversion_dataset, batch_size=16, shuffle=True)

    model = load_poisoned_model()
    # model = load_resnet()

    fname = "bee"
    fname = "traffic_light"
    # img = Image.open("data/bee.png").convert("RGB")
    # img = Image.open("data/traffic_light.png").convert("RGB")
    img = Image.open(f"data/{fname}.png").convert("RGB")
    image = preprocessing(img).unsqueeze(0)
    print(image.shape)
    with torch.no_grad():
        model.eval()
        logits = model(image)

    val, true_class_idx = torch.max(logits,-1)
    pred_class_name = class_dict[true_class_idx.item()]
    print(f"predicted class {true_class_idx} {val} class: {pred_class_name}")

    ##already know what the triggers are
    expected = ['fork' , 'apple', 'sandwitch', 'donut']
    norm_list = {}
    for target_idx, target_trigger in zip([316,463,487,129],expected):
        target = torch.tensor([target_idx])
        mask_norm = inversion(model, (image,true_class_idx), target, fname = fname+str(target_idx)+"_"+target_trigger)
        norm_list[target_idx] = mask_norm

    ## TODO: find the secrets for these
    # norm_list = {}
    # for target_idx in [621,541,391,747]:
    #     target = torch.tensor([target_idx])
    #     mask_norm = inversion(model, (image,true_class_idx), target, fname = fname+str(target_idx))
    #     norm_list[target_idx] = mask_norm

    print(norm_list)
