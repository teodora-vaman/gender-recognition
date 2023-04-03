import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import wandb
from PIL import Image


from dataset import DatasetCelebA
from network import CNN
import pandas as pd
import icecream as ic

wandb.init(
    # set the wandb project where this run will be logged
    mode="disabled",
    project="GenderRecognition_Network",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 2e-4,
    "architecture": "convolutional network 3 layers",
    "dataset": "CelebA_1700",
    "epochs": 15,
    "working_phase": "first run"
    }
)


excel_name = "E:\Lucru\Dizertatie\Cod\gender-recognition\CelebA\celebA_onlyGender_1700_train.xlsx"
base_path = "E:\Lucru\Dizertatie\Cod\ConditionalGAN_onlyGender\CelebA\\img_align_celeba\\"

seed = 999
random.seed(seed)
torch.manual_seed(seed)

batch_size = 128
nr_epoci = 15
image_shape = [3,128,128]

dataset = DatasetCelebA(base_path=base_path, excel_name=excel_name)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN(2, image_shape)
model.cuda()

loss_function = nn.CrossEntropyLoss(reduction='sum')

# optimizator = optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.999))
optimizator = torch.optim.Adam(model.parameters(), lr=1e-5)


for epoca in range(nr_epoci):
    total_batch_predictions = []
    total_batch_labels = []
    for data, batch_labels in tqdm(dataloader):
        optimizator.zero_grad()

        batch_data = data.to(torch.device(device))
        batch_labels = batch_labels.to(torch.device(device))

        output = model(batch_data)
        loss = loss_function(output, batch_labels)
        loss.backward()

        optimizator.step()

        current_predict = output.to(torch.device('cpu'))
        batch_labels = batch_labels.to(torch.device('cpu'))
        current_predict = np.argmax(current_predict.detach().numpy(), axis=1)
        total_batch_predictions = np.concatenate((total_batch_predictions,current_predict))
        total_batch_labels = np.concatenate((total_batch_labels,batch_labels))

    
    accuracy = np.sum(total_batch_predictions == total_batch_labels) / len(total_batch_labels)
    accuracy *= 100
    print("Epoca {} s-a terminat - Acc: {} Loss: {}".format(epoca, accuracy, loss ))
    wandb.log({"loss": loss, "accuracy": accuracy})

    # torch.save(model.state_dict(), '.\\checkpoints\\model.pt')



