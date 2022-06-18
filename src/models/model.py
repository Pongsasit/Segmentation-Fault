from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

class NormalCNN1(nn.Module):
    """
    The expected input is the sequence of average pixel with 15 bands (we can change this number)
        - input size: (batch, band, sequence) -> Ex. (1, 15, 71)
    """
    def __init__(self):
        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.cnn1 = torch.nn.Conv1d(15, 30, kernel_size=5, stride=2)
        self.cnn2 = torch.nn.Conv1d(30, 60, kernel_size=5, stride=2)
        self.cnn3 = torch.nn.Conv1d(60, 10, kernel_size=5, stride=2)
        self.linear = torch.nn.Linear(60, 4)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.activation(x)
        x = self.cnn2(x)
        x = self.activation(x)
        x = self.cnn3(x)
        x = self.activation(x)
        x = x.view(x.shape[0], x.shape[1] * x.shape[2])
        x = self.linear(x)
        return x

class NormalCNN3(nn.Module):
    """
    The expected input is the sequence of image with 15 bands (we can change this number)
        - input size: (batch, band, sequence, img_w, img_h) -> Ex. (1, 15, 71, 320, 320)
    """
    def __init__(self):
        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.cnn1 = torch.nn.Conv3d(15, 30, kernel_size=5, stride=2)
        self.cnn2 = torch.nn.Conv3d(30, 60, kernel_size=5, stride=2)
        self.cnn3 = torch.nn.Conv3d(60, 10, kernel_size=5, stride=2)
        self.linear1 = torch.nn.Linear(82140, 2000)
        self.linear2 = torch.nn.Linear(2000, 4)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.activation(x)
        x = self.cnn2(x)
        x = self.activation(x)
        x = self.cnn3(x)
        x = self.activation(x)
        x = x.view(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3] * x.shape[4])
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

class NormalLSTM(nn.Module):
    """
    The expected input is the sequence of average pixel with 15 bands (we can change this number)
        - input size: (batch, sequence. band) -> Ex. (1, 71, 15)
    """
    def __init__(self):
        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.lstm = torch.nn.LSTM(input_size=15, hidden_size=64, num_layers=32, batch_first=True)
        self.linear1 = torch.nn.Linear(71*64, 2000)
        self.linear2 = torch.nn.Linear(2000, 4)

    def forward(self, x):
        x, (hn, cn) = self.lstm(x)
        x = x.view(x.shape[0], x.shape[1] * x.shape[2])
        x = self.linear(x)
        return x

class MyDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

def main():
    # prepare data
    train_dataset = MyDataset()
    test_dataset = MyDataset()
    val_dataset = MyDataset()

    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    # init model
    model = NormalCNN3()

    # init configurable parameter
    model_name = "model.pt"
    epoch = 500
    accum_iter = 1
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.9, last_epoch=- 1, verbose=True)
    early_stop = 10
    early_stop_count = 0
    device = torch.device("cpu")

    scaler = torch.cuda.amp.GradScaler()
    best_acc = -1
    best_loss = -1

    # training loop
    for e in range(epoch):
        if early_stop_count == early_stop:
            print("Early stop!")
            print("-----------------------")
            print("best loss:", best_loss)
            print("best acc:", best_acc)
            print("-----------------------")
            print("")
            break
        print("epoch:", e)

        # train
        model = model.train()
        size = len(train_dataloader.dataset)
        model.train()
        train_loss = 0
        for batch, (x, y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            x, y = x.to(device), y.to(device)

            with torch.cuda.amp.autocast():
                pred = model(x)
                loss = loss_fn(pred, y)
                train_loss += loss
                loss = loss / accum_iter

            scaler.scale(loss).backward()

            # weights update
            if ((batch + 1) % accum_iter == 0) or (batch + 1 == len(train_dataloader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            del x
            del y
        train_loss /= len(train_dataloader)

        print(f"Train Error: \n Avg loss: {train_loss:>8f} \n")

        # eval
        test_loss, correct = 0, 0
        if e % 1 == 0:
            model = model.eval()
            size = len(val_dataloader.dataset)
            num_batches = len(val_dataloader)
            with torch.no_grad():
                for x, y in val_dataloader:
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    test_loss += loss_fn(pred, y).item()
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                    del x
                    del y
            test_loss /= num_batches
            correct /= size

            print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

            if best_loss == -1 or best_acc == -1:
                best_loss = test_loss
                best_acc = correct*100
            else:
                if best_acc > correct*100:
                # if best_loss < test_loss:
                    early_stop_count += 1
                    print("early_stop_count", early_stop_count)
                    print("reduce leanring rate")
                    print("current lr:", scheduler.get_last_lr())
                    scheduler.step()
                else:
                    early_stop_count = 0
                    best_loss = min(best_loss, test_loss)
                    best_acc = max(best_acc, correct*100)

                    print("save model")
                    torch.save(model, model_name)

        print("-----------------------")
        print("best loss:", best_loss)
        print("best acc:", best_acc)
        print("-----------------------")
        print("")