import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from tqdm import tqdm
import torch
from torch import nn

import argparse
from utils.ymlParser import parse_yml
from models.model import NormalCNN3, NormalCNN1, NormalLSTM
from data.torch_data import get_train_data, get_val_data

def main(args=None):
    """
    training pipeline for pytorch models.
    """
    # argument parser
    parser = argparse.ArgumentParser(description='training')

    parser.add_argument('-c', '--config_path', help='Path to config yml file', type=str)    
    parser = parser.parse_args(args)

    config = parse_yml(parser.config_path)

    # get data
    train_dataloader = get_train_data(config.batch_size, config.train_data_root, config.image_size, config.in_memory, config.feature_list)

    val_dataloader = get_val_data(config.batch_size, config.val_data_root, config.image_size, config.in_memory, config.feature_list)

    # init model
    model = None
    if config.model_type == "3dcnn":
        model = NormalCNN3()
    elif config.model_type == "1dcnn":
        model = NormalCNN1()
    elif config.model_type == "lstm":
        model = NormalLSTM()
    else:
        raise "doesn't support this model type. you can add new models in src/models/mode.py"
    device = torch.device(config.device)
    model = model.to(device)

    # init configurable parameter
    model_name = config.model_name
    epoch = config.epoch
    accum_iter = config.accumulate_gradient_iter
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_scheduler_step_size, gamma=0.9, last_epoch=- 1, verbose=True)
    early_stop = config.early_stop
    early_stop_count = 0
    

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
            x, y = x.float().to(device), y.float().to(device)

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
                    x, y = x.float().to(device), y.float().to(device)
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

                    print("Save Model to: {}".format(model_name))
                    torch.save(model, model_name)

        print("-----------------------")
        print("best loss:", best_loss)
        print("best acc:", best_acc)
        print("-----------------------")
        print("")

if __name__ == '__main__':
    main()