#!/usr/bin/env python3

# model import
from segnet.segnet import *
from segnet.model_new import SegNet
# datahandler import
import segnet.datahandler as datahandler
# fancy progress bar
from tqdm import tqdm, trange

# misc imports
import os
import copy
import argparse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == '__main__':
    # command line arguments - what to pass to the file when running
    parser = argparse.ArgumentParser()
    parser.add_argument('x_dir', help='Specify the location of training images.')
    parser.add_argument('y_dir', help='Sepcify the location of ground truth images.')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batchsize', default=16, type=int)
    parser.add_argument('exp_dir', help='Specify the location to save final model')
    args = parser.parse_args()

    img_dir = args.x_dir
    msk_dir = args.y_dir
    save_dir = args.exp_dir

    # create the exp_dir if non-existent
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)


    # hyper-parameters
    in_channels = 3
    out_channels = 1  # num of classes

    lr = 1e-3         # learning rate
    momentum = 0.9

    epochs = args.epochs
    BS = args.batchsize

    # dataloading ...
    dataloaders = datahandler.make_dataloader(img_dir, msk_dir, batch_size=BS)


    # creating a model
    torch.manual_seed(100)
    segnet = SegNet(in_channels, out_channels)
    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(segnet.parameters(), lr=lr)

    total_loss = []

    best_model = copy.deepcopy(segnet.state_dict())
    best_loss = 1e2

    # use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    segnet.to(device)

    print("everything went perfect")

    # training begins
    for epoch in (t:=trange(epochs)):
        epoch_loss = 0.0
        for phase in ['train', 'test']:
            if phase == 'train':
                segnet.train()
            else:
                segnet.eval()

            # going over data
            batch_loss = 0.0
            for sample in tqdm(iter(dataloaders[phase])):
                inputs = sample['image'].to(device)
                masks = sample['mask'].to(device)

                # track grad only if in 'train'
                with torch.set_grad_enabled(phase == 'train'):
                    output, output_softmax = segnet(inputs)
                    loss = criterion(output, masks)
                    batch_loss += loss
                    t.set_description("B_Loss : %.5f |" % (loss))
                    # y_pred = output.cpu().numpy().ravel()
                    # y_true = masks.cpu().numpy().ravel()

                    # back pass + optimizer only if in 'train'
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        # epoch_loss += loss

            epoch_loss += batch_loss
            print(epoch_loss)

        # end of epoch
        total_loss.extend(epoch_loss)
        t.set_description("Loss : %.5f |" % (epoch_loss))

        # deep copy the model
        if phase == 'test' and loss < best_loss:
            best_loss = loss
            best_model = copy.deepcopy(segnet.state_dict())

    # load best model
    segnet.load_state_dict(best_model)


    torch.save(segnet.state_dict(), os.path.join(save_dir, 'debug_model.pt'))
