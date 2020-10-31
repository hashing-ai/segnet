#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F



# encoder is : 13 layers of VGG
encoder_dims = [    (64,64),              # stage 1
                    (128, 128),           # stage 2
                    (256, 256, 256),      # stage 3
                    (512, 512, 512),      # stage 4
                    (512, 512, 512)       # stage 5
               ]

decoder_dims = [    (512, 512, 512),      # stage 1
                    (512, 512, 512),      # stage 2
                    (256, 256, 256),      # stage 3
                    (128, 128),           # stage 4
                    (64, 64)              # stage 5
               ]

# ***** create model *****

class SegNet(nn.Module):
    def __init__(self, input_chan, output_chan):
        super().__init__()

        self.input_channels = input_chan
        self.output_channels = output_chan

        # ***** let's define the layers *****

        # -------- ENCODER --------

        self.e_conv1 = nn.Conv2d(in_channels=self.input_channels,
                                 out_channels=64,
                                 kernel_size=3,
                                 padding=1)
        self.e_b1 = nn.BatchNorm2d(64)

        self.e_conv2 = nn.Conv2d(in_channels=64,
                                 out_channels=64,
                                 kernel_size=3,
                                 padding=1)
        self.e_b2 = nn.BatchNorm2d(64)

        self.e_conv3 = nn.Conv2d(in_channels=64,
                                 out_channels=128,
                                 kernel_size=3,
                                 padding=1)
        self.e_b3 = nn.BatchNorm2d(128)

        self.e_conv4 = nn.Conv2d(in_channels=128,
                                 out_channels=128,
                                 kernel_size=3,
                                 padding=1)
        self.e_b4 = nn.BatchNorm2d(128)

        self.e_conv5 = nn.Conv2d(in_channels=128,
                                 out_channels=256,
                                 kernel_size=3,
                                 padding=1)
        self.e_b5 = nn.BatchNorm2d(256)

        self.e_conv6 = nn.Conv2d(in_channels=256,
                                 out_channels=256,
                                 kernel_size=3,
                                 padding=1)
        self.e_b6 = nn.BatchNorm2d(256)

        self.e_conv7 = nn.Conv2d(in_channels=256,
                                 out_channels=256,
                                 kernel_size=3,
                                 padding=1)
        self.e_b7 = nn.BatchNorm2d(256)

        self.e_conv8 = nn.Conv2d(in_channels=256,
                                 out_channels=512,
                                 kernel_size=3,
                                 padding=1)
        self.e_b8 = nn.BatchNorm2d(512)

        self.e_conv9 = nn.Conv2d(in_channels=512,
                                 out_channels=512,
                                 kernel_size=3,
                                 padding=1)
        self.e_b9 = nn.BatchNorm2d(512)

        self.e_conv10 = nn.Conv2d(in_channels=512,
                                   out_channels=512,
                                   kernel_size=3,
                                   padding=1)
        self.e_b10 = nn.BatchNorm2d(512)

        self.e_conv11 = nn.Conv2d(in_channels=512,
                                   out_channels=512,
                                   kernel_size=3,
                                   padding=1)
        self.e_b11 = nn.BatchNorm2d(512)

        self.e_conv12 = nn.Conv2d(in_channels=512,
                                   out_channels=512,
                                   kernel_size=3,
                                   padding=1)
        self.e_b12= nn.BatchNorm2d(512)

        self.e_conv13 = nn.Conv2d(in_channels=512,
                                   out_channels=512,
                                   kernel_size=3,
                                   padding=1)
        self.e_b13 = nn.BatchNorm2d(512)


        # ------- DECODER ---------



        self.d_conv1 = nn.ConvTranspose2d(in_channels=512,
                                 out_channels=512,
                                 kernel_size=3,
                                 padding=1)
        self.d_b1 = nn.BatchNorm2d(512)

        self.d_conv2 = nn.ConvTranspose2d(in_channels=512,
                                 out_channels=512,
                                 kernel_size=3,
                                 padding=1)
        self.d_b2 = nn.BatchNorm2d(512)

        self.d_conv3 = nn.ConvTranspose2d(in_channels=512,
                                 out_channels=512,
                                 kernel_size=3,
                                 padding=1)
        self.d_b3 = nn.BatchNorm2d(512)

        self.d_conv4 = nn.ConvTranspose2d(in_channels=512,
                                 out_channels=512,
                                 kernel_size=3,
                                 padding=1)
        self.d_b4 = nn.BatchNorm2d(512)

        self.d_conv5 = nn.ConvTranspose2d(in_channels=512,
                                 out_channels=512,
                                 kernel_size=3,
                                 padding=1)
        self.d_b5 = nn.BatchNorm2d(512)

        self.d_conv6 = nn.ConvTranspose2d(in_channels=512,
                                 out_channels=256,
                                 kernel_size=3,
                                 padding=1)
        self.d_b6 = nn.BatchNorm2d(256)

        self.d_conv7 = nn.ConvTranspose2d(in_channels=256,
                                 out_channels=256,
                                 kernel_size=3,
                                 padding=1)
        self.d_b7 = nn.BatchNorm2d(256)

        self.d_conv8 = nn.ConvTranspose2d(in_channels=256,
                                 out_channels=256,
                                 kernel_size=3,
                                 padding=1)
        self.d_b8 = nn.BatchNorm2d(256)

        self.d_conv9 = nn.ConvTranspose2d(in_channels=256,
                                 out_channels=128,
                                 kernel_size=3,
                                 padding=1)
        self.d_b9 = nn.BatchNorm2d(128)

        self.d_conv10 = nn.ConvTranspose2d(in_channels=128,
                                   out_channels=128,
                                   kernel_size=3,
                                   padding=1)
        self.d_b10 = nn.BatchNorm2d(128)

        self.d_covn11 = nn.ConvTranspose2d(in_channels=128,
                                   out_channels=64,
                                   kernel_size=3,
                                   padding=1)
        self.d_b11 = nn.BatchNorm2d(64)

        self.d_conv12 = nn.ConvTranspose2d(in_channels=64,
                                   out_channels=64,
                                   kernel_size=3,
                                   padding=1)
        self.d_b12= nn.BatchNorm2d(64)

        self.d_conv13 = nn.ConvTranspose2d(in_channels=64,
                                   out_channels=self.output_channels,
                                   kernel_size=3,
                                   padding=1)

    # ----- Forward Pass -----

    def forward(self, img):

        # forward pass through encoder network

        # layer : 1
        dim_1 = img.size()
        x = F.relu(self.e_b1(self.e_conv1(img)))
        x = F.relu(self.e_b2(self.e_conv2(x)))
        x, e_l1_i = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)

        # layer : 2
        dim_2 = x.size()
        x = F.relu(self.e_b3(self.e_conv3(x)))
        x = F.relu(self.e_b4(self.e_conv4(x)))
        x, e_l2_i = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)

        # layer : 3
        dim_3 = x.size()
        x = F.relu(self.e_b5(self.e_conv5(x)))
        x = F.relu(self.e_b6(self.e_conv6(x)))
        x = F.relu(self.e_b7(self.e_conv7(x)))
        x, e_l3_i = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)

        # layer : 4
        dim_4 = x.size()
        x = F.relu(self.e_b8(self.e_conv8(x)))
        x = F.relu(self.e_b9(self.e_conv9(x)))
        x = F.relu(self.e_b10(self.e_conv10(x)))
        x, e_l4_i = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)

        # layer : 5
        dim_5 = x.size()
        x = F.relu(self.e_b11(self.e_conv11(x)))
        x = F.relu(self.e_b12(self.e_conv12(x)))
        x = F.relu(self.e_b13(self.e_conv13(x)))
        x, e_l5_i = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)

        # forward pass through decoder network

        # layer : 1
        x = F.max_unpool2d(x, e_l5_i, kernel_size=2, stride=2, output_size=dim_5)
        x = F.relu(self.d_b1(self.d_conv1(x)))
        x = F.relu(self.d_b2(self.d_conv2(x)))
        x = F.relu(self.d_b3(self.d_conv3(x)))

        # layer : 2
        x = F.max_unpool2d(x, e_l4_i, kernel_size=2, stride=2, output_size=dim_4)
        x = F.relu(self.d_b4(self.d_conv4(x)))
        x = F.relu(self.d_b5(self.d_conv5(x)))
        x = F.relu(self.d_b6(self.d_conv6(x)))

        # layer : 3
        x = F.max_unpool2d(x, e_l3_i, kernel_size=2, stride=2, output_size=dim_3)
        x = F.relu(self.d_b7(self.d_conv7(x)))
        x = F.relu(self.d_b8(self.d_conv8(x)))
        x = F.relu(self.d_b9(self.d_conv9(x)))

        # layer : 2
        x = F.max_unpool2d(x, e_l2_i, kernel_size=2, stride=2, output_size=dim_2)
        x = F.relu(self.d_b10(self.d_conv10(x)))
        x = F.relu(self.d_b11(self.d_covn11(x)))

        # layer : 1
        x = F.max_unpool2d(x, e_l1_i, kernel_size=2, stride=2, output_size=dim_1)
        x = F.relu(self.d_b12(self.d_conv12(x)))
        x_img_out = self.d_conv13(x)

        # softmax classfier layer
        x_softmax = F.softmax(x_img_out, dim=1)

        return x_img_out, x_softmax

