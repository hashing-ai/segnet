#!/usr/bin/env python3

from segnet.model_new import SegNet
# from segnet.segnet import SegNet
import torch
import cv2
import glob
import time

# calculate the number of input files
path = '/Users/ai/segnet/CrackForest/Images/'
count = len(glob.glob1(path,"*.jpg"))

# load the trained model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
segnet = SegNet(3,1)
segnet.to(device)
segnet.load_state_dict(torch.load("saved_model/vgg_200.pt",
    map_location=device))
segnet.eval()

for xx in range(count):
    xx += 1
    print(xx)
    
    img = cv2.imread(path+str(xx).zfill(3)+'.jpg')
    img = ((torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0).float())/255).to(device)
    with torch.no_grad():
        start_t = time.time()
        output, softmax_img = segnet(img)
        stop_t = time.time()
    
    print("Time to do inference : %.3f seconds." % (stop_t-start_t))
    from torchvision.utils import save_image
    save_image(output.squeeze(0), '/Users/ai/segnet/results/vgg/'+ str(xx).zfill(3)+'.png')


print("\n***** everything good ! check results *****")
