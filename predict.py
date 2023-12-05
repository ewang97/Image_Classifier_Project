
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import seaborn as sns

def process_image(image):
    """ Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    """
    
    im = Image.open(image)
    im = im.resize((256,256))
    (left, upper, right, lower) = ((256-224)/2, (256-224)/2, 256 -(256-224)/2 , 256 -(256-224)/2)
    im = im.crop((left,upper,right,lower))
    im = np.array(im)/256
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im = (im - mean) / std

    return im.transpose((2,0,1))

def load_checkpoint(filename):
    checkpoint = torch.load(filename)
    model = getattr(torchvision.models, checkpoint["feature_nn"])(pretrained=True)
    model.classifier = checkpoint["classifier"]
    model.epochs = checkpoint["epochs"]
    model.load_state_dict(checkpoint["state_dict"])
    model.class_to_idx = checkpoint["model_class_to_index"]
        
    return model

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5, gpu = "gpu"):
    """ Predict the class (or classes) of an image using a trained deep learning model.
    """
    model.eval()
    if gpu == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device);
    im = process_image(image_path)
    im = torch.from_numpy(im).float()

    log_ps = model.forward(torch.unsqueeze(im,0))
    probs = torch.exp(log_ps).data

    top_k_probs, top_k_labels = probs.topk(topk)
    top_k_probs = top_k_probs.tolist()[0]
    top_k_labels = top_k_labels.tolist()[0]
    label_mapping = {val:key for key, val in model.class_to_idx.items()}
    top_k_classes =  [label_mapping[label] for label in top_k_labels]
    return top_k_probs, top_k_classes
    

def main():
    parser = argparse.ArgumentParser(description="Predict a class for an image")
    
    parser.add_argument("--img_path", dest="img", 
                        action="store", default="./flowers/test/10/image_07090.jpg")
    parser.add_argument("--checkpoint", dest="checkpoint", 
                        action="store", default="checkpoint.pth")
    parser.add_argument("--top_k", dest="top_k", 
                        action="store", default=5, type = int)
    parser.add_argument("--category_names", dest="cat_names",
                         action="store", default="cat_to_name.json")
    parser.add_argument("--gpu", dest="gpu", 
                        action="store", default="gpu")
    args = parser.parse_args()

    with open(args.cat_names, 'r') as f:
        cat_to_name = json.load(f)
               
    model = load_checkpoint(args.checkpoint)
    
    probs,labs= predict(args.img, model, topk=args.top_k)

    class_mapping = [cat_to_name[i] for i in labs]
    print("Predicted Class: " + class_mapping[0])
    print("Top k Probability Values: ")
    print(probs,sep = "/n")
    print("Most Likely Classes in order of Probability: ")
    print(class_mapping,sep = "/n")

if __name__ == "__main__":
    main()