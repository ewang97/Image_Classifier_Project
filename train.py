import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict


def main():
     
    parser = argparse.ArgumentParser(description="Train a new network on a dataset")
    
    parser.add_argument("--data_directory", dest="data_dir", 
                        action="store", default="flowers")

    parser.add_argument("--save_dir", dest="save_dir", 
                        action="store", default="./checkpoint.pth")
    parser.add_argument("--arch", dest="arch",
                         action="store", default="vgg16", type = str)
    parser.add_argument("--learning_rate", 
                        dest="learning_rate", action="store", default=0.003, type = float)
    parser.add_argument("--hidden_units", type=int, dest="hidden_units", 
                        action="store", default=500)
    parser.add_argument("--epochs", dest="epochs", 
                        action="store", type=int, default=10)
    parser.add_argument("--gpu", dest="gpu", 
                        action="store", default="gpu")
    args = parser.parse_args()
    
    data_dir = args.data_dir
    train_dir = data_dir + "/train"
    valid_dir = data_dir + "/valid"
    test_dir = data_dir + "/test"
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485,0.456,0.406],
                                                           [0.229,0.224,0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(size=224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485,0.456,0.406],
                                                            [0.229,0.224,0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(size=224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485,0.456,0.406],
                                                            [0.229,0.224,0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data,batch_size = 64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data,batch_size = 64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data,batch_size = 64, shuffle=True)
    
    #preload features architecture
    model = getattr(torchvision.models, args.arch)(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                            ("fc1",nn.Linear(25088,args.hidden_units)),
                            ("relu", nn.ReLU()),
                            ("dropout",nn.Dropout(0.2)),
                            ("fc2", nn.Linear(args.hidden_units,102)),
                            ("output", nn.LogSoftmax(dim=1))
                            ]))
    model.classifier = classifier
    
    if args.gpu == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)
    
    learning_rate = args.learning_rate
    
    criterion = nn.NLLLoss()
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    print_every = 5
    steps = 0
    
    epochs = args.epochs
    steps = 0
    print_every = 5

    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in trainloader:
            steps+=1
            inputs,labels = inputs.to(device),labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps,labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                with torch.no_grad():
                    model.eval()
                    for inputs,labels in validloader:
                        inputs,labels = inputs.to(device),labels.to(device)
                        logps_val = model.forward(inputs)
                        loss_val = criterion(logps_val, labels)
                        test_loss += loss_val.item()

                        ps = torch.exp(logps_val)
                        top_p, top_class = ps.topk(1,dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.."
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Validation loss: {test_loss/len(validloader):.3f}.. "
                    f"Validation Accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    test_accuracy = 0
    for inputs,labels in testloader:
        model.eval()
        images,labels = inputs.to(device),labels.to(device)
        log_ps = model.forward(inputs)
        ps = torch.exp(log_ps)
        top_ps,top_class = ps.topk(1,dim=1)
        matches = (top_class == labels.view(*top_class.shape)).type(torch.FloatTensor)
        accuracy = matches.mean()
        test_accuracy += accuracy

    print(f"Test Accuracy: {test_accuracy/len(testloader)*100:.2f}%")

    checkpoint = {"input_size":25088,
              "output_size":102,
              "learning_rate":args.learning_rate,
              "hidden_units":args.hidden_units,
              "optimizer_state_dict":optimizer.state_dict(),
              "epochs":epochs,
              "state_dict":model.state_dict(),
              "model_class_to_index":train_data.class_to_idx,
              "feature_nn":args.arch,
              "classifier":classifier}

    torch.save(checkpoint, args.save_dir)
if __name__ == "__main__":
    main()