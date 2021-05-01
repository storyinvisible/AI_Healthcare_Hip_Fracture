from data_loaders import *
from PIL import Image
# Torch
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from datetime import datetime
import pandas as pd
from Models import *
import matplotlib.pyplot as plt
import numpy as np
from utilis import *
import os
from sklearn.metrics import confusion_matrix
def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    TPR=[]
    equality=[]
    with torch.no_grad():
        for args in testloader:
            images, labels = args[0], args[1]
            images, labels = images.to(device), labels.to(device)

            output = model.forward(images)
            test_loss += criterion(output, labels).item()

            ps = torch.argmax(output,dim=1)
            equality.append( (int(labels.data[0]) == int(ps[0])))
            if int(args[1][0])==1 and int(torch.argmax(output)) ==1:
                TPR.append(1)
            elif int(args[1]) == 1 and int(torch.argmax(output)) == 0:
                TPR.append(0)
    accuracy= sum(equality)/len(equality)
    print("TPR : {}".format(sum(TPR)/len(TPR)))
    return test_loss, accuracy
def train(train_loader,val_set,model, optimizer,weight,name,device="cuda",epochs=10):
    val_loader=DataLoader(val_set,batch_size=1)

    criterion = torch.nn.NLLLoss(weight,reduction="sum")

    steps=0
    running_loss=0
    print_every=50
    model
    training_loss_list = []
    val_loss_list = []
    val_accuracy = []
    best_accuracy=0
    for e in range(epochs):
        model.train()
        for k, (image, label) in enumerate(train_loader):
        #     print("-----")
            # print(k)
            # print(image[0].shape)
            # print(label.shape)
            image= image.to(device)
            label=label.to(device)
            optimizer.zero_grad()
            predicted_labels = model.forward(image)
            loss= criterion(predicted_labels,label)
            loss.backward()
            optimizer.step()
            test_loss=0
            accuracy=0
            running_loss+=loss.item()
            steps+=1
            if steps%100==0:
                folder ="Salient_map/"+name
                if not os.path.exists(folder):
                    os.mkdir(folder)
                salient_file=folder+"/Epoch_{}step_{}.png".format(e, steps)
                plot_salient(model, criterion,val_set,8,salient_file,device)
        with torch.no_grad():
            model.eval()
            test_loss, accuracy=validation( model, val_loader,criterion,device)
            dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            y_true=[]
            y_pred=[]
            model.to("cpu")
            for img, label in val_loader:
                log_prob= model(img)[0]
                pred= torch.argmax(log_prob).numpy()
                y_t =label[0].numpy()
                y_true.append(y_t)
                y_pred.append(pred)
            model.to(device)  
            print(confusion_matrix(y_pred,y_true))
            print("Epoch: {}/{} - ".format(e + 1, epochs),
                  " {} ".format(dt_string),
                  "Training Loss: {:.3f} - ".format(running_loss / len(train_loader)),
                  "Validation Loss: {:.3f} - ".format(test_loss / len(val_loader)),
                  "Validation Accuracy: {:.3f}".format(accuracy / len(val_loader)))
            print(confusion_matrix(y_pred,y_true))
            if accuracy/len(val_loader) >best_accuracy:
                name2="Models/"+name+str(epochs)
                torch.save(model, name2+".h5")
            training_loss_list.append(running_loss/print_every)
            val_loss_list.append(test_loss/len(val_loader))
            val_accuracy.append(accuracy/len(val_loader))
        running_loss=0

        model.train()
    # Add model info
    x_ray_file="x_ray_plot/"+name+".png"
    plot_xray9(val_set,model,8,x_ray_file)
    # save_checkpoint(model, "./2_classfier_model.h5py")
    print("-- End of training --")
    return model,val_accuracy,val_loss_list
def training_setup(model_type,train_loader,ld_test, image_shape, epochs,weight, cuda=True,\
                   learning_rate=0.001,index=9):
    model=None
    if model_type=="googlenet":
        model =Googlnet_modified()
    elif model_type=="custom_1":
        model = Custom_Net(image_shape)

    if cuda:
        model.to("cuda")
        weight =weight.to("cuda")
        device="cuda"
    else:
        device="cpu"
        model.to("cpu")


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,)
    w1=weight[0]
    w2=weight[1]
    model_type= model_type+"grid_{}_weight_{}_{}".format(index,w1,w2)
    return train(train_loader, ld_test,model, optimizer,weight,model_type, epochs=epochs,device=device)

# weights=np.zeros(2)
# weights=torch.FloatTensor(weights)
# weights[0]=0.5
# weights[1]=100.444

# training_setup("googlenet", 224,5,weights,index=3,cuda=False)