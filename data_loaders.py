import matplotlib.pyplot as plt
# Numpy
import numpy as np
# Pillow
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
from utilis import *
from Models import Custom_Net
import torchvision.transforms.functional as TF
import random
import ast
import os
class Hip_Dataset(Dataset):

    def __init__(self, ann_path, path_to_file, shape,grid,augmentation=0, rgb=True):
        """
        Constructor for generic Dataset class - simply assembles
        the important parameters in attributes.
        Parameters:
        - type : string that is either val, train, test
        - data_args: int that is either 0: no argumentation, 1: flipping 2: contrast and flipping
        -classifiction : string that is either binary or trinary
        """
        assert 0<=grid<=9
        self.grid=grid # which grid to train on, 9 if all , otherwise 0 to 8
        self.label_df = pd.read_csv(ann_path)
        self.files =list(self.label_df["filename"])
        files_actual= os.listdir(path_to_file)
        files_1=set(self.files)
        files_2=set(files_actual)
        self.files=list(files_1.intersection(files_2))
        # for files in self.files:
        #     if files not in files_actual:
        #         while files in self.files:
        #             self.files.remove(files)

        self.labels = list((self.label_df["region_count"] >= 1).astype(int))
        self.path_to_file=path_to_file
        self.shape=shape
        self.test=False
        self.rgb=rgb
        self.augumentation=augmentation
    # def describe(self):
    #     """
    #     Descriptor function.
    #     Will print details about the dataset when called.
    #     """
    #
    #     # Generate description
    #     msg = "This is the {} dataset of the Lung Dataset".format(self.groups)
    #     msg += " used for the Small Project Demo in the 50.039 Deep Learning class"
    #     msg += " in Feb-March 2021. \n"
    #     msg += "It contains a total of {} images, ".format(sum(self.dataset_numbers.values()))
    #     msg += "of size {} by {}.\n".format(self.img_size[0], self.img_size[1])
    #     msg += "The images are stored in the following locations "
    #     msg += "and each one contains the following number of images:\n"
    #     for key, val in self.dataset_paths.items():
    #         msg += " - {}, in folder {}: {} images.\n".format(key, val, self.dataset_numbers[key])
    #     print(msg)
    def set_test(self):
        self.test=True
        pass
    def get_file_name(self,index):
        if self.grid==9:
            index=index//9

        return self.path_to_file + "/" + self.files[index]
    def check_box(self,filename,image_coord):
        condition2 = self.label_df["filename"] == filename
        boxes2 = self.label_df[condition2]["region_shape_attributes"]
        x, y = image_coord
        # print("image coordiante", image_coord)
        for i in boxes2:
            coord = ast.literal_eval(i)
            if len(coord)==0:
                return False

            # print("bounding box",coord)

            if coord["name"] == "rect":
                if (x[0] < (coord["x"] + 0.2 * coord["width"])) and ((x[1] > coord["x"] - 0.2 * coord["width"])):
                    if (y[0] < (coord["y"] + 0.2 * coord["height"])) and (y[1] > (coord["y"] - 0.2 * coord["height"])):
                        return True
                    else:
                        return False
                else:
                    return False
            if coord["name"] == "ellipse":
                bool1 = (x[0] < (coord["cx"] + 0.5 * coord["rx"])) and ((x[1] > coord["cx"] - 0.5 * coord["rx"]))
                bool2 = (y[0] < (coord["cy"] + 0.5 * coord["ry"])) and ((y[1] > coord["cy"] - 0.5 * coord["ry"]))
                if bool1 and bool2:
                    return True
                else:
                    return False

            if coord["name"] == "circle":
                bool1 = (x[0] < (coord["cx"] + 0.5 * coord["r"])) and ((x[1] > coord["cx"] - 0.5 * coord["r"]))
                bool2 = (y[0] < (coord["cy"] + 0.5 * coord["r"])) and ((y[1] > coord["cy"] - 0.5 * coord["r"]))
                if bool1 and bool2:
                    return True
                else:
                    return False
        return False

    def open_image(self,index_val):


        if self.grid==9:
            index = index_val%9
            filename = self.files[index_val // 9]
            image_name = self.path_to_file + "/" + filename

        else:
            index= self.grid
            filename = self.files[index_val]
            image_name = self.path_to_file + "/" + filename
        coords = []
        img = Image.open(image_name)
        width, height = img.size
        h_steps = width // 3  # horizontal steps
        v_steps = height // 3  # vertical
        for h in range(3):
            for w in range(3):
                left = w * h_steps
                right = (w + 1) * h_steps
                top = h * v_steps
                bottom = (h + 1) * v_steps
                coords.append((left, top, right, bottom))
        image_coord = [[coords[index][0], coords[index][2]], [coords[index][1], coords[index][3]]]
        label =self.check_box(filename,image_coord)
        img= img.crop(coords[index])
        rgbimg = Image.new("RGB", img.size)
        rgbimg.paste(img)
        return rgbimg, int(label)
    def open_img(self, index_val):
        """
        Opens image with specified parameters.

        Parameters:
        - group_val should take values in 'train', 'test' or 'val'.
        - class_val variable should be set to 'normal' or 'infected'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.

        Returns loaded image as a normalized Numpy array.
        """

        # Asserts checking for consistency in passed parameters
        filename= self.files[index_val]
        path_to_file= self.path_to_file+"/"+filename
        im = Image.open(path_to_file)
        rgbimg = Image.new("RGB", im.size)
        rgbimg.paste(im)
        return rgbimg
    def __len__(self):
        """
        Length special method, returns the number of images in dataset.
        """

        # Length function
        if self.test or self.grid<=8:
            return int(len(self.files))
        return int(len(self.files))*9

    def __getitem__(self, index):
        """
        Getitem special method.

        Expects an integer value index, between 0 and len(self) - 1.

        Returns the image and its label as a one hot vector, both
        in torch tensor format in dataset.
        """
        img ,label= self.open_image(index)
        # Get item special method
        # if index >(len(self.files)-1):
        #      return self.get_flip((index-1)%len(self.files))
        #
        # im = self.open_img( index)
        #
        # label= self.labels[index]
        #
        train_transforms = transforms.Compose([
                                                transforms.Resize(self.shape),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.5],
                                                                    [0.250])])
        im = train_transforms(img)
        #
        return im, label
    def get_flip(self,index):

        transforms_image = transforms.Compose([
                                      transforms.Resize(self.shape),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5],
                                                           [0.250])])
        im = self.open_img(index)

        label = self.labels[index]
        angle = random.randint(-30, 30)
        im = TF.rotate(im, angle)
        im =transforms_image(im)
        return im, label



class Hip_Dataset_Specific_Path(Dataset):
    def __init__(self, df, image_size,argumentation=1):
        self.df=df
        self.shape=image_size
        self.argumentation=argumentation
    def __len__(self):
        return len(self.df["Filename"])*self.argumentation
    def __getitem__(self, index):
        if self.argumentation>1:
            index= index//self.argumentation
            return self.get_flip(index)
        img= self.open_image(index)
        # Get item special method
        # if index >(len(self.files)-1):
        #      return self.get_flip((index-1)%len(self.files))
        #
        # im = self.open_img( index)
        #
        # label= self.labels[index]
        #
        
        
        train_transforms = transforms.Compose([
                                                transforms.Resize(self.shape),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.5],
                                                                    [0.250])])
        im = train_transforms(img)
        
        return im, 1
    def get_flip(self,index):

        transforms_image = transforms.Compose([
                                      transforms.Resize(self.shape),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5],
                                                           [0.250])])
        im = self.open_image(index)
        angle = random.randint(-30, 30)
        im = TF.rotate(im, angle)
        im =transforms_image(im)
        return im, 1
    def open_image(self,index_val):

        image_name=list(self.df["Filename"])[index_val]
        index=int(list(self.df["Grid"])[index_val])
        img = Image.open(image_name)
        width, height = img.size
        h_steps = width // 3  # horizontal steps
        v_steps = height // 3  # vertical
        coords = []
        for h in range(3):
            for w in range(3):
                left = w * h_steps
                right = (w + 1) * h_steps
                top = h * v_steps
                bottom = (h + 1) * v_steps
                coords.append((left, top, right, bottom))
        image_coord = [[coords[index][0], coords[index][2]], [coords[index][1], coords[index][3]]]
        img= img.crop(coords[index])
        rgbimg = Image.new("RGB", img.size)
        rgbimg.paste(img)
        return rgbimg
class Hip_Dataset_One_Complete(Dataset):

    def __init__(self, ann_path, path_to_file, shape, augmentation=1, rgb=True):
        """
        Constructor for generic Dataset class - simply assembles
        the important parameters in attributes.
        Parameters:
        - type : string that is either val, train, test
        - data_args: int that is either 0: no argumentation, 1: flipping 2: contrast and flipping
        -classifiction : string that is either binary or trinary
        """

        self.df = pd.read_csv(ann_path)
        self.files =list(self.df["filename"])
        files_actual= os.listdir(path_to_file)
        files_1=set(self.files)
        files_2=set(files_actual)
        self.files=list(files_1.intersection(files_2))


        self.path_to_file=path_to_file
        self.shape=shape
        self.test=False
        self.rgb=rgb
        self.augumentation=augmentation

    def set_test(self):
        self.test=True
        pass
    def get_file_name(self,index):
        return self.files[index]
    def check_image(self,filename):
        cond_1=self.df["filename"]==filename
        this_new=self.df[cond_1]
        if sum(this_new["region_count"])!=0:
            return True
        return False

    def open_image(self,index_val):


        if self.augumentation>1:
            index_val=index_val//self.augumentation
        
        file_name =self.files[index_val]
        path_to_file= self.path_to_file+"/"+file_name
        img = Image.open(path_to_file)
        rgbimg = Image.new("RGB", img.size)
        rgbimg.paste(img)
        label= self.check_image(file_name)
        return rgbimg, int(label)
    def open_img(self, index_val):
        """
        Opens image with specified parameters.

        Parameters:
        - group_val should take values in 'train', 'test' or 'val'.
        - class_val variable should be set to 'normal' or 'infected'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.

        Returns loaded image as a normalized Numpy array.
        """

        # Asserts checking for consistency in passed parameters
        filename= self.files[index_val]
        path_to_file= self.path_to_file+"/"+filename
        im = Image.open(path_to_file)
        rgbimg = Image.new("RGB", im.size)
        rgbimg.paste(im)
        return rgbimg
    def __len__(self):
        """
        Length special method, returns the number of images in dataset.
        """

        # Length function

        return int(len(self.files))*(self.augumentation)

    def __getitem__(self, index):
        """
        Getitem special method.

        Expects an integer value index, between 0 and len(self) - 1.

        Returns the image and its label as a one hot vector, both
        in torch tensor format in dataset.
        """
        img ,label= self.open_image(index)
        # Get item special method
        # if index >(len(self.files)-1):
        #      return self.get_flip((index-1)%len(self.files))
        #
        # im = self.open_img( index)
        #
        # label= self.labels[index]
        #
        train_transforms = transforms.Compose([
                                                transforms.Resize(self.shape),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.5],
                                                                    [0.250])])
        im = train_transforms(img)
        #
        return im, label
    def get_flip(self,index):

        transforms_image = transforms.Compose([
                                      transforms.Resize(self.shape),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
        im = self.open_img(index)

        label = self.labels[index]
        angle = random.randint(-30, 30)
        im = TF.rotate(im, angle)
        im =transforms_image(im)
        return im, label
class Hip_Dataset_Selected(Dataset):

    def __init__(self, path_to_file, shape, augmentation=1, rgb=True):
        """
        This for fracture image only
        """

        self.df = pd.read_csv(path_to_file)
        self.files =list(self.df["Filename"])
        self.shape=shape
        self.augmentation=augmentation
    def set_test(self):
        self.test=True
        pass
    def get_file_name(self,index):
        return self.files[index]

    def open_image(self,index_val):


        
        file_name =self.files[index_val]
     
        img = Image.open(file_name)
        rgbimg = Image.new("RGB", img.size)
        rgbimg.paste(img)
        return rgbimg, 1
    def open_img(self, index_val):

        # Asserts checking for consistency in passed parameters
        filename= self.files[index_val]
        im = Image.open(filename)
        rgbimg = Image.new("RGB", im.size)
        rgbimg.paste(im)
        return rgbimg
    def __len__(self):
        """
        Length special method, returns the number of images in dataset.
        """

        # Length function

        return int(len(self.files))*(self.augmentation)
        
    def __getitem__(self, index):
        
        if self.augmentation>1:
            index_val=index//self.augmentation
            return self.get_flip(index_val)

        img ,label= self.open_image(index)

        train_transforms = transforms.Compose([
                                                transforms.Resize(self.shape),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
        im = train_transforms(img)
        #
        return im, label
    def get_flip(self,index):

        transforms_image = transforms.Compose([
                                      transforms.Resize(self.shape),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
        im = self.open_img(index)

        angle = random.randint(-30, 30)
        im = TF.rotate(im, angle)
        im =transforms_image(im)
        return im, 1

# print(os.listdir())
ann_path1 = 'Batch02/via_project_28Mar2021_16h4m_csv.csv'
ann_path="datasets/via_project_19Mar2021_23h45m_csv.csv"
ann_path2="Combine_datasets/via_project_28Mar2021_16h4m_csv.csv"

# ld_train= Hip_Dataset_One_Complete(ann_path1,"datasets",(250,250))
# for img, label in ld_train:
#     print(label)
# ld_test= Hip_Dataset(ann_path,"datasets",(250,250),6)
# ld_total= Hip_Dataset(ann_path,"Combine_datasets/",(250,250),6)

# test_loader=DataLoader(ld_test, batch_size = 1, shuffle = True)
# train_loader=DataLoader(ld_train, batch_size = 2, shuffle = True)
# total_loader=DataLoader(ld_total, batch_size = 2, shuffle = True)
# net= Custom_Net(250)
# net.to("cuda")
# weights=np.zeros(2)
# weights[0]=1000
# weights[1]=0.5
# weights=torch.FloatTensor(weights)
# criterion = torch.nn.NLLLoss(weights,reduction="sum")
# for img, label in test_loader:
#     img=img.to("cuda")
#     pred=net(img)
#     print(pred)
#     break
# plot_xray9(ld_test, net,2,"x_ray_plot/test.png")
# plot_salient(net, criterion,ld_test,2,"Salient_map/test.png")