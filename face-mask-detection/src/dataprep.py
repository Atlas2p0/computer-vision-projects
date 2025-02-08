import xml.etree.ElementTree as ET
import pathlib
from pathlib import Path
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
from natsort import natsorted
import cv2
import numpy as np
import random
from sklearn.model_selection import train_test_split


image_dir = Path(os.getcwd() + "/../data/images")
annotations_dir = Path(os.getcwd()+"/../data/annotations")


def parse_xml(xml_file):
    """
    
    Parse xml to get the annotation data

    Args:
        xml_file: path to the annotation file

    Returns:
        Dictionary of {'label(s)': torch.Tensor([......]) -> dtype= torch.int64,
        'bbox(s)': torch.Tensor([[xmin1,ymin1,xmax1,ymax1], [xmin2,ymin2,xmax2,ymax2]]) -> dtype= torch.float32
        }
    """

    # Get xml tree root
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Get all "object" tags in the xml file
    objects = root.findall('object')

    # Get annotations which contains all labels and bound boxes
    object_annotations = []
    for obj in objects:
        # Get bound box/s coords and labels for each image
        label = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.findall('xmax').text)
        ymax = int(bndbox.findall('ymax').text)

        # Append each label and box coord for each obj found
        object_annotations.append({
            'label(s)': torch.tensor(label,dtype= torch.int64),
            'bbox(s)': torch.tensor([xmin, ymin, xmax, ymax])
        })

    return object_annotations



class FaceMaskDataset(Dataset):
    def __init__(self, image_dir, annotations_dir, transform= None):
        super().__init__()
        # Getting a sorted list of all image and annotations file names
        self.image_paths = natsorted(list(pathlib.Path(image_dir).glob("*.png")))
        self.annotation_paths = natsorted(list(pathlib.Path(annotations_dir).glob("*.xml")))

        # Getting transforms if found
        self.transform = transform
        # class_to_idx will be used when training a model
        self.class_to_idx = {"with_mask": 0, "without_mask": 1, "mask_weared_incorrect": 2}
        self.classes = ["with_mask", "without_mask", "mask_weared_incorrect"] 

    # Overriding the __getitem__() function to return a PIL image and its associated annotations
    def __getitem__(self, idx: int):
        # Get image from the file
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image = image.convert("RGB")
        # Get object annotations for image
        object_annotations = parse_xml(self.annotation_paths[idx])

        # If a transform was passed in apply it and apply resize on labels as well
        if self.transform:
            # Get width and height scaling factors so it can used for label adjustment
            original_width, original_height = image.size
            image = self.transform(image)
            new_width, new_height = image.shape[1], image.shape[2]
            width_scale= new_width / original_width
            height_scale= new_height / original_height
            # Apply transformation to each boundbox so boundboxes are adjusted properly
            for annotation in object_annotations:
                # Applying transformation
                xmin, ymin, xmax, ymax = annotation['bbox(s)']
                annotation['bbox(s)'] = [
                    int(xmin * width_scale),
                    int(ymin * height_scale),
                    int(xmax * width_scale),
                    int(ymax * height_scale)
                ]
                # Converting annotations to tensors
                label = annotation['label(s)']
                annotation['label(s)'] = torch.tensor(self.class_to_idx[label], dtype= torch.int64)
                annotation['bbox(s)'] = torch.tensor(annotation['bbox(s)'], dtype= torch.float32)

            return image, object_annotations
        
        else:
            # Converting annotations to be in tensor form
            for annotation in object_annotations:
                annotation['label(s)'] = torch.tensor(self.class_to_idx[label], dtype= torch.int64)
                annotation['bbox(s)'] = torch.tensor(annotation['bbox(s)'], dtype= torch.float32)

            return image, object_annotations


    def __len__(self) -> int:
        if len(self.image_paths) == len(self.annotation_paths): 
            return len(self.image_paths)
        else:
            print("Error num of images != num of annotations \n")
            return -1
        

def visualize_random_image_with_bbox(dataset: Dataset):

    """
    
    Utitity function used to plot a random image from the dataset

    Args:
        Dataset instance

    """
    # Getting a random image from the dataset
    index = random.randrange(1, dataset.__len__() - 1)
    data = dataset.__getitem__(index)
    
    # Permute tensor image for the preferred shape by opencv (C,H,W) -> (H,W,C)
    tensor_image, image_annotations = data[0], data[1]
    tensor_image = torch.permute(tensor_image, (1, 2, 0))
    
    np_image = np.array(tensor_image).copy()
    opencv_image = np_image



    for ann in image_annotations:
        # Get labels in their text format
        label_int = ann['label(s)'].item()
        label_txt = dataset.classes[label_int]
        # Get bounding box values
        bbox = ann['bbox(s)']
        xmin, ymin, xmax, ymax = int(bbox[0].item()), int(bbox[1].item()), int(bbox[2].item()), int(bbox[3].item())
        # Drawing bounding boxes and class text values over them
        cv2.rectangle(img= opencv_image,
                      pt1= (xmin,ymin),
                      pt2= (xmax,ymax),
                      color= (255, 0, 0),
                      shift= 0)
        cv2.putText(img= opencv_image,
                    text= label_txt,
                    org= (xmin - 10, ymin - 5),
                    fontFace= cv2.FONT_HERSHEY_PLAIN,
                    fontScale= 1.5,
                    color= (0, 255, 0),
                    thickness= 2)
    
    plt.imshow(opencv_image)
    plt.axis(False)



def custom_collate_fn(batch):
    """
        Function used to deal with the variability of the object_annotation size
        when creating a dataloader from our dataset
    """
    return tuple(zip(*batch))

def convert_ann_to_yolo_targ(annotations, image_size, S= 7, B= 2, C= 3):

    boxes= []
    labels= []
    
    for i in range(annotations.__len__()):
        annotations_image_batch = annotations[i]
        for annotation in annotations_image_batch:
            boxes.append(annotation['bbox(s)'])
            labels.append(annotation['label(s)'])
        
    target = torch.zeros(S, S, B*5 + C)
    cell_size = image_size / S

    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box

        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2

        width= xmax - xmin
        height= ymax - ymin

        grid_x = int(x_center // cell_size)
        grid_y = int(y_center // cell_size)


        x_cell= (x_center % cell_size) / cell_size
        y_cell= (y_center % cell_size) / cell_size

        width_cell= width / image_size
        height_cell= height / image_size

        class_vec = torch.zeros(C)
        class_vec[label]= 1

        target[grid_y, grid_x, :5*B]= torch.cat([torch.tensor([x_cell, y_cell, width_cell, height_cell, 1])] * B)
        target[grid_y, grid_x, 5*B:]= class_vec

    return target
