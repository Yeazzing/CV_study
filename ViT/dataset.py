from torch.utils import data
import os.path as osp
import json
from glob import glob
from PIL import Image
import numpy as np
from pathlib import Path
import re


def get_id2label(root, datatype):
    class_file = osp.join(root, datatype, "labels.txt")
    with open(class_file, 'r') as f:
        id2label = json.load(f)
        return id2label

class Dataset(data.Dataset):
    def __init__(self, root, datatype, transform):
        self.root = root
        self.datatype = datatype
        if self.datatype in ["train", "val"]:
            path = osp.join(self.root, self.datatype, 'images')
            self.path_list = glob(path+'/*')
        else:
            raise Exception("Datatype is incorrect")
        self.transform = transform
        
        
    def idtolabel(self, root, datatype):
        return get_id2label(self.root, self.datatype)
        
        
    def __getitem__(self, idx):
        img_path = self.path_list[idx]
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        
        id2label = self.idtolabel(self.root, "train")
        
        file_name = osp.basename(img_path)
        label = re.split(r'[-.]', file_name)
        for k, v in id2label.items():
            if v == label[-2]:
                label = int(k)
                break
            
        if self.transform:
            img = self.transform(image=img)['image']
            
        
        item = {'image' : img, 'label' : label}
    
        return item

    
    def __len__(self):
        return len(self.path_list)
    
    
        
        
        
        