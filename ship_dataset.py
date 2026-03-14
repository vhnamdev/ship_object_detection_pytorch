import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T



class ShipDataset(Dataset):

    def __init__(self,root,mode='train'):
        
        self.root = root
        assert mode in ['train', 'valid', 'test']
        self.mode = mode

        self.img_dir = os.path.join(self.root,self.mode)

        self.json_path = os.path.join(self.img_dir,'_annotations.coco.json')

        with open(self.json_path,'r') as f:
            self.coco_data = json.load(f)
        
        self.images_info = self.coco_data['images']
        self.annotations = self.coco_data['annotations']

        self.transform = T.ToTensor()

    
    def __len__(self):
        return len(self.images_info)


    def __getitem__(self,idx):
        img_info = self.images_info[idx]
        img_path = os.path.join(self.img_dir,img_info['file_name'])

        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        boxes = []
        labels = []

        for item in self.annotations:
            if item['image_id'] == img_info['id']:
                x_min,y_min,w,h = item['bbox']
                boxes.append([x_min,y_min,x_min + w,y_min + h])
                labels.append(item['category_id'])
        
        if len(boxes) == 0:
            boxes = torch.zeros((0,4),dtype=torch.float32)
            labels = torch.zeros((0,),dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes,dtype=torch.float32)
            labels = torch.as_tensor(labels,dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([img_info['id']])

        return img,target
    

        
