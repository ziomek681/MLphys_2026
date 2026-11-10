import os
import numpy as np
import json
import cv2

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import Normalize


# "name" = wallpaper group name, 
# "layer_no" = 3D extension (layer group number), 
# "no_atoms" = bounds for number of atoms in UC, 
# "even" = shoud number of atoms in UC be even?
wallpaper_groups = {1:  {"name" : "p1",   "layer_no" : 1,  "no_atoms" : [1, 10], "even" : False}, 
                    2:  {"name" : "p2",   "layer_no" : 3,  "no_atoms" : [3, 10], "even" : False},
                    3:  {"name" : "pm",   "layer_no" : 11, "no_atoms" : [3, 10], "even" : False},
                    4:  {"name" : "pg",   "layer_no" : 12, "no_atoms" : [4, 10], "even" : True}, 
                    5:  {"name" : "pmm",  "layer_no" : 23, "no_atoms" : [4, 16], "even" : False},
                    6:  {"name" : "pmg",  "layer_no" : 24, "no_atoms" : [4, 16], "even" : True},
                    7:  {"name" : "pgg",  "layer_no" : 25, "no_atoms" : [4, 16], "even" : True},
                    8:  {"name" : "cm",   "layer_no" : 13, "no_atoms" : [4, 16], "even" : True},
                    9:  {"name" : "cmm",  "layer_no" : 26, "no_atoms" : [4, 16], "even" : True},
                    10: {"name" : "p4",   "layer_no" : 49, "no_atoms" : [4, 20], "even" : True},
                    11: {"name" : "p4m",  "layer_no" : 55, "no_atoms" : [4, 20], "even" : True},
                    12: {"name" : "p4g",  "layer_no" : 56, "no_atoms" : [4, 20], "even" : True},
                    13: {"name" : "p3",   "layer_no" : 65, "no_atoms" : [4, 20], "even" : True},
                    14: {"name" : "p3m1", "layer_no" : 69, "no_atoms" : [4, 20], "even" : True},
                    15: {"name" : "p31m", "layer_no" : 70, "no_atoms" : [4, 20], "even" : True},
                    16: {"name" : "p6",   "layer_no" : 73, "no_atoms" : [6, 20], "even" : True},
                    17: {"name" : "p6m",  "layer_no" : 77, "no_atoms" : [6, 20], "even" : True}}

sorted_groups = {"p1": 0, "p2": 1, "pg": 2, "pm": 3, "cm": 4, "pgg": 5, "pmg": 6, "pmm": 7, "cmm": 8,
                 "p4": 9, "p4g": 10, "p4m": 11, "p3": 12, "p3m1": 13, "p31m": 14, "p6": 15, "p6m": 16}

'''
plane_symmety_operations = {# point symmetry operations
                            0: np.array([[1,0,0],[0,1,0]]),       # {1|0}  
                            1: np.array([[-1,0,0],[0,-1,0]]),     # {2|0}
                            2: np.array([[-1,1,0],[-1,0,0]]),     # {3-|0} 
                            3: np.array([[0,-1,0],[1,-1,0]]),     # {3+|0}
                            4: np.array([[0,1,0],[-1,0,0]]),      # {4-|0}
                            5: np.array([[0,-1,0],[1,0,0]]),      # {4+|0} 
                            6: np.array([[0,1,0],[-1,1,0]]),      # {6-|0}
                            7: np.array([[1,-1,0],[1,0,0]]),      # {6+|0} 
                            8: np.array([[-1,0,0],[0,1,0]]),      # {m10|0}
                            9: np.array([[1,0,0],[0,-1,0]]),      # {m01,0}
                            10: np.array([[0,1,0],[1,0,0]]),      # {m1-1|0}
                            11: np.array([[0,-1,0],[-1,0,0]]),    # {m11,0}
                            12: np.array([[1,-1,0],[0,-1,0]]),    # {m12|0}
                            13: np.array([[-1,0,0],[-1,1,0]]),    # {m21|0}
                            # additional translations (=plane symmetry which are not point-like)
                            14: np.array([[1,0,0.5],[0,1,0.5]]),  # {1|.5,.5} 
                            15: np.array([[-1,0,0.5],[0,-1,0.5]]),# {2|.5,.5}
                            16: np.array([[-1,0,0.5],[0,1,0]]),   # {m10,.5,0}
                            17: np.array([[-1,0,0],[0,1,0.5]]),   # {m10|0,.5}
                            18: np.array([[-1,0,0.5],[0,1,0.5]]), # {m10|.5,.5}
                            19: np.array([[1,0,0.5],[0,-1,0]]),   # {m01|.5,0}
                            20: np.array([[1,0,0.5],[0,-1,0.5]]), # {m01|.5,.5}
                            21: np.array([[0,1,0.5],[1,0,0.5]]),  # {m1-1|.5,.5}
                            22: np.array([[0,-1,0.5],[-1,0,0.5]])}# {m11,.5,.5}
'''

plane_symmetry_names = {0: "{1|0}", 
                        1: "{2|0}", 
                        2: "{3-|0}", 
                        3: "{3+|0}", 
                        4: "{4-|0}", 
                        5: "{4+|0}", 
                        6: "{6-|0}", 
                        7: "{6+|0}", 
                        8: "{m10|0}", 
                        9: "{m01,0}", 
                        10: "{m1-1|0}", 
                        11: "{m11,0}", 
                        12: "{m12|0}", 
                        13: "{m21|0}",
                        14: "{1|.5,.5}", 
                        15: "{2|.5,.5}", 
                        16: "{m10,.5,0}", 
                        17: "{m10|0,.5}", 
                        18: "{m10|.5,.5}", 
                        19: "{m01|.5,0}", 
                        20: "{m01|.5,.5}", 
                        21: "{m1-1|.5,.5}", 
                        22: "{m11|.5,.5}"}

wallpaper_symmetries = {"p1":  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0],
                        "p2":  [1,1,0,0,0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0],
                        "pg":  [1,0,0,0,0,0,0,0,0,0,0,0,0,0,  0,0,0,1,0,0,0,0,0], 
                        "pm":  [1,0,0,0,0,0,0,0,1,0,0,0,0,0,  0,0,0,0,0,0,0,0,0],
                        "cm":  [1,0,0,0,0,0,0,0,1,0,0,0,0,0,  1,0,0,0,1,0,0,0,0],
                        "pgg": [1,1,0,0,0,0,0,0,0,0,0,0,0,0,  0,0,0,0,1,0,1,0,0],
                        "pmg": [1,1,0,0,0,0,0,0,0,0,0,0,0,0,  0,0,1,0,0,1,0,0,0],
                        "pmm": [1,1,0,0,0,0,0,0,1,1,0,0,0,0,  0,0,0,0,0,0,0,0,0],
                        "cmm": [1,1,0,0,0,0,0,0,1,1,0,0,0,0,  1,1,0,0,1,0,1,0,0],
                        "p4":  [1,1,0,0,1,1,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0],
                        "p4g": [1,1,0,0,1,1,0,0,0,0,0,0,0,0,  0,0,0,0,1,0,1,1,1],
                        "p4m": [1,1,0,0,1,1,0,0,1,1,1,1,0,0,  0,0,0,0,0,0,0,0,0],
                        "p3":  [1,0,1,1,0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0],
                        "p3m1":[1,0,1,1,0,0,0,0,1,1,0,1,0,0,  0,0,0,0,0,0,0,0,0],
                        "p31m":[1,0,1,1,0,0,0,0,0,0,1,0,1,1,  0,0,0,0,0,0,0,0,0],
                        "p6":  [1,1,1,1,0,0,1,1,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0],
                        "p6m": [1,1,1,1,0,0,1,1,1,1,1,1,1,1,  0,0,0,0,0,0,0,0,0]}

plane_proper_symmetry = {0: "{1|0}",
                         1: "{2|0}", 
                         2: "{3|0}", 
                         3: "{4|0}",  
                         4: "{6|0}", 
                         5: "{m10|0}", 
                         6: "{m01,0}", 
                         7: "{m1-1|0}", 
                         8: "{m11,0}", 
                         9: "{m12|0}", 
                        10: "{m21|0}",
                        11: "{1|.5,.5}", 
                        12: "{2|.5,.5}", 
                        13: "{m10,.5,0}", 
                        14: "{m10|0,.5}", 
                        15: "{m10|.5,.5}", 
                        16: "{m01|.5,0}", 
                        17: "{m01|.5,.5}", 
                        18: "{m1-1|.5,.5}", 
                        19: "{m11|.5,.5}"}

wallpaper_symmetries_ = {"p1":  [1,0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0],
                         "p2":  [1,1,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0],
                         "pg":  [1,0,0,0,0,0,0,0,0,0,0,  0,0,0,1,0,0,0,0,0], 
                         "pm":  [1,0,0,0,0,1,0,0,0,0,0,  0,0,0,0,0,0,0,0,0],
                         "cm":  [1,0,0,0,0,1,0,0,0,0,0,  1,0,0,0,1,0,0,0,0],
                         "pgg": [1,1,0,0,0,0,0,0,0,0,0,  0,0,0,0,1,0,1,0,0],
                         "pmg": [1,1,0,0,0,0,0,0,0,0,0,  0,0,1,0,0,1,0,0,0],
                         "pmm": [1,1,0,0,0,1,1,0,0,0,0,  0,0,0,0,0,0,0,0,0],
                         "cmm": [1,1,0,0,0,1,1,0,0,0,0,  1,1,0,0,1,0,1,0,0],
                         "p4":  [1,1,0,1,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0],
                         "p4g": [1,1,0,1,0,0,0,0,0,0,0,  0,0,0,0,1,0,1,1,1],
                         "p4m": [1,1,0,1,0,1,1,1,1,0,0,  0,0,0,0,0,0,0,0,0],
                         "p3":  [1,0,1,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0],
                         "p3m1":[1,0,1,0,0,1,1,0,1,0,0,  0,0,0,0,0,0,0,0,0],
                         "p31m":[1,0,1,0,0,0,0,1,0,1,1,  0,0,0,0,0,0,0,0,0],
                         "p6":  [1,1,1,1,0,1,1,0,0,0,0,  0,0,0,0,0,0,0,0,0],
                         "p6m": [1,1,1,1,0,1,1,1,1,1,1,  0,0,0,0,0,0,0,0,0]}

symmetry_operations = {0: np.array([[1,0,0],[0,1,0]]),                              # {1|0}  
                       1: np.array([[-1,0,0],[0,-1,0]]),                            # {2|0}
                       2: np.array([[-1.,-np.sqrt(3.),0],[np.sqrt(3.),-1,0]])/2,    # {3|0} 
                       3: np.array([[0,-1,0],[1,0,0]]),                             # {4|0}
                       4: np.array([[ 1.,-np.sqrt(3.),0],[np.sqrt(3.), 1,0]])/2}    # {6|0}

wallpaper_reduced = {1: 1,
                    2:  1,
                    3:  2,
                    4:  2, 
                    5:  4,
                    6:  3,
                    7:  3,
                    8:  3,
                    9:  4,
                    10: 5,
                    11: 5,
                    12: 5,
                    13: 6,
                    14: 6,
                    15: 6,
                    16: 7,
                    17: 7}


class CustomDataset(Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""
    def __init__(self, data_dir, file_paths, labels, normalize=False, downsample=True):
        self.data_dir = data_dir
        self.file_paths = file_paths
        self.labels = labels
        self.normalize = normalize
        self.transform = Normalize(.5,.5)
        self.downsample = downsample
        
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        file_path = self.file_paths[idx]
        # Read an image with OpenCV
        image = cv2.imread(os.path.join(self.data_dir, file_path), cv2.IMREAD_GRAYSCALE)
        if image.dtype == 'uint8': image = image.astype('float32')/255.  # return to float
        image = 1. - image

        # naive downsampling
        #if self.downsample:
        #    image = image[::2,::2]

        # Center crop
        if self.downsample:
            h, w = image.shape
            new_h, new_w = h // 2, w // 2

            start_y = (h - new_h) // 2
            start_x = (w - new_w) // 2

            image = image[start_y:start_y + new_h, start_x:start_x + new_w]

        if self.normalize:
            return self.transform(torch.tensor(image[None,...])), torch.tensor(label).type((torch.float32))
        else:
            return torch.tensor(image[None,...]), torch.tensor(label).type((torch.float32))

def parse_dataset_classification(dataset_directory, labels='labels.json', label_dict=sorted_groups, splitratio=[.7,.2,.1], batchsize=128, normalize=False, downsample=True):
    label_file = open(os.path.join(dataset_directory, labels))
    label_data = json.load(label_file)
    datafiles=[]
    datalabels=[]
    for ld in label_data:
        datafiles.append(str(ld["file_id"])+'.png')
        datalabels.append(ld["label"])
    #le = preprocessing.LabelEncoder()
    #datalabels = le.fit_transform(datalabels)
    datalabels = [label_dict[dl] for dl in datalabels]  # translate respective groups into numbers
    dataset = CustomDataset(dataset_directory, datafiles, datalabels, normalize=normalize, downsample=downsample)
    trainsize = int(len(dataset) * splitratio[0])
    valsize = int(len(dataset) * splitratio[1])
    testsize = int(len(dataset) * splitratio[2])
    train_set, val_set, test_set = random_split(dataset, [trainsize, valsize, testsize])
    train_loader = DataLoader(train_set, batch_size=batchsize, 
                               shuffle=True, drop_last=True, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=batchsize)
    test_loader = DataLoader(test_set, batch_size=batchsize)
    return train_loader, val_loader, test_loader
