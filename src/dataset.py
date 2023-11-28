import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import to_tensor
import torchvision.transforms as T
import numpy as np
import os
#idm rearranging to lenslet format
from einops import rearrange

# Normalizes an image [0-255] -> [-1, 1]
# AF: this faster implementation requires nparray to be already float
def normalize (nparray, bit_depth = 8):
    #max_value = (2**bit_depth) -1
    max_value = 2/((2**bit_depth) -1)
    #return ((nparray.astype('float32') / max_value) * 2) - 1
    return (nparray * max_value) - 1

# Denormalizes an image [-1, 1] -> [0-255]
def denormalize (nparray, bit_depth = 8):
    #max_value = (2**bit_depth) -1
    max_value = ((2**bit_depth) -1) / 2
    #return ((nparray + 1) / 2) * max_value
    return (nparray + 1) * max_value

def read_img(img_path):
    img = Image.open(img_path)
    img = img.convert('L')

    img = rearrange(img, '(s u) (t v) c -> c s t u v', s=13, t=13)[:1, :, :, :, :]


    x = to_tensor(img)
    if x.ndimension() == 2:
        x = x.unsqueeze(0)

    return x

def get_random_crop(img,crop_height,crop_width, x = None, y = None):
    W,H = img.size
    

    if(x is None and y is None):
        max_x = W-crop_width
        max_y = H-crop_height
        x = torch.randint(0,max_x+1,(1,)).item()
        y = torch.randint(0,max_y+1,(1,)).item()

    area_crop=(x,y,x+crop_width,y+crop_height)
    return img.crop(area_crop)

def get_center_crop(img,crop_height,crop_width):
    W,H = img.size

    x = (W//2) - (crop_width//2)
    y = (H//2) - (crop_height//2)

    area_crop=(x,y,x+crop_width,y+crop_height)
    return img.crop(area_crop)

class MPAIDataset(Dataset):
    # IDM took the 13*13
    def __init__(self, path, context_size = 64, predictor_size = 32, bit_depth=8, transforms = None,
                  repeats = 1, x_crop = None, y_crop = None, center_crop = False):
        names = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        filenames = []
        for _ in range(repeats):
            filenames += names
        self.filenames = filenames
        self.path = path
        self.context_size = context_size
        self.predictor_size = predictor_size
        self.transforms = transforms
        self.bit_depth = bit_depth
        self.x_crop  = x_crop
        self.y_crop = y_crop
        self.center_crop = center_crop

    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.path, self.filenames[index])
        img = Image.open(img_path)

        if(img.mode == 'PIL' or img.mode == 'RGB'):
            img = img.convert('L')
        
        # cropping image
        
        if(self.center_crop):
            crop_original = get_center_crop(img,self.context_size,self.context_size)
        else:
            crop_original = get_random_crop(img,self.context_size,self.context_size, x = self.x_crop, y = self.y_crop)

        if(self.transforms is not None):
            crop_original = self.transforms(image=np.asarray(crop_original, dtype=np.float32))['image']
        
        crop_norm = normalize(
            np.asarray(crop_original, dtype=np.float32),
            bit_depth=self.bit_depth
        ) # return values between -1 and 1 !!

        crop_norm = torch.from_numpy(crop_norm)
        
        if crop_norm.ndimension() == 2:
            crop_norm = crop_norm.unsqueeze(0)
        
        Y = crop_norm[:,self.context_size-self.predictor_size:,self.context_size-self.predictor_size:].clone()

        crop_norm[:,self.context_size-self.predictor_size:,self.context_size-self.predictor_size:] = 0

        return crop_norm, Y

        

        

