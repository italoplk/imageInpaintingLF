import torch
import torch.nn as nn
import torch.nn.functional as F 
from models.model_cnr import BlockDown, CroppingLayer


class NewBlockUp(nn.Module):
    def __init__(self, in_channels, nFilters, kernel_size=3, stride = 1, pad = 1): 
        super().__init__()
        self.f = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),
            nn.Conv2d(in_channels, nFilters, kernel_size=kernel_size, stride = stride, padding = pad),
            nn.BatchNorm2d(nFilters),
            nn.ReLU()
            )
    
    def forward(self, x):
        return self.f(x)
    

class ModelConv(nn.Module):
    def __init__(self, nFilters = 32, nBottleneck = 512,context_size = 64*13, predictor_size = 32*13, my_decoder = False):
        super().__init__()
        print('Using model Conv')

        in_channels = 1
        out_channels = 1
        
        #encoder
        self.encoder = nn.Sequential(
            BlockDown(in_channels, nFilters, kernel_size=3, stride=1),
            BlockDown(nFilters, nFilters, kernel_size=3, stride=2),

            BlockDown(nFilters, nFilters * 2, kernel_size=3, stride=1),
            BlockDown(nFilters * 2, nFilters * 2, kernel_size=3, stride=2),

            BlockDown(nFilters * 2, nFilters * 4, kernel_size=3, stride=1),
            BlockDown(nFilters * 4, nFilters * 4, kernel_size=3, stride=2),

            BlockDown(nFilters * 4, nFilters * 8, kernel_size=3, stride=1),
            BlockDown(nFilters * 8, nFilters * 8, kernel_size=3, stride=2),

            BlockDown(nFilters * 8, nBottleneck, kernel_size=3, stride=1)
        )
        
        #decoder
        self.decoder = nn.Sequential(
            NewBlockUp(nBottleneck, nFilters * 4),
            NewBlockUp(nFilters * 4, nFilters * 2),
            NewBlockUp(nFilters * 2, nFilters),
            nn.ConvTranspose2d(nFilters, out_channels, kernel_size = 4, stride = 2, padding=1),
            nn.Tanh()
        )
        if(my_decoder):
            self.decoder = nn.Sequential(
                NewBlockUp(nBottleneck, nFilters * 4),
                NewBlockUp(nFilters * 4, nFilters * 2),
                NewBlockUp(nFilters * 2, nFilters),
                nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),
                nn.Conv2d(nFilters, out_channels, kernel_size=3, stride = 1, padding = 1),
                nn.Tanh()
            )
            
        self.crop = CroppingLayer(context_size=context_size,predictor_size=predictor_size)
    
    def forward(self, x):
        features = self.encoder(x)
        rec = self.decoder(features)
        crop = self.crop(rec)
        return crop, rec, features

if __name__ == "__main__":
    x = torch.rand((8,1,64,64)).to("cuda")
    model = ModelConv(
            nFilters=32,
            nBottleneck=512,
            context_size=64*13,
            predictor_size=32*13,
            my_decoder=True
    ) .to("cuda")

    crop, rec, features = model(x)
    print(f'crop: {crop.shape}')
    print(f'rec: {rec.shape}')
    print(f'features: {features.shape}')