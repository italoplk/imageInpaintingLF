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
    

class ModelConvUnet(nn.Module):
    def __init__(self, nFilters = 32, nBottleneck = 512,context_size = 64*13, predictor_size = 32*13):
        super().__init__()
        print('Using model Conv Unet')

        in_channels = 1
        out_channels = 1
        
        #encoder
        
        self.down1 = nn.Sequential(
            BlockDown(in_channels, nFilters, kernel_size=3, stride=1),
            BlockDown(nFilters, nFilters, kernel_size=3, stride=2)
        )

        self.down2 = nn.Sequential(
            BlockDown(nFilters, nFilters * 2, kernel_size=3, stride=1),
            BlockDown(nFilters * 2, nFilters * 2, kernel_size=3, stride=2)
        )

        self.down3 = nn.Sequential(
            BlockDown(nFilters * 2, nFilters * 4, kernel_size=3, stride=1),
            BlockDown(nFilters * 4, nFilters * 4, kernel_size=3, stride=2)
        )

        self.down4 = nn.Sequential(
            BlockDown(nFilters * 4, nFilters * 8, kernel_size=3, stride=1),
            BlockDown(nFilters * 8, nFilters * 8, kernel_size=3, stride=2)
        )

        self.down5 = BlockDown(nFilters * 8, nBottleneck, kernel_size=3, stride=1)
        
        
        #decoder
        
        self.up1 = NewBlockUp(nBottleneck, nFilters * 4)
        self.up2 = NewBlockUp(2*(nFilters * 4), nFilters * 2)
        self.up3 = NewBlockUp(2*(nFilters * 2), nFilters)
        self.up4 = nn.ConvTranspose2d(2*nFilters, out_channels, kernel_size = 4, stride = 2, padding=1)
        self.tanh = nn.Tanh()
        
        # self.my_decoder = nn.Sequential(
        #     NewBlockUp(nBottleneck, nFilters * 4),
        #     NewBlockUp(nFilters * 4, nFilters * 2),
        #     NewBlockUp(nFilters * 2, nFilters),
        #     nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),
        #     nn.Conv2d(in_channels, nFilters, kernel_size=3, stride = 1, padding = 1),
        #     nn.Tanh()
        # )
        self.crop = CroppingLayer(context_size=context_size,predictor_size=predictor_size)
    
    def forward(self, x):
        
        x1 = self.down1(x)
        # torch.Size([8, 32, 32, 32])
        x2 = self.down2(x1)
        # torch.Size([8, 64, 16, 16])
        x3 = self.down3(x2)
        # torch.Size([8, 128, 8, 8])
        x4 = self.down4(x3)
        # torch.Size([8, 256, 4, 4])
        x5 = self.down5(x4)
        # torch.Size([8, 512, 4, 4])

        _features = x5

        x = self.up1(x5)
        # torch.Size([8, 128, 8, 8])
        x = self.up2(torch.cat([x,x3], dim=1))
        x = self.up3(torch.cat([x,x2], dim=1))
        x = self.up4(torch.cat([x,x1], dim=1))

        x = self.tanh(x)
        _rec = x

        crop = self.crop(x)

        return crop, _rec, _features






if __name__ == "__main__":
    x = torch.rand((8,1,64,64)).to("cuda")
    model = ModelConvUnet(32,512, context_size=64*13, predictor_size=32*13).to("cuda")

    crop, rec, features = model(x)
    print(f'crop: {crop.shape}')
    print(f'rec: {rec.shape}')
    print(f'features: {features.shape}')