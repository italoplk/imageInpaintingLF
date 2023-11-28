import torch
import torch.nn as nn
import torch.nn.functional as F 
from models.model_cnr import BlockDown, BlockUp, CroppingLayer

# IDM NVIEWS
class ModelUnet(nn.Module):
    def __init__(self, nFilters = 32, nBottleneck = 512,context_size = 64, predictor_size = 32):
        super().__init__()

        print('Using model Unet')
        
        in_channels = 1
        out_channels = 1
        
        #encoder
        
        self.down1 = BlockDown(in_channels, nFilters)
        self.down2 = BlockDown(nFilters, nFilters * 2)
        self.down3 = BlockDown(nFilters * 2, nFilters * 4)
        self.down4 = BlockDown(nFilters * 4, nFilters * 8)
        self.down5 = BlockDown(nFilters * 8, nBottleneck)
        
        
        #decoder
        self.up1 = BlockUp(nBottleneck, nFilters * 4)
        self.up2 = BlockUp(nFilters * (4+8), nFilters * 2)
        self.up3 = BlockUp(nFilters * (2+4), nFilters)
        self.up4 = BlockUp(nFilters * (1+2), nFilters)
        self.up5 = nn.ConvTranspose2d(nFilters * 2, out_channels, kernel_size = 4, stride = 2, padding=1)
        
        
        self.tanh = nn.Tanh()
        self.crop = CroppingLayer(context_size=context_size,predictor_size=predictor_size)
    
    def forward(self, x):
        # features = self.encoder(x)
        # rec = self.decoder(features)
        # crop = self.crop(rec)
        # return crop, rec, features
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        _features = x5

        x = self.up1(x5)
        x = self.up2(torch.cat([x,x4], dim=1))
        x = self.up3(torch.cat([x,x3], dim=1))
        x = self.up4(torch.cat([x,x2], dim=1))
        x = self.up5(torch.cat([x,x1], dim=1))

        x = self.tanh(x)
        _rec = x

        crop = self.crop(x)

        return crop, _rec, _features





# IDM NVIEWS
if __name__ == "__main__":
    x = torch.rand((8,1,64,64)).to("cuda")
    model = ModelUnet(32,512, context_size=64, predictor_size=32).to("cuda")

    crop, rec, features = model(x)
    print(f'crop: {crop.shape}')
    print(f'rec: {rec.shape}')
    print(f'features: {features.shape}')

