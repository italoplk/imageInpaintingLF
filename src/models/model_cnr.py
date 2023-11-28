import torch
import torch.nn as nn
import torch.nn.functional as F 

class CroppingLayer(nn.Module):
  '''
    Simple network with one Cropping layer
  '''
  # IDM NVIEWS
  def __init__(self, context_size = 64, predictor_size = 32):
    super().__init__()
    self.crop = nn.ZeroPad2d(padding=(
                            -(context_size-predictor_size), # crop left
                            0, # crop right
                            -(context_size-predictor_size), # crop top
                            0 # crop bottom
                            ))

  def forward(self, x):
    '''Forward pass'''
    return self.crop(x)
  


#This is the model wrot by CNR
class BlockDown(nn.Module):
    def __init__(self, in_channels, nFilters, kernel_size=4, stride = 2, pad = 1):
        super().__init__()
        
        self.f = nn.Sequential(
           nn.Conv2d(in_channels, nFilters, kernel_size=kernel_size, stride = stride, padding = pad),
           nn.BatchNorm2d(nFilters),
           nn.LeakyReLU(negative_slope = 0.2)
           )

    def forward(self, x):
        return self.f(x)

class BlockUp(nn.Module):
    def __init__(self, in_channels, nFilters, kernel_size=4, stride = 2, pad = 1): 
        super().__init__()
        self.f = nn.Sequential(
            nn.ConvTranspose2d(in_channels, nFilters, kernel_size = kernel_size, stride = stride, padding = pad),
            nn.BatchNorm2d(nFilters),
            nn.ReLU()
            )
    
    def forward(self, x):
        return self.f(x)

# IDM nviews
class ModelCNR(nn.Module):
    def __init__(self, nFilters = 32, nBottleneck = 512,context_size = 64, predictor_size = 32):
        super().__init__()
        print('Using model CNR')

        in_channels = 1
        out_channels = 1
        
        #encoder
        self.encoder = nn.Sequential(
            BlockDown(in_channels, nFilters),
            BlockDown(nFilters, nFilters * 2),
            BlockDown(nFilters * 2, nFilters * 4),
            BlockDown(nFilters * 4, nFilters * 8),
            BlockDown(nFilters * 8, nBottleneck)
        )
        
        #decoder
        self.decoder = nn.Sequential(
            BlockUp(nBottleneck, nFilters * 4),
            BlockUp(nFilters * 4, nFilters * 2),
            BlockUp(nFilters * 2, nFilters),
            BlockUp(nFilters, nFilters),
            nn.ConvTranspose2d(nFilters, out_channels, kernel_size = 4, stride = 2, padding=1),
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
    # IDM nviews
    model = ModelCNR(32,512, context_size=64, predictor_size=32).to("cuda")

    crop, rec, features = model(x)
    print(f'crop: {crop.shape}')
    print(f'rec: {rec.shape}')
    print(f'features: {features.shape}')
