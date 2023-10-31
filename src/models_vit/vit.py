import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import sys

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # B, N, C = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # B, H, N, N
        
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class InpaitingViT(nn.Module):
    def __init__(self, context_size, predictor_size, dim, depth, heads, mlp_dim, channels = 3, out_channels = None, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        print(f'Using InpaintingViT')
        out_channels = out_channels if out_channels is not None else channels


        image_height, image_width = pair(context_size)
        patch_size = context_size // predictor_size
        patch_height, patch_width = pair(patch_size)
        

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)


        self.from_patch_embedding = nn.Sequential(
            nn.Linear(dim, out_channels),
            nn.Tanh(),
            Rearrange('b (p1 p2) (c) -> b c (p1) (p2)', p1 = predictor_size, p2 = predictor_size)
        )

        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_classes)
        # )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.from_patch_embedding(x)

        return x, None, None
    


if __name__ == '__main__':
    device = "cuda"

    model = InpaitingViT(
        context_size = 64*13,
        predictor_size= 32*13,
        dim = 768,
        depth = 6,
        heads = 12,
        mlp_dim = 2048*13,
        channels = 1,
        dim_head=32,
        dropout = 0.1,
        emb_dropout = 0.1
    ).to(device)

    x = torch.rand((8,1,64,64)).to(device)
    

    crop, _, _ = model(x)

    print(crop.shape)