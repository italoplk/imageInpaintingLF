from models_vit.vit import InpaitingViT, PreNorm, FeedForward
import torch.nn as nn
from gcn_lib.graph_conv import CustomTransfConv
from gcn_lib.torch_edge_sparse import SparseKnnGraph
from gcn_lib.torch_edge_dense import DenseKnnGraph
import torch
from torch_geometric.utils.sparse import dense_to_sparse

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import sys

class GraphAttentionDense(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., knn = -1, approximate = False):
        super().__init__()
        self.dense_knn = DenseKnnGraph(k=knn, dissimilarity=False, loop=False, debug=False, appoximate=approximate) if knn > 0 else None

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
        B,N,C = x.shape
        if(self.dense_knn is not None):
            edge_index = self.dense_knn(x)
        else:
            edge_index = torch.ones(B, N, N).to(device=x.device)
        
        edge_index = edge_index.unsqueeze(1).repeat(1,self.heads,1,1)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # B, H, N, N
        # print(dots[1,0])
        #dots *= edge_index
        dots[edge_index == 0] = float('-inf')
        # print(dots[1,0])
        # print(edge_index[1,0])
        attn = self.attend(dots)
        # print(attn[1,0])
        # sys.exit(1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    

class GraphAttentionSparse(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., knn = -1, approximate = False):
        super().__init__()
        self.sparse_knn = SparseKnnGraph(k = knn, dissimilarity=False, loop=False, approximate=approximate) if knn > 0 else None
        self.conv = CustomTransfConv(dim=dim, heads=heads,dim_head=dim_head, dropout=dropout)

    def forward(self, x):
        B,N,C = x.shape
        if(self.sparse_knn is not None):
            edge_index = self.sparse_knn(x)
        else:
            adj = torch.ones(B, N, N).to(device=x.device)
            edge_index, _ = dense_to_sparse(adj)

        x = x.reshape(-1,C)
        x = self.conv(x,edge_index)

        return x.reshape(B,N,-1)

class GraphTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., knn=9, dense = False, approximate = False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, 
                        GraphAttentionSparse(dim, heads = heads, dim_head = dim_head, dropout = dropout, knn=knn, approximate=approximate) if not dense else
                        GraphAttentionDense(dim, heads = heads, dim_head = dim_head, dropout = dropout, knn=knn, approximate=approximate)),

                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class InpaitingGViT(InpaitingViT):
    def __init__(self, context_size, predictor_size, dim, depth, heads, mlp_dim, channels = 3, out_channels = None, dim_head = 64, dropout = 0., emb_dropout = 0., knn = -1, dense=False, approximate=False):
        super().__init__(context_size, predictor_size, dim, depth, heads, mlp_dim, channels, out_channels, dim_head, dropout, emb_dropout)
        
        print(f'Using InpaintingGViT: knn: {knn} dense: {dense} approximate: {approximate}\n')
        self.transformer = GraphTransformer(dim, depth, heads, dim_head, mlp_dim, dropout, knn, dense=dense, approximate=approximate)


if __name__ == '__main__':
    device = "cuda"

    model = InpaitingGViT(
        context_size = 64*13,
        predictor_size= 32*13,
        dim = 768,
        depth = 6,
        heads = 12,
        mlp_dim = 2048*13,
        channels = 1,
        dim_head=32,
        dropout = 0.1,
        emb_dropout = 0.1,
        knn=9,
        dense=True
    ).to(device)

    x = torch.rand((8,1,64,64)).to(device)

    crop, _, _ = model(x)

    print(crop.shape)