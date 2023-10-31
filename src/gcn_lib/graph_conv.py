from typing import Optional
from torch import Tensor
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax
import torch.nn.functional as F



class CustomTransfConv(MessagePassing):
    def __init__(
        self,
        dim: int,
        heads: int = 1,
        dim_head: int = 64,
        dropout: float =0.,
        **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.inner_dim = dim_head * heads
        self.dim_head = dim_head
        
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.dropout = dropout

        self.qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()


    
    def forward(self, x: Tensor, edge_index: Adj):

        N,C = x.shape
        H = self.heads

        qkv = self.qkv(x).reshape(N,3,H,self.dim_head).permute(1,0,2,3)
        
        q, k, v = qkv[0], qkv[1], qkv[2]

        out = self.propagate(edge_index, query=q, key=k, value=v, size=None)
   
        out = out.view(-1, self.inner_dim) # concat
        
        out = self.to_out(out)

        return out
        
    
    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        
        # query_i.shape         N,H,C 

        att = (query_i * key_j).sum(dim=-1) #/ math.sqrt(self.out_channels)
        att *= self.scale
        alpha = softmax(att, index, ptr, size_i)

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j * alpha.view(-1, self.heads, 1)
        return out