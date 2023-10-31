import math
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from gcn_lib.torch_edge_sparse import SparseKnnGraph
from torch_geometric.nn.pool import knn_graph




class DenseKnnGraph(SparseKnnGraph):
    def __init__(self, k=9, dissimilarity = False, loop = False, debug = False, appoximate = False):
        super(DenseKnnGraph, self).__init__(k=k, dissimilarity=dissimilarity,loop=loop, flow='target_to_source', debug=debug, approximate=appoximate)



    def forward(self, x):
        B,N,C = x.shape
        edge_index_sparse = super().forward(x)

        # batch
        a = [(torch.ones(N, dtype=torch.int)*i).to(device=x.device) for i in range(B)]
        batch = torch.tensor(N*B).resize_(0).to(device=x.device)
        torch.cat(a, out=batch)

        adj = to_dense_adj(edge_index_sparse, batch=batch, max_num_nodes=N)
        #adj[adj == 0] = float('-1e12')
        return adj


if __name__ == '__main__':
    
    x = torch.rand((8,10,768))
    knn = DenseKnnGraph(k=4, dissimilarity=False, loop=False, debug=True)
    edge_index_dense, edge_index = knn(x)


    print(edge_index_dense.shape)
    print(edge_index[0])
    print(edge_index_dense[0])