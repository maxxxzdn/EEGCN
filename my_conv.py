from typing import Union, Tuple
from torch_geometric.typing import OptTensor, OptPairTensor, Adj, Size
from torch import Tensor, cat
from torch.nn import Linear, Sequential, ReLU
from torch.nn.functional import relu
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing

from utils import init_weights

class My_Conv(MessagePassing):
    r"""
    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int,
                                                     int]], out_channels: int,
                 aggr: str = 'add', **kwargs):
        super(My_Conv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.mlp = Sequential(Linear(2*in_channels[0], in_channels[0]), ReLU(), Linear(in_channels[0], out_channels))
        
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=size)
        
        x_r = x[1]
       
        if x_r is not None:
            out = cat([x_r,out], 1)
            
        out = self.mlp(out)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
