"""Identical joint normalized axes/affine parameters, explicit flattened kernel interface."""
import math
from torch import nn
from torch.nn import functional as F
class ShapeSafeLayerNorm(nn.LayerNorm):
 def forward(self,x):
  n=len(self.normalized_shape);shape=x.shape
  z=x.flatten(start_dim=-n)
  y=F.layer_norm(z,(math.prod(self.normalized_shape),),None if self.weight is None else self.weight.reshape(-1),None if self.bias is None else self.bias.reshape(-1),self.eps)
  return y.reshape(shape)
