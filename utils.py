import torch
from torchvision.models.detection.anchor_utils import AnchorGenerator

from sigfig import round


def _default_anchorgen(idx):
    sizeses = ((32,), (64,), (128,), (256,), (512,))
    anchor_sizes = (sizeses[idx])
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    return AnchorGenerator(anchor_sizes, aspect_ratios)

def permute_and_flatten(layer, N, A, C, H, W):
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer

def _tensor_size(tensor):
    return f"{round(tensor.element_size() * tensor.nelement() / 1000000, sigfigs=4)} Mb"

def _size_helper(obj):
    if type(obj) == torch.Tensor:
        return  str(obj.size()).replace(', ', 'x'), _tensor_size(obj)
        # print(name, "::" , obj.shape, _tensor_size(obj) )  
    elif type(obj) == type([1, 2]):
        add = 0
        for tensor in obj:
            if type(tensor) != torch.Tensor:
                assert False, f"Expected a tensor or a list of tensors as input, a list of {type(tensor)} was given."
            add += tensor.element_size() * tensor.nelement() / 1000000
        return "List of Tensors", f"{round(add, sigfigs=4)} Mb" 
    else:
        assert False, f"Expected a tensor or a list of tensors as input, a {type(obj)} was given."