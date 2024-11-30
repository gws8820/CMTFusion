from torchvision.models.vgg import vgg19
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from numpy.core import integer, empty, arange, asarray, roll
from numpy.core.overrides import array_function_dispatch, set_module
import numpy.fft as fft


class CosineSimilarity(nn.Module):
    r"""Returns cosine similarity between :math:`x_1` and :math:`x_2`, computed along dim.
    .. math ::
        \text{similarity} = \dfrac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)}.
    Args:
        dim (int, optional): Dimension where cosine similarity is computed. Default: 1
        eps (float, optional): Small value to avoid division by zero.
            Default: 1e-8
    Shape:
        - Input1: :math:`(\ast_1, D, \ast_2)` where D is at position `dim`
        - Input2: :math:`(\ast_1, D, \ast_2)`, same shape as the Input1
        - Output: :math:`(\ast_1, \ast_2)`
    Examples::
        >>> input1 = torch.randn(100, 128)
        >>> input2 = torch.randn(100, 128)
        >>> cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        >>> output = cos(input1, input2)
    """
    __constants__ = ['dim', 'eps']
    dim: int
    eps: float

    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        super(CosineSimilarity, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        cos = torch.nn.CosineSimilarity(dim=1)
        loss_cos = torch.mean(1 - cos(x1, x2))

        return loss_cos  # F.cosine_similarity(x1, x2, self.dim, self.eps)

class perceptual_loss(nn.Module):
    def __init__(self, requires_grad=False, device=torch.device("cpu")):
        super(perceptual_loss, self).__init__()

        self.maeloss = nn.L1Loss()
        vgg = vgg19(pretrained=True).to(device)

        vgg_pretrained_features = vgg.features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()

        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x].to(device))
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x].to(device))
        for x in range(9, 14):
            self.slice3.add_module(str(x), vgg_pretrained_features[x].to(device))

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X, Y):
        X = X.expand(-1, 3, -1, -1)  # Expand to 3 channels for VGG
        Y = Y.expand(-1, 3, -1, -1)
        fx2 = self.slice1(X)
        fx4 = self.slice2(fx2)
        fx6 = self.slice3(fx4)

        fy2 = self.slice1(Y)
        fy4 = self.slice2(fy2)
        fy6 = self.slice3(fy4)

        loss_p = self.maeloss(fx2, fy2) + self.maeloss(fx4, fy4) + self.maeloss(fx6, fy6)
        return loss_p


class L_spa(nn.Module):
    def __init__(self, device=torch.device("cpu")):
        super(L_spa, self).__init__()
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).to(device).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).to(device).unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).to(device).unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).to(device).unsqueeze(0).unsqueeze(0)

        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enhance):
        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)

        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        device = org_pool.device
        weight_diff = torch.max(
            torch.tensor([1.0], device=device) + 10000 * torch.min(org_pool - 0.3, torch.tensor([0.0], device=device)),
            torch.tensor([0.5], device=device)
        )
        E_1 = torch.mul(torch.sign(enhance_pool - 0.5), enhance_pool - org_pool)

        D_org_left = F.conv2d(org_pool, self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)

        D_enhance_left = F.conv2d(enhance_pool, self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)

        D_left = torch.pow(D_org_left - D_enhance_left, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)

        E = (D_left + D_right + D_up + D_down)
        return E


class frequency(nn.Module):
    def __init__(self, device=torch.device("cpu")):
        super(frequency, self).__init__()
        # Example: Initialize kernels on the specified device
        self.kernel = torch.FloatTensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]).to(device).unsqueeze(0).unsqueeze(0)

    def forward(self, input_tensor, target_tensor1, target_tensor2):
        device = input_tensor.device  # Ensure the computations are on the same device
        diff1 = F.conv2d(input_tensor, self.kernel, padding=1)
        diff2 = F.conv2d(target_tensor1, self.kernel, padding=1)
        diff3 = F.conv2d(target_tensor2, self.kernel, padding=1)
        loss_fre = torch.mean(torch.abs(diff1 - diff2)) + torch.mean(torch.abs(diff1 - diff3))
        return loss_fre