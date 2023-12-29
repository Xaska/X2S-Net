import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils.dca import DCA
def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)
class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True, group=1):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        # assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
class FB(nn.Module):
    def __init__(self,indim,scale_factor=0.5):
        super(FB, self).__init__()
        self.scale_factor = scale_factor
        # Spatial Attention
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv1 = Conv(indim, indim, 1, bn=True, relu=True, bias=False)
    def forward(self,x1, x2):
        # 高维度特征
        l_jump = x1
        max_result, _ = torch.max(x1, dim=1, keepdim=True)
        avg_result = torch.mean(x1, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        x1 = self.spatial(result)
        x1 = self.sigmoid(x1) * l_jump
        x1 = F.interpolate(x1, scale_factor=self.scale_factor, mode="bilinear")
        # 低维度特征
        x2 = self.conv1(x2)
        ouput = torch.cat([x1, x2], dim=1)
        return ouput
# class FeatureNetwork(nn.Module):
#     def __init__(self, in_chans=3, dims=[128, 256, 512, 1024], depths=[3, 3, 9, 3], drop_path_rate=0.,layer_scale_init_value=1e-6,):
#         super(FeatureNetwork, self).__init__()
#
#         ###### Local Branch Setting #######
#         self.downsample_layers = nn.ModuleList()   # stem + 3 stage downsample
#         stem = nn.Sequential(
#             nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
#             LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
#         )
#         self.downsample_layers.append(stem)
#
#
#         for i in range(3):
#             downsample_layer = nn.Sequential(
#                     LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
#                     nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
#             )
#             self.downsample_layers.append(downsample_layer)
#         self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple blocks
#         dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
#         cur = 0
#         for i in range(4):
#             stage = nn.Sequential(
#                 *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
#                 layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
#             )
#             self.stages.append(stage)
#             cur += depths[i]
#
#     def forward(self, x):
#         x_c = self.downsample_layers[0](x)
#         x_c_1 = self.stages[0](x_c)
#         x_c = self.downsample_layers[1](x_c_1)
#         x_c_2 = self.stages[1](x_c)
#         x_c = self.downsample_layers[2](x_c_2)
#         x_c_3 = self.stages[2](x_c)
#         x_c = self.downsample_layers[3](x_c_3)
#         x_c_4 = self.stages[3](x_c)
#
#         return x_c_4
# [128,256,512,1024]取得0.997效果
class FeatureNetwork(nn.Module):
    def __init__(self, in_chans=3, dims=[8, 16, 32, 64], depths=[1, 1, 3, 1], drop_path_rate=0.,layer_scale_init_value=1e-6,
                 spatial_att=True,
                 channel_att=True,
                 spatial_head_dim=[4, 4, 4, 4],
                 channel_head_dim=[1, 1, 1, 1],
                 patch_size_ratio=8,
                 patch_size=12,
                 n=1,
                 ):
        super(FeatureNetwork, self).__init__()
        # 多尺度融合块设置
        self.tb1 = FB(indim=dims[1])
        self.tb2 = FB(indim=dims[3])
        self.tb3 = FB(indim=dims[2] + dims[3],scale_factor=0.25)


        ###### ConvNeXt Setting #######
        self.downsample_layers = nn.ModuleList()   # stem + 3 stage downsample
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)


        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        self.DCA_vgg2 = DCA(n=n,
                            features=dims,
                            strides=[patch_size_ratio, patch_size_ratio // 2, patch_size_ratio // 4,
                                     patch_size_ratio // 8],
                            patch=patch_size,
                            spatial_att=spatial_att,
                            channel_att=channel_att,
                            spatial_head=spatial_head_dim,
                            channel_head=channel_head_dim,
                            )

    def forward(self, x):
        output = []
        x_c = self.downsample_layers[0](x)
        x_c_1 = self.stages[0](x_c)
        x_c = self.downsample_layers[1](x_c_1)
        x_c_2 = self.stages[1](x_c)
        x_c = self.downsample_layers[2](x_c_2)
        x_c_3 = self.stages[2](x_c)
        x_c = self.downsample_layers[3](x_c_3)
        x_c_4 = self.stages[3](x_c)

        output.append(x_c_1)
        output.append(x_c_2)
        output.append(x_c_3)
        output.append(x_c_4)

        output = self.DCA_vgg2(output)

        output1 = self.tb1(output[0], output[1])
        output2 = self.tb2(output[2], output[3])
        output3 = self.tb3(output1, output2)


        return output3


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()

        self.input1 = FeatureNetwork()  # Recursively call FeatureNet to create the siamese network.
        self.input2 = FeatureNetwork()
        self.Conv3 = nn.Conv2d(240, 768, kernel_size=3, stride=1, padding=2)
        self.trans1 = nn.Conv2d(in_channels=768, out_channels=448, kernel_size=(1, 1), stride=2, padding=0)
        self.bnt1 = nn.BatchNorm2d(num_features=448)

        self.trans2 = nn.ConvTranspose3d(in_channels=28, out_channels=512, kernel_size=(1, 1, 1), stride=(1, 1, 1),
                                         padding=(0, 0, 0))
        self.bnt2 = nn.BatchNorm3d(num_features=512)

        # 分割模型1
        self.seg_1 = nn.ConvTranspose2d(in_channels=28, out_channels=128, kernel_size=(5, 5), stride=1, padding=0)
        self.seg_2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(5, 5), stride=2, padding=2, output_padding=1)
        self.seg_3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(5, 5),  stride=2,  padding=2, output_padding=1)

        self.seg1 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(1, 1), stride=1)
        self.se1 = nn.BatchNorm2d(num_features=1)
        # 边缘模型1
        self.edge_1 = nn.ConvTranspose2d(in_channels=28, out_channels=128, kernel_size=(5, 5), stride=1, padding=0)
        self.edge_2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(5, 5), stride=2, padding=2, output_padding=1)
        self.edge_3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(5, 5),  stride=2,  padding=2, output_padding=1)

        self.edge1 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(1, 1), stride=1)
        self.bn_edge1 = nn.BatchNorm2d(num_features=1)
        # 边缘模型2
        self.edge2_1 = nn.ConvTranspose2d(in_channels=28, out_channels=128, kernel_size=(5, 5), stride=1, padding=0)
        self.edge2_2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(5, 5), stride=2, padding=2, output_padding=1)
        self.edge2_3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(5, 5),  stride=2,  padding=2, output_padding=1)

        self.edge2 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(1, 1), stride=1)
        self.bn_edge2 = nn.BatchNorm2d(num_features=1)
        # 分割模型2
        self.seg_11 = nn.ConvTranspose2d(in_channels=28, out_channels=128, kernel_size=(5, 5), stride=1, padding=0)
        self.seg_22 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(5, 5), stride=2, padding=2, output_padding=1)
        self.seg_33 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(5, 5), stride=2, padding=2, output_padding=1)

        self.seg2 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(1, 1), stride=1)
        self.se2 = nn.BatchNorm2d(num_features=1)

        # 生成网络，生成3维图像
        self.g1 = nn.ConvTranspose3d(in_channels=512, out_channels=256,
                                     kernel_size=(1, 1, 1), stride=(2, 2, 2),
                                     padding=(0, 0, 0), output_padding=(1, 1, 1), bias=True, groups=1)
        self.bng1 = nn.BatchNorm3d(num_features=256)

        self.g2 = nn.ConvTranspose3d(
            in_channels=256,
            out_channels=128,
            kernel_size=(5, 5, 5),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            output_padding=(0, 0, 0),
            bias=True,
            groups=1,
            dilation=(1, 1, 1),
        )
        self.bng2 = nn.BatchNorm3d(num_features=128)

        self.g3 = nn.ConvTranspose3d(
            in_channels=128,
            out_channels=128,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            output_padding=(0, 0, 0),
            bias=True,
            groups=1,
            dilation=(1, 1, 1),
        )
        self.bng3 = nn.BatchNorm3d(num_features=128)


        self.g7 = nn.ConvTranspose3d(
            in_channels=128,
            out_channels=64,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            output_padding=(0, 0, 0),
            bias=True,
            groups=1,
            dilation=(1, 1, 1),
        )
        self.bng7 = nn.BatchNorm3d(num_features=64)

        self.g8 = nn.ConvTranspose3d(
            in_channels=64,
            out_channels=64,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            output_padding=(0, 0, 0),
            bias=True,
            groups=1,
            dilation=(1, 1, 1),
        )
        self.bng8 = nn.BatchNorm3d(num_features=64)

        self.g9 = nn.ConvTranspose3d(
            in_channels=10,
            out_channels=1,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            output_padding=(0, 0, 0),
            bias=True,
            groups=1,
            dilation=(1, 1, 1),
        )
        self.bng9 = nn.BatchNorm3d(num_features=1)

        self.Sig = nn.Sigmoid()
        self.relu = nn.ReLU()
    def forward(self,input1,input2):
        batch_size = input1.shape[0]
        input1 = self.input1(input1)
        input2 = self.input2(input2)
        # 拼接
        x = torch.cat([input1, input2], dim=1)
        x = self.Conv3(x)

        x = self.trans1(x)
        x = self.bnt1(x)
        x = self.relu(x)
        x1 = x.reshape((batch_size, 28, 1, 28, 28))

        x2 = self.trans2(x1)
        x2 = self.bnt2(x2)
        x2 = self.relu(x2)

        x3 = x1.reshape((batch_size, 28, 28, 28))
        # 边缘输出1
        edge1 = self.edge_1(x3)
        edge1 = self.edge_2(edge1)
        edge1 = self.edge_3(edge1)
        edge1 = self.edge1(edge1)
        edge1 = self.bn_edge1(edge1)
        edge1 = self.relu(edge1)
        # 边缘输出2
        edge2 = self.edge2_1(x3)
        edge2 = self.edge2_2(edge2)
        edge2 = self.edge2_3(edge2)
        edge2 = self.edge2(edge2)
        edge2 = self.bn_edge2(edge2)
        edge2 = self.relu(edge2)

        # 分割输出1
        output1 = self.seg_1(x3)
        output1 = self.seg_2(output1)
        output1 = self.seg_3(output1)
        output1 = self.seg1(output1)
        output1 = self.se1(output1)
        output1 = self.relu(output1)
        # 分割输出2
        output2 = self.seg_11(x3)
        output2 = self.seg_22(output2)
        output2 = self.seg_33(output2)
        output2 = self.seg2(output2)
        output2 = self.se2(output2)
        output2 = self.relu(output2)
        # 生成3维模型
        output3 = self.g1(x2)
        output3 = self.bng1(output3)
        output3 = self.relu(output3)

        output3 = self.g2(output3)
        output3 = self.bng2(output3)
        output3 = self.relu(output3)

        output3 = self.g3(output3)
        output3 = self.bng3(output3)
        output3 = self.Sig(output3)

        output3 = self.g7(output3)
        output3 = self.bng7(output3)
        output3 = self.Sig(output3)

        output3 = self.g8(output3)
        output3 = self.bng8(output3)
        output3 = self.Sig(output3)

        output3 = torch.transpose(output3, 1, 2)
        output3 = self.g9(output3)
        output3 = self.bng9(output3)
        output3 = self.Sig(output3)

        return output1, output2, output3, edge1, edge2

if __name__ == '__main__':
    model = FeatureNet()
    input1 = torch.randn(1, 3, 384, 384).cuda()
    input2 = torch.randn(1, 3, 384, 384).cuda()
    model.cuda()
    outs = model(input1, input2)
    pass













