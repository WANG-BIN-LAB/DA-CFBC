from functools import partial
from typing import Optional
import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import checkpoint_seq
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            # print(self.weight.size())
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

            return x


class MSBlock(nn.Module):
    """ Multi-Scale Block (MSBlock) with parallel 3D convolutions, dividing input channels equally across branches. """

    def __init__(self, in_channels):
        super(MSBlock, self).__init__()
        
        # Divide input channels into three equal parts for each branch
        branch_channels = in_channels // 3
        
        # Branch 1: 1x1x7, 1x1x9, and 1x1x11 convolutions on a third of the input channels
        self.branch1_conv1 = nn.Conv3d(branch_channels, branch_channels, kernel_size=(1, 1, 7), padding=(0, 0, 3),groups=branch_channels)
        self.branch1_conv2 = nn.Conv3d(branch_channels, branch_channels, kernel_size=(1, 1, 9), padding=(0, 0, 4),groups=branch_channels)
        self.branch1_conv3 = nn.Conv3d(branch_channels, branch_channels, kernel_size=(1, 1, 11), padding=(0, 0, 5),groups=branch_channels)
        self.branch1_norm = nn.BatchNorm3d(branch_channels)
        
        # Branch 2: 1x7x1, 1x9x1, and 1x11x1 convolutions on another third of the input channels
        self.branch2_conv1 = nn.Conv3d(branch_channels, branch_channels, kernel_size=(1, 7, 1), padding=(0, 3, 0),groups=branch_channels)
        self.branch2_conv2 = nn.Conv3d(branch_channels, branch_channels, kernel_size=(1, 9, 1), padding=(0, 4, 0),groups=branch_channels)
        self.branch2_conv3 = nn.Conv3d(branch_channels, branch_channels, kernel_size=(1, 11, 1), padding=(0, 5, 0),groups=branch_channels)
        self.branch2_norm = nn.BatchNorm3d(branch_channels)
        
        # Branch 3: 7x1x1, 9x1x1, and 11x1x1 convolutions on the last third of the input channels
        self.branch3_conv1 = nn.Conv3d(branch_channels, branch_channels, kernel_size=(7, 1, 1), padding=(3, 0, 0),groups=branch_channels)
        self.branch3_conv2 = nn.Conv3d(branch_channels, branch_channels, kernel_size=(9, 1, 1), padding=(4, 0, 0),groups=branch_channels)
        self.branch3_conv3 = nn.Conv3d(branch_channels, branch_channels, kernel_size=(11, 1, 1), padding=(5, 0, 0),groups=branch_channels)
        self.branch3_norm = nn.BatchNorm3d(branch_channels)
        
        # Final 1x1x1 convolution for dimensionality adjustment after concatenation
        self.final_conv1 = nn.Conv3d(branch_channels*3, branch_channels, kernel_size=1)
        self.final_conv2 = nn.Conv3d(branch_channels*3, branch_channels, kernel_size=1)
        self.final_conv3 = nn.Conv3d(branch_channels*3, branch_channels, kernel_size=1)

    def forward(self, x):
        # Split input into three parts for each branch
        x1, x2, x3 = torch.split(x, x.size(1) // 3, dim=1)
        
        # Branch 1: Apply parallel convolutions and concatenate results
        out1 = torch.cat([self.branch1_conv1(x1), self.branch1_conv2(x1), self.branch1_conv3(x1)], dim=1)
        out1 = self.final_conv1(out1)
        out1 = self.branch1_norm(out1)
        
        # Branch 2: Apply parallel convolutions and concatenate results
        out2 = torch.cat([self.branch2_conv1(x2), self.branch2_conv2(x2), self.branch2_conv3(x2)], dim=1)
        out2 = self.final_conv2(out2)
        out2 = self.branch2_norm(out2)
        
        # Branch 3: Apply parallel convolutions and concatenate results
        out3 = torch.cat([self.branch3_conv1(x3), self.branch3_conv2(x3), self.branch3_conv3(x3)], dim=1)
        out3 = self.final_conv3(out3)
        out3 = self.branch3_norm(out3)
        
        # Concatenate all branches and apply final 1x1x1 conv to match original in_channels
        out = torch.cat((out1, out2, out3), dim=1)
        # return self.final_conv(out)
        return out

class DAFCM(nn.Module):
    """Direction Aware Feature Capture Module (DAFCM)"""

    def __init__(self, in_channels, cube_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()

        gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
        self.dwconv_hwd = nn.Conv3d(gc, gc, cube_kernel_size, padding=cube_kernel_size // 2, groups=gc)
        self.dwconv_wd = nn.Conv3d(gc, gc, kernel_size=(1, 1, band_kernel_size), padding=(0, 0, band_kernel_size // 2),
                                   groups=gc)
        self.dwconv_hd = nn.Conv3d(gc, gc, kernel_size=(1, band_kernel_size, 1), padding=(0, band_kernel_size // 2, 0),
                                   groups=gc)
        self.dwconv_hw = nn.Conv3d(gc, gc, kernel_size=(band_kernel_size, 1, 1), padding=(band_kernel_size // 2, 0, 0),
                                   groups=gc)
        self.split_indexes = (in_channels - 4 * gc, gc, 3*gc)
        self.msblock = MSBlock(3*gc)

    def forward(self, x):
        x_id, x_hwd, x_ms = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hwd(x_hwd), self.msblock(x_ms)),
            dim=1,
        )
class ConvModule(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            groups: int = 1,
            norm_cfg: Optional[dict] = None,
            act_cfg: Optional[dict] = None):
        super().__init__()
        layers = []
        # Convolution Layer
        layers.append(nn.Conv3d(in_channels, out_channels, kernel_size, 
                                 (stride, stride, stride), padding, 
                                 groups=groups, bias=(norm_cfg is None)))
        # Normalization Layer
        if norm_cfg:
            norm_layer = self._get_norm_layer(out_channels, norm_cfg)
            layers.append(norm_layer)
        # Activation Layer
        if act_cfg:
            act_layer = self._get_act_layer(act_cfg)
            layers.append(act_layer)
        # Combine all layers
        self.block = nn.Sequential(*layers)


    def forward(self, x):
        return self.block(x)

    def _get_norm_layer(self, num_features, norm_cfg):
        if norm_cfg['type'] == 'BN':
            return nn.BatchNorm3d(num_features, momentum=norm_cfg.get('momentum', 0.1), eps=norm_cfg.get('eps', 1e-5))
        # Add more normalization types if needed
        raise NotImplementedError(f"Normalization layer '{norm_cfg['type']}' is not implemented.")

    def _get_act_layer(self, act_cfg):
        if act_cfg['type'] == 'ReLU':
            return nn.ReLU(inplace=True)
        if act_cfg['type'] == 'SiLU':
            return nn.SiLU(inplace=True)
        # Add more activation types if needed
        raise NotImplementedError(f"Activation layer '{act_cfg['type']}' is not implemented.")

class DASFM(nn.Module):
    """Direction Aware Spatial Focus Module (DASFM)"""
    def __init__(
            self,
            channels: int,
            h_kernel_size: int = 3,
            v_kernel_size: int = 3,
            d_kernel_size: int = 3,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU')):
        super().__init__()
        self.avg_pool = nn.AvgPool3d(5, 1, 2)
        self.conv1 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.h_conv = ConvModule(channels, channels, (1, h_kernel_size, 1), 1,
                                 (0, h_kernel_size // 2, 0), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.v_conv = ConvModule(channels, channels, (v_kernel_size, 1, 1), 1,
                                 (v_kernel_size // 2, 0, 0), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.d_conv = ConvModule(channels, channels, (1, 1, d_kernel_size), 1,
                                 (0, 0, d_kernel_size // 2), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.conv2 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.act = nn.Sigmoid()

    def forward(self, x):
        # attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        attn_factor = self.act(self.conv2(self.d_conv(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x)))))))
        return attn_factor
class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    copied from timm: https://github.com/huggingface/pytorch-image-models/blob/v0.6.11/timm/models/layers/mlp.py
    """
    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU,
            norm_layer=None, bias0=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        #bias = to_2tuple(bias)
        bias = (bias0, bias0)

        self.fc1 = nn.Conv3d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.pwconv1 = nn.Conv3d(in_features, hidden_features, kernel_size=1, groups=in_features)
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.pwconv2 = nn.Conv3d(hidden_features, in_features, kernel_size=1, groups=in_features)
        self.fc2 = nn.Conv3d(hidden_features, out_features, kernel_size=1, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)#[1,16,64,64,64]-->[1,64,64,64,64]
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class MlpHead(nn.Module):
    """ MLP classification head
    """
    def __init__(self, dim, num_classes=1000, mlp_ratio=3, act_layer=nn.GELU,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), drop=0., bias=True):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):#[1,1024,4,4]
        x = x.mean((2, 3)) # global average pooling#[1,1024]
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            # 将 Conv3d 层添加到 ops 列表中
            ops.append(nn.Conv3d(input_channel, n_filters_out, kernel_size=3, padding=1))
            
            # 根据 normalization 参数添加不同的标准化层
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out, track_running_stats=False))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False, "Unsupported normalization type"

            # 添加ReLU激活函数
            ops.append(nn.ReLU(inplace=True))

        # 将所有操作封装到一个 Sequential 容器中
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        # 直接使用 self.conv 来处理输入 x
        #print(type(x)) 
        x = self.conv(x)
        return x

class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out, track_running_stats=False))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out, track_running_stats=False))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class DAFFBlock(nn.Module):
    """ DAFFBlock Block """

    def __init__(
            self,
            dim,
            token_mixer1=nn.Identity,
            token_mixer2=nn.Identity,
            norm_layer=nn.BatchNorm3d,
            mlp_layer=ConvMlp,
            mlp_ratio=4,
            act_layer=nn.GELU,
            ls_init_value=1e-6,
            drop_path=0.,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU')
            
    ):
        super().__init__()
        self.token_mixer1 = token_mixer1(dim)
        self.token_mixer2 = token_mixer2(dim)
        self.pw_conv = ConvModule(dim, dim, 1, 1, 0, 1,
                                  norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.dim = dim

    def forward(self, x):
        shortcut = x
        x1 = self.token_mixer1(x)
        x2 = self.token_mixer2(x)
        x3 = x1*x2
        x = x1+x3
        x = self.pw_conv(x)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x


class DAFFStage(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            #ds_stride=0,
            #n_filters,
            stage_i=0,
            depth=2,
            drop_path_rates=None,
            ls_init_value=1.0,
            token_mixer1=nn.Identity,
            token_mixer2=nn.Identity,
            act_layer=nn.GELU,
            norm_layer=None,
            mlp_ratio=4,
            normalization='batchnorm'
    ):
        super().__init__()
        self.grad_checkpointing = False
        self.stage_i = stage_i
        drop_path_rates = drop_path_rates or [0.] * depth
        stage_blocks = []
        for i in range(depth):
            stage_blocks.append(DAFFBlock(
                dim=out_chs,
                drop_path=drop_path_rates[i],
                ls_init_value=ls_init_value,
                token_mixer1=token_mixer1,
                token_mixer2=token_mixer2,
                act_layer=act_layer,
                norm_layer=norm_layer,
                mlp_ratio=mlp_ratio,
            ))
            in_chs = out_chs
        self.blocks = nn.Sequential(*stage_blocks)
        self.out_chs = out_chs

    def forward(self, x): 

        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x


class DAUNet(nn.Module):
    """ DAUNet"""

    def __init__(
            self,
            in_chans=1,
            n_filters=16,
            num_classes=1000,
            depths=(3, 3, 9, 3),
            dims=(96, 192, 384, 768),
            token_mixers1=nn.Identity,
            token_mixers2=nn.Identity,
            norm_layer=nn.BatchNorm3d,
            act_layer=nn.GELU,
            mlp_ratios=(4, 4, 4, 3),
            head_fn=MlpHead,
            drop_rate=0.,
            drop_path_rate=0.,
            ls_init_value=1e-6,
            normalization='batchnorm',
            has_dropout=False,
            **kwargs,
    ):
        super().__init__()

        num_stage = len(depths)
        if not isinstance(token_mixers1, (list, tuple)):
            token_mixers1 = [token_mixers1] * num_stage
        if not isinstance(token_mixers2, (list, tuple)):
            token_mixers2 = [token_mixers2] * num_stage
        if not isinstance(mlp_ratios, (list, tuple)):
            mlp_ratios = [mlp_ratios] * num_stage


        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.dims = dims
        self.stages = nn.Sequential()
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        stages = []
        prev_chs = self.dims[0]
        # feature resolution stages, each consisting of multiple residual blocks
        for i in range(num_stage):
            out_chs = dims[i]
            stages.append(DAFFStage(
                prev_chs,
                out_chs,
                #ds_stride=2 if i > 0 else 1, 
                #
                stage_i=i,
                depth=depths[i],
                drop_path_rates=dp_rates[i],
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                token_mixer1=token_mixers1[i],
                token_mixer2=token_mixers2[i],
                norm_layer=norm_layer,
                mlp_ratio=mlp_ratios[i],
                normalization='batchnorm'
            ))
            prev_chs = out_chs*(2**i)
        self.stages = nn.Sequential(*stages)
        self.num_features = prev_chs
        self.out_chans = out_chs
        #encoder
        self.has_dropout = has_dropout
        filter = out_chs // 8

        self.block_one = ConvBlock(1, in_chans, 8, normalization=normalization)
        self.block_dw_one = nn.Conv3d(8, 8, kernel_size=(7, 7, 7), stride=1, padding=3,groups=8)
        self.ln1 = LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        self.block_one_dw = DownsamplingConvBlock(8, filter, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(filter, filter*2, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(filter*2, filter*4, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(filter*4, filter*8, normalization=normalization) 
        self.block_five = ConvBlock(3, filter * 8, filter * 8, normalization=normalization)
        #decoder
        self.block_five_up = UpsamplingDeconvBlock(filter * 8, filter * 4, normalization=normalization)
        self.block_six = ConvBlock(3, filter * 4, filter * 4, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(filter * 4, filter * 2, normalization=normalization)
        self.block_seven = ConvBlock(3, filter * 2, filter * 2, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(filter * 2, filter, normalization=normalization)
        self.block_eight = ConvBlock(2, filter, filter, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(filter, 8, normalization=normalization)
        self.out_conv_seg = nn.Conv3d(8, 3, 1, padding=0)
        self.out_conv_edge = nn.Conv3d(8, 3, 1, padding=0)
        self.out_conv_off = nn.Conv3d(8, 3, 1, padding=0)

        self.head = head_fn(self.num_features, num_classes, drop=drop_rate)
        self.apply(self._init_weights)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'norm'}
    

    def forward_features(self, x):
        x1 = self.block_one(x)
        x1 = self.block_dw_one(x1)
        x1_dw = self.block_one_dw(x1)
        x2 = self.stages[0](x1_dw)
        x2_dw= self.block_two_dw(x2)
        x3 = self.stages[1](x2_dw)
        x3_dw= self.block_three_dw(x3)
        x4 = self.stages[2](x3_dw)
        x4_dw = self.block_four_dw(x4)
        x5 = self.stages[3](x4_dw)

        res = [x2,x3,x4,x5]
        return res

    def decoder_seg(self,features):
        x2 = features[0]
        x3 = features[1]
        x4 = features[2]
        x5 = features[3]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2
        
        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)

        out_seg = self.out_conv_seg(x8_up)
        
        return out_seg
    def decoder_off(self,features):
        x2 = features[0]
        x3 = features[1]
        x4 = features[2]
        x5 = features[3]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2
        
        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)

        out_off = self.out_conv_off(x8_up)
        
        return out_off
    def decoder_bd(self,features):
        x2 = features[0]
        x3 = features[1]
        x4 = features[2]
        x5 = features[3]

        x5_up = self.block_five_up(x5)

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2
        
        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)

        out_bd = self.out_conv_edge(x8_up)
        
        return out_bd

    def forward_head(self, x):
        x = self.head(x)
        return x

    def forward(self, x):
        features = self.forward_features(x)
        out_seg = self.decoder_seg(features)
        out_bd = self.decoder_bd(features)
    
        return out_seg, out_bd


    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)



def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.0', 'classifier': 'head.fc',
        **kwargs
    }


default_cfgs = dict(
    daunet_tiny=_cfg(
        url='https://github.com/sail-sg/inceptionnext/releases/download/model/inceptionnext_tiny.pth',
    ),
    daunet_small=_cfg(
        url='https://github.com/sail-sg/inceptionnext/releases/download/model/inceptionnext_small.pth',
    ),
    daunet_base=_cfg(
        url='https://github.com/sail-sg/inceptionnext/releases/download/model/inceptionnext_base.pth',
    ),
    daunet_base_384=_cfg(
        url='https://github.com/sail-sg/inceptionnext/releases/download/model/inceptionnext_base_384.pth',
        input_size=(3, 384, 384), crop_pct=1.0,
    ),
)


@register_model
def daunet_tiny(pretrained=False, **kwargs):
    model = DAUNet(depths=(3, 3, 15, 3), dims=(32, 64, 128, 256), 
                      token_mixers1=DAFCM,token_mixers2=DASFM,
                      **kwargs
    )
    model.default_cfg = default_cfgs['daunet_tiny']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model

@register_model
def daunet_small(pretrained=False, **kwargs):
    model = DAUNet(depths=(3, 3, 27, 3), dims=(96, 192, 384, 768), 
                      token_mixers=DAFCM,token_mixers2=DASFM,
                      **kwargs
    )
    model.default_cfg = default_cfgs['daunet_small']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model

@register_model
def daunet_base(pretrained=False, **kwargs):
    model = DAUNet(depths=(3, 3, 27, 3), dims=(128, 256, 512, 1024), 
                      token_mixers=DAFCM,token_mixers2=DASFM,
                      **kwargs
    )
    model.default_cfg = default_cfgs['daunet_base']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model

@register_model
def daunet_base_384(pretrained=False, **kwargs):
    model = DAUNet(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], 
                      mlp_ratios=[4, 4, 4, 3],
                      token_mixers=DAFCM,token_mixers2=DASFM,
                      **kwargs
    )
    model.default_cfg = default_cfgs['daunet_base_384']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url=model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model

###################################################################################
if __name__ == "__main__":
    input = torch.randint(
        low=0,
        high=1,
        size=(1, 1, 80, 80, 128),
        dtype=torch.float,
    )
    input = input.to("cuda:0")
    model = daunet_tiny().cuda()
    print(model)
    output_seg,out_off = model(input)
    print(output_seg.shape)

    from thop import profile
    from thop import clever_format

    flops, params = profile(model, inputs=(input, ))
    flops, params = clever_format([flops, params], '%.3f')
    print(f"FLOPs: {flops}, 参数量：{params}")