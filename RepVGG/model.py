import torch.nn as nn
import numpy as np
import torch
import copy
import torch.utils.checkpoint as checkpoint
from se_block import SEBlock


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    # 将卷积层和BN层合并为一个分支，方便后续重参数化
    res = nn.Sequential()
    res.add_module("conv", nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups, bias=False))
    res.add_module("bn", nn.BatchNorm2d(num_features=out_channels))
    return res


class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        # deploy: 网络当前所处的状态, 训练时为False, 重参数化后为True
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3, "kernel_size should be 3."
        assert padding == 1, "padding should be 1"
        # padding_11 表示1x1卷积所使用的padding
        padding_11 = padding - kernel_size // 2
        self.nonlinearity = nn.ReLU()

        if use_se:
            self.se = SEBlock(out_channels, internal_neurous=out_channels // 16)
        else:
            self.se = nn.Identity()

        if deploy:
            # 如果当前需要重参数化, 则主线上没有分支, 所有分支合并为一个卷积层
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups,
                                         bias=True, padding_mode=padding_mode)
        else:
            # 训练过程中, 需要分为三个分支
            # 3x3+BN   1x1+BN   BN
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                   stride=stride, padding=padding_11, groups=groups)
            # print("RepVGG Block, identity = ", self.rbr_identity)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_custom_L2(self):
        K3 = self.rbr_dense.conv.weight
        K1 = self.rbr_1x1.conv.weight
        t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
        t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
        l2_loss_circle = (K3**2).sum() - (K3[:, :, 1:2, 1:2]**2).sum()
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1
        l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()
        return l2_loss_eq_kernel + l2_loss_circle

    def get_equivalent_kernel_bias(self):
        # 获取每个分支中混合后的卷积核权重和偏置
        # 返回等效后的卷积核和偏置
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        # 将1x1的卷积核变为3x3的卷积核
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        # 将卷积核和BN层融合
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        # 训练完成后 切换到重参数化模式
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,
                                     out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size,
                                     stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding,
                                     dilation=self.rbr_dense.conv.dilation,
                                     groups=self.rbr_dense.conv.groups,
                                     bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        self.__delattr__("rbr_dense")
        self.__delattr__("rbr_1x1")
        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")
        self.deploy = True


class RepVGG(nn.Module):

    def __init__(self, num_blocks, num_classes=1000, width_multiplier=None,
                 override_groups_map=None, deploy=False, use_se=False, use_checkpoint=False):
        super(RepVGG, self).__init__()
        assert len(width_multiplier) == 4
        self.deploy = deploy
        self.override_groups_map = override_groups_map if override_groups_map else dict()
        assert 0 not in self.override_groups_map
        self.use_se = use_se
        self.use_checkpoint = use_checkpoint

        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.stage0 = RepVGGBlock(
            in_channels=3,
            out_channels=self.in_planes,
            kernel_size=3,
            stride=2,
            padding=1,
            deploy=self.deploy,
            use_se=self.use_se
        )
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64*width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128*width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256*width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512*width_multiplier[3]), num_blocks[3], stride=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512*width_multiplier[3]), num_classes)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            cur_group = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(
                in_channels=self.in_planes,
                out_channels=planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=cur_group,
                deploy=self.deploy,
                use_se=self.use_se
            ))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.ModuleList(blocks)

    def forward(self, x):
        out = self.stage0(x)
        for stage in (self.stage1, self.stage2, self.stage3, self.stage4):
            for block in stage:
                if self.use_checkpoint:
                    out = checkpoint.checkpoint(block, out)
                else:
                    out = block(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}


def create_RepVGG_A0(deploy=False, use_checkpoint=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy, use_checkpoint=use_checkpoint)


def create_RepVGG_A1(deploy=False, use_checkpoint=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy, use_checkpoint=use_checkpoint)


def create_RepVGG_A2(deploy=False, use_checkpoint=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[1.5, 1.5, 1.5, 2.75], override_groups_map=None, deploy=deploy, use_checkpoint=use_checkpoint)


def create_RepVGG_B0(deploy=False, use_checkpoint=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy, use_checkpoint=use_checkpoint)


def create_RepVGG_B1(deploy=False, use_checkpoint=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=None, deploy=deploy, use_checkpoint=use_checkpoint)


def create_RepVGG_B1g2(deploy=False, use_checkpoint=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g2_map, deploy=deploy, use_checkpoint=use_checkpoint)


def create_RepVGG_B1g4(deploy=False, use_checkpoint=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g4_map, deploy=deploy, use_checkpoint=use_checkpoint)


def create_RepVGG_B2(deploy=False, use_checkpoint=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy, use_checkpoint=use_checkpoint)


def create_RepVGG_B2g2(deploy=False, use_checkpoint=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g2_map, deploy=deploy, use_checkpoint=use_checkpoint)


def create_RepVGG_B2g4(deploy=False, use_checkpoint=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g4_map, deploy=deploy, use_checkpoint=use_checkpoint)


def create_RepVGG_B3(deploy=False, use_checkpoint=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=None, deploy=deploy, use_checkpoint=use_checkpoint)


def create_RepVGG_B3g2(deploy=False, use_checkpoint=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g2_map, deploy=deploy, use_checkpoint=use_checkpoint)


def create_RepVGG_B3g4(deploy=False, use_checkpoint=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g4_map, deploy=deploy, use_checkpoint=use_checkpoint)


def create_RepVGG_D2se(deploy=False, use_checkpoint=False):
    return RepVGG(num_blocks=[8, 14, 24, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy, use_se=True, use_checkpoint=use_checkpoint)


func_dict = {
    'RepVGG-A0': create_RepVGG_A0,
    'RepVGG-A1': create_RepVGG_A1,
    'RepVGG-A2': create_RepVGG_A2,
    'RepVGG-B0': create_RepVGG_B0,
    'RepVGG-B1': create_RepVGG_B1,
    'RepVGG-B1g2': create_RepVGG_B1g2,
    'RepVGG-B1g4': create_RepVGG_B1g4,
    'RepVGG-B2': create_RepVGG_B2,
    'RepVGG-B2g2': create_RepVGG_B2g2,
    'RepVGG-B2g4': create_RepVGG_B2g4,
    'RepVGG-B3': create_RepVGG_B3,
    'RepVGG-B3g2': create_RepVGG_B3g2,
    'RepVGG-B3g4': create_RepVGG_B3g4,
    'RepVGG-D2se': create_RepVGG_D2se,
}


def get_RepVGG_func_by_name(name):
    return func_dict[name]


def repvgg_model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return
