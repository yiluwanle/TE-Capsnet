import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from TCN import TemporalConvNet


def squash(input, dim=-1, eps=10e-21):
    n = torch.norm(input, dim=dim, keepdim=True)
    return (1 - 1 / (torch.exp(n) + eps)) * (input / (n + eps))


class PrimaryCapsLayer(nn.Module):
    """创建一个主胶囊层，使用2D深度卷积提取每个胶囊的属性

    Args:
        in_channels (int): 深度卷积的特征数
        kernel_size (int): 深度卷积的核维数
        num_capsules (int): 初级胶囊数
        dim_capsules (int): 初级胶囊维度
        stride (int, optional): 深度卷积的步长，默认值为1
    """

    def __init__(self, in_channels, kernel_size, num_capsules, dim_capsules, stride=1):
        super(PrimaryCapsLayer, self).__init__()
        self.depthwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
            padding="valid",
        )
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules

    def forward(self, x):
        output = self.depthwise_conv(x)
        output = output.view(output.size(0), self.num_capsules, self.dim_capsules)
        return squash(output)


class RoutingLayer(nn.Module):
    """使用完全连接网络的自注意路由层，以创建高级胶囊层。"""

    def __init__(self, args):
        super(RoutingLayer, self).__init__()
        self.W = nn.Parameter(
            torch.Tensor(args['caps2_num'], args['caps1_num'], args['caps1_channels'], args['caps2_channels']))
        self.b = nn.Parameter(torch.zeros(args['caps2_num'], args['caps1_num'], 1))
        self.num_capsules = args['caps2_num']
        self.dim_capsules = args['caps2_channels']
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.W)
        nn.init.zeros_(self.b)

    def forward(self, input):
        u = torch.einsum(
            "...ji,kjiz->...kjz", input, self.W
        )
        c = torch.einsum("...ij,...kj->...i", u, u)[
            ..., None
        ]
        c = c / torch.sqrt(
            torch.Tensor([self.dim_capsules]).type(torch.cuda.FloatTensor)
        )
        c = torch.softmax(c, axis=1)
        c = c + self.b
        s = torch.sum(
            torch.mul(u, c), dim=-2
        )
        return squash(s)


# 自注意路由胶囊层
class CapsuleLayers(nn.Module):
    def __init__(self, args, device):
        super(CapsuleLayers, self).__init__()
        self.args = args

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=args["conv1_channels"], kernel_size=(5, 9), stride=3)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.bn1 = nn.BatchNorm2d(args["conv1_channels"])
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=args['conv1_channels'], out_channels=args["conv2_channels"],
                               kernel_size=(5, 9), stride=3)
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        self.bn2 = nn.BatchNorm2d(args["conv2_channels"])
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels=args["conv2_channels"],
                               out_channels=args['caps1_channels'] * args['caps1_num'], kernel_size=(5, 9), stride=3)
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
        self.bn3 = nn.BatchNorm2d(args['caps1_channels'] * args['caps1_num'])
        self.relu3 = nn.ReLU(inplace=True)

        self.primary_caps = PrimaryCapsLayer(in_channels=args['caps1_channels'] * args['caps1_num'], kernel_size=(5, 9),
                                             num_capsules=args['caps1_num'],
                                             dim_capsules=args['caps1_channels'], stride=2)

        self.traffic_caps = RoutingLayer(args)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.primary_caps(x)
        x = self.traffic_caps(x)
        x = x.unsqueeze(1)
        return x


class Network(nn.Module):
    def __init__(self, args, device):
        super(Network, self).__init__()
        self.args = args
        self.capsule_layers = nn.ModuleList([
            CapsuleLayers(args, device) for _ in range(args['time_step'])
        ])
        self.flatten = nn.Flatten()
        if args["tod_embedding_dim"] > 0:
            self.tod_embedding = nn.Embedding(args["steps_per_day"], args["tod_embedding_dim"])
        if args["dow_embedding_dim"] > 0:
            self.dow_embedding = nn.Embedding(7, args["dow_embedding_dim"])

        self.tcn = TemporalConvNet(
            args['caps2_num'] * args['caps2_channels'] + args['tod_embedding_dim'] + args["dow_embedding_dim"],
            args['tcn_list'])

        self.fc1 = nn.Linear(args["time_end"] * args['tcn_list'][1], 2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.fc3 = nn.Linear(4096, args["time_end"] * args["num_sensor"])

    def forward(self, x, date):
        features = []
        if self.args["tod_embedding_dim"] > 0:
            tod = date[..., 0]
            tod_emb = self.tod_embedding(
                (tod * self.args["steps_per_day"]).long()
            )
            features.append(tod_emb)
        if self.args["dow_embedding_dim"] > 0:
            dow = date[..., 1]
            dow_emb = self.dow_embedding(
                dow.long()
            )
            features.append(dow_emb)

        # 分时间步
        capsule_outputs = [capsule(x[:, i, ...]) for i, capsule in enumerate(self.capsule_layers)]

        capsule_outputs = torch.cat(capsule_outputs, dim=1)
        capsule_outputs = capsule_outputs.contiguous().view(-1, self.args['time_end'],
                                                            self.args['caps2_num'] * self.args['caps2_channels'])
        features.append(capsule_outputs)
        capsule_outputs = torch.cat(features, dim=-1)
        capsule_outputs = capsule_outputs.permute(0, 2, 1)

        capsule_outputs = self.tcn(capsule_outputs)
        capsule_outputs = self.flatten(capsule_outputs)

        capsule_outputs = F.relu(self.fc1(capsule_outputs), inplace=True)
        capsule_outputs = F.relu(self.fc2(capsule_outputs), inplace=True)
        capsule_outputs = self.fc3(capsule_outputs)
        capsule_outputs = capsule_outputs.contiguous().view(-1, self.args['time_end'], self.args['num_sensor'])
        return capsule_outputs
