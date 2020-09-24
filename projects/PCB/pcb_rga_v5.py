from __future__ import division, absolute_import
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Resnet50_Branch(nn.Module):
    def __init__(self, ** kwargs):
        super(Resnet50_Branch, self).__init__()

        # backbone--------------------------------------------------------------------------
        resnet = models.resnet50(pretrained=True)
        # Modifiy the stride of last conv layer----------------------------
        resnet.layer4[0].downsample[0].stride = (1, 1)
        resnet.layer4[0].conv2.stride = (1, 1)
        # Remove avgpool and fc layer of resnet------------------------------
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)

    def forward(self, x):
        return self.backbone(x)


class PCB_RGA(nn.Module):
    def __init__(self, num_classes, loss='softmax', height=384, width=128, **kwargs):
        super(PCB_RGA, self).__init__()
        self.parts = 6
        self.num_classes = num_classes
        self.loss = loss

        # backbone=============================================================================
        self.backbone = Resnet50_Branch()

        # gloab=============================================================================
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gloab = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.gloab.apply(weights_init_kaiming)
        self.global_softmax = nn.Linear(512, num_classes)
        self.global_softmax.apply(weights_init_kaiming)

        # part==============================================================================
        # avgpool--------------------------------------------------------------------------
        self.avgpool = nn.AdaptiveAvgPool2d((self.parts, 1))
        # self.dropout = nn.Dropout(p=0.5)

        # local_conv--------------------------------------------------------------------
        self.local_conv_list = nn.ModuleList()
        for _ in range(self.parts):
            local_conv = nn.Sequential(
                nn.Conv1d(2048, 256, kernel_size=1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True))
            # local_conv.apply(torchtool.weights_init_kaiming)
            self.local_conv_list.append(local_conv)

        # Classifier for each stripe--------------------------------------------------------------------------
        self.fc_list = nn.ModuleList()
        for _ in range(self.parts):
            fc = nn.Linear(256, num_classes)
            nn.init.normal_(fc.weight, std=0.001)
            nn.init.constant_(fc.bias, 0)
            self.fc_list.append(fc)

    def forward(self, x):
        # backbone(Tensor T)([N, 2048, 24, 6]) ========================================================================================
        resnet_features = self.backbone(x)

        # gloab([N, 512]) ========================================================================================
        # att_features = self.rga_att(resnet_features)
        global_avgpool_features = self.global_avgpool(resnet_features)
        gloab_features = self.gloab(global_avgpool_features).squeeze()

        # parts ========================================================================================
        # tensor g([N, 2048, 6, 1])---------------------------------------------------------------------------------
        features_G = self.avgpool(resnet_features)

        # 1x1 conv([N, C=256, H=6, W=1])---------------------------------------------------------------------------------
        features_H = []
        for i in range(self.parts):
            stripe_features_H = self.local_conv_list[i](features_G[:, :, i, :])
            features_H.append(stripe_features_H)

        # Return the features_H([N,1536])***********************************************************************
        if not self.training:
            # features_H.append(gloab_features.unsqueeze_(2))
            v_g = torch.cat(features_H, dim=1)
            v_g = F.normalize(v_g, p=2, dim=1)
            return v_g.view(v_g.size(0), -1)

        # fc（[N, C=num_classes]）---------------------------------------------------------------------------------

        gloab_softmax = self.global_softmax(gloab_features)

        batch_size = x.size(0)
        logits_list = [self.fc_list[i](features_H[i].view(batch_size, -1)) for i in range(self.parts)]

        return logits_list


def build_model(model_name, num_classes, **kwargs):
    model_name=model_name
    return PCB_RGA(
        num_classes=num_classes,
        **kwargs
    )
