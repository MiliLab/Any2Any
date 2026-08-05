"""Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""

import os
import torch
import torch.nn as nn
from torchvision import models
from collections import namedtuple

from taming.util import get_ckpt_path


class LPIPS(nn.Module):
    def __init__(self, use_dropout=True, ckpt_path=None, vgg_weights_path=None, allow_download=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]
        self.net = vgg16(pretrained=(vgg_weights_path is None), requires_grad=False, weights_path=vgg_weights_path)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained(name="vgg_lpips", ckpt_path=ckpt_path, allow_download=allow_download)
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips", ckpt_path=None, allow_download=True):
        ckpt = None
        if ckpt_path is not None and str(ckpt_path).strip() != "":
            ckpt = os.path.abspath(str(ckpt_path))
            if not os.path.exists(ckpt):
                raise FileNotFoundError(f"LPIPS ckpt not found: {ckpt}")
        else:
            candidates = [
                os.path.join("taming", "modules", "autoencoder", "lpips", "vgg.pth"),
                os.path.join("taming-transformers", "taming", "modules", "autoencoder", "lpips", "vgg.pth"),
            ]
            for p in candidates:
                if os.path.exists(p):
                    ckpt = p
                    break
            if ckpt is None:
                if not bool(allow_download):
                    raise FileNotFoundError(
                        "LPIPS weights missing and allow_download=False. Provide --lpips-ckpt-path or place vgg.pth under taming/modules/autoencoder/lpips/"
                    )
                ckpt = get_ckpt_path(name, "taming/modules/autoencoder/lpips")

        try:
            state = torch.load(ckpt, map_location=torch.device("cpu"), weights_only=True)
        except TypeError:
            state = torch.load(ckpt, map_location=torch.device("cpu"))
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        self.load_state_dict(state, strict=False)
        print("loaded pretrained LPIPS loss from {}".format(ckpt))

    @classmethod
    def from_pretrained(cls, name="vgg_lpips"):
        if name != "vgg_lpips":
            raise NotImplementedError
        model = cls()
        ckpt = get_ckpt_path(name)
        try:
            state = torch.load(ckpt, map_location=torch.device("cpu"), weights_only=True)
        except TypeError:
            state = torch.load(ckpt, map_location=torch.device("cpu"))
        model.load_state_dict(state, strict=False)
        return model

    def forward(self, input, target):
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True, weights_path=None):
        super(vgg16, self).__init__()
        try:
            w = models.VGG16_Weights.IMAGENET1K_V1 if bool(pretrained) else None
            m = models.vgg16(weights=w)
        except Exception:
            m = models.vgg16(pretrained=bool(pretrained))
        if weights_path is not None and str(weights_path).strip() != "":
            p = os.path.abspath(str(weights_path))
            if not os.path.exists(p):
                raise FileNotFoundError(f"VGG16 weights not found: {p}")
            try:
                state = torch.load(p, map_location=torch.device("cpu"), weights_only=True)
            except TypeError:
                state = torch.load(p, map_location=torch.device("cpu"))
            if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
                state = state["state_dict"]
            m.load_state_dict(state, strict=False)
        vgg_pretrained_features = m.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2,dim=1,keepdim=True))
    return x/(norm_factor+eps)


def spatial_average(x, keepdim=True):
    return x.mean([2,3],keepdim=keepdim)

