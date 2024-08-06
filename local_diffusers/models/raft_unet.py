import math
import copy
from random import random
from typing import Optional, List, Union
from tqdm.auto import tqdm
from functools import partial, wraps
from contextlib import contextmanager, nullcontext
from collections import namedtuple
from pathlib import Path
from diffusers.utils import BaseOutput

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch import nn, einsum
from torch.cuda.amp import autocast
from torch.special import expm1
import torchvision.transforms as T
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from dataclasses import dataclass
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange, Reduce
from core.extractor import BasicEncoder
from .imagen_unet import Unet, exists, UNet2DOutput
from core.utils.utils import bilinear_sampler, coords_grid, downflow8
from core.corr import CorrBlock
from .init_pre import predictor
from configs.parser import parse_args
try:
    from xformers.ops import memory_efficient_attention, unbind, fmha

    XFORMERS_AVAILABLE = True
except ImportError:
    print("xFormers not available")
    XFORMERS_AVAILABLE = False

try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

    FLASH_AVAILABLE = True
except ImportError:
    print("FLASH ATTENTION2 not available")
    FLASH_AVAILABLE = False
    
from rddm import ResidualDiffusion


# predefined unets, with configs lining up with hyperparameters in appendix of paper
class RAFT_Unet(Unet, ResidualDiffusion, ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, channels, channels_out, sample_size, model_cfg, add_dim=(0, 0, 324, 0), corr_index='noised_flow', **kwargs):
        default_kwargs = dict(
            channels=channels,
            channels_out=channels_out,
            sample_size=sample_size,
            dim=128,
            dim_mults=(1, 2, 4, 8),
            num_resnet_blocks=(2, 4, 8, 8),
            layer_attns=(False, False, True, True),
            layer_cross_attns=(False, False, False, False),
            attn_heads=8,
            ff_mult=2.,
            memory_efficient=True,
            add_dim=add_dim,
            corr_index=corr_index
        )
        super().__init__(**default_kwargs)

        # feature encoder
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=0.0)

        model_cfg = parse_args(model_cfg)
        self.cnet = predictor(args=model_cfg)
        print('[fnet: BasicEncoder]')
        assert self.corr_index in ['orginal', 'noised_flow', None]
        print('[corr_index: ', self.corr_index, ']')

    def forward(
            self,
            sample: torch.FloatTensor,
            noise_scheduler,
            timestep: Union[torch.Tensor, float, int],
            valid,
            class_labels: Optional[torch.Tensor] = None,
            return_dict: bool = True,
            normalize=False,
    ):
        time = timestep
        x = sample

        image1, image2, flow_gt = x[:, :3], x[:, 3:6], x[:, 6:9]
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        image1 = image1.contiguous()
        image2 = image2.contiguous()
        
        init, init_loss = self.cnet(torch.cat([image1 , image2], dim=1), flow_gt)
        res = init - flow_gt
        
        noise = torch.randn(flow_gt.shape, dtype=torch.float32).to(flow_gt.device)         
        noisy_target = noise_scheduler.q_sample(flow_gt, res, time, noise)
        x = torch.cat([image1, image2, res, noisy_target], dim=1)
                     
        fmap1, fmap2 = self.fnet([image1, image2])
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()       
        corr_fn = CorrBlock(fmap1, fmap2, radius=4)
        N, C, H, W = image1.shape
        coords1 = coords_grid(N, H // 8, W // 8, device=image1.device)
        if self.corr_index == 'orginal':
            pass
        elif normalize:
            coords1 = coords1 + x[:, 6:8, ::8, ::8] * torch.tensor([W, H]).view(1, 2, 1, 1).to(x) / 8
        else:
            coords1 = coords1 + x[:, 6:8, ::8, ::8] / 8
        coords1 = coords1.detach()
        corr = corr_fn(coords1)  # index correlation volume

        # initial convolution
        x = self.init_conv(x)           #in channel = 10
        # init conv residual

        if self.init_conv_to_final_conv_residual:
            init_conv_residual = x.clone()

        # time conditioning
        if len(time.shape) == 0:
            time = time.reshape(1).repeat(sample.shape[0])
        time = time.to(x.device)
        time = noise_scheduler.alphas_cumsum[timestep] * noise_scheduler.num_timesteps
        time_hiddens = self.to_time_hiddens(time)

        # derive time tokens

        time_tokens = self.to_time_tokens(time_hiddens)
        t = self.to_time_cond(time_hiddens)

        # add lowres time conditioning to time hiddens
        # and add lowres time tokens along sequence dimension for attention

        # text conditioning

        text_tokens = None

        # main conditioning tokens (c)

        c = time_tokens if not exists(text_tokens) else torch.cat((time_tokens, text_tokens), dim=-2)

        # normalize conditioning tokens

        c = self.norm_cond(c)

        # initial resnet block (for memory efficient unet)

        if exists(self.init_resnet_block):
            x = self.init_resnet_block(x, t)

        # go through the layers of the unet, down and up

        hiddens = []

        for ind, (pre_downsample, init_block, resnet_blocks, attn_block, post_downsample) in enumerate(self.downs):
            if exists(pre_downsample):
                x = pre_downsample(x)

            if self.add_dim[ind] > 0:
                x = torch.cat([x, corr.to(x)], dim=1)

            x = init_block(x, t, c)

            for resnet_block in resnet_blocks:
                x = resnet_block(x, t)
                hiddens.append(x)

            x = attn_block(x, c)
            hiddens.append(x)

            if exists(post_downsample):
                x = post_downsample(x)

        x = self.mid_block1(x, t, c)

        if exists(self.mid_attn):
            x = self.mid_attn(x)

        x = self.mid_block2(x, t, c)

        add_skip_connection = lambda x: torch.cat((x, hiddens.pop() * self.skip_connect_scale), dim=1)

        up_hiddens = []

        for init_block, resnet_blocks, attn_block, upsample in self.ups:
            x = add_skip_connection(x)
            x = init_block(x, t, c)

            for resnet_block in resnet_blocks:
                x = add_skip_connection(x)
                x = resnet_block(x, t)

            x = attn_block(x, c)
            up_hiddens.append(x.contiguous())
            x = upsample(x)

        # whether to combine all feature maps from upsample blocks

        x = self.upsample_combiner(x, up_hiddens)

        # final top-most residual if needed

        if self.init_conv_to_final_conv_residual:
            x = torch.cat((x, init_conv_residual), dim=1)

        if exists(self.final_res_block):
            x = self.final_res_block(x, t)

        pred_res = self.final_conv(x)
        if normalize:
            pred_res = torch.tanh(pred_res)
        # return UNet2DOutput(sample=x)
        pred_noise = self.predict_noise_from_res(noisy_target, timestep, init, pred_res)
        pred = self.q_posterior_from_res_noise(pred_res, pred_noise, pred_res, timestep)   
        
        if self.training:
            nosie_loss = valid[:, None] * (pred_res - res).abs()
            total_loss = 0.7 * init_loss +  nosie_loss           
            return total_loss, pred, pred_res, pred_noise
        else:
            return x, res