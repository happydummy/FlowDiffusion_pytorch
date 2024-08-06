from .extractor import ResNetFPN
from .layer import conv1x1, conv3x3
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class predictor(nn.Module):
    def __init__(self, args):
        super(predictor, self).__init__()
        self.args = args
        self.use_var = True
        self.var_max = 10
        self.var_min = 0
        self.cnet = ResNetFPN(args, input_dim=6, output_dim=2 * self.args['dim'], norm_layer=nn.BatchNorm2d, init_weight=True)
        self.init_conv = conv3x3(2 * args['dim'], 2 * args['dim'])
        self.upsample_weight = nn.Sequential(
            nn.Conv2d(args['dim'], args['dim'] * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(args['dim'] * 2, 64 * 9, 1, padding=0)
        )
        self.flow_head = nn.Sequential(
            # flow(2) + weight(2) + log_b(2)
            nn.Conv2d(args['dim'], 2 * args['dim'], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * args['dim'], 6, 3, padding=1)
        )
    
    def upsample_data(self, flow, info, mask):
        """ Upsample [H/8, W/8, C] -> [H, W, C] using convex combination """
        N, C, H, W = info.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_info = F.unfold(info, [3, 3], padding=1)
        up_info = up_info.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        up_info = torch.sum(mask * up_info, dim=2)
        up_info = up_info.permute(0, 1, 4, 2, 5, 3)
        
        return up_flow.reshape(N, 2, 8*H, 8*W), up_info.reshape(N, C, 8*H, 8*W)
        
    def forward(self, x, flow_gt):
        cnet = self.cnet(x)
        cnet = self.init_conv(cnet)
        net, context = torch.split(cnet, [self.args['dim'], self.args['dim']], dim=1)
        flow_update = self.flow_head(net)
        weight_update = .25 * self.upsample_weight(net)
        flow_8x = flow_update[:, :2]
        info_8x = flow_update[:, 2:]
        flow_up, info_up = self.upsample_data(flow_8x, info_8x, weight_update)
        
        if self.training:
            if not self.use_var:
                    var_max = var_min = 0
            else:
                var_max = self.var_max
                var_min = self.var_min
            raw_b = info_up[:, 2:]
            log_b = torch.zeros_like(raw_b)
            weight = info_up[:, :2]
            # Large b Component                
            log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=var_max)
            # Small b Component
            log_b[:, 1] = torch.clamp(raw_b[:, 1], min=var_min, max=0)
            # term2: [N, 2, m, H, W]
            term2 = ((flow_gt - flow_up).abs().unsqueeze(2)) * (torch.exp(-log_b).unsqueeze(1))
            # term1: [N, m, H, W]
            term1 = weight - math.log(2) - log_b
            init_loss = torch.logsumexp(weight, dim=1, keepdim=True) - torch.logsumexp(term1.unsqueeze(1) - term2, dim=2)

        return flow_up, init_loss
        
