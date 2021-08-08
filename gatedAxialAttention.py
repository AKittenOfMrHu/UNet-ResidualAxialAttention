import math
import torch
import torch.nn as nn


def to_qkv(in_channels, out_channels):
    layer =  nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
        nn.ReLU(),
        nn.BatchNorm1d(out_channels)
    )
    return layer



class GatedAxialAttentionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, head_dim, pos_dim, h_w='h'):
        super(GatedAxialAttentionBlock, self).__init__()

        self.h_w = h_w

        self.to_qkv = to_qkv(in_planes, out_planes*2)
        self.head_dim = head_dim
        self.head_planes = out_planes*2//head_dim
        self.qk_dim = self.head_planes//4
        self.v_dim = self.head_planes//2
        self.pos_dim = pos_dim

        self.gate_qr = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.gate_kr = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.gate_v_pos = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.gate_v = nn.Parameter(torch.tensor([1.]), requires_grad=True)

        self.pos_relative = nn.Parameter(torch.randn(self.head_planes, pos_dim*2-1), requires_grad=True)
        nn.init.normal_(self.pos_relative, mean=0, std=1/math.sqrt(in_planes))
        q_index = torch.arange(0, pos_dim)[:, None]
        k_index = torch.arange(0, pos_dim)[None, :]
        index_select = q_index - k_index + pos_dim - 1
        self.index_select = index_select.view(-1).to(self.pos_relative.device)

        self.norm_attention = nn.BatchNorm2d(head_dim*3)
        self.norm_v = nn.BatchNorm2d(head_dim*2)

    def forward(self, x):
        if self.h_w.upper() == 'H':
            x = x.permute([0, 3, 1, 2]) # B, W, C, H
        else:
            x = x.permute([0, 2, 1, 3]) # B, H, C, W
        b, w, c, h = x.shape

        # step 1: get q, k, v, q_pos, k_pos and v_dim
        x = x.reshape(b*w, c, h)
        qkv = self.to_qkv(x).reshape(-1, self.head_dim, self.qk_dim*2+self.v_dim, h) # B*W, C2, H
        #print(f'qkv: {qkv.shape}, qk_dim: {self.qk_dim}, v_dim: {self.v_dim}')
        q, k, v = torch.split(qkv, [self.qk_dim, self.qk_dim, self.v_dim], dim=2)

        qk_pos = torch.index_select(self.pos_relative, dim=1, index=self.index_select).reshape(-1, self.pos_dim, self.pos_dim)
        q_pos, k_pos, v_pos = qk_pos.split([self.qk_dim, self.qk_dim, self.v_dim], dim=0) # qk_dim/v_dim, pos_dim, pos_dim

        # step 2: q, k with relative position
        qk = torch.einsum('bhcp,bhci->bhpi', q, k)
        qr = torch.einsum('bhcp,cpi->bhpi', q, q_pos)
        kr = torch.einsum('bhcp,cpi->bhpi', k, k_pos)
        qr = torch.mul(qr, self.gate_qr)
        kr = torch.mul(kr, self.gate_kr)

        attention = self.norm_attention(torch.cat([qk, qr, kr], dim=1)).reshape(-1, 3, self.head_dim, h, h).sum(dim=1)
        attention = torch.softmax(attention, dim=-1)

        # step 3: update v, v_pos with attention
        v_pos = torch.mul(v_pos, self.gate_v_pos)
        v = torch.mul(v, self.gate_v)

        att_v = torch.einsum('bhpi,bhci->bhcp', attention, v)
        att_vp = torch.einsum('bhpi,cpi->bhcp', attention, v_pos)

        new_embd = self.norm_v(torch.cat([att_v, att_vp], dim=1)).reshape(b, w, 2, self.head_dim*self.v_dim, h).sum(dim=2)

        # step 4: get updated feature maps
        if self.h_w.upper() == 'H':
            new_embd = torch.permute(new_embd, [0, 2, 3, 1])
        else:
            new_embd = torch.permute(new_embd, [0, 2, 1, 3])

        return new_embd