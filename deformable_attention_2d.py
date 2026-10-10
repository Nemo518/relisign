# Clone from https://github.com/lucidrains/deformable-attention
import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
# helper functions
from typing import Any, Optional, Tuple, Type
import numpy as np

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(numer, denom):
    return (numer % denom) == 0

# tensor helpers

def create_grid_like(t, dim = 0):
    h, w, device = *t.shape[-2:], t.device

    grid = torch.stack(torch.meshgrid(
        torch.arange(w, device = device),
        torch.arange(h, device = device),
    indexing = 'xy'), dim = dim)

    grid.requires_grad = False
    grid = grid.type_as(t)
    return grid

def normalize_grid(grid, dim = 1, out_dim = -1):
    # normalizes a grid to range from -1 to 1
    h, w = grid.shape[-2:]
    grid_h, grid_w = grid.unbind(dim = dim)

    grid_h = 2.0 * grid_h / max(h - 1, 1) - 1.0
    grid_w = 2.0 * grid_w / max(w - 1, 1) - 1.0

    return torch.stack((grid_h, grid_w), dim = out_dim)

def reshape_grid_1d(grid, dim = 1, out_dim = -1):
    # normalizes a grid to range from -1 to 1
    n = grid.shape[-1]
    grid_h, grid_w = grid.unbind(dim = dim)

    return torch.stack((grid_h, grid_w), dim = out_dim)

class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale

# continuous positional bias from SwinV2

class CPB(nn.Module):
    """ https://arxiv.org/abs/2111.09883v1 """

    def __init__(self, dim, *, heads, offset_groups, depth):
        super().__init__()
        self.heads = heads
        self.offset_groups = offset_groups

        self.mlp = nn.ModuleList([])

        self.mlp.append(nn.Sequential(
            nn.Linear(2, dim),
            nn.ReLU()
        ))

        for _ in range(depth - 1):
            self.mlp.append(nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU()
            ))

        self.mlp.append(nn.Linear(dim, heads // offset_groups))

    def forward(self, grid_q, grid_kv):
        device, dtype = grid_q.device, grid_kv.dtype

        if grid_q.ndim == 3:
            raise AttributeError
            grid_q = rearrange(grid_q, 'h w c -> 1 (h w) c')
        elif grid_q.ndim == 4:
            grid_q = rearrange(grid_q, 'b h w c -> b (h w) c')
        else:
            raise AttributeError
        grid_kv = rearrange(grid_kv, 'b h w c -> b (h w) c')

        pos = rearrange(grid_q, 'b i c -> b i 1 c') - rearrange(grid_kv, 'b j c -> b 1 j c')
        bias = torch.sign(pos) * torch.log(pos.abs() + 1)  # log of distance is sign(rel_pos) * log(abs(rel_pos) + 1)

        for layer in self.mlp:
            bias = layer(bias)

        bias = rearrange(bias, '(b g) i j o -> b (g o) i j', g = self.offset_groups)

        return bias


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C

def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return torch.tensor(sinusoid_table,dtype=torch.float, requires_grad=False).unsqueeze(0) 

# main class

class DeformableAttention2D(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        downsample_factor = 1,
        offset_scale = None,
        offset_groups = None,
        offset_kernel_size = 1,
        group_queries = True,
        group_key_values = True
    ):
        super().__init__()
        offset_scale = default(offset_scale, downsample_factor)
        assert offset_kernel_size >= downsample_factor, 'offset kernel size must be greater than or equal to the downsample factor'
        assert divisible_by(offset_kernel_size - downsample_factor, 2)

        offset_groups = default(offset_groups, heads)
        assert divisible_by(heads, offset_groups)

        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.offset_groups = offset_groups

        offset_dims = inner_dim // offset_groups

        self.downsample_factor = downsample_factor
        
        self.to_offsets = nn.Sequential(
            nn.Conv1d(offset_dims, offset_dims, kernel_size=1, groups = offset_dims, stride = 1, padding = 0),
            nn.GELU(),
            nn.Conv1d(offset_dims, 2, 1, bias = False),
            # Rearrange('b 1 n -> b n'),
            nn.Tanh(),
            Scale(4/6)
        )

        self.rel_pos_bias = CPB(dim // 4, offset_groups = offset_groups, heads = heads, depth = 2)

        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Conv1d(dim, inner_dim, 1, groups = offset_groups if group_queries else 1, bias = False)
        self.to_k = nn.Conv2d(dim, inner_dim, 1, groups = offset_groups if group_key_values else 1, bias = False)
        self.to_v = nn.Conv2d(dim, inner_dim, 1, groups = offset_groups if group_key_values else 1, bias = False)
        self.to_out = nn.Conv1d(inner_dim, dim, 1)

        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        # for pose
        self.pe_layer = PositionEmbeddingRandom(dim//2)
        # for rgb
        self.pos_embed = get_sinusoid_encoding_table(4*4, dim)

    def forward(self, pose_feat, rgb_feat, pose_init, return_vgrid = False):
        """
        b - batch
        h - heads
        x - height
        y - width
        d - dimension
        g - offset groups
        """

        heads, b, h, w, downsample_factor, device = self.heads, rgb_feat.shape[0], *rgb_feat.shape[-2:], self.downsample_factor, rgb_feat.device

        # queries
        # pose_feat: bt, c, n
        pose_feat_cross = rearrange(pose_feat, 'b d n -> b n d')
        rgb_feat_cross = rearrange(rgb_feat, 'b d h w -> b (h w) d')
        pose_init_cross = rearrange(pose_init, 'b d n -> b n d')

        point_embedding = self.pe_layer._pe_encoding(pose_init_cross.detach())

        kv = rgb_feat_cross + self.pos_embed.expand(b, -1, -1).type_as(rgb_feat_cross).to(rgb_feat_cross.device).clone().detach()
        
        pose_feat_cross, _ = self.cross_attn(pose_feat_cross + point_embedding, 
                                          kv, kv)

        pose_feat_cross = rearrange(pose_feat_cross, 'b n d -> b d n')

        q = self.to_q(pose_feat + pose_feat_cross)

        # calculate offsets - offset MLP shared across all groups

        group_1d = lambda t: rearrange(t, 'b (g d) n -> (b g) d n', g = self.offset_groups)
        group_2d = lambda t: rearrange(t, 'b (g d) ... -> (b g) d ...', g = self.offset_groups)

        grouped_queries = group_1d(q)
        
        offsets = self.to_offsets(grouped_queries)

        # calculate grid + offsets

        # pose_init --> [0, 1]
        grid = pose_init[:,None].repeat(1, self.offset_groups, 1, 1) * 2 - 1
        # grid --> [-1, 1]
        grid = grid.reshape(-1, *pose_init.shape[-2:])

        vgrid = grid + offsets
        vgrid_scaled = reshape_grid_1d(vgrid)[:,None]
        # vgrid_scaled = normalize_grid(vgrid)

        kv_feats = F.grid_sample(
            group_2d(rgb_feat),
            vgrid_scaled,
        mode = 'bilinear', padding_mode = 'zeros', align_corners = False)

        kv_feats = rearrange(kv_feats, '(b g) d ... -> b (g d) ...', b = b)

        # derive key / values
        k, v = self.to_k(kv_feats), self.to_v(kv_feats)

        # scale queries
        q = q * self.scale

        # split out heads
        q, k, v = map(lambda t: rearrange(t, 'b (h d) ... -> b h (...) d', h = heads), (q, k, v))

        # query / key similarity
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # relative positional bias
        grid = reshape_grid_1d(grid)[:,None]

        rel_pos_bias = self.rel_pos_bias(grid, vgrid_scaled)
        sim = sim + rel_pos_bias

        # numerical stability
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()

        # attention
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # aggregate and combine heads
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b (h d) n')
        out = self.to_out(out)

        if return_vgrid:
            return out, vgrid

        return out
