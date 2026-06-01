from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import repeat, rearrange
from typing import Optional, List, Tuple

import step1x3d_geometry
from step1x3d_geometry.utils.base import BaseModule
from step1x3d_geometry.utils.typing import *
from step1x3d_geometry.utils.misc import get_world_size, get_device
from step1x3d_geometry.utils.checkpoint import checkpoint


import os


def safe_quat_normalization(q: torch.Tensor) -> torch.Tensor:
    """Safely normalize a quaternion tensor, handling potential division by zero."""
    q_norm = torch.norm(q, dim=-1, keepdim=True)
    
    # 使用更高效的方式：直接检查norm是否太小，避免NaN检查
    eps = 1e-8
    small_norm_mask = q_norm < eps
    
    # 对于正常情况，直接归一化
    q_normalized = q / torch.clamp(q_norm, min=eps)
    
    # 只对小norm的情况替换为单位四元数
    if small_norm_mask.any():
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=q.device, dtype=q.dtype)
        q_normalized = torch.where(small_norm_mask, identity_quat, q_normalized)
    
    return q_normalized


class PointEmbed(nn.Module):
    """
    基于3D坐标的位置编码器 - 改进版，支持坐标感知的位置编码
    
    与原始的基于索引的embedding不同，这个实现：
    1. 基于实际3D坐标计算位置编码，而不是顶点索引
    2. 具有几何不变性：相同坐标→相同编码
    3. 支持任意顶点排列：顶点顺序改变不影响结果
    4. 使用sinusoidal编码保证空间关系的连续性
    """
    def __init__(self, hidden_dim=768):
        super().__init__()
        
        assert hidden_dim % 3 == 0, "hidden_dim must be divisible by 3 for x,y,z coordinates"
        self.embedding_dim = hidden_dim // 3 // 2  # 每个坐标轴的sin/cos维度
        
        # 创建频率向量 - 数值稳定版本
        omega = np.arange(self.embedding_dim, dtype=np.float64)
        omega = omega / self.embedding_dim  # [0, 1/N, 2/N, ..., 1]
        # 使用更温和的频率范围：从1到0.01而不是1到0.0001
        omega = 1. / (100**omega)  # 从1到0.01，更安全的范围
        self.register_buffer('omega', torch.from_numpy(omega))
        
    def forward(self, coords):
        """
        Args:
            coords: [B, N, 3] 或 [B, T, N, 3] - 3D坐标
        Returns:
            embed: 对应维度的位置编码
        """
        input_type = coords.dtype
        original_shape = coords.shape
        
        # 处理不同维度的输入
        if len(coords.shape) == 3:
            B, N, D = coords.shape
            pos = coords.reshape(B*N, D)  # (B*N, 3)
        elif len(coords.shape) == 4:
            B, T, N, D = coords.shape
            pos = coords.reshape(B*T*N, D)  # (B*T*N, 3)
        else:
            raise ValueError(f"Unsupported input shape: {coords.shape}")
        
        # 分别对x,y,z坐标计算sinusoidal编码
        from einops import einsum
        emb_x = einsum(pos[:, 0], self.omega, 'm,d->m d')  # (B*N, D/6)
        emb_y = einsum(pos[:, 1], self.omega, 'm,d->m d')  # (B*N, D/6)
        emb_z = einsum(pos[:, 2], self.omega, 'm,d->m d')  # (B*N, D/6)
        
        # 应用sin/cos变换
        emb_x = torch.cat([torch.sin(emb_x), torch.cos(emb_x)], dim=1)  # (B*N, D/3)
        emb_y = torch.cat([torch.sin(emb_y), torch.cos(emb_y)], dim=1)  # (B*N, D/3)
        emb_z = torch.cat([torch.sin(emb_z), torch.cos(emb_z)], dim=1)  # (B*N, D/3)
        
        # 拼接所有坐标的编码
        embed = torch.cat([emb_x, emb_y, emb_z], dim=1)  # (B*N, D)
        
        # 重塑回原始形状
        if len(original_shape) == 3:
            embed = embed.reshape(B, N, -1)
        elif len(original_shape) == 4:
            embed = embed.reshape(B, T, N, -1)
            
        return embed.to(input_type)


def _build_mlp(in_dim, hidden_dim, out_dim, num_layers):
    """构建一个指定层数的MLP"""
    layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for _ in range(num_layers - 2):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)

def _build_cross_attention_layers(width, heads, num_layers):
    """构建多层Cross-attention模块"""
    cross_attn_layers = nn.ModuleList([
        nn.MultiheadAttention(width, heads, batch_first=True) 
        for _ in range(num_layers)
    ])
    ln_layers = nn.ModuleList([
        nn.LayerNorm(width) 
        for _ in range(num_layers)
    ])
    return cross_attn_layers, ln_layers

class StaticParamDecoder(nn.Module):
    """静态参数解码器 - 预测高斯参数"""
    def __init__(self, width, heads=8, num_layers=2, num_cross_attn_layers=1):
        super().__init__()
        self.num_cross_attn_layers = num_cross_attn_layers
        
        # 多层Cross-attention
        self.cross_attn_layers, self.ln_layers = _build_cross_attention_layers(
            width, heads, num_cross_attn_layers
        )
        
        # 静态参数MLP
        self.static_mlp = _build_mlp(
            in_dim=width,
            hidden_dim=width * 2,
            out_dim=10, # center_offset(3) + scale(3) + quat(4)
            num_layers=num_layers
        )
        
        # 初始化：输出接近零，让高斯参数从合理默认值开始
        nn.init.constant_(self.static_mlp[-1].weight, 0.0)
        nn.init.constant_(self.static_mlp[-1].bias, 0.0)
    
    def forward(self, anchor_features, global_features):
        """
        Args:
            anchor_features: [B, K, width] - V0_att_anchor
            global_features: [B, N, width] - V0_att
        Returns:
            gaussians: [B, K, 10] - center_offset(3) + scale(3) + quat(4)
        """
        # 多层Cross-attention: anchor作为query，global作为key/value
        current_features = anchor_features
        for i in range(self.num_cross_attn_layers):
            attn_output, _ = self.cross_attn_layers[i](
                current_features, global_features, global_features
            )
            # 残差连接 + LayerNorm
            current_features = self.ln_layers[i](attn_output + current_features)
        enhanced_features = current_features
        
        # 预测高斯参数
        static_params = self.static_mlp(enhanced_features)  # [B, K, 10]
        
        # 分离参数并添加合理的默认值
        center_offset = static_params[:, :, :3] * 0.01  # 小的偏移
        scale_delta = static_params[:, :, 3:6]
        # 新的scale设计：初始化为0.05，范围(0, 2)
        scale = 0.05 * torch.exp(scale_delta)  # 使用exp确保正值，初始为0.05
        quat_delta = static_params[:, :, 6:]
        
        # 四元数从单位四元数开始
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=quat_delta.device, dtype=quat_delta.dtype)
        orientation = identity_quat + quat_delta
        orientation = safe_quat_normalization(orientation)
        
        # 限制尺度范围 (0, 2.0)
        min_scale = torch.tensor(1e-6, device=scale.device, dtype=scale.dtype)
        max_scale = torch.tensor(2.0, device=scale.device, dtype=scale.dtype)
        scale = torch.clamp(scale, min=min_scale, max=max_scale)
        
        gaussians = torch.cat([center_offset, scale, orientation], dim=-1)  # [B, K, 10]
        return gaussians



class BaseVAEEncoder(nn.Module):
    """基础VAE编码器 - 通用设计"""
    def __init__(self, width, embed_dim, heads=8, name="base", num_cross_attn_layers=1):
        super().__init__()
        self.name = name
        self.num_cross_attn_layers = num_cross_attn_layers
        
        # 多层Cross-attention
        self.cross_attn_layers, self.ln_layers = _build_cross_attention_layers(
            width, heads, num_cross_attn_layers
        )
        
        # VAE编码MLP
        self.vae_mlp = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, embed_dim * 2)  # mu + log_sigma
        )
        
        # 初始化
        nn.init.xavier_uniform_(self.vae_mlp[-1].weight, gain=0.1)
        nn.init.constant_(self.vae_mlp[-1].bias, 0.0)
    
    def _compute_vae_params(self, enhanced_features, kl_weight=1.0):
        """通用VAE参数计算 - 支持KL annealing"""
        vae_params = self.vae_mlp(enhanced_features)
        mu, log_sigma = torch.chunk(vae_params, 2, dim=-1)
        
        # 数值稳定性
        log_sigma = torch.clamp(log_sigma, min=-20.0, max=20.0)
        
        # 采样
        epsilon = torch.randn_like(mu)
        z = mu + torch.exp(0.5 * log_sigma) * epsilon
        
        # 改进的KL损失计算
        sigma = torch.exp(log_sigma)
        sigma = torch.clamp(sigma, min=1e-8, max=1e8)
        
        # 按维度计算KL散度（更精确）
        kl_div_per_dim = 0.5 * (mu**2 + sigma - log_sigma - 1)  # [B, ..., dim]
        
        # 应用KL annealing权重
        kl_loss = kl_weight * torch.mean(kl_div_per_dim)
        
        # 数值稳定性检查
        if torch.isnan(kl_loss) or torch.isinf(kl_loss):
            kl_loss = torch.tensor(0.0, device=z.device, dtype=torch.float32)
        
        # 返回更多信息用于调试
        return mu, log_sigma, z, kl_loss, torch.mean(kl_div_per_dim).detach()


class DynamicVAEEncoder(BaseVAEEncoder):
    """动态参数的VAE编码器 - 局部变换"""
    def __init__(self, width, embed_dim, heads=8, num_cross_attn_layers=1):
        super().__init__(width, embed_dim, heads, name="dynamic", num_cross_attn_layers=num_cross_attn_layers)
    
    def forward(self, anchor_features, global_features, kl_weight=1.0):
        """
        Args:
            anchor_features: [B, K, T-1, width] - VT_att_anchor
            global_features: [B, T-1, N, width] - V_delta_att
            kl_weight: KL annealing权重
        Returns:
            mu: [B, K, T-1, embed_dim]
            log_sigma: [B, K, T-1, embed_dim] 
            z: [B, K, T-1, embed_dim]
            kl_loss: scalar
        """
        B, K, T_minus_1, width = anchor_features.shape
        _, _, N, _ = global_features.shape
        
        # 多层Cross-attention增强特征 - 锚点作为query，全局作为key/value
        anchor_flat = anchor_features.permute(0, 2, 1, 3).reshape(B * T_minus_1, K, width)
        global_flat = global_features.reshape(B * T_minus_1, N, width)
        
        # 多层Cross-attention
        current_features = anchor_flat
        for i in range(self.num_cross_attn_layers):
            attn_output, _ = self.cross_attn_layers[i](
                current_features, global_flat, global_flat
            )
            # 残差连接 + LayerNorm
            current_features = self.ln_layers[i](attn_output + current_features)
        enhanced_flat = current_features
        
        # VAE编码 - 输出局部变换参数
        mu, log_sigma, z, kl_loss, _ = self._compute_vae_params(enhanced_flat, kl_weight)  # [B*T-1, K, embed_dim*2]
        
        # 重塑回时间维度
        mu = mu.reshape(B, T_minus_1, K, -1).permute(0, 2, 1, 3)  # [B, K, T-1, embed_dim]
        log_sigma = log_sigma.reshape(B, T_minus_1, K, -1).permute(0, 2, 1, 3)  # [B, K, T-1, embed_dim]
        z = z.reshape(B, T_minus_1, K, -1).permute(0, 2, 1, 3)  # [B, K, T-1, embed_dim]
        
        return mu, log_sigma, z, kl_loss


class DynamicVAEDecoder(nn.Module):
    """动态参数的VAE解码器"""
    def __init__(self, embed_dim, width, heads=8, num_layers=2):
        super().__init__()
        
        # Latent到特征的投影
        self.latent_proj = nn.Linear(embed_dim, width)
        
        # 变换参数预测
        self.transform_mlp = _build_mlp(
            in_dim=width,
            hidden_dim=width * 2,
            out_dim=7, # quat(4) + trans(3)
            num_layers=num_layers
        )
        
        # 初始化：输出接近零
        nn.init.constant_(self.transform_mlp[-1].weight, 0.0)
        nn.init.constant_(self.transform_mlp[-1].bias, 0.0)
    
    def forward(self, z):
        """
        Args:
            z: [B, K, T-1, embed_dim] - 动态latent
        Returns:
            transforms: [B, K, T-1, 7] - quat(4) + trans(3)
        """
        # 投影latent到特征空间
        features = self.latent_proj(z)  # [B, K, T-1, width]
        
        # 预测变换参数
        transform_params = self.transform_mlp(features)  # [B, K, T-1, 7]
        
        # 分离四元数和平移
        quat_delta = transform_params[:, :, :, :4]
        trans_delta = transform_params[:, :, :, 4:]
        
        # 四元数从单位四元数开始
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=quat_delta.device, dtype=quat_delta.dtype)
        quat = identity_quat + quat_delta
        quat = safe_quat_normalization(quat)
        
        # 平移限制范围
        max_trans = torch.tensor(1.0, device=trans_delta.device, dtype=trans_delta.dtype)
        trans = torch.clamp(trans_delta, min=-max_trans, max=max_trans)
        
        transforms = torch.cat([quat, trans], dim=-1)
        return transforms


class RootVAEEncoder(BaseVAEEncoder):
    """根变换的VAE编码器 - 全局变换"""
    def __init__(self, width, embed_dim, heads=8, num_cross_attn_layers=1):
        super().__init__(width, embed_dim, heads, name="root", num_cross_attn_layers=num_cross_attn_layers)
    
    def forward(self, anchor_features, global_features, kl_weight=1.0):
        """
        Args:
            anchor_features: [B, K, T-1, width] - VT_att_anchor
            global_features: [B, T-1, N, width] - V_delta_att (与DynamicVAEEncoder保持一致)
            kl_weight: KL annealing权重
        Returns:
            mu: [B, T-1, embed_dim]
            log_sigma: [B, T-1, embed_dim]
            z: [B, T-1, embed_dim]
            kl_loss: scalar
        """
        B, K, T_minus_1, width = anchor_features.shape
        _, _, N, _ = global_features.shape
        
        # 重塑为批量处理
        anchor_flat = anchor_features.permute(0, 2, 1, 3).reshape(B * T_minus_1, K, width)
        global_flat = global_features.reshape(B * T_minus_1, N, width)
        
        # 步骤1：从锚点特征聚合得到全局查询 (按用户建议)
        # 最简单的方法：直接取均值
        global_query = torch.mean(anchor_flat, dim=1, keepdim=True)  # [B*T-1, 1, width]
        
        # 步骤2：多层Cross-Attention
        current_features = global_query
        for i in range(self.num_cross_attn_layers):
            attn_output, _ = self.cross_attn_layers[i](
                current_features, global_flat, global_flat
            )
            # 残差连接 + LayerNorm
            current_features = self.ln_layers[i](attn_output + current_features)
        enhanced_feat = current_features
        
        # VAE编码 - 输出全局变换参数
        mu, log_sigma, z, kl_loss, _ = self._compute_vae_params(enhanced_feat.squeeze(1), kl_weight)  # [B*T-1, embed_dim*2]
        
        # 重塑回时间维度
        mu = mu.reshape(B, T_minus_1, -1)  # [B, T-1, embed_dim]
        log_sigma = log_sigma.reshape(B, T_minus_1, -1)  # [B, T-1, embed_dim]
        z = z.reshape(B, T_minus_1, -1)  # [B, T-1, embed_dim]
        
        return mu, log_sigma, z, kl_loss


class RootVAEDecoder(nn.Module):
    """根变换的VAE解码器"""
    def __init__(self, embed_dim, width, heads=8, num_layers=2):
        super().__init__()
        
        # Latent到特征的投影
        self.latent_proj = nn.Linear(embed_dim, width)
        
        # 根变换预测
        self.root_mlp = _build_mlp(
            in_dim=width,
            hidden_dim=width,
            out_dim=7, # quat(4) + trans(3)
            num_layers=num_layers
        )
        
        # 初始化：输出接近零
        nn.init.constant_(self.root_mlp[-1].weight, 0.0)
        nn.init.constant_(self.root_mlp[-1].bias, 0.0)
    
    def forward(self, z):
        """
        Args:
            z: [B, T-1, embed_dim] - 根变换latent
        Returns:
            root_quat: [B, T-1, 4]
            root_trans: [B, T-1, 3]
        """
        # 投影latent到特征空间
        features = self.latent_proj(z)  # [B, T-1, width]
        
        # 预测根变换
        root_params = self.root_mlp(features)  # [B, T-1, 7]
        
        # 分离四元数和平移
        quat_delta = root_params[:, :, :4]
        trans_delta = root_params[:, :, 4:]
        
        # 四元数从单位四元数开始
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=quat_delta.device, dtype=quat_delta.dtype)
        root_quat = identity_quat + quat_delta
        root_quat = safe_quat_normalization(root_quat)
        
        # 平移限制范围
        max_trans = torch.tensor(2.0, device=trans_delta.device, dtype=trans_delta.dtype)
        root_trans = torch.clamp(trans_delta, min=-max_trans, max=max_trans)
        
        return root_quat, root_trans


class TemporalTransformerBlock(nn.Module):
    """Temporal self-attention across the time dimension.

    Operates on tensors of shape [B, K, T, C] (bone-level) or [B, T, C] (root-level).
    Reshapes so that T becomes the sequence dimension, applies N transformer layers,
    then reshapes back. Includes learnable temporal position embeddings.
    """

    def __init__(self, dim: int, heads: int = 8, num_layers: int = 2,
                 max_time_steps: int = 64, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.max_time_steps = max_time_steps

        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, max_time_steps, dim)
        )
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.ln_out = nn.LayerNorm(dim)

        self._init_weights()

    def _init_weights(self):
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, K, T, C] (bone-level) or [B, T, C] (root-level)
        Returns:
            same shape as input, with temporal context mixed in
        """
        if x.ndim == 4:
            B, K, T, C = x.shape
            x_flat = x.reshape(B * K, T, C)
        elif x.ndim == 3:
            B, T, C = x.shape
            K = None
            x_flat = x
        else:
            raise ValueError(f"Expected 3D or 4D input, got {x.ndim}D")

        x_flat = x_flat + self.temporal_pos_embed[:, :T, :]
        x_out = self.transformer(x_flat)
        x_out = self.ln_out(x_out + x_flat)

        if K is not None:
            x_out = x_out.reshape(B, K, T, C)
        return x_out



class TopologyAwareAttention(nn.Module):
    """拓扑感知自注意力模块"""
    def __init__(
        self,
        width: int,
        heads: int,
        k_neighbors: int = 16,
        init_scale: float = 0.25,
        qkv_bias: bool = True,
        qk_norm: bool = True,
    ):
        super().__init__()
        self.width = width
        self.heads = heads
        self.k_neighbors = k_neighbors
        self.dim_head = width // heads
        # separate Q, K, V projections for sparse neighbor attention
        self.to_q = nn.Linear(width, width, bias=qkv_bias)
        self.to_k = nn.Linear(width, width, bias=qkv_bias)
        self.to_v = nn.Linear(width, width, bias=qkv_bias)
        self.to_out = nn.Linear(width, width)
        
        # optional normalization
        self.q_norm = nn.LayerNorm(self.dim_head) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.dim_head) if qk_norm else nn.Identity()
        
        # 初始化
        for lin in (self.to_q, self.to_k, self.to_v, self.to_out):
            nn.init.normal_(lin.weight, std=init_scale)
            if lin.bias is not None:
                nn.init.constant_(lin.bias, 0.0)

    def forward(self, x: torch.Tensor, neighbor_idx: Optional[torch.Tensor] = None, context: Optional[torch.Tensor] = None):
        """
        Args:
            x: [B, N, width]
            neighbor_idx: [B, N, k_neighbors] - 邻居索引列表
            context: [B, M, width] - 上下文特征（用于交叉注意力）
        """
        B, N, _ = x.shape
        heads, dim_head = self.heads, self.dim_head

        q = self.to_q(x).view(B, N, heads, dim_head)
        q = self.q_norm(q)

        if context is not None:
            # Dense cross-attention
            k = self.to_k(context).view(B, context.shape[1], heads, dim_head)
            v = self.to_v(context).view(B, context.shape[1], heads, dim_head)
            k = self.k_norm(k)

            scale = 1 / math.sqrt(dim_head)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)

            sim = torch.einsum('bhid,bhjd->bhij', q, k) * scale
            attn = sim.softmax(dim=-1)
            out = torch.einsum('bhij,bhjd->bhid', attn, v)
            out = out.permute(0, 2, 1, 3).reshape(B, N, self.width)
            return self.to_out(out)
        else:
            # Sparse self-attention
            k = self.to_k(x).view(B, N, heads, dim_head)
            k = self.k_norm(k)
            v = self.to_v(x).view(B, N, heads, dim_head)
            
            K = neighbor_idx.shape[-1]
            
            k_flat = k.reshape(B * N, heads, dim_head)
            v_flat = v.reshape(B * N, heads, dim_head)

            batch_offset = torch.arange(B, device=x.device).view(B, 1, 1) * N
            flat_neighbor_idx = (neighbor_idx + batch_offset).view(-1)

            k_neigh_flat = k_flat[flat_neighbor_idx]
            v_neigh_flat = v_flat[flat_neighbor_idx]

            k_neigh = k_neigh_flat.view(B, N, K, heads, dim_head)
            v_neigh = v_neigh_flat.view(B, N, K, heads, dim_head)

            scores = (q.unsqueeze(2) * k_neigh).sum(-1) / math.sqrt(dim_head)
            # 数值稳定性：限制注意力分数的范围
            min_score = torch.tensor(-100.0, device=scores.device, dtype=scores.dtype)
            max_score = torch.tensor(100.0, device=scores.device, dtype=scores.dtype)
            scores = torch.clamp(scores, min=min_score, max=max_score)
            weights = F.softmax(scores, dim=2)
            attn_out = (weights.unsqueeze(-1) * v_neigh).sum(2).view(B, N, self.width)
            return self.to_out(attn_out)


class TopologyAwareEncoder(nn.Module):
    """双路径拓扑感知编码器 - 几何路径 + 运动路径 + 层次化采样"""
    def __init__(
        self,
        input_dim: int = 3,
        width: int = 768,
        heads: int = 12,
        layers: int = 8,
        num_tokens: int = 64,  # 最终骨骼数量
        embed_dim: int = 64,
        init_scale: float = 0.25,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        use_flash: bool = False,
        use_checkpoint: bool = False,
        motion_layers: int = 4,  # 运动路径层数
        num_vertices: int = 5000, # 顶点数量
        num_frames: int = 30, # 帧数
    ):
        super().__init__()
        self.width = width
        self.num_tokens = num_tokens
        self.motion_layers = motion_layers
        self.use_checkpoint = use_checkpoint
        self.num_vertices = num_vertices
        self.num_frames = num_frames
        
        # ==================== 路径1: 静态几何编码 ====================
        # V0 输入投影和位置嵌入
        self.geometry_input_proj = nn.Linear(input_dim, width)
        # 恢复简单MLP - 事实证明简单最有效
        self.geometry_pos_encoder = nn.Sequential(
            nn.Linear(input_dim, width // 2),
            nn.ReLU(),
            nn.Linear(width // 2, width)
        )
        
        # 几何路径的拓扑感知自注意力层
        self.geometry_attention_layers = nn.ModuleList([
            TopologyAwareAttention(
                width=width,
                heads=heads,
                k_neighbors=16,
                init_scale=init_scale,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
            ) for _ in range(layers)
        ])
        
        self.geometry_ln_layers = nn.ModuleList([
            nn.LayerNorm(width) for _ in range(layers)
        ])
        
        self.geometry_mlp_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(width, width * 4),
                nn.GELU(),
                nn.Linear(width * 4, width)
            ) for _ in range(layers)
        ])
        
        self.geometry_ln_mlp_layers = nn.ModuleList([
            nn.LayerNorm(width) for _ in range(layers)
        ])
        
        # ==================== 路径2: 纯运动编码 ====================
        # V_delta 特征投影和位置嵌入
        self.motion_input_proj = nn.Linear(input_dim, width)
        # 按用户建议：位置编码基于canonical shape (V0)，在时间维度复制
        # 优势：1) 保持顶点身份一致性 2) 时空信息分离 3) 更清晰的设计
        
        # 时间编码：区分不同时间步
        self.motion_temporal_embed = nn.Parameter(torch.empty(1, self.num_frames - 1, 1, self.width))
        nn.init.trunc_normal_(self.motion_temporal_embed, std=0.02)
        
        # 运动路径的拓扑感知自注意力层
        self.motion_attention_layers = nn.ModuleList([
            TopologyAwareAttention(
                width=width,
                heads=heads,
                k_neighbors=16,
                init_scale=init_scale,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
            ) for _ in range(motion_layers)
        ])
        
        self.motion_ln_layers = nn.ModuleList([
            nn.LayerNorm(width) for _ in range(motion_layers)
        ])
        
        self.motion_mlp_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(width, width * 4),
                nn.GELU(),
                nn.Linear(width * 4, width)
            ) for _ in range(motion_layers)
        ])
        
        self.motion_ln_mlp_layers = nn.ModuleList([
            nn.LayerNorm(width) for _ in range(motion_layers)
        ])
        
        # 注意：解码器将在MeshMotionVAE中实例化，这里只负责编码

        # 自定义初始化以提高数值稳定性
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)  # 小方差初始化
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)  # 偏置为0
        elif isinstance(m, nn.TransformerEncoderLayer):
            for p in m.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p, gain=0.1)  # 小增益Xavier初始化

    def _forward(self, vertices: torch.Tensor, neighbor_idx: Optional[torch.Tensor]=None):
        """
        双路径编码 + FPS采样
        Args:
            vertices: [B, T, N, 3] - dynamic mesh sequence
        Returns:
            V0_att: [B, N, width] - 静态几何特征
            V_delta_att: [B, T-1, N, width] - 动态运动特征
            V0_att_anchor: [B, K, width] - 锚点几何特征
            VT_att_anchor: [B, T-1, K, width] - 锚点运动特征
            anchor_coords: [B, K, 3] - 锚点坐标
            anchor_indices: [B, K] - 锚点索引
        """
        B, T, N, _ = vertices.shape
        K = self.num_tokens
        
        if neighbor_idx is None:
            raise ValueError("neighbor_idx is required")

        # ==================== 路径1: 静态几何编码 ====================
        V0 = vertices[:, 0]  # [B, N, 3]
        
        # 几何特征投影和位置编码
        V0_proj = self.geometry_input_proj(V0)  # [B, N, width]
        # 使用基于坐标的位置编码
        V0_pos_embed = self.geometry_pos_encoder(V0)  # [B, N, width]
        V0_encoded = V0_proj + V0_pos_embed
        
        # 几何路径自注意力 (per-layer checkpointing)
        V0_att = V0_encoded
        for attn, ln_attn, mlp, ln_mlp in zip(
            self.geometry_attention_layers, self.geometry_ln_layers, 
            self.geometry_mlp_layers, self.geometry_ln_mlp_layers
        ):
            if self.use_checkpoint:
                V0_att = V0_att + torch.utils.checkpoint.checkpoint(
                    attn, ln_attn(V0_att), neighbor_idx, use_reentrant=False)
                V0_att = V0_att + torch.utils.checkpoint.checkpoint(
                    mlp, ln_mlp(V0_att), use_reentrant=False)
            else:
                V0_att = V0_att + attn(ln_attn(V0_att), neighbor_idx)
                V0_att = V0_att + mlp(ln_mlp(V0_att))

        # ==================== 路径2: 纯运动编码 ====================
        V_delta = vertices[:, 1:] - vertices[:, :-1]  # [B, T-1, N, 3]
        B, T_minus_1, N, _ = V_delta.shape
        
        # 运动特征投影和位置编码
        V_delta_reshaped = V_delta.reshape(B * T_minus_1, N, 3)
        V_delta_proj = self.motion_input_proj(V_delta_reshaped)
        V_delta_proj_reshaped = V_delta_proj.reshape(B, T_minus_1, N, self.width)
        
        # 按用户建议：位置编码基于canonical shape (V0)，保持顶点身份一致性
        # 1. 复用V0的位置编码（避免重复计算）
        # 在时间维度复制位置编码
        spatial_embed = V0_pos_embed.unsqueeze(1).expand(-1, T_minus_1, -1, -1)  # [B, T-1, N, width]
        # 2. 添加时间编码区分不同时间步
        time_embed = self.motion_temporal_embed.expand(B, T_minus_1, N, self.width)  # [B, T-1, N, width]
        
        V_delta_encoded = V_delta_proj_reshaped + spatial_embed + time_embed
        
        # 运动路径自注意力（逐帧处理）
        V_delta_flat = V_delta_encoded.reshape(B * T_minus_1, N, self.width)
        neighbor_idx_expanded = neighbor_idx.unsqueeze(1).expand(-1, T_minus_1, -1, -1).reshape(B * T_minus_1, N, -1)
        
        # 运动路径自注意力 (per-layer checkpointing — this is the memory bottleneck)
        V_delta_att = V_delta_flat
        for attn, ln_attn, mlp, ln_mlp in zip(
            self.motion_attention_layers, self.motion_ln_layers,
            self.motion_mlp_layers, self.motion_ln_mlp_layers
        ):
            if self.use_checkpoint:
                V_delta_att = V_delta_att + torch.utils.checkpoint.checkpoint(
                    attn, ln_attn(V_delta_att), neighbor_idx_expanded, use_reentrant=False)
                V_delta_att = V_delta_att + torch.utils.checkpoint.checkpoint(
                    mlp, ln_mlp(V_delta_att), use_reentrant=False)
            else:
                V_delta_att = V_delta_att + attn(ln_attn(V_delta_att), neighbor_idx_expanded)
                V_delta_att = V_delta_att + mlp(ln_mlp(V_delta_att))
        
        V_delta_att = V_delta_att.reshape(B, T_minus_1, N, self.width)

        # ==================== FPS采样锚点 ====================
        # 直接对V0_att进行FPS采样获得K个锚点
        anchor_indices = self._fps_sampling_indices_large(V0_att, K)  # [B, K]
        
        # 提取锚点的几何特征和坐标
        V0_att_anchor = torch.gather(V0_att, 1, 
            anchor_indices.unsqueeze(-1).expand(-1, -1, self.width))  # [B, K, width]
        anchor_coords = torch.gather(V0, 1,
            anchor_indices.unsqueeze(-1).expand(-1, -1, 3))  # [B, K, 3]
            
        # 提取锚点的运动特征（保留时间维度）
        VT_att_anchor = torch.gather(V_delta_att, 2,
            anchor_indices.unsqueeze(1).unsqueeze(-1).expand(-1, T_minus_1, -1, self.width))  # [B, T-1, K, width]
        VT_att_anchor = VT_att_anchor.permute(0, 2, 1, 3)  # [B, K, T-1, width] - 调整为期望的维度顺序

        return V0_att, V_delta_att, V0_att_anchor, VT_att_anchor, anchor_coords, anchor_indices

    def _fps_sampling_indices_large(self, vertices: torch.Tensor, num_points: int) -> torch.Tensor:
        """FPS采样任意数量的点"""
        B, N, _ = vertices.shape
        device = vertices.device
        
        vertices_fp32 = vertices.float()
        
        centroids = torch.zeros(B, num_points, dtype=torch.long, device=device)
        distances = torch.full((B, N), 1e10, device=device, dtype=torch.float32)
        
        farthest = torch.zeros(B, dtype=torch.long, device=device)
        batch_indices = torch.arange(B, dtype=torch.long, device=device)
        
        for i in range(num_points):
            centroids[:, i] = farthest
            centroid_xyz = vertices_fp32[batch_indices, farthest].unsqueeze(1)
            dist = torch.sum((vertices_fp32 - centroid_xyz) ** 2, dim=-1)
            mask = dist < distances
            distances[mask] = dist[mask]
            farthest = torch.max(distances, dim=1)[1]
        
        return centroids


    def _fps_sampling_indices(self, vertices: torch.Tensor) -> torch.Tensor:
        """Farthest Point Sampling (向量化 GPU 实现)
        Args:
            vertices: [B, N, 3]
        Returns:
            indices:  [B, K]  选出的顶点索引
        """
        B, N, _ = vertices.shape
        device = vertices.device
        K = self.num_tokens

        vertices_fp32 = vertices.float()

        centroids = torch.zeros(B, K, dtype=torch.long, device=device)
        distances = torch.full((B, N), 1e10, device=device, dtype=torch.float32)

        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
        batch_indices = torch.arange(B, dtype=torch.long, device=device)

        for i in range(K):
            centroids[:, i] = farthest
            centroid_xyz = vertices_fp32[batch_indices, farthest].unsqueeze(1)
            dist = torch.sum((vertices_fp32 - centroid_xyz) ** 2, dim=-1)
            mask = dist < distances
            distances[mask] = dist[mask]
            farthest = torch.max(distances, dim=1)[1]

        return centroids

    def forward(self, vertices: torch.Tensor, neighbor_idx: Optional[torch.Tensor] = None):
        return self._forward(vertices, neighbor_idx)






class GaussianSkinningLBS(nn.Module):
    """高斯蒙皮线性混合蒙皮"""
    def __init__(self):
        super().__init__()
    
    def quaternion_to_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """四元数转旋转矩阵"""
        # 更安全的四元数归一化
        q = safe_quat_normalization(q)
        
        # 提取分量
        w, x, y, z = q.unbind(-1)
        
        # 计算旋转矩阵
        matrix = torch.stack([
            1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y,
            2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x,
            2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y
        ], dim=-1).view(*q.shape[:-1], 3, 3)
        
        return matrix
    
    def compute_gaussian_weights(
        self, 
        vertices: torch.Tensor, 
        gaussian_centers: torch.Tensor,
        gaussian_scales: torch.Tensor,
        gaussian_rotations: torch.Tensor
    ) -> torch.Tensor:
        """
        计算高斯蒙皮权重 - 简化版马氏距离计算
        Args:
            vertices: [B, N, 3] - 初始顶点位置
            gaussian_centers: [B, K, 3] - 高斯中心
            gaussian_scales: [B, K, 3] - 高斯尺度
            gaussian_rotations: [B, K, 4] - 高斯旋转（四元数）
        Returns:
            weights: [B, N, K] - 蒙皮权重
        """
        B, N, _ = vertices.shape
        K = gaussian_centers.shape[1]
        
        # 简化版马氏距离计算（数学等价但更稳定高效）
        # 1. 计算相对位置
        rel_pos = vertices.unsqueeze(2) - gaussian_centers.unsqueeze(1)  # [B, N, K, 3]
        
        # 2. 计算旋转矩阵
        R = self.quaternion_to_matrix(gaussian_rotations)  # [B, K, 3, 3]
        
        # 3. 转换到椭球局部坐标系
        # rel_pos: [B, N, K, 3], R: [B, K, 3, 3]
        # 需要将rel_pos与R.transpose(-2, -1)相乘
        rel_pos_local = torch.einsum('bnki,bkij->bnkj', rel_pos, R.transpose(-2, -1))  # [B, N, K, 3]
        
        # 4. 按椭球尺度缩放（确保尺度为正且合理）
        epsilon = 1e-6
        epsilon_tensor = torch.tensor(epsilon, device=gaussian_scales.device, dtype=gaussian_scales.dtype)
        max_scale_val = torch.tensor(2.0, device=gaussian_scales.device, dtype=gaussian_scales.dtype)  # 从10.0降到2.0
        gaussian_scales_safe = torch.clamp(gaussian_scales, min=epsilon_tensor, max=max_scale_val)
        
        # 直接除法缩放（避免平方和矩阵运算）
        scaled_pos = rel_pos_local / (gaussian_scales_safe.unsqueeze(1) + epsilon)  # [B, N, K, 3]
        
        # 5. 计算马氏距离平方
        mahalanobis_dist_sq = torch.sum(scaled_pos ** 2, dim=-1)  # [B, N, K]
        
        # 6. 计算未归一化权重 - 使用温和的参数
        # 温和的指数衰减
        unnormalized_weights = torch.exp(-0.5 * mahalanobis_dist_sq)  # 从-1.0改为-0.5，减少衰减强度
        
        # 7. 温和的温度缩放
        temperature = torch.tensor(1.0, device=unnormalized_weights.device, dtype=unnormalized_weights.dtype)  # 从0.1改为1.0
        power_val = torch.tensor(1.0, device=unnormalized_weights.device, dtype=unnormalized_weights.dtype) / temperature
        unnormalized_weights = torch.pow(unnormalized_weights, power_val)  # 现在power=1，相当于无额外缩放
        
        # 8. 归一化权重
        weight_sum = unnormalized_weights.sum(dim=-1, keepdim=True)
        min_val = torch.tensor(1e-8, device=weight_sum.device, dtype=weight_sum.dtype)
        weight_sum = torch.clamp(weight_sum, min=min_val)  # 确保分母不为零
        weights = unnormalized_weights / weight_sum
        
        return weights
    
    def forward(
        self,
        vertices: torch.Tensor,
        token_indices: torch.Tensor,
        decoded_params: dict,
        num_frames: int
    ) -> torch.Tensor:
        """
        执行高斯蒙皮线性混合蒙皮 - 向量化优化版本
        Args:
            vertices: [B, N, 3] - 初始顶点位置
            token_indices: [B, K] - token对应的顶点索引
            decoded_params: 解码的参数字典
            num_frames: 帧数
        Returns:
            animated_vertices: [B, T, N, 3] - 动画后的顶点
        """
        B, N, _ = vertices.shape
        K = token_indices.shape[1]
        
        # 提取参数
        gaussians = decoded_params['gaussians']  # [B, K, 10]
        transforms = decoded_params['transforms']  # [B, K, T, 7]
        bone_offset = decoded_params['bone_offset']  # [B, K, 3] New
        root_quat = decoded_params['root_quat']   # [B, T, 4] - 新增
        root_trans = decoded_params['root_trans'] # [B, T, 3] - 新增
        gaussian_offset = gaussians[:, :, :3]  # [B, K, 3]
        gaussian_scales = gaussians[:, :, 3:6]  # [B, K, 3]
        gaussian_quat = gaussians[:, :, 6:]  # [B, K, 4]
        motion_quat = transforms[:, :, :, :4]  # [B, K, T, 4]
        motion_trans = transforms[:, :, :, 4:]  # [B, K, T, 3]
        
        # 计算高斯中心
        anchor_vertices = torch.gather(vertices, 1, token_indices.unsqueeze(-1).expand(-1, -1, 3))  # [B, K, 3]
        gaussian_centers = anchor_vertices + gaussian_offset  # [B, K, 3]
        
        # New: 计算骨骼中心用于旋转中心
        bone_centers = anchor_vertices + bone_offset  # [B, K, 3]
        
        # 计算蒙皮权重 (使用 gaussian_centers)
        weights = self.compute_gaussian_weights(
            vertices, gaussian_centers, gaussian_scales, gaussian_quat
        )  # [B, N, K]
        
        # # Sparsity metric: % of weights > threshold (e.g., 0.1)
        # import pdb
        # def check_weights(weights):
        #     print('weights shape: ', weights.shape)
            
        #     w1 = weights[0]
        #     print(np.unique(w1.sum(dim=1).cpu().numpy()))
        #     pdb.set_trace()
        #     return

        # check_weights(weights)


        # output_dir = '/home/hzhang12/disk/code/Step1X-3D/outputs/step1x-3d-mesh-motion-vae/mesh_motion_vae/n20+kl0.0001+lr2e-05/test'
        # os.makedirs(output_dir, exist_ok=True)
        # np.save(os.path.join(output_dir, 'gaussian_centers.npy'), gaussian_centers.cpu().numpy())
        # np.save(os.path.join(output_dir, 'bone_centers.npy'), bone_centers.cpu().numpy())
        # np.save(os.path.join(output_dir, 'token_vertices.npy'), anchor_vertices.cpu().numpy())
        # np.save(os.path.join(output_dir, 'vertices.npy'), vertices.cpu().numpy())
        # print(aaaa)

        # 向量化优化：一次性计算所有帧的旋转矩阵
        # motion_quat: [B, K, T, 4] -> [B*K*T, 4]
        motion_quat_flat = motion_quat.reshape(B * K * num_frames, 4)
        motion_rot_mats_flat = self.quaternion_to_matrix(motion_quat_flat)  # [B*K*T, 3, 3]
        motion_rot_mats = motion_rot_mats_flat.reshape(B, K, num_frames, 3, 3)  # [B, K, T, 3, 3]
        
        # 准备顶点数据用于批量计算
        vertices_expanded = vertices.unsqueeze(2).unsqueeze(3).expand(-1, -1, K, num_frames, -1)  # [B, N, K, T, 3]
        
        # 新增: 以 bone_centers 为中心计算局部坐标
        bone_centers_expanded = bone_centers.unsqueeze(1).unsqueeze(3).expand(-1, N, -1, num_frames, -1)  # [B, N, K, T, 3]
        local_vertices = vertices_expanded - bone_centers_expanded
        
        # 批量矩阵乘法：一次性计算所有帧的旋转
        vertices_for_rot = local_vertices.unsqueeze(-1)  # [B, N, K, T, 3, 1]
        rot_mats_expanded = motion_rot_mats.unsqueeze(1).expand(-1, N, -1, -1, -1, -1)  # [B, N, K, T, 3, 3]
        
        rotated_vertices = torch.matmul(rot_mats_expanded, vertices_for_rot).squeeze(-1)  # [B, N, K, T, 3]
        
        # 添加平移和中心点
        motion_trans_expanded = motion_trans.unsqueeze(1).expand(-1, N, -1, -1, -1)  # [B, N, K, T, 3]
        
        # The transformation for each bone should be a rotation around the bone's center,
        # followed by a translation. The correct formula is R * (v - p) + p + t.
        # `rotated_vertices` is R * (v - p). We need to add back p (`bone_centers_expanded`)
        # and the translation t (`motion_trans_expanded`).
        transformed_vertices = rotated_vertices + bone_centers_expanded + motion_trans_expanded  # [B, N, K, T, 3]
        
        # 线性混合蒙皮：一次性计算所有帧
        weights_expanded = weights.unsqueeze(-1).unsqueeze(-1)  # [B, N, K, 1, 1]
        animated_vertices_local = torch.sum(weights_expanded * transformed_vertices, dim=2)  # [B, N, T, 3]
        
        # === 新增: 应用全局根变换 ===
        # 1. 将根四元数转为旋转矩阵
        root_rot_mats = self.quaternion_to_matrix(root_quat) # [B, T, 3, 3]
        
        # 2. 应用旋转
        # animated_vertices_local: [B, N, T, 3] -> [B, T, N, 3]
        animated_vertices_local_t_first = animated_vertices_local.transpose(1, 2)
        # 使用 einsum 进行批量矩阵乘法
        # root_rot_mats: [B, T, 3, 3], animated_vertices_local_t_first: [B, T, N, 3]
        animated_vertices_global = torch.einsum('btij,btnj->btni', root_rot_mats, animated_vertices_local_t_first)
        
        # 3. 应用平移
        # root_trans: [B, T, 3] -> [B, T, 1, 3]
        animated_vertices_global = animated_vertices_global + root_trans.unsqueeze(2)
        
        return animated_vertices_global, weights  # 同时返回权重用于正则化


@step1x3d_geometry.register("mesh-motion-vae")
class MeshMotionVAE(BaseModule):
    """Mesh Motion VAE with Gaussian Skinning Decoder"""
    
    @dataclass
    class Config(BaseModule.Config):
        # 编码器配置
        width: int = 768
        heads: int = 12
        num_encoder_layers: int = 8
        num_tokens: int = 64
        embed_dim: int = 64
        num_vertices: int = 5000  # 顶点数量
        
        # 运动路径编码配置
        motion_layers: int = 4  # 运动路径层数
        
        # 解码器配置
        num_decoder_layers: int = 2 # 解码器MLP层数 (deprecated, kept for compatibility)
        num_cross_attn_layers: int = 1 # Cross-attention层数 (deprecated, kept for compatibility)
        num_frames: int = 30
        
        # 细粒度解码器配置
        static_num_decoder_layers: int = 2      # 静态解码器MLP层数
        dynamic_num_decoder_layers: int = 2     # 动态解码器MLP层数
        root_num_decoder_layers: int = 2        # 根解码器MLP层数
        static_num_cross_attn_layers: int = 1   # 静态Cross-attention层数
        dynamic_num_cross_attn_layers: int = 1  # 动态Cross-attention层数
        root_num_cross_attn_layers: int = 1     # 根Cross-attention层数
        
        # 训练配置
        sample_posterior: bool = True
        lambda_reconstruction: float = 1.0
        lambda_motion_smooth: float = 0.1
        lambda_diversity: float = 0.1  # 新增多样性损失权重
        lambda_sparsity: float = 0.01  # 稀疏性损失权重
        lambda_kl: float = 0.001
        lambda_root_smooth: float = 0.1 # 新增根变换平滑损失权重
        
        # KL Annealing配置
        kl_annealing_warmup_steps: int = 5000  # KL annealing预热步数
        kl_annealing_schedule: str = "linear"  # linear, cosine, sigmoid
        kl_min_weight: float = 0.0  # KL权重最小值
        kl_max_weight: float = 1.0  # KL权重最大值
        
        # 其他配置
        init_scale: float = 0.25
        qkv_bias: bool = True
        qk_norm: bool = True
        use_flash: bool = False
        use_checkpoint: bool = False
        z_scale_factor: float = 1.0
        
        # Temporal attention config
        use_temporal_attn: bool = False
        temporal_attn_layers: int = 2
        temporal_attn_heads: int = 8

    cfg: Config

    def configure(self) -> None:
        super().configure()
        
        # KL annealing训练步数跟踪
        self.register_buffer('training_step', torch.tensor(0, dtype=torch.long))
        
        # 编码器 - 简化的FPS架构
        self.encoder = TopologyAwareEncoder(
            width=self.cfg.width,
            heads=self.cfg.heads,
            layers=self.cfg.num_encoder_layers,
            num_tokens=self.cfg.num_tokens,
            embed_dim=self.cfg.embed_dim,
            init_scale=self.cfg.init_scale,
            qkv_bias=self.cfg.qkv_bias,
            qk_norm=self.cfg.qk_norm,
            use_flash=self.cfg.use_flash,
            use_checkpoint=self.cfg.use_checkpoint,
            # 保留motion_layers用于运动路径
            motion_layers=self.cfg.motion_layers,
            num_vertices=self.cfg.num_vertices,
            num_frames=self.cfg.num_frames,
        )
        
        # ==================== 解码器组件 ====================
        # 静态参数解码器（高斯参数）- 直接预测
        self.static_decoder = StaticParamDecoder(
            self.cfg.width, 
            heads=self.cfg.heads, 
            num_layers=self.cfg.static_num_decoder_layers,
            num_cross_attn_layers=self.cfg.static_num_cross_attn_layers
        )
        
        # 动态参数VAE编码器和解码器
        self.dynamic_vae_encoder = DynamicVAEEncoder(
            self.cfg.width, 
            self.cfg.embed_dim, 
            heads=self.cfg.heads, 
            num_cross_attn_layers=self.cfg.dynamic_num_cross_attn_layers
        )
        self.dynamic_vae_decoder = DynamicVAEDecoder(
            self.cfg.embed_dim, 
            self.cfg.width, 
            heads=self.cfg.heads, 
            num_layers=self.cfg.dynamic_num_decoder_layers
        )
        
        # 根变换VAE编码器和解码器
        root_embed_dim = self.cfg.embed_dim // 2  # 根变换用较小的latent
        self.root_vae_encoder = RootVAEEncoder(
            self.cfg.width, 
            root_embed_dim, 
            heads=self.cfg.heads, 
            num_cross_attn_layers=self.cfg.root_num_cross_attn_layers
        )
        self.root_vae_decoder = RootVAEDecoder(
            root_embed_dim, 
            self.cfg.width, 
            heads=self.cfg.heads, 
            num_layers=self.cfg.root_num_decoder_layers
        )
        
        # Temporal attention (cross-frame) — encoder and decoder side
        if self.cfg.use_temporal_attn:
            self.enc_temporal_attn = TemporalTransformerBlock(
                dim=self.cfg.width,
                heads=self.cfg.temporal_attn_heads,
                num_layers=self.cfg.temporal_attn_layers,
                max_time_steps=self.cfg.num_frames,
            )
            self.dec_temporal_attn = TemporalTransformerBlock(
                dim=self.cfg.embed_dim,
                heads=min(self.cfg.temporal_attn_heads, self.cfg.embed_dim // 16),
                num_layers=self.cfg.temporal_attn_layers,
                max_time_steps=self.cfg.num_frames,
            )
            self.root_enc_temporal_attn = TemporalTransformerBlock(
                dim=self.cfg.width,
                heads=self.cfg.temporal_attn_heads,
                num_layers=self.cfg.temporal_attn_layers,
                max_time_steps=self.cfg.num_frames,
            )
            self.root_dec_temporal_attn = TemporalTransformerBlock(
                dim=root_embed_dim,
                heads=min(self.cfg.temporal_attn_heads, root_embed_dim // 8),
                num_layers=self.cfg.temporal_attn_layers,
                max_time_steps=self.cfg.num_frames,
            )

        # 高斯蒙皮LBS
        self.gaussian_lbs = GaussianSkinningLBS()
        
    def compute_kl_annealing_weight(self) -> float:
        """计算KL annealing权重"""
        if self.cfg.kl_annealing_warmup_steps <= 0:
            return self.cfg.kl_max_weight
        
        # 计算进度比例
        progress = min(1.0, float(self.training_step) / self.cfg.kl_annealing_warmup_steps)
        
        # 根据调度策略计算权重
        if self.cfg.kl_annealing_schedule == "linear":
            weight = self.cfg.kl_min_weight + progress * (self.cfg.kl_max_weight - self.cfg.kl_min_weight)
        elif self.cfg.kl_annealing_schedule == "cosine":
            import math
            weight = self.cfg.kl_min_weight + 0.5 * (self.cfg.kl_max_weight - self.cfg.kl_min_weight) * (1 - math.cos(math.pi * progress))
        elif self.cfg.kl_annealing_schedule == "sigmoid":
            import math
            # Sigmoid annealing: 更平滑的过渡
            sigmoid_input = (progress - 0.5) * 12  # 调整陡度
            sigmoid_output = 1 / (1 + math.exp(-sigmoid_input))
            weight = self.cfg.kl_min_weight + sigmoid_output * (self.cfg.kl_max_weight - self.cfg.kl_min_weight)
        else:
            weight = self.cfg.kl_max_weight
        
        return weight

    def encode(
        self,
        vertices: torch.Tensor,
        neighbor_idx: Optional[torch.Tensor] = None,
        sample_posterior: bool = True,
    ):
        """
        编码动态网格序列 - 新架构
        Args:
            vertices: [B, T, N, 3] - 动态网格序列
            neighbor_idx: [B, N, k] - 邻居索引列表  
            sample_posterior: 是否采样后验分布
        Returns:
            V0_att: [B, N, width] - 静态几何特征
            V_delta_att: [B, T-1, N, width] - 动态运动特征
            V0_att_anchor: [B, K, width] - 锚点几何特征
            VT_att_anchor: [B, T-1, K, width] - 锚点运动特征
            anchor_coords: [B, K, 3] - 锚点坐标
            anchor_indices: [B, K] - 锚点索引
        """
        # 使用新的简化编码器
        return self.encoder(vertices, neighbor_idx)

    def decode(self, dynamic_z: torch.Tensor, root_z: torch.Tensor):
        """
        解码潜在表示 - 新架构
        Args:
            dynamic_z: [B, K, T-1, embed_dim] - 动态变换latent
            root_z: [B, T-1, embed_dim] - 根变换latent
        Returns:
            transforms: [B, K, T-1, 7] - 骨骼变换
            root_quat: [B, T-1, 4] - 根旋转
            root_trans: [B, T-1, 3] - 根平移
        """
        dynamic_z = dynamic_z / self.cfg.z_scale_factor
        root_z = root_z / self.cfg.z_scale_factor
        
        if self.cfg.use_temporal_attn:
            dynamic_z = self.dec_temporal_attn(dynamic_z)
            root_z = self.root_dec_temporal_attn(root_z)
        
        transforms = self.dynamic_vae_decoder(dynamic_z)
        root_quat, root_trans = self.root_vae_decoder(root_z)
        
        return transforms, root_quat, root_trans

    def forward(
        self,
        vertices: torch.Tensor,
        neighbor_idx: Optional[torch.Tensor] = None,
        sample_posterior: bool = True,
    ):
        """
        前向传播 - VAE+直接预测混合架构
        Args:
            vertices: [B, T, N, 3] - 动态网格序列
            neighbor_idx: [B, N, k] - 邻居索引列表
            sample_posterior: 是否采样后验分布
        Returns:
            dict: 包含各种输出
        """
        # 编码 - 简化的FPS采样架构
        V0_att, V_delta_att, V0_att_anchor, VT_att_anchor, anchor_coords, anchor_indices = self.encoder(
            vertices, neighbor_idx
        )
        
        # 1. 静态参数 - 直接预测（无VAE）
        gaussians = self.static_decoder(V0_att_anchor, V0_att)  # [B, K, 10]
        
        # 2. 动态参数 - VAE编码解码（支持KL annealing）
        kl_weight = self.compute_kl_annealing_weight()
        
        # Encoder-side temporal attention: each bone attends across all time steps
        vt_for_dynamic = VT_att_anchor
        if self.cfg.use_temporal_attn:
            vt_for_dynamic = self.enc_temporal_attn(VT_att_anchor)  # [B, K, T-1, width]
        
        dynamic_mu, dynamic_log_sigma, dynamic_z, dynamic_kl_loss = self.dynamic_vae_encoder(
            vt_for_dynamic, V_delta_att, kl_weight
        )
        
        # Decoder-side temporal attention on latent z
        dynamic_z_dec = dynamic_z
        dynamic_mu_dec = dynamic_mu
        if self.cfg.use_temporal_attn:
            dynamic_z_dec = self.dec_temporal_attn(dynamic_z)
            dynamic_mu_dec = self.dec_temporal_attn(dynamic_mu)
        
        if sample_posterior:
            transforms = self.dynamic_vae_decoder(dynamic_z_dec)  # [B, K, T-1, 7]
        else:
            transforms = self.dynamic_vae_decoder(dynamic_mu_dec)  # [B, K, T-1, 7]
        
        # 3. 根变换 - VAE编码解码（统一接口，支持KL annealing）
        vt_for_root = VT_att_anchor
        if self.cfg.use_temporal_attn:
            vt_for_root = self.root_enc_temporal_attn(VT_att_anchor)  # [B, K, T-1, width]
        
        root_mu, root_log_sigma, root_z, root_kl_loss = self.root_vae_encoder(vt_for_root, V_delta_att, kl_weight)
        
        # Decoder-side temporal attention on root latent
        root_z_dec = root_z
        root_mu_dec = root_mu
        if self.cfg.use_temporal_attn:
            root_z_dec = self.root_dec_temporal_attn(root_z)
            root_mu_dec = self.root_dec_temporal_attn(root_mu)
        
        if sample_posterior:
            root_quat, root_trans = self.root_vae_decoder(root_z_dec)
        else:
            root_quat, root_trans = self.root_vae_decoder(root_mu_dec)
        
        # 总KL损失
        total_kl_loss = dynamic_kl_loss + root_kl_loss
        
        # 组织解码参数
        decoded_params = {
            'gaussians': gaussians,
            'transforms': transforms,
            'bone_offset': torch.zeros_like(anchor_coords),  # bone center和gaussian center一致
            'root_quat': root_quat,
            'root_trans': root_trans
        }
        
        # 高斯蒙皮LBS
        animated_vertices, skinning_weights = self.gaussian_lbs(
            vertices[:, 0],  # 初始帧顶点
            anchor_indices,  # 直接使用FPS采样的锚点索引
            decoded_params,
            self.cfg.num_frames-1
        )  # [B, T-1, N, 3], [B, N, K]
        
        # 检查最终输出的数值稳定性
        if torch.isnan(animated_vertices).any() or torch.isinf(animated_vertices).any():
            print("WARNING: NaN/Inf detected in animated_vertices!")
            # 用初始顶点替换有问题的帧 - 修复数据类型匹配问题
            nan_mask = torch.isnan(animated_vertices) | torch.isinf(animated_vertices)
            initial_vertices = vertices[:, 0].unsqueeze(1).expand_as(animated_vertices).to(animated_vertices.dtype)
            animated_vertices[nan_mask] = initial_vertices[nan_mask]
        
        return {
            'V0_att_anchor': V0_att_anchor,
            'VT_att_anchor': VT_att_anchor,
            'anchor_coords': anchor_coords,
            'anchor_indices': anchor_indices,
            # VAE相关输出
            'dynamic_mu': dynamic_mu,
            'dynamic_log_sigma': dynamic_log_sigma,
            'dynamic_z': dynamic_z,
            'root_mu': root_mu,
            'root_log_sigma': root_log_sigma,
            'root_z': root_z,
            'kl_loss': total_kl_loss,
            # 解码输出
            'decoded_params': decoded_params,
            'animated_vertices': animated_vertices,
            'skinning_weights': skinning_weights,
        }

    def compute_losses(self, output: dict, target_vertices: torch.Tensor):
        """
        计算损失函数
        Args:
            output: 模型输出
            target_vertices: [B, T, N, 3] - 目标顶点序列
        Returns:
            dict: 损失字典
        """
        # 更新训练步数（仅在训练时）
        if self.training:
            self.training_step += 1
        
        B, T, N, _ = target_vertices.shape
        current_kl_weight = self.compute_kl_annealing_weight()
        
        # 1. 重建损失 - 添加数值稳定性检查
        pred_vertices = output['animated_vertices']  # [B, T-1, N, 3]
        target_vertices = target_vertices[:, 1:] # [B, T-1, N, 3]
        # 这部分需要确定一下，我们预测出来的是定点的1-T帧的位置，所以要跟gt位置求loss，不能跟位置差求loss
        # 检查NaN和无穷大
        if torch.isnan(pred_vertices).any() or torch.isinf(pred_vertices).any():
            print("WARNING: NaN or Inf detected in pred_vertices!")
            pred_vertices = torch.nan_to_num(pred_vertices, nan=0.0, posinf=1e6, neginf=-1e6)
        
        reconstruction_loss = F.mse_loss(pred_vertices.float(), target_vertices.float())
        
        # 检查重建损失
        if torch.isnan(reconstruction_loss) or torch.isinf(reconstruction_loss):
            print("WARNING: NaN or Inf in reconstruction_loss!")
            reconstruction_loss = torch.tensor(1e6, device=pred_vertices.device, dtype=torch.float32)
        
        # 2. 运动平滑损失 - 添加数值稳定性
        motion_smooth_loss = torch.tensor(0.0, device=pred_vertices.device, dtype=torch.float32)
        if T - 1 > 1:
            try:
                decoded_params = output['decoded_params']
                gaussians = decoded_params['gaussians']
                transforms = decoded_params['transforms']
                
                # 检查transforms的数值稳定性
                if torch.isnan(transforms).any() or torch.isinf(transforms).any():
                    print("WARNING: NaN or Inf detected in transforms!")
                    transforms = torch.nan_to_num(transforms, nan=0.0, posinf=1e6, neginf=-1e6)
                
                motion_quat = transforms[:, :, :, :4].float()  # [B, K, T-1, 4]
                motion_trans = transforms[:, :, :, 4:].float()  # [B, K, T-1, 3]
                
                # 归一化四元数以确保数值稳定性
                motion_quat = safe_quat_normalization(motion_quat)
                
                # 计算相邻帧之间的差异
                q1 = motion_quat[:, :, :-1]
                q2 = motion_quat[:, :, 1:]
                
                # 确保 q1 和 q2 在同一个半球上
                dot_product = torch.sum(q1 * q2, dim=-1)
                q2_corrected = torch.where(dot_product.unsqueeze(-1) < 0, -q2, q2)
                
                # 使用更稳定的L2范数
                quat_dist_sq = torch.sum((q1 - q2_corrected)**2, dim=-1)
                max_quat_dist = torch.tensor(4.0, device=quat_dist_sq.device, dtype=quat_dist_sq.dtype)
                quat_dist_sq = torch.clamp(quat_dist_sq, max=max_quat_dist)  # 限制最大值

                trans_diff = motion_trans[:, :, 1:] - motion_trans[:, :, :-1]  # [B, K, T-1, 3]
                trans_diff_sq = torch.sum(trans_diff ** 2, dim=-1)
                trans_diff_sq = torch.clamp(trans_diff_sq, max=1e6)  # 限制最大值
                
                motion_smooth_loss = (
                    torch.mean(quat_dist_sq) + torch.mean(trans_diff_sq)
                )
                
                # 检查运动平滑损失
                if torch.isnan(motion_smooth_loss) or torch.isinf(motion_smooth_loss):
                    print("WARNING: NaN or Inf in motion_smooth_loss!")
                    motion_smooth_loss = torch.tensor(0.0, device=pred_vertices.device, dtype=torch.float32)
                
                # 调试信息
                # if torch.rand(1).item() < 0.01:  # 1% 概率打印调试信息
                #     print(f"Motion smooth loss debug - Δquat²: {torch.mean(quat_dist_sq).item():.6f}, Δtrans²: {torch.mean(trans_diff_sq).item():.6f}")
                #     # 添加更详细的调试信息
                #     print(f"Motion debug - quat range: [{motion_quat.min().item():.6f}, {motion_quat.max().item():.6f}]")
                #     print(f"Motion debug - trans range: [{motion_trans.min().item():.6f}, {motion_trans.max().item():.6f}]")
                #     print(f"Motion debug - quat std across time: {torch.std(motion_quat, dim=2).mean().item():.6f}")
                #     print(f"Motion debug - trans std across time: {torch.std(motion_trans, dim=2).mean().item():.6f}")
                    
            except Exception as e:
                print(f"Error in motion smooth loss calculation: {e}")
                motion_smooth_loss = torch.tensor(0.0, device=pred_vertices.device, dtype=torch.float32)
        
        # 3. KL损失 - 添加数值稳定性检查
        kl_loss = output['kl_loss']
        if torch.isnan(kl_loss) or torch.isinf(kl_loss):
            print("WARNING: NaN or Inf in kl_loss!")
            kl_loss = torch.tensor(0.0, device=pred_vertices.device, dtype=torch.float32)
        
        # 新增: 根变换平滑损失
        root_smooth_loss = torch.tensor(0.0, device=pred_vertices.device, dtype=torch.float32)
        if T - 1 > 1 and self.cfg.lambda_root_smooth > 0:
            try:
                root_quat = output['decoded_params']['root_quat'].float() # [B, T-1, 4]
                root_trans = output['decoded_params']['root_trans'].float() # [B, T-1, 3]
                
                # 归一化四元数
                root_quat = safe_quat_normalization(root_quat)
                
                # 计算相邻帧差异
                q1 = root_quat[:, :-1]
                q2 = root_quat[:, 1:]
                dot_product = torch.sum(q1 * q2, dim=-1)
                q2_corrected = torch.where(dot_product.unsqueeze(-1) < 0, -q2, q2)
                quat_dist_sq = torch.sum((q1 - q2_corrected)**2, dim=-1)

                trans_diff_sq = torch.sum((root_trans[:, 1:] - root_trans[:, :-1])**2, dim=-1)
                
                root_smooth_loss = torch.mean(quat_dist_sq) + torch.mean(trans_diff_sq)
                
                if torch.isnan(root_smooth_loss) or torch.isinf(root_smooth_loss):
                    print("WARNING: NaN or Inf in root_smooth_loss!")
                    root_smooth_loss = torch.tensor(0.0, device=pred_vertices.device, dtype=torch.float32)
            except Exception as e:
                print(f"Error in root smooth loss calculation: {e}")
                root_smooth_loss = torch.tensor(0.0, device=pred_vertices.device, dtype=torch.float32)

        # 4. SE(3)变换多样性损失 - 鼓励不同bone有不同变换
        diversity_loss = torch.tensor(0.0, device=pred_vertices.device, dtype=torch.float32)
        if T - 1 > 1 and self.cfg.lambda_diversity > 0:
            try:
                decoded_params = output['decoded_params']
                transforms = decoded_params['transforms']  # [B, K, T-1, 7]
                
                # 检查transforms的数值稳定性
                if torch.isnan(transforms).any() or torch.isinf(transforms).any():
                    print("WARNING: NaN or Inf detected in transforms for diversity loss!")
                    transforms = torch.nan_to_num(transforms, nan=0.0, posinf=1e6, neginf=-1e6)
                
                motion_quat = transforms[:, :, :, :4].float()  # [B, K, T-1, 4]
                motion_trans = transforms[:, :, :, 4:].float()  # [B, K, T-1, 3]
                
                # 归一化四元数
                quat_norm = torch.norm(motion_quat, dim=-1, keepdim=True)
                quat_norm = torch.clamp(quat_norm, min=1e-8)
                motion_quat = motion_quat / quat_norm
                
                # 计算骨骼间变换差异的负熵（鼓励多样性）
                # 对于四元数：计算每个时间步骨骼间的差异
                B, K, T_minus_1, _ = motion_quat.shape
                
                # 四元数多样性：计算骨骼间的最小旋转角度差异
                quat_expanded_i = motion_quat.unsqueeze(2)  # [B, K, 1, T-1, 4]
                quat_expanded_j = motion_quat.unsqueeze(1)  # [B, 1, K, T-1, 4]
                
                # 计算四元数之间的点积（角度相似性）
                quat_dot = torch.sum(quat_expanded_i * quat_expanded_j, dim=-1)  # [B, K, K, T-1]
                quat_dot = torch.abs(quat_dot)  # 处理负值情况
                max_quat_dot = torch.tensor(1.0, device=quat_dot.device, dtype=quat_dot.dtype)
                quat_dot = torch.clamp(quat_dot, max=max_quat_dot)  # 确保在有效范围内
                
                # 排除自己与自己的比较
                mask = torch.eye(K, device=motion_quat.device).bool()
                quat_dot_masked = quat_dot.masked_fill(mask.unsqueeze(0).unsqueeze(-1), 0)
                
                # 计算旋转多样性损失（惩罚过于相似的旋转）
                quat_diversity_loss = torch.mean(quat_dot_masked ** 2)
                
                # 平移多样性：计算骨骼间的平移差异
                trans_expanded_i = motion_trans.unsqueeze(2)  # [B, K, 1, T-1, 3]
                trans_expanded_j = motion_trans.unsqueeze(1)  # [B, 1, K, T-1, 3]
                trans_diff = trans_expanded_i - trans_expanded_j  # [B, K, K, T-1, 3]
                trans_dist = torch.norm(trans_diff, dim=-1)  # [B, K, K, T-1]
                
                # 排除自己与自己的比较，计算平移多样性
                trans_dist_masked = trans_dist.masked_fill(mask.unsqueeze(0).unsqueeze(-1), float('inf'))
                # 鼓励最小距离不要太小（避免所有平移都相同）
                min_trans_dist = torch.min(trans_dist_masked, dim=2)[0]  # [B, K, T-1]
                trans_diversity_loss = torch.mean(torch.exp(-min_trans_dist))  # 距离越小，损失越大
                
                diversity_loss = quat_diversity_loss + trans_diversity_loss
                
                # 检查多样性损失
                if torch.isnan(diversity_loss) or torch.isinf(diversity_loss):
                    print("WARNING: NaN or Inf in diversity_loss!")
                    diversity_loss = torch.tensor(0.0, device=pred_vertices.device, dtype=torch.float32)
                    
            except Exception as e:
                print(f"Error in diversity loss calculation: {e}")
                diversity_loss = torch.tensor(0.0, device=pred_vertices.device, dtype=torch.float32)

        # 5. 权重稀疏化损失 - 鼓励局部化的蒙皮权重
        sparsity_loss = torch.tensor(0.0, device=pred_vertices.device, dtype=torch.float32)
        if 'skinning_weights' in output and self.cfg.lambda_sparsity > 0:
            try:
                weights = output['skinning_weights']  # [B, N, K]
                # 计算权重的熵，熵越高说明权重越分散，我们希望权重更集中
                weights_log = torch.log(weights + 1e-8)
                entropy = -torch.sum(weights * weights_log, dim=-1)  # [B, N]
                # 我们希望最小化熵（鼓励稀疏权重）
                sparsity_loss = torch.mean(entropy)
                
                # 检查稀疏性损失
                if torch.isnan(sparsity_loss) or torch.isinf(sparsity_loss):
                    print("WARNING: NaN or Inf in sparsity_loss!")
                    sparsity_loss = torch.tensor(0.0, device=pred_vertices.device, dtype=torch.float32)
                    
            except Exception as e:
                print(f"Error in sparsity loss calculation: {e}")
                sparsity_loss = torch.tensor(0.0, device=pred_vertices.device, dtype=torch.float32)

        # 6. 总损失 - 添加梯度裁剪
        lambda_diversity = getattr(self.cfg, 'lambda_diversity', 0.1)  # 默认权重
        lambda_sparsity = getattr(self.cfg, 'lambda_sparsity', 0.01)  # 稀疏性损失权重
        lambda_root_smooth = getattr(self.cfg, 'lambda_root_smooth', 0.1)
        total_loss = (
            self.cfg.lambda_reconstruction * reconstruction_loss +
            self.cfg.lambda_motion_smooth * motion_smooth_loss +
            self.cfg.lambda_kl * kl_loss +
            lambda_diversity * diversity_loss +
            lambda_sparsity * sparsity_loss +
            lambda_root_smooth * root_smooth_loss
        )
        
        # 最终检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("WARNING: NaN or Inf in total_loss! Using fallback loss.")
            total_loss = torch.tensor(1e6, device=pred_vertices.device, dtype=torch.float32)
        
        # 添加调试信息（每100次迭代打印一次）
        if torch.rand(1).item() < 0.01:  # 1%概率打印
            print(f"Loss breakdown - Recon: {reconstruction_loss.item():.6f}, "
                  f"Motion: {motion_smooth_loss.item():.6f}, "
                  f"KL: {kl_loss.item():.6f} (weight: {current_kl_weight:.4f}), "
                  f"Diversity: {diversity_loss.item():.6f}, "
                  f"Sparsity: {sparsity_loss.item():.6f}, "
                  f"Root: {root_smooth_loss.item():.6f}")
            print(f"KL Annealing - Step: {self.training_step.item()}, Weight: {current_kl_weight:.4f}")
        
        return {
            'total_loss': total_loss.float(),
            'reconstruction_loss': reconstruction_loss.float(),
            'motion_smooth_loss': motion_smooth_loss.float(),
            'kl_loss': kl_loss.float(),
            'diversity_loss': diversity_loss.float(),
            'sparsity_loss': sparsity_loss.float(),
            'root_smooth_loss': root_smooth_loss.float(),
            'kl_weight': current_kl_weight,  # 新增：KL权重监控
            'training_step': self.training_step.item(),  # 新增：训练步数监控
        } 