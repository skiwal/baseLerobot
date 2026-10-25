#!/usr/bin/env python
"""IDQL 模型(M1: 无力觉插装的残差扩散 RL, 无图像版)。

组件(v1 设计规格):
  * 观测编码器: MLP(31 维本体+感知派生特征 -> latent), LayerNorm;
  * Q1,Q2(s,a): MLP, Polyak 目标网; TD 目标用 V(s')(IDQL: 不进 actor);
  * V(s): MLP, IQL 期望回归(tau=0.7);
  * 扩散策略 pi(a|s): epsilon 预测 MLP(条件注入 latent + 时间嵌入),
    训练 = AWR 优势加权去噪损失(权重 exp(A/beta), A=Q-V 标准化);
    推理 = DDIM 10 步, 动作 clamp 到 [-1,1](归一化残差空间)。

自包含 nn.Module(不依赖 lerobot 策略接口); 验证通过后再按插件规范移植。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


def mlp(sizes: list[int], layer_norm: bool = True) -> nn.Sequential:
    """标准 MLP: Linear + LayerNorm + SiLU(最后一层无 LN/激活)。"""
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class SinusoidalPosEmb(nn.Module):
    """扩散时间步的正弦嵌入。"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class IDQLModel(nn.Module):
    """IDQL: 双 Q + V(期望回归) + AWR 加权扩散策略。"""

    def __init__(self, obs_dim: int = 31, act_dim: int = 5,
                 latent_dim: int = 256, hidden_dim: int = 256,
                 n_diffusion_train_steps: int = 100, n_diffusion_infer_steps: int = 10,
                 beta_start: float = 1e-4, beta_end: float = 2e-2,
                 discount: float = 0.99, expectile_tau: float = 0.7,
                 awr_beta: float = 1.0, target_update_rate: float = 0.005):
        super().__init__()
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.discount = discount
        self.expectile_tau = expectile_tau
        self.awr_beta = awr_beta
        self.target_update_rate = target_update_rate
        self.n_train_steps = n_diffusion_train_steps
        self.n_infer_steps = n_diffusion_infer_steps

        # --- 网络 ---
        self.encoder = mlp([obs_dim, latent_dim, latent_dim])
        self.q1 = mlp([latent_dim + act_dim, hidden_dim, hidden_dim, 1])
        self.q2 = mlp([latent_dim + act_dim, hidden_dim, hidden_dim, 1])
        self.v = mlp([latent_dim, hidden_dim, hidden_dim, 1])
        t_emb_dim = 64
        self.t_emb = SinusoidalPosEmb(t_emb_dim)
        self.actor = mlp([latent_dim + act_dim + t_emb_dim,
                          hidden_dim * 2, hidden_dim * 2, hidden_dim * 2, act_dim])

        # --- 目标网(Q1/Q2 的 Polyak 副本) ---
        self.q1_target = mlp([latent_dim + act_dim, hidden_dim, hidden_dim, 1])
        self.q2_target = mlp([latent_dim + act_dim, hidden_dim, hidden_dim, 1])
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        for p in list(self.q1_target.parameters()) + list(self.q2_target.parameters()):
            p.requires_grad_(False)

        # --- 扩散调度(线性 beta) ---
        betas = torch.linspace(beta_start, beta_end, n_diffusion_train_steps)
        alphas = 1.0 - betas
        ab = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", ab)

    # ------------------------------------------------------------- 基础前向
    def encode(self, obs: Tensor) -> Tensor:
        return self.encoder(obs)

    def q_values(self, z: Tensor, action: Tensor, use_target: bool = False):
        qa = torch.cat([z, action], dim=-1)
        if use_target:
            return self.q1_target(qa), self.q2_target(qa)
        return self.q1(qa), self.q2(qa)

    def q_min(self, z: Tensor, action: Tensor, use_target: bool = False) -> Tensor:
        q1, q2 = self.q_values(z, action, use_target)
        return torch.min(q1, q2)

    def value(self, z: Tensor) -> Tensor:
        return self.v(z)

    # ------------------------------------------------------------- 扩散
    def _diffuse(self, action: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """前向扩散: a_t = sqrt(ab_t) a_0 + sqrt(1-ab_t) eps。"""
        ab = self.alphas_cumprod[t][:, None]
        return ab.sqrt() * action + (1.0 - ab).sqrt() * noise

    def actor_loss(self, z: Tensor, action: Tensor, weights: Tensor) -> Tensor:
        """AWR 加权去噪损失: w * ||eps_theta(a_t, t, z) - eps||^2。"""
        bsz = action.shape[0]
        t = torch.randint(0, self.n_train_steps, (bsz,), device=action.device)
        noise = torch.randn_like(action)
        a_noisy = self._diffuse(action, t, noise)
        t_e = self.t_emb(t)
        eps_pred = self.actor(torch.cat([z, a_noisy, t_e], dim=-1))
        per = (eps_pred - noise).pow(2).mean(dim=-1)
        return (weights * per).mean(), per.mean().detach()

    @torch.no_grad()
    def _ddim_sample(self, z: Tensor, n_infer: int | None = None) -> Tensor:
        """DDIM 采样(给定 latent, 返回动作)。"""
        n_infer = n_infer or self.n_infer_steps
        bsz = z.shape[0]
        a = torch.randn(bsz, self.act_dim, device=z.device)
        t_seq = torch.linspace(self.n_train_steps - 1, 0, n_infer).round().long().clamp(0, self.n_train_steps - 1)
        t_seq = torch.unique(t_seq)
        for i in range(len(t_seq) - 1, -1, -1):
            t = int(t_seq[i])
            ab_t = self.alphas_cumprod[t]
            eps = self.actor(torch.cat([z, a, self.t_emb(torch.full((bsz,), t, device=z.device, dtype=torch.long))], dim=-1))
            x0 = ((a - (1.0 - ab_t).sqrt() * eps) / ab_t.sqrt()).clamp(-1.0, 1.0)
            if i > 0:
                ab_prev = self.alphas_cumprod[int(t_seq[i - 1])]
                a = ab_prev.sqrt() * x0 + (1.0 - ab_prev).sqrt() * eps
            else:
                a = x0
        return a.clamp(-1.0, 1.0)

    @torch.no_grad()
    def sample_action(self, obs: Tensor, n_infer: int | None = None) -> Tensor:
        """DDIM 单样本(策略随机)。"""
        return self._ddim_sample(self.encode(obs), n_infer)

    @torch.no_grad()
    def sample_action_best_of_k(self, obs: Tensor, k: int = 8,
                                n_infer: int | None = None) -> Tensor:
        """扩散多模态提议 + Q 筛选: 采样 k 个候选, 取 Q 最大的。

        扩散策略天然多模态(直推/微摆/回退), 单次采样可能抽到歪插类坏动作;
        用训练好的 Q 在线筛选, 把"模式选择"交给价值网络。推理期专用。
        """
        z = self.encode(obs)
        cands = [self._ddim_sample(z, n_infer) for _ in range(k)]
        qs = torch.cat([self.q_min(z, c) for c in cands], dim=0)   # (k, 1) or (k,)
        best = int(torch.argmax(qs.squeeze(-1)))
        return cands[best]

    # ------------------------------------------------------------- 目标网更新
    def update_targets(self) -> None:
        with torch.no_grad():
            for p, p_t in zip(self.q1.parameters(), self.q1_target.parameters()):
                p_t.data.mul_(1 - self.target_update_rate).add_(self.target_update_rate * p.data)
            for p, p_t in zip(self.q2.parameters(), self.q2_target.parameters()):
                p_t.data.mul_(1 - self.target_update_rate).add_(self.target_update_rate * p.data)

    # ------------------------------------------------------------- 损失
    def compute_losses(self, obs: Tensor, action: Tensor, reward: Tensor,
                       next_obs: Tensor, done: Tensor) -> dict[str, Tensor]:
        """一次 IDQL 全损失前向(Q/V/actor 三项, 分别 backward 由训练器处理)。"""
        z = self.encode(obs)
        with torch.no_grad():
            z_next = self.encode(next_obs)
            v_next = self.value(z_next)
            q_target = reward + self.discount * (1.0 - done) * v_next.squeeze(-1)

        # Q: TD 回归(目标来自 V(s'), detach)
        q1_pred, q2_pred = self.q_values(z, action)
        q1_loss = torch.nn.functional.mse_loss(q1_pred.squeeze(-1), q_target)
        q2_loss = torch.nn.functional.mse_loss(q2_pred.squeeze(-1), q_target)
        q_loss = q1_loss + q2_loss

        # V: 期望回归 toward min(Q1,Q2)(s, a_data)(online Q, detach)
        with torch.no_grad():
            q_data = self.q_min(z, action)
        v_pred = self.value(z)
        diff = q_data - v_pred.squeeze(-1)
        weight = torch.where(diff > 0, self.expectile_tau, 1.0 - self.expectile_tau)
        v_loss = (weight * diff.pow(2)).mean()

        # actor: AWR 加权扩散 BC(优势 = Q - V, 标准化后 exp(A/beta))
        with torch.no_grad():
            adv = (q_data - v_pred.squeeze(-1))
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)
            w = torch.exp(adv / self.awr_beta).clamp(max=100.0)
            w = w / w.mean()
        actor_loss, raw_diff = self.actor_loss(z, action, w)

        return {
            "q_loss": q_loss,
            "v_loss": v_loss,
            "actor_loss": actor_loss,
            "q_mean": q1_pred.mean().detach(),
            "v_mean": v_pred.mean().detach(),
            "adv_std": adv.std().detach(),
            "raw_diffusion_loss": raw_diff,
        }
