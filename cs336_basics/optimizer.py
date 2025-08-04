from __future__ import annotations

import os
from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int
import traceback
import math
import numpy.typing as npt
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
import torch.optim as optim
import math
from einops import einsum, rearrange
from cs336_basics.layer import *



class LR_Cosine_Schedule(object):
    def __init__(self,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int
    ):
        self.max_lr = max_learning_rate
        self.min_lr = min_learning_rate
        self.warmup_iters = warmup_iters
        self.cosine_cycle_iters = cosine_cycle_iters

    def get_lr_at_it(self, it):
        current_lr = 0
        if it <= self.warmup_iters:
            current_lr = it/self.warmup_iters * self.max_lr
        elif it < self.cosine_cycle_iters:
            degree = (it-self.warmup_iters)  / (self.cosine_cycle_iters-self.warmup_iters) * math.pi
            current_lr = math.cos(degree) * (self.max_lr - self.min_lr) / 2 + (self.max_lr + self.min_lr) / 2
        else:
            current_lr = self.min_lr
        return current_lr




class CustomOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, **kwargs):
        defaults = dict(lr=lr, **kwargs)
        super(CustomOptimizer, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """执行一步优化"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # 在这里实现具体的参数更新逻辑
                grad = p.grad.data
                # 更新参数
                p.data.add_(grad, alpha=-group['lr'])
        
        return loss




# 1. 定义自定义优化器类，继承自 torch.optim.Optimizer
class CustomSGD(optim.Optimizer):
    """
    一个自定义的、最简化的 SGD 优化器实现。
    """
    
    # 2. 实现构造函数 __init__
    def __init__(self, params, lr=0.01):
        """
        Args:
            params (iterable): 模型参数的可迭代对象。
            lr (float, optional): 学习率 (默认: 0.01)。
        """
        # 检查学习率是否有效
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
            
        # 定义优化器的默认超参数
        defaults = dict(lr=lr)
        
        # 调用父类的构造函数，这是必须的步骤
        super(CustomSGD, self).__init__(params, defaults)

    # 3. 实现核心方法 step
    @torch.no_grad() # 使用 @torch.no_grad() 装饰器，因为我们是在更新参数，而不是计算梯度
    def step(self, closure=None):
        """
        执行单步优化。

        Args:
            closure (callable, optional): 一个可以重新评估模型并返回损失的闭包。
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 遍历所有的参数组 (param_group)
        # 在我们的简单例子中，只有一个参数组
        for group in self.param_groups:
            # 获取该组的学习率
            lr = group['lr']
            
            # 遍历该组中的所有参数 (p)
            for p in group['params']:
                # 检查参数是否有梯度
                if p.grad is None:
                    continue
                
                # 获取梯度
                grad = p.grad
                
                # 应用 SGD 更新规则
                # p.add_(value, alpha=...) 等价于 p = p + alpha * value
                # 这里我们执行 p = p - lr * grad
                p.add_(grad, alpha=-lr)
                
        return loss


class CustomAdam(optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        """
        初始化函数。
        Args:
            params (iterable): 模型参数的可迭代对象。
            lr (float, optional): 学习率 (默认: 0.001)。
            betas (Tuple[float, float], optional): 用于计算梯度及其平方的运行平均值的系数 (默认: (0.9, 0.999))。
            eps (float, optional): 为了数值稳定性而添加到分母的一项 (默认: 1e-8)。
            weight_decay (float, optional): 权重衰减 (L2惩罚) (默认: 0)。
        """
        # --- 参数合法性检查 ---
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        # --- 定义默认超参数字典 ---
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        
        # --- 调用父类构造函数 ---
        super(CustomAdam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        执行单步优化。

        Args:
            closure (callable, optional): 一个可以重新评估模型并返回损失的闭包。
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 遍历所有参数组 (通常只有一个)
        for group in self.param_groups:
            # 获取当前参数组的超参数
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            # 遍历该组中的所有参数 (p)
            for p in group['params']:
                if p.grad is None:
                    continue # 如果参数没有梯度，则跳过
                
                grad = p.grad # 获取梯度

                # --- Adam 核心逻辑 ---

                # 获取该参数的状态 (state)
                # self.state 是 Optimizer 基类提供的字典，用于存储每个参数的持久化状态
                state = self.state[p]

                # 延迟初始化状态 (如果第一次遇到该参数)
                if len(state) == 0:
                    state['step'] = 0
                    # 一阶矩向量 (m)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # 二阶矩向量 (v)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                # 获取该参数的一阶矩和二阶矩估计值
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                # 增加时间步
                state['step'] += 1
                
                # 0. (可选) 应用权重衰减
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # 1. 更新一阶矩估计 (m_t)
                # exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # 2. 更新二阶矩估计 (v_t)
                # exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 3. 计算偏差校正系数
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # 计算校正后的 m_hat 和 v_hat 的分母部分
                # denorm = (sqrt(v_hat) + eps)
                # 我们先计算 sqrt(v_t) / sqrt(bias_correction2)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                # 4. 计算最终的步长并更新参数
                # step_size = lr / bias_correction1
                step_size = lr / bias_correction1
                
                # 更新参数: p = p - step_size * (m_t / denom)
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class CustomAdamW(optim.Optimizer):
    """
    一个从头实现的自定义 AdamW 优化器。
    它实现了与 torch.optim.AdamW 相同的解耦权重衰减。
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        """
        初始化函数。

        Args:
            params (iterable): 模型参数的可迭代对象。
            lr (float, optional): 学习率 (默认: 0.001)。
            betas (Tuple[float, float], optional): 用于计算梯度及其平方的运行平均值的系数 (默认: (0.9, 0.999))。
            eps (float, optional): 为了数值稳定性而添加到分母的一项 (默认: 1e-8)。
            weight_decay (float, optional): 权重衰减系数 (默认: 0.01)。
        """
        # --- 参数合法性检查 ---
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        # --- 定义默认超参数字典 ---
        # 注意这里的 weight_decay 是 AdamW 的核心
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        
        # --- 调用父类构造函数 ---
        super(CustomAdamW, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        执行单步优化。
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # 获取当前参数组的超参数
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad

                # --- 核心区别 1: AdamW 不将 weight_decay 添加到梯度中 ---
                # 在标准的 Adam+L2 实现中，会有类似下面这行代码，这里没有：
                # grad = grad.add(p, alpha=weight_decay)

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                
                # 执行 Adam 的标准动量和二阶矩更新
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 计算偏差校正
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                step_size = lr / bias_correction1
                
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                
                # --- AdamW 的更新分为两步 ---
                
                # 步骤 A: 执行 Adam 的主要更新（不含权重衰减）
                p.addcdiv_(exp_avg, denom, value=-step_size)

                # 步骤 B: 执行解耦的权重衰减
                # 核心区别 2: 直接在参数上应用衰减
                # 这个操作等价于: p.data = p.data - lr * weight_decay * p.data
                # 注意：权重衰减的量与学习率 lr 相关联，这是 PyTorch 官方 AdamW 的标准做法。
                if weight_decay != 0:
                    p.add_(p, alpha=-weight_decay * lr)

        return loss
