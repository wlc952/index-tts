#!/usr/bin/env python3
"""
终极版 BigVGAN ONNX 导出 - 保留完整功能并解决 ONNX 兼容性
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import numpy as np
from omegaconf import OmegaConf

from indextts.BigVGAN.models import BigVGAN as Generator
import indextts.BigVGAN.activations as activations
from indextts.BigVGAN.alias_free_torch.filter import kaiser_sinc_filter1d

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️ ONNXRuntime not available. Install it with: pip install onnxruntime")

# 关闭警告
warnings.filterwarnings("ignore")


class ONNXSnakeBeta(nn.Module):
    """ONNX 兼容的 SnakeBeta - 与原始功能完全一致"""
    
    def __init__(self, original_snakebeta):
        super().__init__()
        self.in_features = original_snakebeta.in_features
        self.alpha_logscale = original_snakebeta.alpha_logscale
        self.no_div_by_zero = original_snakebeta.no_div_by_zero
        
        self.alpha = nn.Parameter(original_snakebeta.alpha.data.clone())
        self.beta = nn.Parameter(original_snakebeta.beta.data.clone())
    
    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        
        x = x + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(torch.sin(x * alpha), 2)
        return x


class ONNXLowPassFilter1d(nn.Module):
    """ONNX 兼容的静态 LowPassFilter1d"""
    
    def __init__(self, original_filter, channels):
        super().__init__()
        self.stride = original_filter.stride
        self.pad_left = original_filter.pad_left
        self.pad_right = original_filter.pad_right
        self.padding_mode = original_filter.padding_mode
        self.channels = channels
        
        # 创建静态卷积层，预先为指定通道数扩展权重
        filter_weight = original_filter.filter.data.clone()  # [1, 1, kernel_size]
        # 扩展到指定通道数 [channels, 1, kernel_size]
        expanded_weight = filter_weight.expand(channels, -1, -1)
        
        # 创建静态卷积层
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=filter_weight.shape[-1],
            stride=self.stride,
            groups=channels,
            bias=False
        )
        
        # 设置权重
        with torch.no_grad():
            self.conv.weight.copy_(expanded_weight)
    
    def forward(self, x):
        # Padding
        if self.pad_left > 0 or self.pad_right > 0:
            x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        
        # 使用静态卷积
        out = self.conv(x)
        return out


class ONNXUpSample1d(nn.Module):
    """ONNX 兼容的静态 UpSample1d"""
    
    def __init__(self, original_upsample, channels):
        super().__init__()
        self.ratio = original_upsample.ratio
        self.stride = original_upsample.stride
        self.pad = original_upsample.pad
        self.pad_left = original_upsample.pad_left
        self.pad_right = original_upsample.pad_right
        self.channels = channels
        
        # 创建静态转置卷积层
        filter_weight = original_upsample.filter.data.clone()  # [1, 1, kernel_size]
        # 扩展到指定通道数 [channels, 1, kernel_size]
        expanded_weight = filter_weight.expand(channels, -1, -1)
        
        # 创建静态转置卷积层
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=filter_weight.shape[-1],
            stride=self.stride,
            groups=channels,
            bias=False
        )
        
        # 设置权重
        with torch.no_grad():
            self.conv_transpose.weight.copy_(expanded_weight)
    
    def forward(self, x):
        # Padding
        x = F.pad(x, (self.pad, self.pad), mode='replicate')
        
        # 使用静态转置卷积
        x = self.ratio * self.conv_transpose(x)
        
        # 截取输出
        if self.pad_right > 0:
            x = x[..., self.pad_left:-self.pad_right]
        else:
            x = x[..., self.pad_left:]
        
        return x


class ONNXDownSample1d(nn.Module):
    """ONNX 兼容的静态 DownSample1d"""
    
    def __init__(self, original_downsample, channels):
        super().__init__()
        self.lowpass = ONNXLowPassFilter1d(original_downsample.lowpass, channels)
    
    def forward(self, x):
        return self.lowpass(x)


class ONNXActivation1d(nn.Module):
    """ONNX 兼容的静态 Activation1d"""
    
    def __init__(self, original_activation, channels):
        super().__init__()
        self.act = original_activation.act  # 保留原始激活函数（已被替换为ONNX兼容版本）
        self.upsample = ONNXUpSample1d(original_activation.upsample, channels)
        self.downsample = ONNXDownSample1d(original_activation.downsample, channels)
    
    def forward(self, x):
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)
        return x



def verify_onnx_model(onnx_path, pytorch_model, test_inputs):
    """使用ONNXRuntime验证ONNX模型"""
    if not ONNX_AVAILABLE:
        print("   ❌ ONNXRuntime不可用，跳过验证")
        return False
    
    print(f"7. 验证ONNX模型: {onnx_path}")
    
    try:
        # 加载ONNX模型
        print("   正在加载ONNX模型...")
        ort_session = ort.InferenceSession(onnx_path)
        
        # 获取输入输出信息
        input_names = [input.name for input in ort_session.get_inputs()]
        output_names = [output.name for output in ort_session.get_outputs()]
        
        print(f"   ONNX输入名称: {input_names}")
        print(f"   ONNX输出名称: {output_names}")
        
        # 准备输入数据
        if isinstance(test_inputs, tuple):
            inputs_dict = {}
            for i, input_tensor in enumerate(test_inputs):
                if i < len(input_names):
                    inputs_dict[input_names[i]] = input_tensor.numpy()
        else:
            inputs_dict = {input_names[0]: test_inputs.numpy()}
        
        # ONNX推理
        print("   正在运行ONNX推理...")
        onnx_outputs = ort_session.run(output_names, inputs_dict)
        onnx_output = torch.tensor(onnx_outputs[0])
        
        # PyTorch推理
        print("   正在运行PyTorch推理...")
        with torch.no_grad():
            if isinstance(test_inputs, tuple):
                pytorch_output = pytorch_model(*test_inputs)
            else:
                pytorch_output = pytorch_model(test_inputs)
        
        # 比较输出
        print("   正在比较输出...")
        print(f"   PyTorch输出形状: {pytorch_output.shape}")
        print(f"   ONNX输出形状: {onnx_output.shape}")
        print(f"   PyTorch输出范围: [{pytorch_output.min():.6f}, {pytorch_output.max():.6f}]")
        print(f"   ONNX输出范围: [{onnx_output.min():.6f}, {onnx_output.max():.6f}]")
        
        # 计算差异
        diff = torch.norm(pytorch_output - onnx_output).item()
        pytorch_norm = torch.norm(pytorch_output).item()
        diff_ratio = diff / pytorch_norm if pytorch_norm > 0 else 0
        
        print(f"   输出差异: {diff:.8f}")
        print(f"   输出差异比例: {diff_ratio:.8f} ({diff_ratio*100:.6f}%)")
        
        # 判断验证结果
        tolerance = 1e-4  # 容差
        if diff_ratio < tolerance:
            print(f"   ✅ ONNX模型验证成功！差异在容差范围内 (<{tolerance*100:.4f}%)")
            return True
        else:
            print(f"   ⚠️ ONNX模型验证失败！差异超出容差范围 (>{tolerance*100:.4f}%)")
            return False
            
    except Exception as e:
        print(f"   ❌ ONNX验证过程中出错: {str(e)}")
        return False


def create_npu_compatible_model():
    """创建NPU兼容的模型，解决浮点异常问题"""
    print("=== 创建NPU兼容的BigVGAN模型 ===")
    
    # 1. 加载原始模型
    print("1. 加载原始模型...")
    cfg = OmegaConf.load("checkpoints/config.yaml")
    
    model = Generator(cfg.bigvgan, use_cuda_kernel=False)
    checkpoint = torch.load("checkpoints/bigvgan_generator.pth", map_location="cpu")
    model.load_state_dict(checkpoint["generator"])
    model.eval()
    model.remove_weight_norm()
    
    # 2. 替换所有不兼容的模块
    print("2. 替换NPU不兼容的模块...")
    
    # 定义通道数映射
    channels_map = {
        0: 768, 1: 768, 2: 768,
        3: 384, 4: 384, 5: 384,
        6: 192, 7: 192, 8: 192,
        9: 96, 10: 96, 11: 96,
        12: 48, 13: 48, 14: 48,
        15: 24, 16: 24, 17: 24,
        'post': 24
    }
    
    def replace_for_npu(module, module_path=""):
        for name, child in module.named_children():
            current_path = f"{module_path}.{name}" if module_path else name
            
            if isinstance(child, activations.SnakeBeta):
                setattr(module, name, NPUSnakeBeta(child))
                print(f"   替换了 SnakeBeta: {current_path}")
            elif hasattr(child, '__class__') and 'Activation1d' in child.__class__.__name__:
                # 确定通道数
                channels = 768  # 默认值
                if 'resblocks' in current_path:
                    parts = current_path.split('.')
                    for i, part in enumerate(parts):
                        if part == 'resblocks' and i + 1 < len(parts):
                            try:
                                idx = int(parts[i + 1])
                                channels = channels_map.get(idx, 768)
                                break
                            except (ValueError, IndexError):
                                pass
                elif 'activation_post' in current_path:
                    channels = channels_map['post']
                
                # 替换为NPU兼容版本
                npu_activation = NPUActivation1d(child, channels)
                if isinstance(npu_activation.act, activations.SnakeBeta):
                    npu_activation.act = NPUSnakeBeta(npu_activation.act)
                setattr(module, name, npu_activation)
                print(f"   替换了 Activation1d: {current_path} (通道数: {channels})")
            else:
                replace_for_npu(child, current_path)
    
    replace_for_npu(model)
    return model


class NPUSnakeBeta(nn.Module):
    """NPU兼容的SnakeBeta，极度保守的数值稳定性"""
    
    def __init__(self, original_snakebeta):
        super().__init__()
        self.in_features = original_snakebeta.in_features
        self.alpha_logscale = original_snakebeta.alpha_logscale
        self.no_div_by_zero = original_snakebeta.no_div_by_zero
        
        # 限制参数范围，防止极值
        alpha_data = torch.clamp(original_snakebeta.alpha.data.clone(), -5.0, 5.0)
        beta_data = torch.clamp(original_snakebeta.beta.data.clone(), -5.0, 5.0)
        
        self.alpha = nn.Parameter(alpha_data)
        self.beta = nn.Parameter(beta_data)
        
        # 极严格的数值稳定性参数
        self.eps = 1e-6
        self.max_val = 5.0  # 更严格的范围限制
        self.min_val = -5.0
        self.sin_scale = 0.1  # 减小sin函数的影响
    
    def forward(self, x):
        # 极严格的输入裁剪
        x = torch.clamp(x, self.min_val, self.max_val)
        
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        
        # 严格限制参数范围
        alpha = torch.clamp(alpha, -3.0, 3.0)
        beta = torch.clamp(beta, -3.0, 3.0)
        
        if self.alpha_logscale:
            # 更严格的指数范围限制
            alpha = torch.clamp(alpha, -5, 5)
            beta = torch.clamp(beta, -5, 5)
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        
        # 确保beta不会太小或太大
        beta = torch.clamp(beta, 0.1, 10.0) + self.eps
        alpha = torch.clamp(alpha, 0.01, 5.0)
        
        # 限制sin输入，使用更小的scale
        sin_input = torch.clamp(x * alpha * self.sin_scale, -10, 10)
        
        # 使用更稳定的sin计算
        sin_val = torch.sin(sin_input)
        sin_squared = sin_val * sin_val  # 避免pow操作
        
        # 限制除法结果
        div_result = torch.clamp(1.0 / beta, 0.01, 100.0)
        
        # 计算结果并严格限制范围
        result = x + div_result * sin_squared * 0.1  # 进一步减小激活强度
        result = torch.clamp(result, self.min_val, self.max_val)
        
        return result


class NPULowPassFilter1d(nn.Module):
    """NPU兼容的静态LowPassFilter1d，极度保守的数值稳定性，精确控制输出尺寸"""
    
    def __init__(self, original_filter, channels):
        super().__init__()
        self.stride = max(1, original_filter.stride)  # 确保stride >= 1
        self.pad_left = max(0, original_filter.pad_left)
        self.pad_right = max(0, original_filter.pad_right)
        self.padding_mode = 'constant'
        self.channels = channels
        
        # 创建极保守的滤波器权重
        filter_weight = original_filter.filter.data.clone()
        
        # 权重归一化和稳定化
        weight_sum = torch.sum(torch.abs(filter_weight)) + 1e-8
        filter_weight = filter_weight / weight_sum
        filter_weight = torch.clamp(filter_weight, -0.1, 0.1)  # 严格限制权重范围
        
        # 确保权重和为1（归一化）
        filter_weight = filter_weight / (torch.sum(filter_weight) + 1e-8)
        
        expanded_weight = filter_weight.expand(channels, -1, -1)
        
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=filter_weight.shape[-1],
            stride=self.stride,
            groups=channels,
            bias=False,
            padding=0  # 手动处理padding
        )
        
        with torch.no_grad():
            self.conv.weight.copy_(expanded_weight)
            # 进一步限制权重
            self.conv.weight.clamp_(-0.05, 0.05)
    
    def forward(self, x):
        # 记录输入长度，用于计算期望输出长度
        input_length = x.shape[-1]
        expected_output_length = input_length // self.stride
        
        # 极严格的输入范围限制
        x = torch.clamp(x, -3.0, 3.0)
        
        # 手动padding，使用更小的pad值
        actual_pad_left = min(self.pad_left, 10)
        actual_pad_right = min(self.pad_right, 10)
        
        if actual_pad_left > 0 or actual_pad_right > 0:
            x = F.pad(x, (actual_pad_left, actual_pad_right), mode='constant', value=0.0)
        
        out = self.conv(x)
        
        # 确保输出长度符合预期
        current_length = out.shape[-1]
        if current_length > expected_output_length:
            # 如果过长，从中心裁剪
            excess = current_length - expected_output_length
            start = excess // 2
            out = out[..., start:start + expected_output_length]
        elif current_length < expected_output_length and expected_output_length > 0:
            # 如果过短，填充
            deficit = expected_output_length - current_length
            pad_left = deficit // 2
            pad_right = deficit - pad_left
            out = F.pad(out, (pad_left, pad_right), mode='constant', value=0.0)
        
        # 极严格的输出范围限制
        out = torch.clamp(out, -3.0, 3.0)
        return out


class NPUUpSample1d(nn.Module):
    """NPU兼容的静态UpSample1d，极保守设计，精确控制输出尺寸"""
    
    def __init__(self, original_upsample, channels):
        super().__init__()
        self.ratio = min(original_upsample.ratio, 4)  # 限制ratio避免过度放大
        self.stride = max(1, original_upsample.stride)
        self.pad = max(0, min(original_upsample.pad, 5))  # 限制pad大小
        self.pad_left = max(0, min(original_upsample.pad_left, 10))
        self.pad_right = max(0, min(original_upsample.pad_right, 10))
        self.channels = channels
        
        # 极保守的权重处理
        filter_weight = original_upsample.filter.data.clone()
        
        # 权重归一化和限制
        weight_abs_sum = torch.sum(torch.abs(filter_weight)) + 1e-8
        filter_weight = filter_weight / weight_abs_sum
        filter_weight = torch.clamp(filter_weight, -0.05, 0.05)
        
        # 再次归一化确保稳定
        filter_weight = filter_weight / (torch.sum(torch.abs(filter_weight)) + 1e-8)
        
        expanded_weight = filter_weight.expand(channels, -1, -1)
        
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=filter_weight.shape[-1],
            stride=self.stride,
            groups=channels,
            bias=False,
            padding=0  # 手动处理padding
        )
        
        with torch.no_grad():
            self.conv_transpose.weight.copy_(expanded_weight)
            # 进一步限制权重
            self.conv_transpose.weight.clamp_(-0.02, 0.02)
    
    def forward(self, x):
        # 记录输入尺寸，用于计算期望的输出尺寸
        input_length = x.shape[-1]
        expected_output_length = input_length * self.ratio
        
        # 极严格的输入范围限制
        x = torch.clamp(x, -2.0, 2.0)
        
        # 保守的padding
        actual_pad = min(self.pad, 3)
        if actual_pad > 0:
            x = F.pad(x, (actual_pad, actual_pad), mode='constant', value=0.0)
        
        # 限制ratio避免溢出，使用更小的放大倍数
        safe_ratio = min(self.ratio, 2.0)
        x = safe_ratio * self.conv_transpose(x)
        
        # 精确控制输出尺寸
        current_length = x.shape[-1]
        
        # 如果输出过长，从中心裁剪到期望长度
        if current_length > expected_output_length:
            excess = current_length - expected_output_length
            start = excess // 2
            x = x[..., start:start + expected_output_length]
        # 如果输出过短，在两端填充
        elif current_length < expected_output_length:
            deficit = expected_output_length - current_length
            pad_left = deficit // 2
            pad_right = deficit - pad_left
            x = F.pad(x, (pad_left, pad_right), mode='constant', value=0.0)
        
        # 极严格的输出范围限制
        x = torch.clamp(x, -2.0, 2.0)
        return x


class NPUDownSample1d(nn.Module):
    """NPU兼容的静态DownSample1d"""
    
    def __init__(self, original_downsample, channels):
        super().__init__()
        self.lowpass = NPULowPassFilter1d(original_downsample.lowpass, channels)
    
    def forward(self, x):
        return self.lowpass(x)


class NPUActivation1d(nn.Module):
    """NPU兼容的静态Activation1d，极保守设计，确保维度匹配"""
    
    def __init__(self, original_activation, channels):
        super().__init__()
        self.act = original_activation.act
        self.upsample = NPUUpSample1d(original_activation.upsample, channels)
        self.downsample = NPUDownSample1d(original_activation.downsample, channels)
        
        # 添加额外的稳定性参数
        self.eps = 1e-7
        self.max_val = 2.0
        self.min_val = -2.0
    
    def _match_dimensions(self, processed, original):
        """确保处理后的张量与原始张量维度匹配"""
        if processed.shape != original.shape:
            # 如果时间维度不匹配，进行调整
            if len(processed.shape) == 3 and len(original.shape) == 3:
                if processed.shape[2] > original.shape[2]:
                    # 裁剪到原始大小
                    diff = processed.shape[2] - original.shape[2]
                    start = diff // 2
                    processed = processed[..., start:start + original.shape[2]]
                elif processed.shape[2] < original.shape[2]:
                    # 填充到原始大小
                    diff = original.shape[2] - processed.shape[2]
                    pad_left = diff // 2
                    pad_right = diff - pad_left
                    processed = F.pad(processed, (pad_left, pad_right), mode='constant', value=0.0)
        
        return processed
    
    def forward(self, x):
        # 保存原始形状用于维度匹配
        original_shape = x.shape
        
        # 每步都进行极严格的范围限制
        x = torch.clamp(x, self.min_val, self.max_val)
        
        # 检查是否有异常值
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.zeros_like(x)
        
        # 保存激活前的值用于残差连接
        input_for_residual = x.clone()
        
        x = self.upsample(x)
        x = torch.clamp(x, self.min_val, self.max_val)
        
        # 再次检查异常值
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.zeros_like(x)
        
        x = self.act(x)
        x = torch.clamp(x, self.min_val, self.max_val)
        
        # 再次检查异常值
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.zeros_like(x)
        
        x = self.downsample(x)
        x = torch.clamp(x, self.min_val, self.max_val)
        
        # 确保输出维度与输入匹配
        x = self._match_dimensions(x, input_for_residual)
        
        # 最终检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.zeros_like(x)
            
        return x


def export_npu_compatible_onnx():
    """导出NPU兼容的ONNX模型"""
    print("=== BigVGAN NPU兼容 ONNX 导出 ===")
    
    # 创建NPU兼容模型
    model = create_npu_compatible_model()
    
    # 准备测试数据
    print("3. 准备测试数据...")
    latent = torch.randn(1, 256, 1280)
    # 限制输入范围
    latent = torch.clamp(latent, -3.0, 3.0)
    speaker_embedding = torch.randn(1, 512, 1)
    speaker_embedding = torch.clamp(speaker_embedding, -3.0, 3.0)
    
    # 创建包装器
    class NPUBigVGANWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.eps = 1e-7
            self.max_val = 2.0
            self.min_val = -2.0
        
        def _safe_clamp(self, x, desc=""):
            """安全的数值裁剪，包含异常值检查"""
            # 检查异常值
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"警告: 在{desc}检测到异常值，已清零")
                x = torch.zeros_like(x)
            
            # 裁剪到安全范围
            x = torch.clamp(x, self.min_val, self.max_val)
            return x
        
        def forward(self, x, embedding):
            # 输入范围极严格限制
            x = self._safe_clamp(x, "输入x")
            embedding = self._safe_clamp(embedding, "输入embedding")
            
            x = x.transpose(1, 2)
            x = self._safe_clamp(x, "转置后")
            
            x = self.model.conv_pre(x)
            x = self._safe_clamp(x, "conv_pre后")
            
            cond_out = self.model.cond_layer(embedding)
            cond_out = self._safe_clamp(cond_out, "cond_layer后")
            
            x = x + cond_out
            x = self._safe_clamp(x, "加cond后")
            
            for i in range(self.model.num_upsamples):
                # upsampling
                for i_up in range(len(self.model.ups[i])):
                    x = self.model.ups[i][i_up](x)
                    x = self._safe_clamp(x, f"up_{i}_{i_up}后")
                
                cond_up = self.model.conds[i](embedding)
                cond_up = self._safe_clamp(cond_up, f"cond_{i}后")
                
                x = x + cond_up
                x = self._safe_clamp(x, f"加cond_{i}后")
                
                # AMP blocks with safer aggregation
                xs_list = []
                for j in range(self.model.num_kernels):
                    block_out = self.model.resblocks[i * self.model.num_kernels + j](x)
                    block_out = self._safe_clamp(block_out, f"resblock_{i}_{j}后")
                    xs_list.append(block_out)
                
                # 更安全的求平均
                if xs_list:
                    x = torch.stack(xs_list, dim=0).mean(dim=0)
                    x = self._safe_clamp(x, f"resblock平均_{i}后")
            
            # post conv
            x = self.model.activation_post(x)
            x = self._safe_clamp(x, "activation_post后")
            
            x = self.model.conv_post(x)
            x = self._safe_clamp(x, "conv_post后")
            
            # 最终tanh前再次限制到更小范围
            x = torch.clamp(x, -0.99, 0.99)
            x = torch.tanh(x)
            
            return x
    
    # 导出ONNX
    os.makedirs("onnx", exist_ok=True)
    wrapped_model = NPUBigVGANWrapper(model)
    
    print("4. 导出NPU兼容的ONNX模型...")
    torch.onnx.export(
        wrapped_model,
        (latent, speaker_embedding),
        "onnx/bigvgan_npu_compatible.onnx",
        input_names=["latent", "speaker_embedding"],
        output_names=["audio"],
        opset_version=11,  # 使用更低版本以提高NPU兼容性
        do_constant_folding=True,
        verbose=False,
        dynamic_axes=None,  # 禁用动态轴
        export_params=True,
        keep_initializers_as_inputs=False,
        training=torch.onnx.TrainingMode.EVAL
    )
    print("   NPU兼容ONNX模型导出成功！")
    
    # 验证模型
    if ONNX_AVAILABLE:
        print("5. 验证NPU兼容模型...")
        verify_onnx_model("onnx/bigvgan_npu_compatible.onnx", wrapped_model, (latent, speaker_embedding))
    
    return True


def export_ultimate_onnx():
    print("=== BigVGAN ONNX 导出 ===")
    
    # 1. 加载原始模型
    print("1. 加载原始模型...")
    cfg = OmegaConf.load("checkpoints/config.yaml")
    
    model = Generator(cfg.bigvgan, use_cuda_kernel=False)
    checkpoint = torch.load("checkpoints/bigvgan_generator.pth", map_location="cpu")
    model.load_state_dict(checkpoint["generator"])
    model.eval()
    model.remove_weight_norm()
    
    print(f"   模型加载成功，激活函数类型: {cfg.bigvgan.activation}")
    
    # 2. 准备测试数据
    print("2. 准备测试数据...")
    latent = torch.randn(1, 256, 1280)
    mel_ref = torch.randn(1, 300, 100)
    
    # 3. 获取原始输出作为参考
    print("3. 获取原始模型输出...")
    with torch.no_grad():
        original_output = model(latent, mel_ref)
        if isinstance(original_output, tuple):
            original_output = original_output[0]
    
    print(f"   原始输出形状: {original_output.shape}")
    print(f"   原始输出范围: [{original_output.min():.4f}, {original_output.max():.4f}]")
    
    # 4. 替换所有不兼容的模块
    print("4. 替换ONNX不兼容的模块...")
    replaced_count = 0
    activation_count = 0
    
    # 定义BigVGAN中不同层的通道数
    # 根据模型架构确定每层的通道数
    channels_map = {
        # resblocks中的通道数，按层级排列
        0: 768,   # 第一层 resblocks
        1: 768,   
        2: 768,   
        3: 384,   # 第二层 resblocks
        4: 384,   
        5: 384,   
        6: 192,   # 第三层 resblocks
        7: 192,   
        8: 192,   
        9: 96,    # 第四层 resblocks
        10: 96,   
        11: 96,   
        12: 48,   # 第五层 resblocks
        13: 48,   
        14: 48,   
        15: 24,   # 第六层 resblocks
        16: 24,   
        17: 24,   
        'post': 24  # activation_post
    }
    
    def replace_incompatible_modules(module, module_path=""):
        nonlocal replaced_count, activation_count
        for name, child in module.named_children():
            current_path = f"{module_path}.{name}" if module_path else name
            
            if isinstance(child, activations.SnakeBeta):
                setattr(module, name, ONNXSnakeBeta(child))
                replaced_count += 1
                print(f"   替换了 SnakeBeta: {current_path}")
            elif hasattr(child, '__class__') and 'Activation1d' in child.__class__.__name__:
                # 根据路径确定通道数
                channels = None
                if 'resblocks' in current_path:
                    # 从路径中提取resblock索引
                    parts = current_path.split('.')
                    for part in parts:
                        if part.startswith('resblocks') and '.' in current_path:
                            try:
                                idx = int(parts[parts.index(part) + 1])
                                channels = channels_map.get(idx, 768)  # 默认768
                                break
                            except (ValueError, IndexError):
                                pass
                elif 'activation_post' in current_path:
                    channels = channels_map['post']
                
                if channels is None:
                    channels = 768  # 默认通道数
                
                # 替换 Activation1d 模块
                onnx_activation = ONNXActivation1d(child, channels)
                # 如果激活函数是 SnakeBeta，也需要替换
                if isinstance(onnx_activation.act, activations.SnakeBeta):
                    onnx_activation.act = ONNXSnakeBeta(onnx_activation.act)
                    replaced_count += 1
                setattr(module, name, onnx_activation)
                activation_count += 1
                print(f"   替换了 Activation1d: {current_path} (通道数: {channels})")
            else:
                replace_incompatible_modules(child, current_path)
    
    replace_incompatible_modules(model)
    print(f"   总共替换了 {replaced_count} 个 SnakeBeta")
    print(f"   总共替换了 {activation_count} 个 Activation1d")
    
    # 5. 测试输出差异
    print("5. 测试输出差异...")
    with torch.no_grad():
        new_output = model(latent, mel_ref)
        if isinstance(new_output, tuple):
            new_output = new_output[0]
    
    print(f"   新输出形状: {new_output.shape}")
    print(f"   新输出范围: [{new_output.min():.4f}, {new_output.max():.4f}]")
    
    diff = torch.norm(original_output - new_output).item()
    orig_norm = torch.norm(original_output).item()
    diff_ratio = diff / orig_norm if orig_norm > 0 else 0
    
    print(f"   输出差异: {diff:.6f}")
    print(f"   输出差异比例: {diff_ratio:.6f} ({diff_ratio*100:.4f}%)")
    
    if diff_ratio < 0.001:
        print("   ✅ SnakeBeta 替换完美！")
    else:
        print("   ⚠️ SnakeBeta 替换有差异，需要检查")
    
    # 6. ONNX 导出
    os.mkdir("onnx") if not os.path.exists("onnx") else None

    # 6.1 导出speaker_encoder
    print("6.1 导出 speaker_encoder...")
    torch.onnx.export(
        model.speaker_encoder,
        mel_ref,
        "onnx/bigvgan_speaker_encoder.onnx",
        input_names=["mel_ref"],
        output_names=["speaker_embedding"],
        opset_version=16,
        do_constant_folding=True,
        verbose=False
    )

    speaker_embedding = torch.randn(1, 512, 1)

    # 6.2 导出 bigvgan
    print("6.2 导出 BigVGAN 主模型...")
    class BigVGANWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, x, emdedding):
            x = x.transpose(1, 2)  # 转置以匹配输入形状
            x = self.model.conv_pre(x)

            x = x + self.model.cond_layer(speaker_embedding)

            for i in range(self.model.num_upsamples):
                # upsampling
                for i_up in range(len(self.model.ups[i])):
                    x = self.model.ups[i][i_up](x)

                x = x + self.model.conds[i](speaker_embedding)

                # AMP blocks
                xs = None
                for j in range(self.model.num_kernels):
                    if xs is None:
                        xs = self.model.resblocks[i * self.model.num_kernels + j](x)
                    else:
                        xs += self.model.resblocks[i * self.model.num_kernels + j](x)
                x = xs / self.model.num_kernels

            # post conv
            x = self.model.activation_post(x)
            x = self.model.conv_post(x)
            x = torch.tanh(x)

            return x
    
    wrapped_model = BigVGANWrapper(model)
    torch.onnx.export(
        wrapped_model,
        (latent, speaker_embedding),
        "onnx/bigvgan.onnx",
        input_names=["latent", "speaker_embedding"],
        output_names=["audio"],
        opset_version=16,
        do_constant_folding=True,
        verbose=False
    )
    print("   BigVGAN 主模型导出成功！")
    
    # 7. 验证ONNX模型
    print("\n=== ONNX模型验证 ===")
    verify_success = verify_onnx_model("onnx/bigvgan.onnx", wrapped_model, (latent, speaker_embedding))
    
    if verify_success:
        print("   🎉 ONNX模型验证通过！")
    else:
        print("   ⚠️ ONNX模型验证未通过，建议检查模型")

    return True



if __name__ == "__main__":
    print("🔄 开始导出标准ONNX模型...")
    success = export_ultimate_onnx()
    if success:
        print("\n✅ 标准ONNX模型导出成功！")
        
    # print("\n🔄 开始导出NPU兼容ONNX模型...")
    # npu_success = export_npu_compatible_onnx()
    # if npu_success:
    #     print("\n✅ NPU兼容ONNX模型导出成功！")
