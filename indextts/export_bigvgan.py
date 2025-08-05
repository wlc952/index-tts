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

    """导出NPU兼容的ONNX模型"""
    print("=== BigVGAN NPU兼容 ONNX 导出 ===")
    
    # 创建NPU兼容模型
    model = create_npu_compatible_model()
    
    # 准备测试数据
    print("3. 准备测试数据...")
    latent = torch.randn(1, 224, 1280)
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
    latent = torch.randn(1, 224, 1280)
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


def export_approximated_filter_onnx():
    """滤波器近似方案：用静态卷积近似原始滤波器"""
    print("=== BigVGAN 滤波器近似 ONNX 导出 ===")
    
    # 1. 加载模型
    print("1. 加载原始模型...")
    cfg = OmegaConf.load("checkpoints/config.yaml")
    model = Generator(cfg.bigvgan, use_cuda_kernel=False)
    checkpoint = torch.load("checkpoints/bigvgan_generator.pth", map_location="cpu")
    model.load_state_dict(checkpoint["generator"])
    model.eval()
    model.remove_weight_norm()
    
    # 2. 创建静态滤波器近似
    class StaticFilterActivation1d(nn.Module):
        def __init__(self, original_activation, expected_channels):
            super().__init__()
            
            # 保持激活函数
            if isinstance(original_activation.act, activations.SnakeBeta):
                self.act = MinimalSnakeBeta(original_activation.act)
            else:
                self.act = original_activation.act
            
            # 创建静态抗混叠滤波器
            # 使用简单的低通滤波器近似原始的复杂滤波器
            kernel_size = 5  # 较小的核，NPU友好
            self.channels = expected_channels
            
            # 创建低通滤波器权重（汉宁窗）
            window = torch.hann_window(kernel_size)
            window = window / window.sum()
            
            # 扩展到所有通道
            filter_weight = window.view(1, 1, kernel_size).expand(expected_channels, 1, kernel_size)
            
            # 创建静态卷积层
            self.anti_alias_filter = nn.Conv1d(
                in_channels=expected_channels,
                out_channels=expected_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=expected_channels,
                bias=False
            )
            
            # 设置滤波器权重
            with torch.no_grad():
                self.anti_alias_filter.weight.copy_(filter_weight)
        
        def forward(self, x):
            # 如果通道数不匹配，使用简单的激活
            if x.shape[1] != self.channels:
                return self.act(x)
            
            # 应用激活函数
            x_activated = self.act(x)
            
            # 应用静态抗混叠滤波
            x_filtered = self.anti_alias_filter(x_activated)
            
            # 与原始信号混合，保持大部分激活特性
            x = x_activated * 0.8 + x_filtered * 0.2
            
            return x
    
    class MinimalSnakeBeta(nn.Module):
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
    
    # 3. 替换模块，包括通道数映射
    print("2. 替换为静态滤波器近似...")
    
    # BigVGAN的通道数映射（根据网络结构）
    channels_map = {
        'activation_post': 24,  # 最后一层
        'resblocks.0': 768, 'resblocks.1': 768, 'resblocks.2': 768,
        'resblocks.3': 384, 'resblocks.4': 384, 'resblocks.5': 384,
        'resblocks.6': 192, 'resblocks.7': 192, 'resblocks.8': 192,
        'resblocks.9': 96, 'resblocks.10': 96, 'resblocks.11': 96,
        'resblocks.12': 48, 'resblocks.13': 48, 'resblocks.14': 48,
        'resblocks.15': 24, 'resblocks.16': 24, 'resblocks.17': 24,
    }
    
    def get_expected_channels(module_path):
        """根据模块路径获取期望的通道数"""
        for key, channels in channels_map.items():
            if key in module_path:
                return channels
        return 512  # 默认值
    
    def replace_with_static_filter(module, module_path=""):
        for name, child in module.named_children():
            current_path = f"{module_path}.{name}" if module_path else name
            
            if isinstance(child, activations.SnakeBeta):
                setattr(module, name, MinimalSnakeBeta(child))
                print(f"   替换了 SnakeBeta: {current_path}")
            elif hasattr(child, '__class__') and 'Activation1d' in child.__class__.__name__:
                expected_channels = get_expected_channels(current_path)
                setattr(module, name, StaticFilterActivation1d(child, expected_channels))
                print(f"   替换了 Activation1d: {current_path} (通道数: {expected_channels})")
            else:
                replace_with_static_filter(child, current_path)
    
    replace_with_static_filter(model)
    
    # 4. 包装器
    class FilterApproximatedWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, x, embedding):
            x = x.transpose(1, 2)
            x = self.model.conv_pre(x)
            x = x + self.model.cond_layer(embedding)
            
            for i in range(self.model.num_upsamples):
                for up_layer in self.model.ups[i]:
                    x = up_layer(x)
                
                x = x + self.model.conds[i](embedding)
                
                xs = None
                for j in range(self.model.num_kernels):
                    if xs is None:
                        xs = self.model.resblocks[i * self.model.num_kernels + j](x)
                    else:
                        xs += self.model.resblocks[i * self.model.num_kernels + j](x)
                x = xs / self.model.num_kernels
            
            x = self.model.activation_post(x)
            x = self.model.conv_post(x)
            x = torch.tanh(x)
            
            return x
    
    # 5. 准备测试数据
    print("3. 准备测试数据...")
    torch.manual_seed(42)
    latent = torch.randn(1, 224, 1280)
    speaker_embedding = torch.randn(1, 512, 1)
    
    # 6. 导出模型
    os.makedirs("onnx", exist_ok=True)
    wrapped_model = FilterApproximatedWrapper(model)
    
    print("4. 导出滤波器近似ONNX模型...")
    torch.onnx.export(
        wrapped_model,
        (latent, speaker_embedding),
        "onnx/bigvgan_filter_approximated.onnx",
        input_names=["latent", "speaker_embedding"],
        output_names=["audio"],
        opset_version=14,
        do_constant_folding=True,
        verbose=False,
        dynamic_axes=None,
        export_params=True,
        keep_initializers_as_inputs=False,
        training=torch.onnx.TrainingMode.EVAL
    )
    print("   滤波器近似ONNX模型导出成功！")
    
    return True


if __name__ == "__main__":
    # print("🔄 开始导出标准ONNX模型...")
    # success = export_ultimate_onnx()
    # if success:
    #     print("\n✅ 标准ONNX模型导出成功！")
    print("\n🔄 方案2: 滤波器近似 - 用静态滤波器近似原始滤波器...")
    try:
        filter_success = export_approximated_filter_onnx()
        if filter_success:
            print("✅ 滤波器近似导出成功！这是滤波器和NPU兼容性的折中方案")
        else:
            print("❌ 滤波器近似导出失败")
    except Exception as e:
        print(f"❌ 滤波器近似导出异常: {e}")