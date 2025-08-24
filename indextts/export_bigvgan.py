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

        def _match_time(self, cond: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            """Tile cond on time axis to match x to avoid broadcast add at export."""
            if cond.dim() >= 3:
                t = x.shape[-1]
                if cond.shape[-1] == t:
                    return cond
                if cond.shape[-1] == 1:
                    return cond.repeat(1, 1, t)
                # Fallback crop/pad
                if cond.shape[-1] > t:
                    return cond[..., :t]
                pad = t - cond.shape[-1]
                return torch.nn.functional.pad(cond, (0, pad))
            return cond
        
        def _safe_add_4d(self, x: torch.Tensor, y: torch.Tensor, max_w: int = 256) -> torch.Tensor:
            """Numerically identical to x + y, but reshape T to HxW with small W to avoid backend stride limits."""
            y = self._match_time(y, x)
            n, c, t = x.shape
            pad_len = (max_w - (t % max_w)) % max_w
            if pad_len > 0:
                x_pad = torch.nn.functional.pad(x, (0, pad_len))
                y_pad = torch.nn.functional.pad(y, (0, pad_len))
            else:
                x_pad = x
                y_pad = y
            t_pad = x_pad.shape[-1]
            h = t_pad // max_w
            x4 = x_pad.view(n, c, h, max_w)
            y4 = y_pad.view(n, c, h, max_w)
            z4 = x4 + y4
            z = z4.reshape(n, c, t_pad)
            if pad_len > 0:
                z = z[..., :t]
            return z

        def forward(self, x, emdedding):
            x = x.transpose(1, 2)  # 转置以匹配输入形状
            x = self.model.conv_pre(x)

            # 使用传入的嵌入张量，而不是外部变量，避免将 speaker_embedding 固定为常量
            cond0 = self.model.cond_layer(emdedding)
            x = self._safe_add_4d(x, cond0, max_w=256)

            for i in range(self.model.num_upsamples):
                # upsampling
                for i_up in range(len(self.model.ups[i])):
                    x = self.model.ups[i][i_up](x)

                # 同样使用传入的嵌入张量
                condi = self.model.conds[i](emdedding)
                x = self._safe_add_4d(x, condi, max_w=256)

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
    # print("\n=== ONNX模型验证 ===")
    # verify_success = verify_onnx_model("onnx/bigvgan.onnx", wrapped_model, (latent, speaker_embedding))
    
    # if verify_success:
    #     print("   🎉 ONNX模型验证通过！")
    # else:
    #     print("   ⚠️ ONNX模型验证未通过，建议检查模型")

    # # 8. 生成测试输入/输出 NPZ，供后续对比使用
    # try:
    #     # 保存测试输入
    #     np.savez("bigvgan_test_input.npz", latent=latent.numpy(), speaker_embedding=speaker_embedding.numpy())
    #     print("   ✅ 已保存测试输入: bigvgan_test_input.npz")
    #     # 保存参考输出
    #     if ONNX_AVAILABLE:
    #         ort_session = ort.InferenceSession("onnx/bigvgan.onnx")
    #         input_names = [i.name for i in ort_session.get_inputs()]
    #         output_names = [o.name for o in ort_session.get_outputs()]
    #         outs = ort_session.run(output_names, {
    #             input_names[0]: latent.numpy(),
    #             input_names[1]: speaker_embedding.numpy(),
    #         })
    #         np.savez("bigvgan_test_output.npz", **{output_names[0]: outs[0]})
    #         print("   ✅ 已保存测试输出: bigvgan_test_output.npz (键名为 ONNX 输出名)")
    #     else:
    #         print("   ⚠️ 未安装 onnxruntime，跳过保存测试输出")
    # except Exception as e:
    #     print(f"   ❌ 生成测试 NPZ 失败: {e}")

    return True


if __name__ == "__main__":
    print("🔄 开始导出标准ONNX模型...")
    success = export_ultimate_onnx()
    if success:
        print("\n✅ 标准ONNX模型导出成功！")
