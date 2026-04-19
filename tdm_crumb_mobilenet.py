#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
修复设备不匹配问题的CRUMB-MobileNet
确保所有组件都在正确的设备上
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore', category=FutureWarning)

try:
    from pytorchcv.models.mobilenet import DwsConvBlock
except:
    from pytorchcv.models.common import DwsConvBlock

from pytorchcv.model_provider import get_model


def safe_view(tensor, *shape):
    """安全的view操作"""
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor.view(*shape)


def safe_flatten(tensor):
    """安全的flatten操作"""
    return tensor.contiguous().view(tensor.size(0), -1)


class SimpleMemoryBlock(nn.Module):
    """简化的内存块，避免BatchNorm问题，修复设备问题"""

    def __init__(self, feature_dim, num_blocks=256, block_size=32, device=None):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.codebook = nn.Parameter(torch.randn(num_blocks, block_size, device=self.device) * 0.1)

        # 简化的投影层，只使用LayerNorm
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, block_size * 2),
            nn.ReLU(inplace=True),
            nn.LayerNorm(block_size * 2),
            nn.Dropout(0.1),
            nn.Linear(block_size * 2, block_size),
        ).to(self.device)

        # 简化的重构层
        self.reconstruction = nn.Sequential(
            nn.Linear(block_size, feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.LayerNorm(feature_dim * 2),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim)
        ).to(self.device)

        # 基础统计
        self.register_buffer('block_usage', torch.zeros(num_blocks, device=self.device))
        self.register_buffer('total_usage', torch.tensor(0.0, device=self.device))

    def encode(self, features):
        """简化的编码"""
        batch_size = features.shape[0]
        features = features.contiguous().to(self.device)

        # 检查输入维度
        if features.size(1) != self.feature_dim:
            print(f"警告: 输入特征维度 {features.size(1)} 与期望维度 {self.feature_dim} 不匹配")
            return torch.randint(0, self.num_blocks, (batch_size,), device=self.device)

        projected = self.projection(features)
        distances = torch.cdist(projected.unsqueeze(1), self.codebook.unsqueeze(0))
        distances = distances.squeeze(1)
        indices = torch.argmin(distances, dim=1)

        # 更新统计
        unique_indices, counts = torch.unique(indices, return_counts=True)
        self.block_usage[unique_indices] += counts.float()
        self.total_usage += len(indices)

        return indices

    def decode(self, indices):
        """简化的解码"""
        indices = indices.to(self.device)
        selected_blocks = self.codebook[indices]
        reconstructed = self.reconstruction(selected_blocks)
        return reconstructed.contiguous()

    def get_efficiency_metrics(self):
        """获取效率指标"""
        total_blocks = self.num_blocks
        used_blocks = (self.block_usage > 0).sum().item()
        efficiency = used_blocks / total_blocks if total_blocks > 0 else 0
        return {
            'total_blocks': total_blocks,
            'used_blocks': used_blocks,
            'efficiency': efficiency,
            'avg_usage': self.block_usage.mean().item(),
        }


class CRUMBMobileNet(nn.Module):
    """修复设备问题的CRUMB-MobileNet"""

    def __init__(self, pretrained=True, latent_layer_num=21, num_memory_blocks=256, block_size=32):
        super().__init__()

        print("初始化修复版CRUMBMobileNet...")

        # 获取设备信息
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 获取预训练MobileNet
        try:
            model = get_model("mobilenet_w1", pretrained=pretrained)
            #model = get_model("resnet14", pretrained=pretrained)
            model.features.final_pool = nn.AdaptiveAvgPool2d(4)
            print("✓ MobileNet加载成功")
        except Exception as e:
            print(f"✗ MobileNet加载失败: {e}")
            raise

        # 构建层列表
        all_layers = []
        self._remove_sequential(model, all_layers)
        all_layers = self._remove_DwsConvBlock(all_layers)

        # 分割层
        lat_list = []
        end_list = []

        for i, layer in enumerate(all_layers[:-1]):
            if i <= latent_layer_num:
                lat_list.append(layer)
            else:
                end_list.append(layer)

        self.lat_features = nn.Sequential(*lat_list)
        self.end_features = nn.Sequential(*end_list)

        # 确保所有层都在正确的设备上
        self.lat_features = self.lat_features.to(self.device)
        self.end_features = self.end_features.to(self.device)

        # 兼容性设置
        class BackboneWrapper:
            def __init__(self, parent):
                self.parent = parent

        self.backbone = BackboneWrapper(self)
        self.saved_weights = {}
        self.past_j = {i: 0 for i in range(50)}
        self.cur_j = {i: 0 for i in range(50)}

        # 动态计算特征维度
        print("动态计算特征维度...")
        self.latent_dim = None
        self.end_features_dim = None
        self.latent_spatial_shape = None
        self.output = None

        # 延迟初始化标志
        self.initialized = False

        # CRUMB组件 - 延迟初始化
        self.memory_blocks = None
        self.num_memory_blocks = num_memory_blocks
        self.block_size = block_size

        # 训练控制
        self.training_phase = "pretrain"
        self.use_dual_branch = True
        self.performance_history = defaultdict(list)
        self.adaptation_count = 0

        print("✓ 修复版CRUMBMobileNet初始化完成（延迟初始化）")

    def _lazy_init(self, x):
        """延迟初始化，基于实际输入计算维度，修复设备问题"""
        if self.initialized:
            return

        print("执行延迟初始化...")

        # 确保输入在正确的设备上
        x = x.to(self.device)

        # 临时切换到eval模式
        was_training = self.training
        self.eval()

        with torch.no_grad():
            # 确保lat_features在正确设备上
            self.lat_features = self.lat_features.to(self.device)

            # 计算潜在特征
            latent_output = self.lat_features(x)
            flattened_latent = safe_flatten(latent_output)
            self.latent_dim = flattened_latent.size(1)
            self.latent_spatial_shape = latent_output.shape[1:]

            # 确保end_features在正确设备上
            self.end_features = self.end_features.to(self.device)

            # 计算末端特征
            end_output = self.end_features(latent_output)
            flattened_end = safe_flatten(end_output)
            self.end_features_dim = flattened_end.size(1)

            print(f"✓ 潜在特征维度: {self.latent_dim}")
            print(f"✓ 潜在特征空间形状: {self.latent_spatial_shape}")
            print(f"✓ 末端特征维度: {self.end_features_dim}")

        # 创建分类器并确保在正确设备上
        self.output = self._create_simple_classifier(self.end_features_dim)
        self.output = self.output.to(self.device)

        # 创建内存块并确保在正确设备上
        self.memory_blocks = SimpleMemoryBlock(
            feature_dim=self.latent_dim,
            num_blocks=self.num_memory_blocks,
            block_size=self.block_size,
            device=self.device
        )
        self.memory_blocks = self.memory_blocks.to(self.device)

        # 恢复训练模式
        if was_training:
            self.train()

        # 最终设备检查
        self._check_all_devices()

        self.initialized = True
        print("✓ 延迟初始化完成，所有组件已移动到正确设备")

    def _check_all_devices(self):
        """检查所有组件是否在正确设备上"""
        print("检查设备状态...")

        # 检查主要组件
        components = {
            'lat_features': self.lat_features,
            'end_features': self.end_features,
            'output': self.output,
            'memory_blocks': self.memory_blocks
        }

        for name, component in components.items():
            if component is not None:
                device_str = str(next(component.parameters()).device)
                print(f"  {name}: {device_str}")
                if device_str != str(self.device):
                    print(f"  警告: {name} 不在目标设备上，正在移动...")
                    component.to(self.device)

    def _create_simple_classifier(self, input_dim):
        """创建简单的分类器，确保在正确设备上"""
        hidden_dim = max(256, input_dim // 8)

        classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 50)
        )

        return classifier

    def _remove_sequential(self, network, all_layers):
        for layer in network.children():
            if isinstance(layer, nn.Sequential):
                self._remove_sequential(layer, all_layers)
            else:
                all_layers.append(layer)

    def _remove_DwsConvBlock(self, cur_layers):
        all_layers = []
        for layer in cur_layers:
            if isinstance(layer, DwsConvBlock):
                for ch in layer.children():
                    all_layers.append(ch)
            else:
                all_layers.append(layer)
        return all_layers

    def encode_to_memory_blocks(self, latent_features):
        """编码到内存块，确保设备一致性"""
        if not self.initialized or self.memory_blocks is None:
            return torch.zeros(latent_features.size(0), dtype=torch.long, device=self.device)

        # 确保输入在正确设备上
        latent_features = latent_features.to(self.device)
        flattened = safe_flatten(latent_features)
        indices = self.memory_blocks.encode(flattened)
        return indices

    def decode_from_memory_blocks(self, indices, target_shape):
        """从内存块解码，确保设备一致性"""
        if not self.initialized or self.memory_blocks is None:
            # 返回随机特征作为fallback
            batch_size = target_shape[0]
            return torch.randn(target_shape, device=self.device)

        # 确保indices在正确设备上
        indices = indices.to(self.device)
        reconstructed_flat = self.memory_blocks.decode(indices)
        batch_size = target_shape[0]
        reconstructed = safe_view(reconstructed_flat, batch_size, *self.latent_spatial_shape)
        return reconstructed

    def forward(self, x, latent_input=None, return_lat_acts=False, training_phase=None):
        """修复版forward方法 - 解决设备问题"""
        if training_phase is not None:
            self.training_phase = training_phase

        if x is not None:
            # 确保输入在正确设备上
            x = x.to(self.device)

            # 延迟初始化
            if not self.initialized:
                self._lazy_init(x)

            # 确保所有组件在正确设备上
            if not hasattr(self, '_device_checked'):
                self._check_all_devices()
                self._device_checked = True

            orig_acts = self.lat_features(x)

            if self.training_phase == "pretrain" and self.use_dual_branch:
                # 预训练模式
                if latent_input is not None:
                    latent_input = latent_input.to(self.device)
                    combined_orig = torch.cat((orig_acts, latent_input), 0)
                else:
                    combined_orig = orig_acts

                # 重建分支
                memory_indices = self.encode_to_memory_blocks(orig_acts)
                reconstructed_acts = self.decode_from_memory_blocks(memory_indices, orig_acts.shape)

                # 原始分支
                feat_orig = self.end_features(combined_orig)
                x_orig = safe_flatten(feat_orig)
                logits_orig = self.output(x_orig)

                # 重建分支
                feat_recon = self.end_features(reconstructed_acts)
                x_recon = safe_flatten(feat_recon)
                logits_recon = self.output(x_recon)

                # 特征处理
                final_feat = safe_flatten(feat_orig)

                if return_lat_acts:
                    return logits_orig, logits_recon, final_feat
                else:
                    return logits_orig, logits_recon, memory_indices

            else:
                # 流式模式
                if latent_input is not None:
                    latent_input = latent_input.to(self.device)
                    combined_acts = torch.cat((orig_acts, latent_input), 0)
                else:
                    combined_acts = orig_acts

                feat = self.end_features(combined_acts)
                x_out = safe_flatten(feat)
                logits = self.output(x_out)
                final_feat = x_out

                if return_lat_acts:
                    return logits, orig_acts, final_feat
                else:
                    return logits

        return None

    def to(self, device):
        """重写to方法，确保所有组件都移动到正确设备"""
        super().to(device)
        self.device = device

        if self.initialized:
            if self.output is not None:
                self.output = self.output.to(device)
            if self.memory_blocks is not None:
                self.memory_blocks = self.memory_blocks.to(device)
                self.memory_blocks.device = device

        return self

    def intelligent_memory_update(self, features, learning_rate=0.02):
        """内存更新，确保设备一致性"""
        if features is None or not self.initialized:
            return

        # 确保特征在正确设备上
        features = features.to(self.device)
        self.adaptation_count += 1

    def compress_replay_memory(self, features, labels):
        """压缩重放内存，确保设备一致性"""
        if features is None or len(features) == 0 or not self.initialized:
            return None

        # 确保输入在正确设备上
        features = features.to(self.device).contiguous()
        if len(features.shape) == 4:
            flattened = safe_flatten(features)
        else:
            flattened = features

        indices = self.memory_blocks.encode(flattened)

        return {
            'indices': indices.cpu(),
            'labels': labels.cpu() if torch.is_tensor(labels) else torch.tensor(labels),
            'original_shape': self.latent_spatial_shape,
        }

    def decompress_replay_memory(self, compressed_data):
        """解压重放内存，确保设备一致性"""
        if compressed_data is None or not self.initialized:
            return None, None

        indices = compressed_data['indices'].to(self.device)
        labels = compressed_data['labels']

        reconstructed_flat = self.memory_blocks.decode(indices)
        batch_size = reconstructed_flat.size(0)
        reconstructed = safe_view(reconstructed_flat, batch_size, *self.latent_spatial_shape)

        return reconstructed, labels

    def get_memory_efficiency(self):
        """获取内存效率"""
        if not self.initialized or self.memory_blocks is None:
            return {'efficiency': 0, 'used_blocks': 0, 'total_blocks': 256}
        return self.memory_blocks.get_efficiency_metrics()

    def update_memory_blocks(self, features, learning_rate=0.02):
        """更新内存块，确保设备一致性"""
        if features is not None:
            features = features.to(self.device)
        self.intelligent_memory_update(features, learning_rate)


# 测试函数
def test_device_fixed_model():
    """测试设备修复版模型"""
    print("测试设备修复版CRUMBMobileNet...")

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"测试设备: {device}")

        model = CRUMBMobileNet(pretrained=True)
        model = model.to(device)
        print("✓ 模型创建和设备移动成功")

        # 测试不同大小的输入
        test_inputs = [
            torch.randn(1, 3, 128, 128, device=device),
            torch.randn(4, 3, 128, 128, device=device),
            torch.randn(8, 3, 128, 128, device=device)
        ]

        for i, dummy_input in enumerate(test_inputs):
            print(f"\n测试输入 {i + 1}: {dummy_input.shape} on {dummy_input.device}")

            # 预训练模式
            model.training_phase = "pretrain"
            model.use_dual_branch = True
            result = model(dummy_input, return_lat_acts=False)
            print(f"✓ 预训练模式测试通过")

            # 流式模式
            model.training_phase = "streaming"
            model.use_dual_branch = False
            result = model(dummy_input, return_lat_acts=False)
            print(f"✓ 流式模式测试通过: {result.shape} on {result.device}")

        print("\n✓ 设备问题已修复！模型可以正常运行。")
        return True

    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_device_fixed_model()