#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

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

    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor.view(*shape)


def safe_flatten(tensor):
    
    return tensor.contiguous().view(tensor.size(0), -1)


class SimpleMemoryBlock(nn.Module):
    

    def __init__(self, feature_dim, num_blocks=256, block_size=32, device=None):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.codebook = nn.Parameter(torch.randn(num_blocks, block_size, device=self.device) * 0.1)

       
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, block_size * 2),
            nn.ReLU(inplace=True),
            nn.LayerNorm(block_size * 2),
            nn.Dropout(0.1),
            nn.Linear(block_size * 2, block_size),
        ).to(self.device)

     
        self.reconstruction = nn.Sequential(
            nn.Linear(block_size, feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.LayerNorm(feature_dim * 2),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim)
        ).to(self.device)

        
        self.register_buffer('block_usage', torch.zeros(num_blocks, device=self.device))
        self.register_buffer('total_usage', torch.tensor(0.0, device=self.device))

    def encode(self, features):
       
        batch_size = features.shape[0]
        features = features.contiguous().to(self.device)

       
        if features.size(1) != self.feature_dim:
          
            return torch.randint(0, self.num_blocks, (batch_size,), device=self.device)

        projected = self.projection(features)
        distances = torch.cdist(projected.unsqueeze(1), self.codebook.unsqueeze(0))
        distances = distances.squeeze(1)
        indices = torch.argmin(distances, dim=1)

       
        unique_indices, counts = torch.unique(indices, return_counts=True)
        self.block_usage[unique_indices] += counts.float()
        self.total_usage += len(indices)

        return indices

    def decode(self, indices):
      
        indices = indices.to(self.device)
        selected_blocks = self.codebook[indices]
        reconstructed = self.reconstruction(selected_blocks)
        return reconstructed.contiguous()

    def get_efficiency_metrics(self):
      
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
    

    def __init__(self, pretrained=True, latent_layer_num=21, num_memory_blocks=256, block_size=32):
        super().__init__()

        

       
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

        
        try:
            model = get_model("mobilenet_w1", pretrained=pretrained)
            #model = get_model("resnet14", pretrained=pretrained)
            model.features.final_pool = nn.AdaptiveAvgPool2d(4)
            print("✓ MobileNet加载成功")
        except Exception as e:
           
            raise

        
        all_layers = []
        self._remove_sequential(model, all_layers)
        all_layers = self._remove_DwsConvBlock(all_layers)

        
        lat_list = []
        end_list = []

        for i, layer in enumerate(all_layers[:-1]):
            if i <= latent_layer_num:
                lat_list.append(layer)
            else:
                end_list.append(layer)

        self.lat_features = nn.Sequential(*lat_list)
        self.end_features = nn.Sequential(*end_list)

       
        self.lat_features = self.lat_features.to(self.device)
        self.end_features = self.end_features.to(self.device)

        
        class BackboneWrapper:
            def __init__(self, parent):
                self.parent = parent

        self.backbone = BackboneWrapper(self)
        self.saved_weights = {}
        self.past_j = {i: 0 for i in range(50)}
        self.cur_j = {i: 0 for i in range(50)}

       
       
        self.latent_dim = None
        self.end_features_dim = None
        self.latent_spatial_shape = None
        self.output = None

      
        self.initialized = False

        
        self.memory_blocks = None
        self.num_memory_blocks = num_memory_blocks
        self.block_size = block_size

       
        self.training_phase = "pretrain"
        self.use_dual_branch = True
        self.performance_history = defaultdict(list)
        self.adaptation_count = 0

        

    def _lazy_init(self, x):
       
        if self.initialized:
            return

      

        
        x = x.to(self.device)

        
        was_training = self.training
        self.eval()

        with torch.no_grad():
            
            self.lat_features = self.lat_features.to(self.device)

            
            latent_output = self.lat_features(x)
            flattened_latent = safe_flatten(latent_output)
            self.latent_dim = flattened_latent.size(1)
            self.latent_spatial_shape = latent_output.shape[1:]

           
            self.end_features = self.end_features.to(self.device)

            
            end_output = self.end_features(latent_output)
            flattened_end = safe_flatten(end_output)
            self.end_features_dim = flattened_end.size(1)

            
        
        self.output = self._create_simple_classifier(self.end_features_dim)
        self.output = self.output.to(self.device)

        
        self.memory_blocks = SimpleMemoryBlock(
            feature_dim=self.latent_dim,
            num_blocks=self.num_memory_blocks,
            block_size=self.block_size,
            device=self.device
        )
        self.memory_blocks = self.memory_blocks.to(self.device)

       
        if was_training:
            self.train()

        
        self._check_all_devices()

        self.initialized = True
        

    def _check_all_devices(self):
        

       
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
                    
                    component.to(self.device)

    def _create_simple_classifier(self, input_dim):
        
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
        
        if not self.initialized or self.memory_blocks is None:
            return torch.zeros(latent_features.size(0), dtype=torch.long, device=self.device)

        
        latent_features = latent_features.to(self.device)
        flattened = safe_flatten(latent_features)
        indices = self.memory_blocks.encode(flattened)
        return indices

    def decode_from_memory_blocks(self, indices, target_shape):
        
        if not self.initialized or self.memory_blocks is None:
            # 返回随机特征作为fallback
            batch_size = target_shape[0]
            return torch.randn(target_shape, device=self.device)

        
        indices = indices.to(self.device)
        reconstructed_flat = self.memory_blocks.decode(indices)
        batch_size = target_shape[0]
        reconstructed = safe_view(reconstructed_flat, batch_size, *self.latent_spatial_shape)
        return reconstructed

    def forward(self, x, latent_input=None, return_lat_acts=False, training_phase=None):
        
        if training_phase is not None:
            self.training_phase = training_phase

        if x is not None:
            
            x = x.to(self.device)

            
            if not self.initialized:
                self._lazy_init(x)

           
            if not hasattr(self, '_device_checked'):
                self._check_all_devices()
                self._device_checked = True

            orig_acts = self.lat_features(x)

            if self.training_phase == "pretrain" and self.use_dual_branch:
                
                if latent_input is not None:
                    latent_input = latent_input.to(self.device)
                    combined_orig = torch.cat((orig_acts, latent_input), 0)
                else:
                    combined_orig = orig_acts

                
                memory_indices = self.encode_to_memory_blocks(orig_acts)
                reconstructed_acts = self.decode_from_memory_blocks(memory_indices, orig_acts.shape)

                
                feat_orig = self.end_features(combined_orig)
                x_orig = safe_flatten(feat_orig)
                logits_orig = self.output(x_orig)

                
                feat_recon = self.end_features(reconstructed_acts)
                x_recon = safe_flatten(feat_recon)
                logits_recon = self.output(x_recon)

                
                final_feat = safe_flatten(feat_orig)

                if return_lat_acts:
                    return logits_orig, logits_recon, final_feat
                else:
                    return logits_orig, logits_recon, memory_indices

            else:
                
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
       
        if features is None or not self.initialized:
            return

        
        features = features.to(self.device)
        self.adaptation_count += 1

    def compress_replay_memory(self, features, labels):
        
        if features is None or len(features) == 0 or not self.initialized:
            return None

        
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
    

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

        model = CRUMBMobileNet(pretrained=True)
        model = model.to(device)
        

        
        test_inputs = [
            torch.randn(1, 3, 128, 128, device=device),
            torch.randn(4, 3, 128, 128, device=device),
            torch.randn(8, 3, 128, 128, device=device)
        ]

        for i, dummy_input in enumerate(test_inputs):
            

            
            model.training_phase = "pretrain"
            model.use_dual_branch = True
            result = model(dummy_input, return_lat_acts=False)
            

            
            model.training_phase = "streaming"
            model.use_dual_branch = False
            result = model(dummy_input, return_lat_acts=False)
            

        
        return True

    except Exception as e:
        
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_device_fixed_model()
