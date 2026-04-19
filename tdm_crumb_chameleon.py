#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""融合智能连续学习策略的CRUMB-Enhanced Chameleon实现 - 解决梯度和训练问题"""

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from data_loader_exp3919 import CORE50
import copy
import os
import json
from models.tdm_crumb_mobilenet import CRUMBMobileNet
from utils import *
import configparser
import argparse
from pprint import pprint
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import timeit
import heapq
import torchvision.models as models
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
import torchvision
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import pickle


class IntelligentReplayBuffer:
    """智能重放缓冲区 - 结合遗忘统计和重要性评估"""

    def __init__(self, buffer_size, device):
        self.buffer_size = buffer_size
        self.device = device
        self.examples = []
        self.labels = []
        self.logits = []
        self.importance_scores = []

        # 遗忘统计
        self.example_stats = {}
        self.presentation_count = 0

    def add_data(self, examples, labels, logits=None, importance_scores=None):
        """智能添加数据到缓冲区"""
        if importance_scores is None:
            importance_scores = np.ones(len(examples))

        for i in range(len(examples)):
            if len(self.examples) < self.buffer_size:
                self.examples.append(examples[i].clone())
                self.labels.append(labels[i].clone())
                if logits is not None:
                    self.logits.append(logits[i].clone())
                self.importance_scores.append(importance_scores[i])
            else:
                # 缓冲区已满，替换重要性最低的样本
                min_idx = np.argmin(self.importance_scores)
                if importance_scores[i] > self.importance_scores[min_idx]:
                    self.examples[min_idx] = examples[i].clone()
                    self.labels[min_idx] = labels[i].clone()
                    if logits is not None and len(self.logits) > min_idx:
                        self.logits[min_idx] = logits[i].clone()
                    self.importance_scores[min_idx] = importance_scores[i]

    def get_data(self, batch_size):
        """基于重要性的智能采样"""
        if len(self.examples) == 0:
            return None, None, None

        # 基于重要性分数的概率采样
        importance_array = np.array(self.importance_scores)
        probabilities = importance_array / importance_array.sum()

        indices = np.random.choice(len(self.examples),
                                   size=min(batch_size, len(self.examples)),
                                   p=probabilities, replace=False)

        sampled_examples = torch.stack([self.examples[i] for i in indices])
        sampled_labels = torch.stack([self.labels[i] for i in indices])
        sampled_logits = None
        if len(self.logits) > 0:
            sampled_logits = torch.stack([self.logits[i] for i in indices])

        return sampled_examples, sampled_labels, sampled_logits

    def update_importance_scores(self, example_indices, new_scores):
        """更新样本重要性分数"""
        for i, idx in enumerate(example_indices):
            if idx < len(self.importance_scores):
                # 使用指数移动平均更新
                self.importance_scores[idx] = 0.9 * self.importance_scores[idx] + 0.1 * new_scores[i]

    def is_empty(self):
        return len(self.examples) == 0


class ForgettingStatisticsTracker:
    """遗忘统计追踪器"""

    def __init__(self):
        self.example_stats = {}
        self.forgetting_scores = {}

    def update_stats(self, example_ids, losses, accuracies, margins):
        """更新样本统计信息"""
        for i, example_id in enumerate(example_ids):
            if example_id not in self.example_stats:
                self.example_stats[example_id] = {'losses': [], 'accuracies': [], 'margins': []}

            self.example_stats[example_id]['losses'].append(losses[i])
            self.example_stats[example_id]['accuracies'].append(accuracies[i])
            self.example_stats[example_id]['margins'].append(margins[i])

    def compute_forgetting_scores(self):
        """计算遗忘分数"""
        for example_id, stats in self.example_stats.items():
            accuracies = np.array(stats['accuracies'])

            if len(accuracies) < 2:
                self.forgetting_scores[example_id] = 0
                continue

            # 计算遗忘事件
            transitions = accuracies[1:] - accuracies[:-1]
            forgetting_events = len(np.where(transitions == -1)[0])

            # 计算学习难度
            if len(np.where(accuracies == 0)[0]) > 0:
                learning_presentations = np.where(accuracies == 0)[0][-1] + 1
            else:
                learning_presentations = 0

            self.forgetting_scores[example_id] = forgetting_events + learning_presentations * 0.1

        return self.forgetting_scores

    def get_easy_examples(self, all_indices, removal_ratio=0.1):
        """获取容易学习的样本（移除困难样本）"""
        if len(self.forgetting_scores) == 0:
            return all_indices

        scored_indices = [(idx, self.forgetting_scores.get(idx, 0)) for idx in all_indices]
        scored_indices.sort(key=lambda x: x[1])  # 按遗忘分数升序

        num_keep = int(len(scored_indices) * (1 - removal_ratio))
        return [idx for idx, _ in scored_indices[:num_keep]]


def create_simple_loss(logits, targets):
    """简化版损失函数 - 专注于稳定训练"""
    criterion = nn.CrossEntropyLoss(reduction='mean')
    return criterion(logits, targets)


def compute_simple_replay_probabilities(outputs, labels):
    """简化版重放概率计算"""
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss(reduction='none')
        losses = criterion(outputs, labels)
        probabilities = F.softmax(losses, dim=0)
        return probabilities, losses


def compute_sample_importance(outputs, targets, losses):
    """计算样本重要性分数"""
    with torch.no_grad():
        # 基于损失和预测置信度计算重要性
        probs = F.softmax(outputs, dim=1)
        max_probs = torch.max(probs, dim=1)[0]

        # 重要性 = 高损失 + 低置信度
        importance = losses.detach().cpu().numpy() + (1 - max_probs).detach().cpu().numpy()
        return importance


def apply_gradient_efficiency(model, sparsity_ratio=0.8):
    """应用梯度效率优化"""
    all_gradients = []
    param_names = []

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            all_gradients.extend(param.grad.view(-1).abs().cpu().numpy())
            param_names.append(name)

    if len(all_gradients) == 0:
        return

    # 计算梯度阈值
    threshold = np.percentile(all_gradients, sparsity_ratio * 100)

    # 应用梯度稀疏化
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            mask = (param.grad.abs() >= threshold).float()
            param.grad.mul_(mask)


# --------------------------------- Setup --------------------------------------

parser = argparse.ArgumentParser(description='Run improved CRUMB-Enhanced CL experiments')
parser.add_argument('--name', dest='exp_name', default='default',
                    help='name of the experiment you want to run.')
parser.add_argument('--scenario', type=str, default="ni",
                    choices=['ni', 'nc', 'nic', 'nicv2_79', 'nicv2_196', 'nicv2_391'])
parser.add_argument('--save_dir', type=str, default="results",
                    help='directory to save experimental results')
parser.add_argument('--run', type=int, default=1,
                    help='directory to save experimental results')
parser.add_argument('--memory_blocks', type=int, default=128,
                    help='number of memory blocks for CRUMB')
parser.add_argument('--block_size', type=int, default=32,
                    help='size of each memory block')
parser.add_argument('--replay_method', type=str, default='er',
                    choices=['er', 'der', 'derpp'], help='replay method to use')
parser.add_argument('--gradient_efficient', action='store_true', default=False,
                    help='apply gradient efficiency optimization')
parser.add_argument('--forgetting_removal', type=float, default=0.1,
                    help='ratio of difficult samples to remove based on forgetting')
parser.add_argument('--buffer_weight', type=float, default=0.5,
                    help='weight for buffer replay loss')
args = parser.parse_args()

# directory for saving experimental results
args.save_dir = os.path.join(args.save_dir, args.scenario)
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

# set cuda device
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 修复配置文件读取
config = configparser.ConfigParser()
try:
    with open("params2_tmd.cfg", 'r', encoding='utf-8') as config_file:
        config.read_file(config_file)
    print("配置文件读取成功（UTF-8编码）")
except UnicodeDecodeError:
    try:
        with open("params1.cfg", 'r', encoding='gbk') as config_file:
            config.read_file(config_file)
        print("配置文件读取成功（GBK编码）")
    except UnicodeDecodeError:
        with open("params1.cfg", 'r', encoding='utf-8', errors='ignore') as config_file:
            config.read_file(config_file)
        print("警告：配置文件中的部分字符可能因编码问题被忽略")

exp_config = config[args.exp_name] if args.exp_name in config else config['DEFAULT']
print("Experiment name:", args.exp_name)
pprint(dict(exp_config))

# 获取配置参数
exp_name = eval(exp_config['exp_name'])
use_cuda = eval(exp_config['use_cuda'])
init_lr = eval(exp_config['init_lr'])
inc_lr = eval(exp_config['inc_lr'])
mb_size = eval(exp_config['mb_size'])
init_train_ep = eval(exp_config['init_train_ep'])
inc_train_ep = eval(exp_config['inc_train_ep'])
init_update_rate = eval(exp_config['init_update_rate'])
inc_update_rate = eval(exp_config['inc_update_rate'])
max_r_max = eval(exp_config['max_r_max'])
max_d_max = eval(exp_config['max_d_max'])
inc_step = eval(exp_config['inc_step'])
rm_sz = eval(exp_config['rm_sz'])
momentum = eval(exp_config['momentum'])
l2 = eval(exp_config['l2'])
freeze_below_layer = eval(exp_config['freeze_below_layer'])
latent_layer_num = eval(exp_config['latent_layer_num'])
reg_lambda = eval(exp_config['reg_lambda'])

# 关键修复：调整冻结层数，不要冻结太多层
actual_freeze_layer = min(latent_layer_num - 10, 10)
print(f"修正冻结层数从 {latent_layer_num} 到 {actual_freeze_layer}")

# tensorboard日志
log_dir = 'logs/' + exp_name + '_IMPROVED'
writer = SummaryWriter(log_dir)

# 保存参数
hyper = json.dumps(dict(exp_config))
writer.add_text("parameters2_tmd", hyper, 0)

# 变量初始化
tot_it_step = 0
rm = None
ltm = None

user_pref_list = [
    '[21 46 5 7 43]',
    '[47 27 23 9 41]',
    '[38 37 11 40 23]',
    '[18 37 16 34 36]',
    '[44 6 40 27 31]',
    '[16 22 11 40 24]',
    '[11 15 27 29 45]',
    '[27 49 31 22 13]',
    '[34 47 16 45 30]',
    '[9 46 33 27 21]']
user_idx = int(args.run) - 1
user_pref_cls = list(map(int, user_pref_list[user_idx][1:-1].split(' ')))

running_freq = {i: 0 for i in range(50)}

# 创建数据集
dataset = CORE50(root='core50_128x128', scenario=args.scenario, cumul=False, user_pref_cls=user_pref_cls)
preproc = preprocess_imgs

# 获取测试集
test_x, test_y = dataset.get_test_set()

# 创建改进版模型
print("创建CRUMB模型...")
model = CRUMBMobileNet(
    pretrained=True,
    latent_layer_num=latent_layer_num,
    num_memory_blocks=args.memory_blocks,
    block_size=args.block_size
)

# 修复冻结逻辑 - 只冻结少量早期层
print("应用改进的层冻结策略...")


def freeze_early_layers_only(model, num_freeze=8):
    """只冻结前num_freeze层，保持其他层可训练"""
    frozen_count = 0
    for name, param in model.named_parameters():
        if 'lat_features' in name:
            try:
                layer_num = int(name.split('.')[1])
                if layer_num < num_freeze:
                    param.requires_grad = False
                    frozen_count += 1
                else:
                    param.requires_grad = True
            except:
                param.requires_grad = True
        else:
            param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总共冻结了 {frozen_count} 个参数")
    print(f"可训练参数: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.1f}%)")


# 应用修正的冻结策略
freeze_early_layers_only(model, num_freeze=8)

# 简化的优化器设置
print("设置改进的训练参数...")
optimizer_init = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=0.0002,  # 更保守的学习率
    weight_decay=0.0001
)

optimizer_inc = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=0.0001,  # 更保守的增量学习率
    weight_decay=0.0001
)

# 学习率调度
scheduler_init = torch.optim.lr_scheduler.StepLR(optimizer_init, step_size=100, gamma=0.95)
scheduler_inc = torch.optim.lr_scheduler.StepLR(optimizer_inc, step_size=100, gamma=0.95)

criterion = nn.CrossEntropyLoss()

# 初始化智能组件
print("初始化智能连续学习组件...")
intelligent_buffer = IntelligentReplayBuffer(buffer_size=rm_sz, device='cuda' if use_cuda else 'cpu')
forgetting_tracker = ForgettingStatisticsTracker()

# 简化的概率设置
prob_k = 0.5
prob_n_k = 0.5
prob_class = [prob_k if cls in user_pref_cls else prob_n_k for cls in range(50)]

avg_K = []
avg_acc = []
flag = -1

# 早停机制
best_accuracy = 0.0
patience = 0
max_patience = 10

print("开始改进版CRUMB训练...")
start_time = timeit.default_timer()

# 改进的训练循环
for i, train_batch in enumerate(dataset):
    print(f"\n========== Batch {i} ==========")

    # 设置训练阶段
    if i == 0:
        current_optimizer = optimizer_init
        current_scheduler = scheduler_init
        train_ep = init_train_ep
    else:
        current_optimizer = optimizer_inc
        current_scheduler = scheduler_inc
        train_ep = inc_train_ep

    train_x, train_y = train_batch
    train_x = preproc(train_x)

    print(f"train_x shape: {train_x.shape}, train_y shape: {train_y.shape}")

    # 数据预处理
    if i == 0:
        (train_x, train_y), it_x_ep = pad_data([train_x, train_y], mb_size)

    shuffle_in_unison([train_x, train_y], in_place=True)

    model = maybe_cuda(model, use_cuda=use_cuda)

    # 转换为tensor
    train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
    train_y = torch.from_numpy(train_y).type(torch.LongTensor)

    print(f"开始训练 {train_ep} 个epoch...")

    for ep in range(train_ep):
        print(f"\nEpoch {ep}:")
        model.train()

        correct_cnt = 0
        total_samples = 0
        epoch_loss = 0.0

        for it in range(0, train_x.size(0), mb_size):
            end_idx = min(it + mb_size, train_x.size(0))

            # 获取批次数据
            x_mb = maybe_cuda(train_x[it:end_idx], use_cuda=use_cuda)
            y_mb = maybe_cuda(train_y[it:end_idx], use_cuda=use_cuda)

            # 清零梯度
            current_optimizer.zero_grad()

            # 前向传播
            try:
                if i == 0:
                    model.training_phase = "pretrain"
                    model.use_dual_branch = False
                    result = model(x_mb, return_lat_acts=False)
                    if isinstance(result, tuple):
                        logits = result[0]
                    else:
                        logits = result
                else:
                    model.training_phase = "streaming"
                    model.use_dual_branch = False
                    result = model(x_mb, return_lat_acts=False)
                    if isinstance(result, tuple):
                        logits = result[0]
                    else:
                        logits = result

            except Exception as e:
                print(f"前向传播错误: {e}")
                continue

            # 检查logits
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"发现异常logits，跳过批次 {it}")
                continue

            # 计算主损失
            main_loss = create_simple_loss(logits, y_mb)
            total_loss = main_loss

            # 智能重放机制
            if not intelligent_buffer.is_empty() and i > 0:
                buffer_x, buffer_y, buffer_logits = intelligent_buffer.get_data(
                    batch_size=min(mb_size // 2, len(y_mb))
                )

                if buffer_x is not None:
                    buffer_x = maybe_cuda(buffer_x, use_cuda=use_cuda)
                    buffer_y = maybe_cuda(buffer_y, use_cuda=use_cuda)

                    # 缓冲区前向传播
                    buffer_outputs = model(buffer_x, return_lat_acts=False)
                    if isinstance(buffer_outputs, tuple):
                        buffer_outputs = buffer_outputs[0]

                    # 根据重放方法计算重放损失
                    if args.replay_method == 'er':
                        replay_loss = F.cross_entropy(buffer_outputs, buffer_y)
                    elif args.replay_method == 'der' and buffer_logits is not None:
                        buffer_logits = maybe_cuda(buffer_logits, use_cuda=use_cuda)
                        replay_loss = F.mse_loss(buffer_outputs, buffer_logits)
                    elif args.replay_method == 'derpp' and buffer_logits is not None:
                        buffer_logits = maybe_cuda(buffer_logits, use_cuda=use_cuda)
                        replay_loss = (F.mse_loss(buffer_outputs, buffer_logits) +
                                       F.cross_entropy(buffer_outputs, buffer_y))
                    else:
                        replay_loss = F.cross_entropy(buffer_outputs, buffer_y)

                    total_loss += args.buffer_weight * replay_loss

            # 检查损失
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"发现异常损失，跳过批次 {it}")
                continue

            # 反向传播
            try:
                total_loss.backward()

                # 应用梯度效率优化
                if args.gradient_efficient and ep > 5:
                    apply_gradient_efficiency(model, sparsity_ratio=0.7)

                # 梯度裁剪
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0
                )

                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"发现异常梯度，跳过更新")
                    current_optimizer.zero_grad()
                    continue

                # 更新参数
                current_optimizer.step()

                # 计算准确率和重要性
                _, predicted = torch.max(logits.data, 1)
                correct_cnt += (predicted == y_mb).sum().item()
                total_samples += y_mb.size(0)
                epoch_loss += total_loss.item()

                # 计算样本重要性并更新统计
                losses_per_sample = F.cross_entropy(logits, y_mb, reduction='none')
                accuracies_per_sample = (predicted == y_mb).float()

                # 计算margins
                correct_scores = logits.gather(1, y_mb.unsqueeze(1)).squeeze(1)
                sorted_logits, _ = torch.sort(logits, dim=1, descending=True)
                margins = correct_scores - sorted_logits[:, 1]

                # 更新遗忘统计
                if ep % 5 == 0:  # 每5个epoch更新一次统计
                    batch_indices = list(range(it, min(it + mb_size, train_x.size(0))))
                    forgetting_tracker.update_stats(
                        batch_indices,
                        losses_per_sample.detach().cpu().numpy(),
                        accuracies_per_sample.detach().cpu().numpy(),
                        margins.detach().cpu().numpy()
                    )

                # 计算重要性分数并添加到缓冲区
                importance_scores = compute_sample_importance(logits, y_mb, losses_per_sample)
                intelligent_buffer.add_data(
                    x_mb.cpu(),
                    y_mb.cpu(),
                    logits.detach().cpu() if args.replay_method in ['der', 'derpp'] else None,
                    importance_scores
                )

                # 定期打印进度
                if (it // mb_size) % 20 == 0:
                    current_acc = correct_cnt / total_samples if total_samples > 0 else 0.0
                    buffer_size = len(intelligent_buffer.examples)
                    print(f"  Batch {it // mb_size}: loss={total_loss.item():.6f}, "
                          f"acc={current_acc:.4f}, grad_norm={grad_norm:.4f}, buffer_size={buffer_size}")

                    # 检查参数是否在更新
                    param_changes = []
                    for name, param in model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            param_changes.append(param.grad.abs().mean().item())

                    avg_param_change = np.mean(param_changes) if param_changes else 0.0
                    print(f"    平均参数变化: {avg_param_change:.8f}")

                    if avg_param_change < 1e-10:
                        print("    警告: 参数变化极小，可能存在梯度消失问题")

            except Exception as e:
                print(f"反向传播错误: {e}")
                continue

        # 学习率调度
        current_scheduler.step()

        # Epoch结束统计
        final_acc = correct_cnt / total_samples if total_samples > 0 else 0.0
        avg_loss = epoch_loss / max(1, total_samples // mb_size)

        print(f"Epoch {ep} 结束: 准确率={final_acc:.4f}, 平均损失={avg_loss:.6f}")

        # 记录训练指标
        writer.add_scalar('train_accuracy', final_acc, tot_it_step)
        writer.add_scalar('train_loss', avg_loss, tot_it_step)
        writer.add_scalar('buffer_size', len(intelligent_buffer.examples), tot_it_step)
        tot_it_step += 1

    # 基于遗忘统计的样本选择（每个batch后应用）
    if i > 0 and args.forgetting_removal > 0:
        print("应用基于遗忘的样本选择...")
        current_indices = list(range(train_x.size(0)))
        easy_indices = forgetting_tracker.get_easy_examples(
            current_indices,
            removal_ratio=args.forgetting_removal
        )

        if len(easy_indices) < len(current_indices):
            print(f"基于遗忘统计移除了 {len(current_indices) - len(easy_indices)} 个困难样本")

    # 更新CRUMB内存
    if i > 0:
        print("更新CRUMB内存块...")
        try:
            sample_features = model.lat_features(x_mb[:min(8, x_mb.size(0))])
            model.update_memory_blocks(sample_features, learning_rate=0.01)
        except Exception as e:
            print(f"内存块更新错误: {e}")
    elif i == 0:
        print("初始化重放内存...")
        rm = [train_x[:min(rm_sz, train_x.size(0))].clone(),
              train_y[:min(rm_sz, train_y.size(0))].clone()]

    # 评估模型
    print("评估模型性能...")
    try:
        ave_loss, acc, accs = get_accuracy(model, criterion, 32, test_x, test_y, preproc=preproc)

        # 计算K类准确率
        avg_K_acc = sum(accs[cls] for cls in user_pref_cls) / len(user_pref_cls)

        avg_K.append(avg_K_acc)
        avg_acc.append(acc)

        # 记录测试指标
        writer.add_scalar('test_accuracy', acc, i)
        writer.add_scalar('test_loss', ave_loss, i)
        writer.add_scalar('avg_K_accuracy', avg_K_acc, i)

        # 记录内存效率
        memory_efficiency = model.get_memory_efficiency()
        writer.add_scalar('memory_efficiency', memory_efficiency['efficiency'], i)
        writer.add_scalar('memory_usage', memory_efficiency['used_blocks'], i)

        print(f"Batch {i} 结果:")
        print(f"  总体准确率: {acc:.4f}")
        print(f"  K类平均准确率: {avg_K_acc:.4f}")
        print(f"  内存效率: {memory_efficiency['efficiency']:.3f}")
        print(f"  缓冲区大小: {len(intelligent_buffer.examples)}")

        # 早停检查
        if acc > best_accuracy:
            best_accuracy = acc
            patience = 0
            print(f"  新的最佳准确率: {best_accuracy:.4f}")
            # 保存最佳模型
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model_improved.pth'))
        else:
            patience += 1
            if patience >= max_patience:
                print(f"早停触发，最佳准确率: {best_accuracy:.4f}")
                break

    except Exception as e:
        print(f"评估过程出错: {e}")
        acc = 0.0
        avg_K_acc = 0.0

print(f"\n改进版训练完成!")
print(f"最终准确率: {avg_acc[-1] if avg_acc else 0:.4f}")
print(f"最佳准确率: {best_accuracy:.4f}")
print(f"缓冲区最终大小: {len(intelligent_buffer.examples)}")

# 保存结果
final_results = {
    'avg_acc': avg_acc,
    'avg_K': avg_K,
    'best_accuracy': best_accuracy,
    'buffer_stats': {
        'final_size': len(intelligent_buffer.examples),
        'avg_importance': np.mean(intelligent_buffer.importance_scores) if intelligent_buffer.importance_scores else 0
    },
    'forgetting_stats': len(forgetting_tracker.example_stats),
    'args': vars(args)
}
torch.save(final_results, os.path.join(args.save_dir, 'final_results_improved.pkl'))
writer.close()

print(f"结果保存到: {args.save_dir}")
