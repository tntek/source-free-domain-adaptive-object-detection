import torch
import torch.autograd as ag
import torch.nn as nn
import torch.nn.functional as F


class MemoryBank(nn.Module):
    def __init__(self, num_classes=8, capacity=10):
        super(MemoryBank, self).__init__()
        self.num_classes = num_classes
        self.capacity = capacity
        # 初始化memory bank，每个类别一个队列
        self.memory = {label: [] for label in range(num_classes)}
        
    def update(self, features, labels):
        """更新memory bank，features和labels是相同长度的列表或batch"""
        for feature, label in zip(features, labels):
            # 确保label是整数
            label = int(label)
            # 检查类别是否在memory bank中
            if label not in self.memory:
                continue
            # 更新对应类别的队列
            if len(self.memory[label]) < self.capacity:
                self.memory[label].append(feature)
            else:
                # 如果队列已满，先出队一个元素再入队
                self.memory[label].pop(0)
                self.memory[label].append(feature)
                
    def read(self, num_samples_per_class=5):
        """随机抽取特定数量的特征和标签，每个类别抽取num_samples_per_class个"""
        from random import sample
        features = []
        labels = []
        for label in self.memory.keys():
            # 获取该类别可用样本数
            available_samples = len(self.memory[label])
            # 确保不超出每类的样本数
            num_samples = min(num_samples_per_class, available_samples)
            if num_samples > 0:
                sampled_features = sample(self.memory[label], num_samples)
                features.extend(sampled_features)
                labels.extend([label] * num_samples)
        return features, labels
    def reset(self):
        """重置memory bank，清空所有类别的存储队列"""
        self.memory = {label: [] for label in range(self.num_classes)}