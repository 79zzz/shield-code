import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        # 使用 softmax 获取概率
        BCE_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')

        # 获取每个样本的预测概率
        pt = torch.exp(-BCE_loss)

        # 计算 Focal Loss
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        return F_loss.mean()  # 返回均值损失
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, features, labels):
        # 假设 features 是形状为 [N, D] 的嵌入向量，labels 是 [N]
        batch_size = features.size(0)
        features = F.normalize(features, dim=1)  # L2 归一化
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # 正样本对的标签匹配
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.T).float().cuda()

        # 计算 InfoNCE 损失
        exp_sim = torch.exp(similarity_matrix) * (1 - torch.eye(batch_size).cuda())  # 排除对角线
        log_prob = similarity_matrix - torch.log(exp_sim.sum(1, keepdim=True) + 1e-9)

        loss = -(mask * log_prob).sum(1) / mask.sum(1)
        return loss.mean()


class CEWithEM(nn.Module):
    def __init__(self, reduction='mean', ignore_index=255, gamma=0.1):
        super(CEWithEM, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, inputs, targets, mask=None):
        _targets = targets.clone()
        if mask is not None:
            _targets[mask] = self.ignore_index

        ce_loss = F.cross_entropy(inputs, _targets, ignore_index=self.ignore_index, reduction=self.reduction)

        # 计算预测结果的概率分布
        p = F.softmax(inputs, dim=1)

        # 计算熵
        entropy = -torch.sum(p * torch.log(p + 1e-10), dim=1).mean()

        # 将熵正则化项加入到总损失中
        loss = ce_loss + self.gamma * entropy

        return loss


from skimage.measure import label

class SparseRegionEnhanceLoss(nn.Module):
    def __init__(self, sparse_weight=3.0, base_weight=1.0, max_region_size=500):
        """
        sparse_weight: 对稀疏区域的额外权重提升
        base_weight: 基础权重
        max_region_size: 最大区域大小，用于归一化
        """
        super(SparseRegionEnhanceLoss, self).__init__()
        self.sparse_weight = sparse_weight
        self.base_weight = base_weight
        self.max_region_size = max_region_size

    def forward(self, logits, targets):
        """
        logits: 模型预测的 [B, C, H, W]
        targets: 真实标签 [B, H, W]
        """
        # 计算基础交叉熵损失
        ce_loss = F.cross_entropy(logits, targets, reduction='none')  # [B, H, W]

        # 初始化权重
        weights = torch.ones_like(ce_loss) * self.base_weight

        # 遍历 batch，检测前景区域连通性
        for b in range(targets.size(0)):
            target_np = targets[b].cpu().numpy()
            for cls in range(1, logits.size(1)):  # 跳过背景类
                # 检测前景类别的连通区域
                binary_mask = (target_np == cls)
                labeled_regions = label(binary_mask)

                # 对每个连通区域计算权重
                for region_id in range(1, labeled_regions.max() + 1):
                    region_mask = (labeled_regions == region_id)
                    region_size = region_mask.sum()

                    # 动态权重计算：使用对数变换来调整权重
                    dynamic_weight = self.sparse_weight * (torch.log(torch.tensor(region_size + 1.0)) / torch.log(torch.tensor(self.max_region_size + 1.0)))

                    # 小区域给予稀疏增强权重
                    if region_size < self.max_region_size:  # 设定区域大小阈值
                        weights[b, region_mask] *= dynamic_weight

        # 加权损失
        weighted_loss = ce_loss * weights

        # 归一化损失
        return weighted_loss.mean()


def dice_loss(pred, target, epsilon=1e-6):
    """
    计算Dice损失
    pred: 预测的标签，shape 为 [B, C, H, W]，B为batch size，C为类别数，H和W为图像尺寸
    target: 真实标签，shape 为 [B, C, H, W]，与pred形状一致
    epsilon: 防止除零的常数，默认值为1e-6
    """
    # 将预测值和目标标签转换为二值化
    pred = torch.sigmoid(pred)  # 如果是多分类问题使用softmax
    pred = pred > 0.5  # 二值化，假设只有一个感兴趣的类别

    intersection = torch.sum(pred * target, dim=(2, 3))  # 求交集
    union = torch.sum(pred, dim=(2, 3)) + torch.sum(target, dim=(2, 3))  # 求并集

    dice = 2. * intersection / (union + epsilon)  # 计算Dice系数

    return 1 - torch.mean(dice)  # 返回损失，值越小越好

def dice_ce_loss(pred, target, epsilon=1e-6):
    """
    计算Dice + 交叉熵损失的组合
    pred: 预测的标签，shape 为 [B, C, H, W]
    target: 真实标签，shape 为 [B, C, H, W]
    epsilon: 防止除零的常数，默认值为1e-6
    """
    # 计算Dice损失
    pred_sigmoid = torch.sigmoid(pred)
    pred_bin = pred_sigmoid > 0.5  # 假设一个二分类问题

    intersection = torch.sum(pred_bin * target, dim=(2, 3))
    union = torch.sum(pred_bin, dim=(2, 3)) + torch.sum(target, dim=(2, 3))
    dice = 2. * intersection / (union + epsilon)
    dice_loss = 1 - torch.mean(dice)

    # 计算交叉熵损失
    ce_loss = F.binary_cross_entropy_with_logits(pred, target)

    # 组合两个损失，权重可以根据需要调整
    total_loss = dice_loss + ce_loss

    return total_loss


import torch
import torch.nn as nn
from torch import Tensor
from torch import einsum


# 计算概率分布是否符合条件
def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


class SurfaceLoss(nn.Module):
    def __init__(self):
        super(SurfaceLoss, self).__init__()
        self.idc = [1]  # 只关注类1，忽略背景

    def forward(self, probs: Tensor, dist_maps: Tensor, conf_u_w_cutmixed: Tensor, dynamic_threshold: float, ignore_mask_cutmixed: Tensor) -> Tensor:
        # 确保probs是一个概率分布
        assert simplex(probs), f"Invalid probs detected: sum of probabilities is not 1."

        # 从probs和dist_maps中提取出关心的类别（这里假设类别为1）
        pc = probs[:, self.idc, ...].type(torch.float32)  # 预测的类别1的概率
        dc = dist_maps[:, self.idc, ...].type(torch.float32)  # 目标距离图
        dc = dc.unsqueeze(-1)

        # 应用动态阈值，遮蔽低置信度的区域
        valid_mask = (conf_u_w_cutmixed >= dynamic_threshold) & (ignore_mask_cutmixed != 255)
        pc = pc * valid_mask  # 对预测概率进行遮蔽
        dc = dc * valid_mask  # 对目标距离图进行遮蔽

        # 计算加权损失，计算概率图与距离图的乘积
        multiplied = einsum("bcwh,bcwh->bcwh", pc, dc)

        # 返回平均损失
        loss = multiplied.mean()

        return loss

import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        """
        smooth: 避免除零错误的小常数，默认1e-6
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        preds: 模型预测的概率图 (B, C, H, W)
        targets: 真实标签 (B, H, W)，包含类别索引（如裂缝为1，背景为0）
        """
        # 确保预测值是二值化的概率图 (0-1之间)
        preds = torch.sigmoid(preds)  # 对预测进行sigmoid处理，得到0-1之间的概率

        # 假设我们关注类别 1（裂缝区域），背景为 0
        targets = (targets == 1).float()  # 将目标区域转换为 1，其它为 0

        # 转换为整数类型，并生成 one-hot 编码
        targets = targets.to(torch.int64)  # 转换为整数类型
        targets = F.one_hot(targets, num_classes=3).permute(0, 3, 1, 2).float()

        # 计算交集（preds * targets）
        intersection = torch.sum(preds * targets)

        # 计算预测和真实的总和
        total = torch.sum(preds) + torch.sum(targets)

        # 计算Dice系数，防止除零错误
        dice = (2 * intersection + self.smooth) / (total + self.smooth)

        # 返回损失值，通常Dice Loss是1-Dice系数，所以我们用1减去Dice系数
        return 1 - dice





