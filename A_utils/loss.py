import torch
import torch.nn as nn

# class FirstDiffLoss(nn.Module):
#     def __init__(self,lambda2=1.0, temperature=0.07):
#         """
#         初始化 SceneDiffLoss 类
#         :param lambda: 损失的权重（生成隐空间特征与图像隐空间特征的对比损失）
#         :param temperature: 温度参数，用于调节对比损失中余弦相似度的尺度
#         """
#         super(FirstDiffLoss, self).__init__()
#         self.lambda2 = lambda2
#         self.temperature = temperature

#     def infoNCE(self, anchor, positive, negative):
#         """
#         计算 InfoNCE 损失（用于对比学习）
#         :param anchor: 锚点样本（特征向量）
#         :param positive: 正样本，与锚点应该相似的特征向量
#         :param negative: 负样本，与锚点不相似的特征向量集合
#         :return: 平均 InfoNCE 损失值
#         """
#         # 计算锚点与正样本的余弦相似度，并除以温度参数
#         pos_similarity = torch.matmul(anchor, positive.t()) / self.temperature
#         # 计算锚点与负样本的余弦相似度，并除以温度参数
#         neg_similarity = torch.matmul(anchor, negative.t()) / self.temperature
#         # 计算 InfoNCE 损失： -log(exp(pos) / (exp(pos) + ∑exp(neg)))
#         loss = -torch.log(torch.exp(pos_similarity) / (torch.exp(pos_similarity) + torch.sum(torch.exp(neg_similarity), dim=1)))
#         return loss.mean()

#     def forward(self, z_image_gen, z_image, z_image_gen_pos, z_image_pos):
#         """
#         前向传播，计算总损失。
#         1. 生成隐空间特征与图像隐空间特征之间的对比损失（LzI2zI' 和 LzI'2zI）
        
#         :param F_image: 图像特征（通过图像编码器获得）
#         :param z_image_gen: 生成的隐空间融合特征（通过扩散模型得到的 latent 融合特征）
#         :param z_image: 图像隐空间特征（通过预训练 autoencoder 得到）
#         :param F_image_pos: 与 F_image 对应的正样本融合特征
#         :param z_image_gen_pos: 与 z_image_gen 对应的正样本图像隐特征
#         :param z_image_pos: 与 z_image 对应的正样本生成特征
#         :return: 总损失 L_total
#         """

#         # 计算由文本条件指导的生成隐空间特征与图像隐空间特征的对比损失，这样来调图像CLIP，帮助CLIP捕捉细节纹理等低级视觉特征。
#         # LzI2zI_gen: 图像隐空间特征到生成融合特征的 InfoNCE 损失
#         LzI2zI_gen = self.infoNCE(z_image, z_image_gen, z_image_pos)
#         # LzI_gen2zI: 生成融合特征到图像隐空间特征的 InfoNCE 损失
#         LzI_gen2zI = self.infoNCE(z_image_gen, z_image, z_image_gen_pos)

#         # 组合两部分损失，得到总损失
#         L_total = self.lambda2 * (LzI2zI_gen + LzI_gen2zI)
#         return L_total

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLossWithLabel(nn.Module):
    """
    基于 InfoNCE 损失的对比学习损失类，支持接收图文对的标签，
    按照标签来区分正负样本对。
    
    假设：
      - batch 中第 i 个图像和第 i 个文本构成正样本对。
      - 负样本对则是从同一个 batch 中随机选择一个与当前样本标签不同的样本作为负样本。
    
    本损失用于使图像经过 CLIP 图像编码器、adapter、VAE 获得的潜在特征
    与文本经过 CLIP 文本编码器、SD 产生的潜在特征之间对齐。    
    """
    def __init__(self, temperature=0.07):
        """
        初始化对比损失类
        
        :param temperature: 温度参数，用于缩放余弦相似度，控制损失函数中 logits 的平滑程度
        """
        super(ContrastiveLossWithLabel, self).__init__()
        self.temperature = temperature

    def forward(self, z_image, z_text, labels):
        """
        前向传播，计算基于标签的对比学习损失。
        
        :param z_image: 图像潜在特征，形状 (B, D)，通过 CLIP 图像编码器、adapter、VAE 得到
        :param z_text: 文本潜在特征，形状 (B, D)，通过 CLIP 文本编码器、SD 得到
        :param labels: 图文对的标签，形状 (B, )，其中第 i 个图像与第 i 个文本对应应具有相同标签
        :return: 对比损失（标量）
        """
        # 对图像和文本潜在特征进行 L2 归一化，保证每个向量为单位向量，从而便于计算余弦相似度
        z_image_norm = F.normalize(z_image, p=2, dim=1)
        z_text_norm  = F.normalize(z_text, p=2, dim=1)
        
        # 计算余弦相似度矩阵，形状 (B, B)；其中 sim_matrix[i, j] 表示第 i 个图像和第 j 个文本之间的相似度
        sim_matrix = torch.matmul(z_image_norm, z_text_norm.t()) / self.temperature
        
        batch_size = z_image.size(0)
        loss_i2t = 0.0   # 图像到文本方向的损失
        loss_t2i = 0.0   # 文本到图像方向的损失
        
        # 对于每个样本，使用标签来区分正负样本
        for i in range(batch_size):
            # 当前样本的标签
            current_label = labels[i]
            
            # 正样本：假设 batch 中第 i 个图像与第 i 个文本为正对，
            # 则正样本的相似度为 sim_matrix[i, i]
            pos_sim = sim_matrix[i, i]
            
            # 负样本选择：在文本方向上，从 batch 中找出所有标签与当前样本不同的文本索引
            neg_indices = (labels != current_label).nonzero(as_tuple=False).reshape(-1)
            if len(neg_indices) == 0:
                # 如果没有负样本（理论上不应发生），则跳过本样本
                continue
            # 随机从负样本集合中选取一个负样本
            rand_idx = neg_indices[torch.randint(0, len(neg_indices), (1,)).item()]
            neg_sim = sim_matrix[i, rand_idx]
            
            # 构建图像到文本方向的 logits，正样本放在第一个位置，
            # logits 为一个二分类问题：[正样本相似度, 负样本相似度]
            logits_i2t = torch.stack([pos_sim, neg_sim], dim=0)
            # 目标标签为 0，即第一个位置为正样本
            target = torch.tensor(0, device=z_image.device).unsqueeze(0)
            loss_i2t += F.cross_entropy(logits_i2t.unsqueeze(0), target)
            
            # 同理，构建文本到图像方向的损失：
            # 正样本：同样认为当前文本和第 i 个图像对应为正样本，对应相似度 sim_matrix[i, i]
            pos_sim_t2i = sim_matrix[i, i]
            # 在图像方向上，从 batch 中找出所有标签与当前样本不同的图像索引
            neg_indices_img = (labels != current_label).nonzero(as_tuple=False).reshape(-1)
            if len(neg_indices_img) == 0:
                continue
            rand_idx_img = neg_indices_img[torch.randint(0, len(neg_indices_img), (1,)).item()]
            neg_sim_t2i = sim_matrix[rand_idx_img, i]
            
            logits_t2i = torch.stack([pos_sim_t2i, neg_sim_t2i], dim=0)
            loss_t2i += F.cross_entropy(logits_t2i.unsqueeze(0), target)
        
        # 平均每个样本的损失
        loss_i2t = loss_i2t / batch_size
        loss_t2i = loss_t2i / batch_size
        
        # 最终损失为两个方向损失的平均值
        loss = (loss_i2t + loss_t2i) / 2.0
        return loss


# class ContrastiveLossWithLabel(nn.Module):
#     def __init__(self, temperature=0.07, use_attention=True):
#         super().__init__()
#         self.temperature = temperature
#         self.use_attention = use_attention

#         if use_attention:
#             self.channel_att = ChannelAttention(channels=4)  # for [B, 4, 64, 64]
#         else:
#             self.channel_att = nn.Identity()

#         self.ce = nn.CrossEntropyLoss()

#     def forward(self, image_latent, text_latent):
#         """
#         image_latent: z_I ∈ [B, 4, 64, 64] ← 来自 image→CLIP→adapter→VAE（可训练）
#         text_latent:  z_T ∈ [B, 4, 64, 64] ← 来自 text→CLIP→SD（冻结）
#         """

#         B = image_latent.size(0)

#         # Channel Attention
#         z_i = self.channel_att(image_latent)   # [B, 4, 64, 64]
#         z_t = self.channel_att(text_latent)

#         # Flatten
#         z_i = F.normalize(z_i.view(B, -1), dim=-1)  # [B, D]
#         z_t = F.normalize(z_t.view(B, -1), dim=-1)

#         # Similarity logits
#         logits = torch.matmul(z_i, z_t.T) / self.temperature  # [B, B]
#         labels = torch.arange(B).to(z_i.device)

#         # Symmetric InfoNCE loss
#         loss_i2t = self.ce(logits, labels)       # image -> text
#         loss_t2i = self.ce(logits.T, labels)     # text -> image

#         return (loss_i2t + loss_t2i) / 2


# class ChannelAttention(nn.Module):
#     """SE-style attention used to weigh important channels in latent features."""
#     def __init__(self, channels, reduction=16):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channels, channels // reduction),
#             nn.ReLU(inplace=True),
#             nn.Linear(channels // reduction, channels),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         B, C, _, _ = x.size()
#         y = self.avg_pool(x).view(B, C)       # [B, C]
#         y = self.fc(y).view(B, C, 1, 1)       # [B, C, 1, 1]
#         return x * y                          # broadcasting