import torch
import torch.nn as nn

# class FirstDiffLoss(nn.Module):
#     def __init__(self,lambda2=1.0, temperature=0.07):
#         super(FirstDiffLoss, self).__init__()
#         self.lambda2 = lambda2
#         self.temperature = temperature

#     def infoNCE(self, anchor, positive, negative):
#         pos_similarity = torch.matmul(anchor, positive.t()) / self.temperature
#         neg_similarity = torch.matmul(anchor, negative.t()) / self.temperature
#         loss = -torch.log(torch.exp(pos_similarity) / (torch.exp(pos_similarity) + torch.sum(torch.exp(neg_similarity), dim=1)))
#         return loss.mean()

#     def forward(self, z_image_gen, z_image, z_image_gen_pos, z_image_pos):

#         LzI2zI_gen = self.infoNCE(z_image, z_image_gen, z_image_pos)
#         LzI_gen2zI = self.infoNCE(z_image_gen, z_image, z_image_gen_pos)

#         L_total = self.lambda2 * (LzI2zI_gen + LzI_gen2zI)
#         return L_total

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLossWithLabel(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLossWithLabel, self).__init__()
        self.temperature = temperature

    def forward(self, z_image, z_text, labels):
        z_image_norm = F.normalize(z_image, p=2, dim=1)
        z_text_norm  = F.normalize(z_text, p=2, dim=1)
        
        sim_matrix = torch.matmul(z_image_norm, z_text_norm.t()) / self.temperature
        
        batch_size = z_image.size(0)
        loss_i2t = 0.0
        loss_t2i = 0.0
        
        for i in range(batch_size):
            current_label = labels[i]
            
            pos_sim = sim_matrix[i, i]
            
            neg_indices = (labels != current_label).nonzero(as_tuple=False).reshape(-1)
            if len(neg_indices) == 0:
                continue
            rand_idx = neg_indices[torch.randint(0, len(neg_indices), (1,)).item()]
            neg_sim = sim_matrix[i, rand_idx]
            
            logits_i2t = torch.stack([pos_sim, neg_sim], dim=0)
            target = torch.tensor(0, device=z_image.device).unsqueeze(0)
            loss_i2t += F.cross_entropy(logits_i2t.unsqueeze(0), target)
            
            pos_sim_t2i = sim_matrix[i, i]
            neg_indices_img = (labels != current_label).nonzero(as_tuple=False).reshape(-1)
            if len(neg_indices_img) == 0:
                continue
            rand_idx_img = neg_indices_img[torch.randint(0, len(neg_indices_img), (1,)).item()]
            neg_sim_t2i = sim_matrix[rand_idx_img, i]
            
            logits_t2i = torch.stack([pos_sim_t2i, neg_sim_t2i], dim=0)
            loss_t2i += F.cross_entropy(logits_t2i.unsqueeze(0), target)
        
        loss_i2t = loss_i2t / batch_size
        loss_t2i = loss_t2i / batch_size
        
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