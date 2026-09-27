import torch
import torch.nn as nn

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