import torch
import torch.nn as nn
import torch.nn.functional as F

class EntityPredictor(nn.Module):
    def __init__(self, hidden_size, entity_num, device):
        super(EntityPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.entity_num = entity_num
        self.device = device
        
        # 实体嵌入层
        self.entity_emb = nn.Embedding(entity_num, hidden_size, padding_idx=0)
        
        # 增加注意力机制处理多路径
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4)
        
        # 调整网络结构
        self.fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_out = nn.Linear(hidden_size, entity_num)
        
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(0.3)  # 从0.2提高到0.3
        
        # 增加L2正则化权重
        self.entity_emb = nn.Embedding(entity_num, hidden_size, padding_idx=0)
        
        # 添加额外的正则化层
        self.batch_norm1 = nn.BatchNorm1d(hidden_size * 2)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        
    def forward(self, inputs):
        batch_size, path_num, _ = inputs.shape
        path_embeddings = []
        
        for p in range(path_num):
            path_batch_emb = []
            for b in range(batch_size):
                entities = torch.where(inputs[b, p] == 1)[0].to(self.device)
                if len(entities) > 0:
                    entities_emb = self.entity_emb(entities)
                    path_emb = entities_emb.mean(0)
                else:
                    path_emb = torch.zeros(self.hidden_size, device=self.device)
                path_batch_emb.append(path_emb)
            
            path_batch_emb = torch.stack(path_batch_emb)
            path_embeddings.append(path_batch_emb)
        
        path_embeddings = torch.stack(path_embeddings, dim=1)
        
        # 使用注意力机制替代简单平均
        path_embeddings = path_embeddings.transpose(0, 1)  # [path_num, batch, hidden]
        attn_output, _ = self.attention(path_embeddings, path_embeddings, path_embeddings)
        x = attn_output.mean(0)  # [batch, hidden]
        
        # 残差连接和层归一化
        x = self.layer_norm1(x)
        x = F.relu(self.fc1(x))
        x = self.layer_norm2(x)
        x = self.dropout(x)
        x = self.batch_norm1(x)
        x = F.relu(self.fc1(x))
        x = self.layer_norm2(x)
        x = self.dropout(x)
        x = self.batch_norm2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        logits = self.fc_out(x)
        
        return torch.sigmoid(logits)  # 改用sigmoid激活