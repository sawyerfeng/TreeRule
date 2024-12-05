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
        self.entity_emb = nn.Embedding(entity_num + 1, hidden_size)
        
        # 注意力层参数
        self.fc_q = nn.Linear(hidden_size, hidden_size)
        self.fc_k = nn.Linear(hidden_size, hidden_size)
        self.fc_v = nn.Linear(hidden_size, hidden_size)
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, inputs):
        """
        输入: [batch, path_num, entity_num]
        输出: [batch, entity_num]
        """
        batch_size, path_num, _ = inputs.shape
        
        # 生成实体候选集
        idx = torch.arange(self.entity_num).repeat(batch_size, 1).to(self.device)
        candidates_emb = self.entity_emb(idx)  # [batch, entity_num, hidden]
        
        # 将输入转换为嵌入
        inputs = inputs.view(-1, self.entity_num)
        inputs = torch.matmul(inputs, self.entity_emb.weight)  # [batch*path_num, hidden]
        inputs = inputs.view(batch_size, path_num, -1)  # [batch, path_num, hidden]
        
        # 注意力机制
        query = self.fc_q(inputs)  # [batch, path_num, hidden]
        key = self.fc_k(candidates_emb)  # [batch, entity_num, hidden]
        
        # 计算注意力分数
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.hidden_size ** 0.5)
        scores = F.softmax(scores, dim=-1)  # [batch, path_num, entity_num]
        
        # 聚合所有路径的预测结果
        scores = scores.mean(dim=1)  # [batch, entity_num]
        
        return scores