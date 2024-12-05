from torch import nn
import torch
from torch.nn import functional as F
import math
class EntityEncoder(nn.Module):
    def __init__(self, hidden_size, entity_num, device):
        super(EntityEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.entity_num = entity_num 
        self.device = device
        
        # 实体嵌入层
        self.entity_emb = nn.Embedding(entity_num, hidden_size, padding_idx=0)
        
        # 调整网络结构，确保输出维度正确
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.3)
        
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
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
        
        # 堆叠并处理所有路径的嵌入
        path_embeddings = torch.stack(path_embeddings, dim=1)  # [batch_size, path_num, hidden_size]
        
        # 对每个batch的所有路径取平均
        x = path_embeddings.mean(dim=1)  # [batch_size, hidden_size]
        
        # 应用变换
        x = self.layer_norm1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.batch_norm1(x)
        
        x = self.layer_norm2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.batch_norm2(x)
        
        return x  # [batch_size, hidden_size]

class PathEncoder(nn.Module):
    def __init__(self, hidden_size, rel_num, device):
        super(PathEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.rel_num = rel_num
        self.device = device
        
        # ���系嵌入层
        self.rel_emb = nn.Embedding(rel_num, hidden_size)
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.3
        )
        
        # 递归MLP
        self.MLP = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size*2, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )
        
        # 投影层
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.dropout = nn.Dropout(0.3)

    def body_recursive(self, inputs):
        """递归编码路径表示
        inputs: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = inputs.shape
        for i in range(seq_len - 1):
            j = i + 1
            if i == 0:
                emb_1 = inputs[:,i,:]
                emb_2 = inputs[:,j,:]
                emb = self.MLP(torch.cat((emb_1, emb_2), dim=1))
            else:
                emb_1 = emb
                emb_2 = inputs[:,j,:]
                emb = self.MLP(torch.cat((emb_1, emb_2), dim=1))
        return emb

    def forward(self, x, contrast=True):
        """
        输入: 
            x - [batch_size, seq_len]的关系ID序列
            contrast - 是否用于对比学习
        """
        # 获取关系嵌入
        rel_embeds = self.rel_emb(x)  # [batch_size, seq_len, hidden_size]
        
        # LSTM编码
        lstm_output, _ = self.lstm(rel_embeds)  # [batch_size, seq_len, hidden_size]
        
        # 递归编码
        path_repr = self.body_recursive(lstm_output)
        path_repr = self.dropout(path_repr)
        
        if contrast:
            # 投影,用于对比学习
            proj_repr = self.proj(path_repr)
            proj_repr = F.normalize(proj_repr, dim=-1)
            # 计算比损失
            loss = self.cal_loss(proj_repr)
            return loss, proj_repr
        else:
            return path_repr

    def cal_loss(self, sentence_embedding, tao=0.5):
        """计算对比学习损失
        Args:
            sentence_embedding: [batch_size, hidden_size] 路径表示
            tao: 温度参数
        """
        idxs = torch.arange(sentence_embedding.size(0))
        y_true = idxs + 1 - idxs % 2 * 2  # 构建标签：相邻样本互为正样本
        y_true = y_true.to(sentence_embedding.device)
        
        # 计算余弦相似度矩阵
        sim = F.cosine_similarity(sentence_embedding.unsqueeze(1), 
                                sentence_embedding.unsqueeze(0), dim=-1)
        
        # 移除自身相似度
        I = torch.eye(sim.size(0)).to(sim.device)
        sim = sim - I
        
        # 应用温度系数
        sim = sim / tao
        
        # 计算对比损失
        loss = F.cross_entropy(sim, y_true)
        return torch.mean(loss)
class TreeEncoder(nn.Module):
    def __init__(self, hidden_size, rel_num, ent_num, device):
        super(TreeEncoder, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.path_encoder = PathEncoder(hidden_size, rel_num, device).to(device)
        
        # 投影头
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        ).to(device)
        
        # 添加实体编码器
        self.entity_encoder = EntityEncoder(hidden_size, ent_num, device).to(device)
        
        # 添加cross attention层
        self.cross_attention = nn.MultiheadAttention(hidden_size, num_heads=4)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, paths, entities=None, contrast=True):
        # 确保输入数据在GPU上
        paths = paths.to(self.device)
        
        # 对每条路径进行编码
        path_embeddings = [self.path_encoder(path, contrast=False) for path in paths]
        path_embeddings = torch.stack(path_embeddings)
        
        # 计算树嵌入
        tree_embedding = torch.mean(path_embeddings, dim=1)
        
        # 如果有实体信息，进行cross attention
        if entities is not None:
            # 首先通过EntityEncoder处理实体信息
            entity_embeddings = self.entity_encoder(entities)  # [batch_size, hidden_size]
            
            # 调整维度以适应attention
            tree_embedding = tree_embedding.unsqueeze(0)  # [1, batch_size, hidden_size]
            entity_embeddings = entity_embeddings.unsqueeze(0)  # [1, batch_size, hidden_size]
            
            # 进行cross attention
            attn_output, _ = self.cross_attention(
                tree_embedding,  # [1, batch_size, hidden_size]
                entity_embeddings,  # [1, batch_size, hidden_size]
                entity_embeddings   # [1, batch_size, hidden_size]
            )
            
            # 残差连接和层归一化
            tree_embedding = tree_embedding + attn_output
            tree_embedding = self.layer_norm1(tree_embedding)
            tree_embedding = self.dropout(tree_embedding)
            
            # 转回[batch_size, hidden_size]格式
            tree_embedding = tree_embedding.squeeze(0)
        
        if contrast:
            # 通过投影头进行映射
            projected_embedding = self.projection_head(tree_embedding)
            projected_embedding = F.normalize(projected_embedding, dim=-1)
            
            # 计算树级别的对比损失
            loss = self.cal_loss(projected_embedding)
            
            return loss, projected_embedding
        else:
            return tree_embedding

    def cal_loss(self, sentence_embedding, tao=0.5):
        """计算对比学习损失
        Args:
            sentence_embedding: [batch_size, hidden_size] 路径表示
            tao: 温度参数
        """
        idxs = torch.arange(sentence_embedding.size(0))
        y_true = idxs + 1 - idxs % 2 * 2  # 构建标签：相邻样本互为正样本
        y_true = y_true.to(sentence_embedding.device)
        
        # 计算余弦相似度阵
        sim = F.cosine_similarity(sentence_embedding.unsqueeze(1), 
                                sentence_embedding.unsqueeze(0), dim=-1)
        
        # 移除自身相似度
        I = torch.eye(sim.size(0)).to(sim.device)
        sim = sim - I
        
        # 应用温度系数
        sim = sim / tao
        
        # 计算对比损失
        loss = F.cross_entropy(sim, y_true)
        return torch.mean(loss)



        
class RulePredictor(nn.Module):
    def __init__(self, hidden_size, rel_num, entity_num, device):
        super(RulePredictor, self).__init__()
        self.hidden_size = hidden_size
        self.rel_num = rel_num
        self.entity_num = entity_num
        self.device = device
        
        # 关系和实体的嵌入层 - 注意实体数量不需要+1
        self.rel_emb = nn.Embedding(rel_num+1, hidden_size)
        self.entity_emb = nn.Embedding(entity_num, hidden_size)  # 移除+1
        
        # 共享的特征处理层
        self.bn_linear = nn.Linear(hidden_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        
        # 关系预测相关层
        self.fc_k_rel = nn.Linear(hidden_size, hidden_size)
        self.fc_q_rel = nn.Linear(hidden_size, hidden_size)
        
        # 实体预测相关层
        self.fc_k_ent = nn.Linear(hidden_size, hidden_size)
        self.fc_q_ent = nn.Linear(hidden_size, hidden_size)
        
        # MLP分类器 - 实体分类器输出维度修正
        self.rel_classifier = nn.Sequential(
            nn.Linear(hidden_size, rel_num+1),
            nn.Softmax(dim=-1)
        )
        self.entity_classifier = nn.Sequential(
            nn.Linear(hidden_size, entity_num),  # 移除+1
            nn.Softmax(dim=-1)
        )

    def forward(self, inputs, mode='mlp'):
        # 特征处理
        inputs = self.bn_linear(inputs)
        inputs = self.bn(inputs)
        
        if mode == 'mlp':
            # MLP模式
            rel_logits = self.rel_classifier(inputs)
            entity_logits = self.entity_classifier(inputs)
        else:
            # Transformer模式
            rel_scores = self.transformer_attention_rel(inputs)
            entity_scores = self.transformer_attention_entity(inputs)
            rel_logits = rel_scores.squeeze(1)
            entity_logits = entity_scores.squeeze(1)
            
        return rel_logits, entity_logits
    
    def transformer_attention_rel(self, inputs):
        inputs = inputs.unsqueeze(1)
        batch_size = inputs.shape[0]
        
        # 获取所有关系的嵌入
        idx_rel = torch.LongTensor(range(self.rel_num+1)).repeat(batch_size, 1).to(self.device)
        relation_emb = self.rel_emb(idx_rel)
        
        # 计算注意力分数
        query = self.fc_q_rel(inputs)
        key = self.fc_k_rel(relation_emb)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.hidden_size)
        
        return scores
    
    def transformer_attention_entity(self, inputs):
        inputs = inputs.unsqueeze(1)
        batch_size = inputs.shape[0]
        
        # 获取所有实体的嵌入 - 修正范围
        idx_ent = torch.LongTensor(range(self.entity_num)).repeat(batch_size, 1).to(self.device)  # 移除+1
        entity_emb = self.entity_emb(idx_ent)
        
        # 计算注意力分数
        query = self.fc_q_ent(inputs)
        key = self.fc_k_ent(entity_emb)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.hidden_size)
        
        return scores

