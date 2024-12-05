from torch import nn
import torch
from torch.nn import functional as F
import math

class PathEncoder(nn.Module):
    def __init__(self, hidden_size, rel_num, device):
        super(PathEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.rel_num = rel_num
        self.device = device
        
        # 关系嵌入层
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
            # 计算��比损失
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
    def __init__(self, hidden_size, rel_num, device):
        super(TreeEncoder, self).__init__()
        self.device = device
        self.path_encoder = PathEncoder(hidden_size, rel_num, device).to(device)
        
        # 投影头
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        ).to(device)

    def forward(self, paths, contrast=True):
        # 确保输入数据在GPU上
        paths = paths.to(self.device)
        
        # 对每条路径进行编码 (不需要对比损失)
        path_embeddings = [self.path_encoder(path, contrast=False) for path in paths]
        path_embeddings = torch.stack(path_embeddings)

        # 计算树嵌入
        tree_embedding = torch.mean(path_embeddings, dim=1)
        
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
        y_true = idxs + 1 - idxs % 2 * 2  # 构��标签：相邻样本互为正样本
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

class EntityEncoder(nn.Module):
    def __init__(self, hidden_size, entity_num, device):
        super(EntityEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.entity_num = entity_num
        self.device = device
        
        # 实体嵌入层
        self.entity_emb = nn.Embedding(entity_num, hidden_size)
        
        # 注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=2,
            dropout=0.1,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # 投影层
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, contrast=True):
        """
        输入: 
            x - [batch_size, seq_len]的实体ID序列
            contrast - 是否用于对比学习
        """
        # 获取实体嵌入
        entity_embeds = self.entity_emb(x)  # [batch_size, seq_len, hidden_size]
        
        # 自注意力处理实体序列
        entity_embeds = self.layer_norm(entity_embeds)
        attn_output, _ = self.attention(
            entity_embeds, 
            entity_embeds, 
            entity_embeds
        )
        
        # 平均池化得到整体表示
        path_repr = torch.mean(attn_output, dim=1)
        path_repr = self.dropout(path_repr)
        
        if contrast:
            # 投影,用于对比学习
            proj_repr = self.proj(path_repr)
            proj_repr = F.normalize(proj_repr, dim=-1)
            # 计算对比损失
            loss = self.cal_loss(proj_repr)
            return loss, proj_repr
        else:
            return path_repr

    def cal_loss(self, sentence_embedding, tao=0.5):
        """计算对比学习损失"""
        idxs = torch.arange(sentence_embedding.size(0))
        y_true = idxs + 1 - idxs % 2 * 2
        y_true = y_true.to(sentence_embedding.device)
        
        sim = F.cosine_similarity(sentence_embedding.unsqueeze(1), 
                                sentence_embedding.unsqueeze(0), dim=-1)
        
        I = torch.eye(sim.size(0)).to(sim.device)
        sim = sim - I
        sim = sim / tao
        
        loss = F.cross_entropy(sim, y_true)
        return torch.mean(loss)


class TreeRule(nn.Module):
    def __init__(self, hidden_size, rel_num, device, model_type='base'):
        super(TreeRule, self).__init__()
        self.hidden_size = hidden_size
        self.rel_num = rel_num
        self.device = device
        self.model_type = model_type
        
        # 基础层
        self.emb = nn.Embedding(self.rel_num, self.hidden_size)
        
        # 根据模型类型初始化所需层
        if model_type in ['base', 'hybrid']:
            self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True, dropout=0.3)
            
        if model_type in ['attention', 'hybrid']:
            self.path_attention = nn.MultiheadAttention(hidden_size, num_heads=1)
            self.seq_attention = nn.MultiheadAttention(hidden_size, num_heads=1)
            self.layer_norm1 = nn.LayerNorm(hidden_size)
            self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        # 添加dropout层
        self.dropout = nn.Dropout(0.3)  # 可以调整dropout率

    def forward(self, inputs, mode='train'):
        batch_size, seq_len, path_len = inputs.size()
        inputs = inputs.view(batch_size * seq_len, path_len)
        emb = self.emb(inputs)
        
        if self.model_type == 'base':
            return self._forward_base(emb, batch_size, seq_len)
        elif self.model_type == 'attention':
            return self._forward_attention(emb, batch_size, seq_len)
        elif self.model_type == 'hybrid':
            return self._forward_hybrid(emb, batch_size, seq_len)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _forward_base(self, emb, batch_size, seq_len):
        """基础模式：只使用LSTM"""
        out, (h, c) = self.lstm(emb)
        out = out.mean(dim=1)
        out = out.view(batch_size, seq_len, -1)
        sentence_emb = out.mean(dim=1)
        sentence_emb = self.dropout(sentence_emb)
        # 在最终输出前添加dropout
        sentence_emb = self.dropout(sentence_emb)
        return sentence_emb
    
    def _forward_attention(self, emb, batch_size, seq_len):
        """纯Attention模式"""
        # 1. Path内部attention
        emb = emb.transpose(0, 1)
        path_attn_out, _ = self.path_attention(emb, emb, emb)
        path_attn_out = path_attn_out.transpose(0, 1)
        path_attn_out = self.layer_norm1(path_attn_out)
        
        # Path平均池化
        path_emb = path_attn_out.mean(dim=1)
        
        # 2. Paths之间attention
        path_emb = path_emb.view(batch_size, seq_len, -1)
        path_emb = path_emb.transpose(0, 1)
        seq_attn_out, _ = self.seq_attention(path_emb, path_emb, path_emb)
        seq_attn_out = seq_attn_out.transpose(0, 1)
        seq_attn_out = self.layer_norm2(seq_attn_out)
        
        sentence_emb = seq_attn_out.mean(dim=1)
        return sentence_emb
    
    def _forward_hybrid(self, emb, batch_size, seq_len):
        """混合模式：LSTM + Attention"""
        # 1. Path内部attention
        emb = emb.transpose(0, 1)
        path_attn_out, _ = self.path_attention(emb, emb, emb)
        path_attn_out = path_attn_out.transpose(0, 1)
        path_attn_out = self.layer_norm1(path_attn_out)
        
        # 2. LSTM处理
        out, (h, c) = self.lstm(path_attn_out)
        out = out.mean(dim=1)
        out = out.view(batch_size, seq_len, -1)
        
        # 3. Paths之间attention
        out = out.transpose(0, 1)
        seq_attn_out, _ = self.seq_attention(out, out, out)
        seq_attn_out = seq_attn_out.transpose(0, 1)
        seq_attn_out = self.layer_norm2(seq_attn_out)
        
        sentence_emb = seq_attn_out.mean(dim=1)
        return sentence_emb
        
class RulePredictor(nn.Module):
    def __init__ (self,hidden_size,rel_num,device):
        super(RulePredictor,self).__init__()
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(rel_num+1,hidden_size)
        self.rel_num = rel_num
        self.device = device
        self.bn_linear = nn.Linear(hidden_size,hidden_size)
        
        self.fc_k = nn.Linear(hidden_size,hidden_size)
        self.fc_v = nn.Linear(hidden_size,hidden_size)
        self.fc_q = nn.Linear(hidden_size,hidden_size)

        self.bn = nn.BatchNorm1d(hidden_size)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size,rel_num+1),
            nn.Softmax(dim=-1)
        )
    def forward(self,inputs,mode = 'mlp'):
        # Batch Normalization层
        inputs = self.bn_linear(inputs)
        inputs = self.bn(inputs)
        #如果是MLP模式，就直接输出分类结果
        if mode == 'mlp':
            logits = self.classifier(inputs)
            return logits
        #如果是transformer模式，就进行transformer attention
        else:
            logits = self.transformer_attention(inputs)
            logits = logits.squeeze(1)
            return logits
    
    def transformer_attention(self, inputs):
        inputs = inputs.unsqueeze(1)
        batch_size, seq_len, emb_size = inputs.shape#创建一个全0 全relation的embedding，并且共享参数
        idx_ = torch.LongTensor(range(self.rel_num+1)).repeat(batch_size, 1).to(self.device)
        relation_emb = self.emb(idx_) # (batch_size, |R|, emb_size)
        # key= torch.cat((relation_emb, inputs), dim=1) # (batch_size, |R|+seq_len, emb_size)，再加上自己
        key = relation_emb## 不加上自己
        #q为inputs,key为relation，
        query = self.fc_q(inputs) # (batch_size, seq_len, emb_size)
        key = self.fc_k(key) # (batch_size, |R|+seq_len, emb_size)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
        / math.sqrt(self.hidden_size)   # (batch_size, seq_len, |R|+seq_len)
        #表示seq_len到|R|+seq_len的相似度
        return scores # unnormalized，表示输入时间步与其他时间步和关系之间的注意力关系

