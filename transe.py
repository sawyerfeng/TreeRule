import torch
import torch.nn as nn
import torch.nn.functional as F

class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, margin=1):
        super(TransE, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.embedding_dim = embedding_dim
        self.margin = margin
        
        # 初始化embeddings
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)
        
        # 归一化实体embeddings
        self.entity_embeddings.weight.data = F.normalize(
            self.entity_embeddings.weight.data, p=2, dim=1
        )
    
    def forward(self, pos_triples, neg_triples):
        # pos_triples: [batch_size, 3]
        pos_heads = self.entity_embeddings(pos_triples[:, 0])
        pos_relations = self.relation_embeddings(pos_triples[:, 1]) 
        pos_tails = self.entity_embeddings(pos_triples[:, 2])
        
        neg_heads = self.entity_embeddings(neg_triples[:, 0])
        neg_relations = self.relation_embeddings(neg_triples[:, 1])
        neg_tails = self.entity_embeddings(neg_triples[:, 2])
        
        pos_score = torch.norm(pos_heads + pos_relations - pos_tails, p=2, dim=1)
        neg_score = torch.norm(neg_heads + neg_relations - neg_tails, p=2, dim=1)
        
        loss = torch.mean(torch.relu(self.margin + pos_score - neg_score))
        return loss 

def generate_tree_paths_from_transe(model, rdict, body_len, top_k=100):
    """使用训练好的TransE模型生成高质量的tree paths
    
    Args:
        model: 训练好的TransE模型
        rdict: 关系字典
        body_len: path长度
        top_k: 每个关系保留的最相似关系数量
    """
    # 获取所有关系embeddings
    rel_embeds = model.relation_embeddings.weight.data
    
    # 计算关系间相似度矩阵
    sim_matrix = torch.mm(rel_embeds, rel_embeds.t())
    
    # 对每个关系找出top_k个最相似的关系
    _, topk_indices = torch.topk(sim_matrix, k=top_k, dim=1)
    
    all_paths = []
    for rel_idx in range(len(rdict.idx2rel)):
        similar_rels = topk_indices[rel_idx].tolist()
        
        # 生成长度为body_len的路径
        paths = []
        for i in range(body_len):
            paths.append(similar_rels[:top_k//body_len])
            
        # 组合paths生成tree结构
        import itertools
        tree_paths = list(itertools.combinations(paths, 2))
        all_paths.extend(tree_paths)
    
    # 转换为与原enumerate_tree相同的格式
    formatted_paths = []
    for path_pair in all_paths:
        path1 = [rdict.idx2rel[idx] for idx in path_pair[0]]
        path2 = [rdict.idx2rel[idx] for idx in path_pair[1]]
        formatted_paths.append([path1, path2])
        
    return formatted_paths