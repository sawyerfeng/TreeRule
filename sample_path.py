# 从anchor扩展路径
from random import sample
from tqdm import tqdm
import itertools
import torch

def construct_path_seq(opt,rdf_data, anchor_rdf, entity2desced, max_path_len=2, PRINT=False):
    anchor_h, anchor_r, anchor_t = parse_rdf(anchor_rdf)
    stack = [(anchor_h, anchor_r, anchor_t)]
    stack_print = ['{}-{}-{}'.format(anchor_h, anchor_r, anchor_t)]
    rule_seq, expended_node = [], []
    record = []
    path_num = 0
    while len(stack) > 0:
        cur_h, cur_r, cur_t = stack.pop(-1)
        cur_print = stack_print.pop(-1)
        deced_list = []
        
        if cur_t in entity2desced:
            deced_list = entity2desced[cur_t]  
        # 
        if len(cur_r.split('|')) < max_path_len and len(deced_list) > 0 and cur_t not in expended_node:
            for r_, t_ in deced_list:
                if t_ != cur_h and t_ != anchor_h:
                    stack.append((cur_t, cur_r+'|'+r_, t_))
                    stack_print.append(cur_print+'-{}-{}'.format(r_, t_))
        expended_node.append(cur_t)
        
        if len(cur_r.split('|')) == max_path_len:
            rule = cur_r + '-' + anchor_r
            rule_seq.append(rule)
            path_num += 1
            if (cur_h,r_,t_) not in record:
                record.append((cur_h,r_,t_))
    return rule_seq, record


def parse_rdf(rdf):
    """
    return: head, relation, tail
    """

    rdf_head, rdf_rel, rdf_tail= rdf
    return rdf_head, rdf_rel, rdf_tail

def construct_fact_dict(fact_rdf):
    """
    input:fact_rdf
    return:fact_dict: {r:[(h,t),...],...}
    """
    fact_dict = {}
    for rdf in fact_rdf:
        fact = parse_rdf(rdf)
        h, r, t = fact
        if r not in fact_dict:
            fact_dict[r] = []
        fact_dict[r].append(rdf)
    return fact_dict 

def sample_anchor_rdf(rdf_data, num=1):
    if num < len(rdf_data):
        return sample(rdf_data, num)
    else:
        return rdf_data

def rule2idx(rule, head_rdict):
    """
    Input a rule (string) and idx it
    """
    body, head = rule.split('-')
    body_path = body.split('|')
    # indexs include body idx seq + notation + head idx
    indexs = []
    for rel in body_path+[-1, head]:
        indexs.append(head_rdict.rel2idx[rel] if rel != -1 else -1)
    return indexs

def sample_tree_data(opt, max_path_len, anchor_num, fact_rdf, entity2desced, head_rdict, ent2idx, tree_num, sample_ratio=1.0):
    """
    采样训练数据的主函数
    Args:
        opt: 配置参数
        max_path_len: 最大路径长度
        anchor_num: 锚点数量
        fact_rdf: 事实三元组
        entity2desced: 实体到后代的映射字典
        head_rdict: 头实体关系字典
        sample_ratio: 采样比例，范围[0,1]，默认为1.0表示使用全部数据
    Returns:
        train_path_idx: 训练路径的索引列表
    """
    print("Sampling training data...")
    
    # 第一步:为每个关系类型采样锚点三元组
    anchors_rdf = []
    # 计算每个关系类型需要采样的数量
    relation_num = (head_rdict.__len__() -1) // 2  # 关系数量(不包含逆关系)
    # relation_num = 2
    per_anchor_num = anchor_num // relation_num  # 每个关系采样的anchor数量
    print("Number of head relation:{}".format(relation_num))
    print("Number of per_anchor_num: {}".format(per_anchor_num))
    
    # 构建fact字典,key为关系,value为包含该关系的三元组列表
    fact_dict = construct_fact_dict(fact_rdf)
    
    # 对每个非逆关系采样anchor三元组
    for head in head_rdict.rel2idx:
        if head != "None" and "inv_" not in head:
            sampled_rdf = sample_anchor_rdf(fact_dict[head], num=per_anchor_num)
            anchors_rdf.extend(sampled_rdf)
    print("Total anchor triples:", len(anchors_rdf))

    # 第二步:为每个anchor三元组构建路径
    all_path_idx = []  # 存储所有生成的路径对
    sample_number = 0
    pbar = tqdm(range(len(anchors_rdf)))
    
    for i in pbar:
        path_seq, record = construct_path_seq(opt,fact_rdf, anchors_rdf[i], 
                                            entity2desced, max_path_len, PRINT=False)
        sample_number += len(record)
        
        if len(path_seq) > 0:
            path_tmp = []
            for rule in path_seq:
                rule_idx = rule2idx(rule, head_rdict)
                # 添加头实体索引
                rule_idx.append(ent2idx[parse_rdf(anchors_rdf[i])[0]])
                idx = torch.LongTensor(rule_idx)
                path_tmp.append(idx)
            
            tree_paths = list(itertools.combinations(path_tmp, tree_num))
            
            for tree in tree_paths:
                tree_idx = torch.stack(tree)
                all_path_idx.append(tree_idx)
    
    # 根据采样比例随机采样
    total_samples = len(all_path_idx)
    sample_size = int(total_samples * sample_ratio)
    train_path_idx = sample(all_path_idx, sample_size) if sample_ratio < 1.0 else all_path_idx
    
    print(f"Total paths: {total_samples}, Sampled paths: {len(train_path_idx)}")
    return train_path_idx

            
            
            
    
    
