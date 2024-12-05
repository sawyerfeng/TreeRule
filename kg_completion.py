from audioop import reverse
from wsgiref import headers
from xml.dom.minidom import Element
from dataset import *
import copy
import re
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import numpy as np
from scipy import sparse
from collections import defaultdict
import argparse
from utils import *
from tqdm import tqdm
from models.entity_predictor import EntityPredictor
import sys
import io
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

head2mrr = defaultdict(list)
head2hit_10 = defaultdict(list)
head2hit_1 = defaultdict(list)
head2hit_3 = defaultdict(list)

def kg_completion(opt, rules, dataset):
    """
    Input a set of rules
    Complete Queries from test_rdf based on rules and fact_rdf 
    """
    # 准备数据
    fact_rdf, train_rdf, valid_rdf, test_rdf = dataset.fact_rdf, dataset.train_rdf, dataset.valid_rdf, dataset.test_rdf
    gt = get_gt(dataset)
    idx2ent, ent2idx = dataset.idx2ent, dataset.ent2idx
    e_num = len(idx2ent)
    idx2rel = dataset.rdict.idx2rel
    r2mat = construct_rmat(idx2rel, idx2ent, ent2idx, fact_rdf+train_rdf+valid_rdf)
    entity_predictor = EntityPredictor(opt.emb_size, dataset.entity_num, opt.device).to(opt.device)
    entity_predictor = torch.load(opt.exp_path + '/entity_model.pt')
    
    # 评估指标
    mrr, hits_1, hits_3, hits_10 = [], [], [], []
    
    # 批处理大小
    batch_size = 128
    
    # 直接处理测试集
    pbar = tqdm(test_rdf, desc="处理测试查询")
    for query_rdf in pbar:
        q_h, q_r, q_t = parse_rdf(query_rdf)
        if q_r not in rules:
            continue
            
        h_idx = ent2idx[q_h]
        t_idx = ent2idx[q_t]
        
        # 收集所有规则的预测分数
        final_scores = np.zeros(e_num)
        
        # 对每个规则进行处理
        for rule in rules[q_r]:
            head, tree_body, conf_1, conf_2 = rule
            
            # 收集路径实体
            path_entities = []
            for branch in tree_body:
                current = sparse.eye(e_num, format='csr')[h_idx]  # 只取查询实体行
                for b_rel in branch:
                    current = current * r2mat[b_rel]
                path_entities.append(current.toarray().flatten())  # 展平为1维数组
            
            # 如果任何路径没有实体，跳过这条规则
            if any(not path.any() for path in path_entities):
                continue
                
            # 转换为正确的维度 [batch, path_num, entity_num]
            path_tensor = torch.tensor(path_entities, dtype=torch.float32, device=opt.device)
            path_tensor = path_tensor.unsqueeze(0)  # 添加batch维度 [1, path_num, entity_num]
            
            with torch.no_grad():
                pred_scores = entity_predictor(path_tensor)  # 输出应该是 [1, entity_num]
                final_scores += pred_scores[0].cpu().numpy() * conf_1
        
        # 计算排名
        pred_ranks = np.argsort(final_scores)[::-1]
        truth = gt[(q_h, q_r)]
        truth = [t for t in truth if t != t_idx]
        
        truth_idx = []
        filtered_ranks = []
        for idx in pred_ranks:
            if idx not in truth and final_scores[idx] > final_scores[t_idx]:
                filtered_ranks.append(idx)
            if idx in truth:
                truth_idx.append(idx)
        
        rank = len(filtered_ranks) + 1
        
        # 更新指标
        mrr.append(1.0/rank)
        head2mrr[q_r].append(1.0/rank)
        hits_1.append(1 if rank <= 1 else 0)
        hits_3.append(1 if rank <= 3 else 0)
        hits_10.append(1 if rank <= 10 else 0)
        head2hit_1[q_r].append(1 if rank <= 1 else 0)
        head2hit_3[q_r].append(1 if rank <= 3 else 0)
        head2hit_10[q_r].append(1 if rank <= 10 else 0)
        
        pbar.set_postfix({
            'MRR': f'{np.mean(mrr):.4f}',
            'Hits@1': f'{np.mean(hits_1):.4f}',
            'Hits@3': f'{np.mean(hits_3):.4f}',
            'Hits@10': f'{np.mean(hits_10):.4f}',
            'Truth_rate': len(truth_idx) / len(pred_ranks)
        })

    msg = "MRR: {} Hits@1: {} Hits@3: {} Hits@10: {}".format(
        np.mean(mrr), np.mean(hits_1), np.mean(hits_3), np.mean(hits_10)
    )
    return msg

def sortSparseMatrix(m, r, rev=True, only_indices=False):
    """ Sort a row in matrix row and return column index
    """
    d = m.getrow(r)
    s = zip(d.indices, d.data)
    sorted_s = sorted(s, key=lambda v: v[1], reverse=rev)
    if only_indices:
        res = [element[0] for element in sorted_s]
    else:
        res = sorted_s
    return res


def remove_var(r):
    """R1(A, B), R2(B, C) --> R1, R2"""
    r = re.sub(r"\(\D?, \D?\)", "", r)
    return r


def parse_rule(r):
    """parse a rule into body and head"""
    r = remove_var(r)
    head, tree = r.split(" <-- ")
    body_list = tree.split(";")
    body = []
    for item in body_list:
        body.append(item.split(", "))
    return head, body


def load_rules(rule_path,all_rules,all_heads):
    with open(rule_path, 'r') as f:
        rules = f.readlines()
        for i_, rule in enumerate(rules):
            conf, r = rule.strip('\n').split('\t')
            conf_1, conf_2 = float(conf[0:5]), float(conf[-6:-1])
            head, body = parse_rule(r)
            # rule item: (head, body, conf_1, conf_2)
            if head not in all_rules:
                all_rules[head] = []
            all_rules[head].append((head, body, conf_1, conf_2))
            
            if head not in all_heads:
                all_heads.append(head)


def construct_rmat(idx2rel, idx2ent, ent2idx, fact_rdf):
    e_num = len(idx2ent)
    r2mat = {}
    for idx, rel in idx2rel.items():
        mat = sparse.dok_matrix((e_num, e_num))
        r2mat[rel] = mat
    # fill rmat
    for rdf in tqdm(fact_rdf, desc="constructing rmat"):
        fact = parse_rdf(rdf)
        h, r, t = fact
        h_idx, t_idx = ent2idx[h], ent2idx[t]
        r2mat[r][h_idx, t_idx] = 1
    for rel, mat in r2mat.items():
        r2mat[rel] = mat.tocsr()
    return r2mat      


def get_gt(dataset):
    # entity
    idx2ent, ent2idx = dataset.idx2ent, dataset.ent2idx
    fact_rdf, train_rdf, valid_rdf, test_rdf = dataset.fact_rdf, dataset.train_rdf, dataset.valid_rdf, dataset.test_rdf
    gt = defaultdict(list)
    all_rdf = fact_rdf + train_rdf + valid_rdf + test_rdf
    for rdf in all_rdf:
        h, r, t = parse_rdf(rdf)
        gt[(h, r)].append(ent2idx[t])
    return gt

def feq(relation, fact_rdf):
    count = 0
    for rdf in fact_rdf:
        h, r, t= parse_rdf(rdf)
        if r == relation:
            count += 1
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="family")
    parser.add_argument("--rule", default="family")
    parser.add_argument('--cpu_num', type=int, default=mp.cpu_count()//2)   
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--top", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0)
    args = parser.parse_args()
    dataset = Dataset(data_root='/{}/'.format(args.data), inv=True)
    all_rules = {}
    all_rule_heads = []
   
    for L in range(2,4):
        file = "/home/fxy/paper/NCRL/saves/rules500_4.txt"
        load_rules("{}".format(file), all_rules, all_rule_heads)
    
    for head in all_rules:
        all_rules[head] = all_rules[head][:args.top]
    
    fact_rdf, train_rdf, valid_rdf, test_rdf = dataset.fact_rdf, dataset.train_rdf, dataset.valid_rdf, dataset.test_rdf

    print_msg("distribution of test query")
    for head in all_rule_heads:
        count = feq(head, test_rdf)
        print("Head: {} Count: {}".format(head, count))
    
    print_msg("distribution of train query")
    for head in all_rule_heads:
        count = feq(head, fact_rdf+valid_rdf+train_rdf)
        print("Head: {} Count: {}".format(head, count))


    kg_completion(all_rules, dataset,args)
    
    print_msg("Stat on head and hit@1")
    for head, hits in head2hit_1.items():
        print(head, np.mean(hits))

    print_msg("Stat on head and hit@10")
    for head, hits in head2hit_10.items():
        print(head, np.mean(hits))
    print_msg("Stat on head and hit@3")
    for head, hits in head2hit_3.items():
        print(head, np.mean(hits))
    print_msg("Stat on head and mrr")
    for head, mrr in head2mrr.items():
        print(head, np.mean(mrr))
