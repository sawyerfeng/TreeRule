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

import sys
import io
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

head2mrr = defaultdict(list)
head2hit_10 = defaultdict(list)
head2hit_1 = defaultdict(list)
head2hit_3 = defaultdict(list)

def kg_completion(rules, dataset, args):
    """
    Input a set of rules
    Complete Queries from test_rdf based on rules and fact_rdf 
    """
    # 准备数据
    fact_rdf, train_rdf, valid_rdf, test_rdf = dataset.fact_rdf, dataset.train_rdf, dataset.valid_rdf, dataset.test_rdf
    gt = get_gt(dataset)
    idx2ent, ent2idx = dataset.idx2ent, dataset.ent2idx
    rdict = dataset.get_relation_dict()
    rel2idx, idx2rel = rdict.rel2idx, rdict.idx2rel
    e_num = len(idx2ent)
    r2mat = construct_rmat(idx2rel, idx2ent, ent2idx, fact_rdf+train_rdf+valid_rdf)
    
    # 存储每个头部关系的推理结果
    head2score = {}
    
    # 对每个头部关系进行规则推理
    for head_rel, rule_list in tqdm(rules.items()):
        score_matrix = sparse.dok_matrix((e_num, e_num))
        
        for rule in rule_list:
            head, tree_body, conf_1, conf_2 = rule
            
            # 计算所有分支路径的存在性矩阵
            branch_exists = []
            for branch in tree_body:
                # 创建单位矩阵作为起始状态
                current = sparse.eye(e_num, format='csr')
                
                # 沿着路径连续应用关系矩阵
                for b_rel in branch:
                    current = current * r2mat[b_rel]
                
                # 将结果转换为布尔矩阵,表示从每个起点出发是否存在该路径
                branch_exists.append((current > 0).astype(int))
            
            # 对于每个起点,检查是否所有分支路径都存在
            path_validity = branch_exists[0]
            for mat in branch_exists[1:]:
                # 对每个起点,只要有一个路径不存在,结果就为0
                path_validity = path_validity.multiply(mat > 0)
            
            # 应用规则和置信度
            # 只有当路径有效时(path_validity=1),才应用头部关系的预测
            score_matrix += path_validity.multiply(r2mat[head]) * conf_1
            
        head2score[head_rel] = score_matrix.tocsr()

    # 评估
    mrr, hits_1, hits_3, hits_10 = [], [], [], []
    
    # 添加进度条并实时显示指标
    pbar = tqdm(test_rdf, desc="评估中")
    for query_rdf in pbar:
        q_h, q_r, q_t = parse_rdf(query_rdf)
        if q_r not in head2score:
            continue
            
        pred = head2score[q_r][ent2idx[q_h]].toarray().flatten()
            
        pred_ranks = np.argsort(pred)[::-1]    

        truth = gt[(q_h, q_r)]
        truth = [t for t in truth if t!=ent2idx[q_t]]
        
        filtered_ranks = []
        for i in range(len(pred_ranks)):
            idx = pred_ranks[i]
            if idx not in truth and pred[idx] > pred[ent2idx[q_t]]:
                filtered_ranks.append(idx)
                
        rank = len(filtered_ranks)+1
        
        # 更新评估指标
        mrr.append(1.0/rank)
        head2mrr[q_r].append(1.0/rank)
        hits_1.append(1 if rank<=1 else 0)
        hits_3.append(1 if rank<=3 else 0)
        hits_10.append(1 if rank<=10 else 0)
        head2hit_1[q_r].append(1 if rank<=1 else 0)
        head2hit_3[q_r].append(1 if rank<=3 else 0)
        head2hit_10[q_r].append(1 if rank<=10 else 0)
        
        # 更新进度条显示当前指标
        pbar.set_postfix({
            'MRR': f'{np.mean(mrr):.4f}',
            'Hits@1': f'{np.mean(hits_1):.4f}',
            'Hits@3': f'{np.mean(hits_3):.4f}',
            'Hits@10': f'{np.mean(hits_10):.4f}'
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
