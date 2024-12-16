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

import json
import os
from datetime import datetime

def save_prediction_details(save_path, head_rel, rule, source_entity, target_entity, confidence):
    """记录单个预测的详细信息"""
    details = {
        "head_relation": head_rel,
        "rule_body": rule[1],  # tree_body
        "confidence": float(confidence),
        "source_entity": source_entity,
        "target_entity": target_entity,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return details

def save_rule_application_stats(save_path, rule_stats):
    """保存规则应用统计信息"""
    with open(os.path.join(save_path, "rule_stats.json"), "w") as f:
        json.dump(rule_stats, f, indent=2)

def save_evaluation_results(save_path, results):
    """保存评估结果"""
    with open(os.path.join(save_path, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

def kg_completion(args,rules, dataset):
    """
    Input a set of rules
    Complete Queries from test_rdf based on rules and fact_rdf 
    Return:
        msg: 评估结果信息
        inferred_paths: 推理出的路径
        inferred_triples: 推理出的三元组（在测试集中的）
        metrics: 包含所有评估指标的字典
    """
    # 创建保存结果的目录
    save_path = os.path.join(args.exp_path, "inference_results")
    os.makedirs(save_path, exist_ok=True)
    
    # 初始化统计信息和返回结果
    rule_stats = {}
    prediction_details = []
    inferred_paths = []  # 存储推理路径
    inferred_triples = []  # 存储推理出的三元组
    evaluation_results = {
        "per_relation": {},
        "overall": {}
    }
    
    # 初始化评估指标字典
    metrics = {
        'head2mrr': defaultdict(list),
        'head2hit_1': defaultdict(list),
        'head2hit_3': defaultdict(list),
        'head2hit_10': defaultdict(list)
    }
    
    # 准备数据
    fact_rdf, train_rdf, valid_rdf, test_rdf = dataset.fact_rdf, dataset.train_rdf, dataset.valid_rdf, dataset.test_rdf
    
    # 将测试集转换为set，方便查找
    test_triples = set()
    for test_sample in test_rdf:
        h, r, t = test_sample
        test_triples.add((h, r, t))
    
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
        # 初始化该关系的规则统计
        rule_stats[head_rel] = {
            "total_rules": len(rule_list),
            "rules_applied": 0,
            "new_predictions": 0
        }
        
        # 创建新的分数矩阵（不依赖已知边）
        score_matrix = sparse.dok_matrix((e_num, e_num))
        
        for rule in rule_list:
            head, tree_body, conf_1, conf_2 = rule
            
            # 计算所有分支路径的可达性矩阵
            branch_reachable = []
            for branch in tree_body:
                # 创建单位矩阵作为起始状态
                current = sparse.eye(e_num, format='csr')
                
                # 沿着路径连续应用关系矩阵来找到可达实体
                for b_rel in branch:
                    current = current * r2mat[b_rel]
                
                # 将结果转换为布尔矩阵,表示从每个起点出发可以到达哪些实体
                branch_reachable.append((current > 0).astype(int))
            
            # 对于每个起点，找到通过所有分支都可达的实体
            path_validity = branch_reachable[0]
            for mat in branch_reachable[1:]:
                path_validity = path_validity.multiply(mat > 0)
            
            # 获取有效的推理路径
            rows, cols = path_validity.nonzero()
            
            # 记录规则应用情况
            if len(rows) > 0:
                rule_stats[head_rel]["rules_applied"] += 1
            
            for i, j in zip(rows, cols):
                score_matrix[i, j] += conf_1
                
                # 如果是新预测的边
                if r2mat[head][i, j] == 0:
                    rule_stats[head_rel]["new_predictions"] += 1
                    
                    # 记录推理路径
                    path_info = {
                        "head_relation": head_rel,
                        "rule_body": [",".join(branch) for branch in tree_body],
                        "source_entity": idx2ent[i],
                        "target_entity": idx2ent[j],
                        "confidence": float(conf_1)
                    }
                    inferred_paths.append(path_info)
                    
                    # 检查是否在测试集中
                    if (idx2ent[i], head_rel, idx2ent[j]) in test_triples:
                        triple = {
                            "head": idx2ent[i],
                            "relation": head_rel,
                            "tail": idx2ent[j],
                            "confidence": float(conf_1),
                            "rule_body": [",".join(branch) for branch in tree_body]
                        }
                        inferred_triples.append(triple)
            
        head2score[head_rel] = score_matrix.tocsr()
    
    print(f"\n总共找到 {len(inferred_triples)} 个在测试集中的推理三元组")
    
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
        metrics['head2mrr'][q_r].append(1.0/rank)
        hits_1.append(1 if rank<=1 else 0)
        hits_3.append(1 if rank<=3 else 0)
        hits_10.append(1 if rank<=10 else 0)
        metrics['head2hit_1'][q_r].append(1 if rank<=1 else 0)
        metrics['head2hit_3'][q_r].append(1 if rank<=3 else 0)
        metrics['head2hit_10'][q_r].append(1 if rank<=10 else 0)
        
        # 更新进度条显示当前指标
        pbar.set_postfix({
            'MRR': f'{np.mean(mrr):.4f}',
            'Hits@1': f'{np.mean(hits_1):.4f}',
            'Hits@3': f'{np.mean(hits_3):.4f}',
            'Hits@10': f'{np.mean(hits_10):.4f}'
        })
    
    # 记录每个关系的评估结果
    for rel in metrics['head2mrr']:
        evaluation_results["per_relation"][rel] = {
            "MRR": float(np.mean(metrics['head2mrr'][rel])),
            "Hits@1": float(np.mean(metrics['head2hit_1'][rel])),
            "Hits@3": float(np.mean(metrics['head2hit_3'][rel])),
            "Hits@10": float(np.mean(metrics['head2hit_10'][rel]))
        }
    
    # 记录总体评估结果
    evaluation_results["overall"] = {
        "MRR": float(np.mean(mrr)),
        "Hits@1": float(np.mean(hits_1)),
        "Hits@3": float(np.mean(hits_3)),
        "Hits@10": float(np.mean(hits_10))
    }
    
    # 保存统计信息和评估结果
    save_rule_application_stats(save_path, rule_stats)
    save_evaluation_results(save_path, evaluation_results)

    msg = "MRR: {} Hits@1: {} Hits@3: {} Hits@10: {}".format(
        np.mean(mrr), np.mean(hits_1), np.mean(hits_3), np.mean(hits_10)
    )
    
    return msg, inferred_paths, inferred_triples, metrics

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


    msg, inferred_paths, inferred_triples, metrics = kg_completion(all_rules, dataset,args)
    
    print_msg("Stat on head and hit@1")
    for head, hits in metrics['head2hit_1'].items():
        print(head, np.mean(hits))

    print_msg("Stat on head and hit@10")
    for head, hits in metrics['head2hit_10'].items():
        print(head, np.mean(hits))
    print_msg("Stat on head and hit@3")
    for head, hits in metrics['head2hit_3'].items():
        print(head, np.mean(hits))
    print_msg("Stat on head and mrr")
    for head, mrr in metrics['head2mrr'].items():
        print(head, np.mean(mrr))
