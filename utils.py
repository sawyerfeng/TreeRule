import torch
from tqdm import tqdm
from random import sample 
import math
import random
import numpy as np
import os
import concurrent.futures
import multiprocessing as mp
import time
import functools
def seed_everything(seed=3407):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
def print_msg(msg):
    msg = "## {} ##".format(msg)
    length = len(msg) 
    msg = "\n{}\n".format(msg)
    print(length*"#" + msg + length * "#")

def parse_rdf(rdf):
    """
    return: head, relation, tail
    """
    rdf_tail, rdf_rel, rdf_head= rdf
    return rdf_head, rdf_rel, rdf_tail
# 构建一个字典，key是head，value是对应的triple
def construct_descendant(rdf_data):
    """
    take entity as h, map it to its r, t
    input: rdf_data
    return:entity2desced: {h:[(r,t),...],...}
    """
    entity2desced = {}
    for rdf_ in rdf_data:
        h_, r_, t_ = parse_rdf(rdf_)
        if h_ not in entity2desced.keys():
            entity2desced[h_] = [(r_, t_)]
        else:
            entity2desced[h_].append((r_, t_))
    sorted_entity2desced = dict(sorted(entity2desced.items(), key=lambda x: len(x[1]), reverse=True))
    sorted_len = [len(x) for x in sorted_entity2desced.values()]
    top_n_keys = list(sorted_entity2desced.keys())[:1000]
    return entity2desced

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
    

def connected(entity2desced, head, tail):
    if head in entity2desced:
        decedents = entity2desced[head]
        for d in decedents:
            d_relation_, d_tail_ = d
            if d_tail_ == tail:
                return d_relation_
        return False
    else:
        return False
    
def rdf2idx(rdf,head_rdict):
    rdf2id = []
    for triple in rdf:
        # triple[1] = str(head_rdict.rel2idx[triple[1]])
        h, r, t = parse_rdf(triple)
        r = str(head_rdict.rel2idx[r])
        rdf2id.append((h, r, t))
    return rdf2id   

def idrule2idx(rule, head_rdict):
    """
    Input a rule (string) and idx it
    """
    body, head = rule.split('-')
    body_path = body.split('|')
    # indexs include body idx seq + notation + head idx
    indexs = []
    for rel in body_path+[-1, head]:
        indexs.append(int(rel) if rel != -1 else -1)
    return indexs

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

def get_topk_rule_seq(rdf_data, anchor_rdf, entity2desced,head_rdict, max_path_len=2, PRINT=False): 
    len2seq = {}
    anchor_h, anchor_r, anchor_t = parse_rdf(anchor_rdf)
    anchor_r = head_rdict.rel2idx[anchor_r]
    # Search
    stack = [(anchor_h, anchor_r, anchor_t)]
    stack_print = ['{}-{}-{}'.format(anchor_h, anchor_r, anchor_t)]
    pre_path = anchor_h
    rule_seq, expended_node = [], []
    record = []
    while len(stack) > 0 :
        cur_h, cur_r, cur_t = stack.pop(-1)
        cur_print = stack_print.pop(-1)
        deced_list = []
        if cur_t in entity2desced:
            deced_list = entity2desced[cur_t]  

        if len(cur_r.split('|')) < max_path_len and len(deced_list) > 0 and cur_t not in expended_node:
            for r_, t_ in deced_list:
                if t_ != cur_h and t_ != anchor_h:
                    stack.append((cur_t, cur_r+'|'+r_, t_))
                    stack_print.append(cur_print+'-{}-{}'.format(r_, t_))
        expended_node.append(cur_t)
        
        rule_head_rel = connected(entity2desced, anchor_h, cur_t)
        if rule_head_rel and cur_t != anchor_t:
            rule = cur_r + '-' + rule_head_rel  
            rule_seq.append(rule)
            if (cur_h, r_, t_) not in record:
                record.append((cur_h, r_, t_))
            if PRINT:
                print('rule body:\n{}'.format(cur_print))
                print('rule head:\n{}-{}-{}'.format(anchor_h, rule_head_rel, cur_t))
                print('rule:\n{}\n'.format(rule))
        elif rule_head_rel == False and random.random() > 0.6:
            rule = cur_r + '-' + str(head_rdict.rel2idx['None'])
            rule_seq.append(rule)
            if (cur_h, r_, t_) not in record:
                record.append((cur_h, r_, t_))
            if PRINT:
                print('rule body:\n{}'.format(cur_print))
                print('rule head:\n{}-{}-{}'.format(anchor_h, rule_head_rel, cur_t))
                print('rule:\n{}\n'.format(rule))
    return rule_seq, record

def sample_training_data_ratio(opt,max_path_len, ratio, fact_rdf, entity2desced, head_rdict):
    print("Sampling training data...")
    anchors_rdf = []
    per_anchor_num ={}
    fact_dict = construct_fact_dict(fact_rdf)#fact_dict每个key是一个head，list是所有的head对应的triple
    
    for key,value in fact_dict.items():
        per_anchor_num[key] = math.ceil(len(value)*ratio)
    for head in head_rdict.rel2idx:
        if head != "None" and "inv_" not in head:
            sampled_rdf = sample_anchor_rdf(fact_dict[head], num=per_anchor_num[head])
            anchors_rdf.extend(sampled_rdf)#找到anchor三元组
    print ("Total_anchor_num",len(anchors_rdf))#anchor三元组的数量
    train_rule, train_rule_dict = [],{}
    len2train_rule_idx = {}
    sample_number = 0
    pbar = tqdm(range(len(anchors_rdf)))
    # for anchor_rdf in anchors_rdf:
    for i in pbar:
        # rule_seq, record = construct_rule_seq(fact_rdf, anchors_rdf[i], entity2desced, max_path_len, PRINT=False)
        rule_seq, record = construct_rule_seq(opt,fact_rdf, anchors_rdf[i], entity2desced, max_path_len, PRINT=False)
        sample_number += len(record)
        if len(rule_seq) > 0:
            train_rule += rule_seq
            for rule in rule_seq:
                idx = torch.LongTensor(rule2idx(rule, head_rdict))
                h = head_rdict.idx2rel[idx[-1].item()]
                if h not in train_rule_dict:
                    train_rule_dict[h] = []
                train_rule_dict[h].append(idx)
                # cluster rules according to its length
                body_len = len(idx) - 2
                assert body_len == max_path_len
                # body_len = opt.max_path_len
                if body_len in len2train_rule_idx.keys():
                    len2train_rule_idx[body_len] += [idx]
                else:
                    len2train_rule_idx[body_len] = [idx]
        pbar.update()
        pbar.set_description(f'anchor: {i}, rule_num: {len(train_rule)}')
                    
    print ("# train_rule:{}".format(sample_number))
    print ("# head:{}".format(len(train_rule_dict)))
    sample_rel_num =0
    sample_none_num =0
    sample_info = {}
    for h in train_rule_dict:
        print ("head {}:{}".format(h,len(train_rule_dict[h])))
        # sample_info ["head {}".format(h)] = len(train_rule_dict[h])
        if h != "None":
            sample_rel_num += len(train_rule_dict[h])
        else:
            sample_none_num += len(train_rule_dict[h])
    ## Writer sample info
    sample_info["ratio"] = ratio
    sample_info["relation_num"] = sample_rel_num
    sample_info["none_num"] = sample_none_num
    sample_info["sample_number"] = sample_number
    sample_info["head_num"] = len(train_rule_dict)
    # sample_info = '<br>'.join(['{}: {}'.format(k, v) for k, v in sample_info.items()])
    # opt.writer(file_name="sample_info",info=sample_info)
    len2train_rule_idx['sample_info'] = sample_info

    return len2train_rule_idx
def sample_anchor(fact_dict,per_anchor_num,head):
    return sample_anchor_rdf(fact_dict[head], num=per_anchor_num[head])


    
def sample_training_data_ratio_parral(opt,max_path_len, ratio, fact_rdf, entity2desced, head_rdict):
    def process_anchor(anchor):
        rule_seq, record = construct_rule_seq(opt,fact_rdf, anchor, entity2desced, max_path_len, PRINT=False)
        return rule_seq, record
    print("Sampling training data...")
    anchors_rdf = []
    # per_anchor_num = anchor_num//((head_rdict.__len__() -1) //2)# 对每个anchor进行采样，数量为anchor_num/rel_num
    per_anchor_num ={}
    # print("Number of head relation:{}".format((head_rdict.__len__() -1) // 2))
    # print ("Number of per_anchor_num: {}".format(per_anchor_num))
    fact_dict = construct_fact_dict(fact_rdf)#fact_dict每个key是一个head，list是所有的head对应的triple

    for key,value in fact_dict.items():
        per_anchor_num[key] = math.ceil(len(value)*ratio)
        # print("head:{} per_anchor_num:{}".format(key,per_anchor_num[key]))
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        partial_sample_anchor = functools.partial(sample_anchor, fact_dict, per_anchor_num)
        sampled_results = list(executor.map(partial_sample_anchor,(head for head in head_rdict.rel2idx if head != "None" and "inv_" not in head)))
    anchors_rdf.extend([result for result_list in sampled_results for result in result_list])
    print ("Total_anchor_num",len(anchors_rdf))#anchor三元组的数量
    train_rule, train_rule_dict = [],{}
    len2train_rule_idx = {}
    sample_number = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(process_anchor,(anchor_rdf for anchor_rdf in anchors_rdf)))
    for rule_seq, record in tqdm(results):
        sample_number += len(record)
        if len(rule_seq) > 0:
            train_rule += rule_seq
            for rule in rule_seq:
                idx = torch.LongTensor(rule2idx(rule, head_rdict))
                h = head_rdict.idx2rel[idx[-1].item()]
                if h not in train_rule_dict:
                    train_rule_dict[h] = []
                train_rule_dict[h].append(idx)
                # cluster rules according to its length
                body_len = len(idx) - 2
                assert body_len == max_path_len
                # body_len = opt.max_path_len
                if body_len in len2train_rule_idx.keys():
                    len2train_rule_idx[body_len] += [idx]
                else:
                    len2train_rule_idx[body_len] = [idx]

                    
    print ("# train_rule:{}".format(sample_number))
    print ("# head:{}".format(len(train_rule_dict)))
    sample_rel_num =0
    sample_none_num =0
    sample_info = {}
    for h in train_rule_dict:
        print ("head {}:{}".format(h,len(train_rule_dict[h])))
        # sample_info ["head {}".format(h)] = len(train_rule_dict[h])
        if h != "None":
            sample_rel_num += len(train_rule_dict[h])
        else:
            sample_none_num += len(train_rule_dict[h])
    ## Writer sample info
    sample_info["ratio"] = ratio
    sample_info["relation_num"] = sample_rel_num
    sample_info["none_num"] = sample_none_num
    sample_info["sample_number"] = sample_number
    sample_info["head_num"] = len(train_rule_dict)
    # sample_info = '<br>'.join(['{}: {}'.format(k, v) for k, v in sample_info.items()])
    # opt.writer(file_name="sample_info",info=sample_info)
    len2train_rule_idx['sample_info'] = sample_info
    # if opt.contrast_ratio == ratio:
    #     opt.writer.add_text("contrast_sample_info",sample_info)
    # else:
    #     opt.writer.add_text("predict_sample_info",sample_info)

    return len2train_rule_idx

def sample_training_data(opt,max_path_len, anchor_num, fact_rdf, entity2desced, head_rdict):
    print("Sampling training data...")
    anchors_rdf = []
    per_anchor_num = anchor_num//((head_rdict.__len__() -1) //2)# 对每个anchor进行采样，数量为anchor_num/rel_num
    print("Number of head relation:{}".format((head_rdict.__len__() -1) // 2))
    print ("Number of per_anchor_num: {}".format(per_anchor_num))
    fact_dict = construct_fact_dict(fact_rdf)#fact_dict每个key是一个head，list是所有的head对应的triple
    for head in head_rdict.rel2idx:
        if head != "None" and "inv_" not in head:
            sampled_rdf = sample_anchor_rdf(fact_dict[head], num=per_anchor_num)
            anchors_rdf.extend(sampled_rdf)#找到anchor三元组
    print ("Total_anchor_num",len(anchors_rdf))#anchor三元组的数量
    train_rule, train_rule_dict = [],{}
    len2train_rule_idx = {}
    sample_number = 0
    pbar = tqdm(range(len(anchors_rdf)))

    # for anchor_rdf in anchors_rdf:
    for i in pbar:
        # rule_seq, record = construct_rule_seq(fact_rdf, anchors_rdf[i], entity2desced, max_path_len, PRINT=False)
        rule_seq, record = construct_rule_seq(opt,fact_rdf, anchors_rdf[i], entity2desced, max_path_len, PRINT=False)
        sample_number += len(record)
        if len(rule_seq) > 0:
            train_rule += rule_seq
            for rule in rule_seq:
                idx = torch.LongTensor(rule2idx(rule, head_rdict))
                h = head_rdict.idx2rel[idx[-1].item()]
                if h not in train_rule_dict:
                    train_rule_dict[h] = []
                train_rule_dict[h].append(idx)
                # cluster rules according to its length
                body_len = len(idx) - 2
                if body_len in len2train_rule_idx.keys():
                    len2train_rule_idx[body_len] += [idx]
                else:
                    len2train_rule_idx[body_len] = [idx]
        pbar.update()
        pbar.set_description(f'anchor: {i}, rule_num: {len(train_rule)}')
                    
    print ("# train_rule:{}".format(sample_number))
    print ("# head:{}".format(len(train_rule_dict)))
    sample_rel_num =0
    sample_none_num =0
    sample_info = {}
    for h in train_rule_dict:
        print ("head {}:{}".format(h,len(train_rule_dict[h])))
        if h != "None":
            sample_rel_num += len(train_rule_dict[h])
        else:
            sample_none_num += len(train_rule_dict[h])
    print("sampled relation number:{}".format(sample_rel_num))
    print("sampled None number:{}".format(sample_none_num))
    sample_info["relation_num"] = sample_rel_num
    sample_info["none_num"] = sample_none_num
    sample_info["sample_number"] = sample_number
    sample_info["head_num"] = len(train_rule_dict)
    len2train_rule_idx['sample_info'] = sample_info
    # path_info = "sampled relation number:{}\n".format(sample_rel_num)+"sampled None number:{}\n".format(sample_none_num)
    # opt.writer.add_text("sampled path info",path_info)
    # with open("/home/fxy/paper/ContrastRule/datasets/sample_path/sampled_path_info.txt",'a') as f:
        # f.write(path_info)
    rule_len_range = list(len2train_rule_idx.keys())
    print("Fact set number:{} Sample number:{}".format(len(fact_rdf), sample_number))
    for rule_len in rule_len_range:
        print("sampled examples for rule of length {}: {}".format(rule_len, len(len2train_rule_idx[rule_len])))
    return len2train_rule_idx

def enumerate_body(relation_num, rdict, body_len,pos_only=True):
    import itertools
    max_rel_num = relation_num
    all_body_idx = list(list(x) for x in itertools.product(range(max_rel_num), repeat=body_len))
    # transfer index to relation name
    idx2rel = rdict.idx2rel
    all_body = []
    for b_idx_ in all_body_idx:
        b_ = [idx2rel[x] for x in b_idx_]
        all_body.append(b_)
    return all_body_idx, all_body

def enumerate_tree(relation_num, rdict, body_len, tree_num=2):
    import itertools
    max_rel_num = relation_num
    all_body_idx = list(list(x) for x in itertools.product(range(max_rel_num), repeat=body_len))
    all_body = []
    idx2rel = rdict.idx2rel
    for b_idx_ in all_body_idx:
        b_ = [idx2rel[x] for x in b_idx_]
        all_body.append(b_)
    tree_paths = list(itertools.combinations(all_body, tree_num))
    return tree_paths

def body2idx(body_list, head_rdict):
    """
    Input a rule (string) and idx it
    """
    res = []
    for body in body_list:
        body_path = body.split('|')
        # indexs include body idx seq + notation + head idx
        indexs = []
        for rel in body_path:
            indexs.append(head_rdict.rel2idx[rel])
        res.append(indexs)
    return res

def tree2idx(tree_list, head_rdict):
    res = []
    for trees in tree_list:
        tree = trees.split(',')
        idx_pair = [list(head_rdict.rel2idx[rel] for rel in pair.split('|')) for pair in tree]
        res.append(idx_pair)
    return res

# 从anchor 开始拓展
def construct_rule_seq(opt,rdf_data, anchor_rdf, entity2desced, max_path_len=2, PRINT=False):    
    len2seq = {}
    anchor_h, anchor_r, anchor_t = parse_rdf(anchor_rdf)
    # Search
    stack = [(anchor_h, anchor_r, anchor_t)]
    stack_print = ['{}-{}-{}'.format(anchor_h, anchor_r, anchor_t)]
    pre_path = anchor_h
    rule_seq, expended_node = [], []
    record = []
    none_rule_number = 0
    close_rule_number = 0
    while len(stack) > 0:
        cur_h, cur_r, cur_t = stack.pop(-1)
        cur_print = stack_print.pop(-1)
        deced_list = []
        
        if cur_t in entity2desced:
            deced_list = entity2desced[cur_t]  

        if len(cur_r.split('|')) < max_path_len and len(deced_list) > 0 and cur_t not in expended_node:
            for r_, t_ in deced_list:
                if t_ != cur_h and t_ != anchor_h:
                    stack.append((cur_t, cur_r+'|'+r_, t_))
                    stack_print.append(cur_print+'-{}-{}'.format(r_, t_))
        expended_node.append(cur_t)
        
        rule_head_rel = connected(entity2desced, anchor_h, cur_t)
        if rule_head_rel  and len(cur_r.split('|')) == max_path_len and cur_t != anchor_t:
            rule = cur_r + '-' + rule_head_rel  
            rule_seq.append(rule)
            close_rule_number += 1
            if (cur_h, r_, t_) not in record:
                record.append((cur_h, r_, t_))
            if PRINT:
                print('rule body:\n{}'.format(cur_print))
                print('rule head:\n{}-{}-{}'.format(anchor_h, rule_head_rel, cur_t))
                print('rule:\n{}\n'.format(rule))
        elif rule_head_rel == False and len(cur_r.split('|')) == max_path_len and random.random() > opt.none_drop and none_rule_number < opt.none_ratio*close_rule_number:
            rule = cur_r + '-' + "None"
            rule_seq.append(rule)
            none_rule_number += 1
            if (cur_h, r_, t_) not in record:
                record.append((cur_h, r_, t_))
            if PRINT:
                print('rule body:\n{}'.format(cur_print))
                print('rule head:\n{}-{}-{}'.format(anchor_h, rule_head_rel, cur_t))
                print('rule:\n{}\n'.format(rule))
    # print("len:",len(rule_seq))
    return rule_seq, record

def construct_rule_seq_balance(opt,rdf_data, anchor_rdf, entity2desced, max_path_len=2, PRINT=False):    
    len2seq = {}
    anchor_h, anchor_r, anchor_t = parse_rdf(anchor_rdf)
    # Search
    stack = [(anchor_h, anchor_r, anchor_t)]
    stack_print = ['{}-{}-{}'.format(anchor_h, anchor_r, anchor_t)]
    pre_path = anchor_h
    rule_seq, expended_node = [], []
    record = []
    none_rule_number = 0
    close_rule_number = 0
    while len(stack) > 0:
        cur_h, cur_r, cur_t = stack.pop(-1)
        cur_print = stack_print.pop(-1)
        deced_list = []
        
        if cur_t in entity2desced:
            deced_list = entity2desced[cur_t]  

        if len(cur_r.split('|')) < max_path_len and len(deced_list) > 0 and cur_t not in expended_node:
            for r_, t_ in deced_list:
                if t_ != cur_h and t_ != anchor_h:
                    stack.append((cur_t, cur_r+'|'+r_, t_))
                    stack_print.append(cur_print+'-{}-{}'.format(r_, t_))
        expended_node.append(cur_t)
        
        rule_head_rel = connected(entity2desced, anchor_h, cur_t)
        if rule_head_rel and cur_t != anchor_t:
            rule = cur_r + '-' + rule_head_rel  
            rule_seq.append(rule)
            if (cur_h, r_, t_) not in record:
                record.append((cur_h, r_, t_))
                close_rule_number += 1
            if PRINT:
                print('rule body:\n{}'.format(cur_print))
                print('rule head:\n{}-{}-{}'.format(anchor_h, rule_head_rel, cur_t))
                print('rule:\n{}\n'.format(rule))
        elif rule_head_rel == False and random.random() > opt.none_drop and none_rule_number < 2*close_rule_number:
            rule = cur_r + '-' + "None"
            rule_seq.append(rule)
            none_rule_number += 1
            if (cur_h, r_, t_) not in record:
                record.append((cur_h, r_, t_))
            if PRINT:
                print('rule body:\n{}'.format(cur_print))
                print('rule head:\n{}-{}-{}'.format(anchor_h, rule_head_rel, cur_t))
                print('rule:\n{}\n'.format(rule))
    # print("len:",len(rule_seq))
    return rule_seq, record

#  将rdf数据转换为idx数据
def rdf2idx(rdf,head_rdict):
    rdf2id = []
    for triple in rdf:
        h, r, t = parse_rdf(triple)
        r = str(head_rdict.rel2idx[r])
        rdf2id.append((h, r, t))
    return rdf2id