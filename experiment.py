from collections import namedtuple
from utils import *
import time
import os
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from kg_completion import load_rules,kg_completion
import torch.multiprocessing as mp
import pickle
from sample_path import *
import sys
sys.path.append('.')  # 将当前目录添加到Python路径
from models.entity_predictor import EntityPredictor
from models.model_agg import TreeEncoder, RulePredictor
import torch.nn as nn
from scipy import sparse
# 定义命名元组，用于存储超参数
training_option = namedtuple(
    'training_option',
    [
        'exp_name',
        'batch_size',
        'emb_size',
        'load_contrast_model',
        'load_predict_model',
        'joint_train',
        'contrast_n_epoch',
        'predict_n_epoch',
        'joint_n_epoch',
        'contrast_lr',
        'predict_lr',
        'joint_contrast_lr',
        'joint_predict_lr',
        'drop_prob',
        'gamma',
        'body_len_range',
        'device',
        'writer',
        'inverse_negative',
        'model',
        'topk',
        'cpu_num',
        'learned_path_len',
        'get_hits',
        'max_path_len',
        'anchor',
        'exp_path',
        'none_drop',
        'tao',
        'contrast_ratio',
        'predict_ratio',
        'none_ratio',
        'load_path',
        'predict_mode',
        'augment',
        'bn_hidden_size',
    ]
)

# 创建一个函数用于返回指定超参数的实例
def create_training_option(opts):
    return training_option(**opts)


def train_three_stage(args, opt, dataset):
    # 获取实验路径
    exp_path = opt.exp_path
    
    # 获取头实体-关系字典和所有三元组数据
    head_rdict = dataset.get_head_relation_dict()
    all_rdf = dataset.fact_rdf + dataset.train_rdf + dataset.valid_rdf
    all_id_rdf = rdf2idx(all_rdf, head_rdict)
    
    print_msg("Sampling Tree")
    # 构建实体的后代字典
    entity2desced = construct_descendant(all_rdf)
    max_path_len = args.max_path_len
    
    # 采样树形数据
    train_data = sample_tree_data(opt, 2, opt.anchor, all_rdf, entity2desced, head_rdict, dataset.ent2idx, args.tree_num, sample_ratio=0.5)
    
    # 获取关系字典和关系数量
    rdict = dataset.get_relation_dict()
    relation_num = rdict.__len__()
    
    # 初始化模型
    tree_model = TreeEncoder(opt.emb_size, relation_num, dataset.entity_num, opt.device).to(opt.device)
    predict_model = RulePredictor(opt.emb_size, relation_num, dataset.entity_num, opt.device).to(opt.device)
    entity_predictor = EntityPredictor(opt.emb_size, dataset.entity_num, opt.device).to(opt.device)
   
    # 初始化优化器
    tree_optimizer = torch.optim.AdamW(tree_model.parameters(), lr=opt.contrast_lr)
    predict_optimizer = torch.optim.AdamW(predict_model.parameters(), lr=opt.predict_lr)
    entity_optimizer = torch.optim.AdamW(entity_predictor.parameters(), lr=0.001)
    # 定义损失函数
    criterion = nn.BCEWithLogitsLoss()  # 对于实体预测使用BCE损失
    
    # 准备数据加载器
    path_data = torch.stack(train_data, 0)
    rule_data = TensorDataset(path_data)
    path_loader = DataLoader(rule_data, batch_size=opt.batch_size, shuffle=True)
    
    # 第一阶段：Tree Encoder预训练（包含了PathEncoder的预训练）
    print_msg("Stage 1: Pre-training Tree Encoder")
    # pbar = tqdm(range(opt.contrast_n_epoch))
    pbar = tqdm(range(0))
    if os.path.exists(exp_path+'/tree_model.pt'):
        print("加载已有的Tree Encoder模型...")
        tree_model = torch.load(exp_path+'/tree_model.pt')
    else:
        for epoch in pbar:
            loss = train_contrast(tree_model, tree_optimizer, path_loader, opt)
            pbar.set_description(f'Tree Encoder Loss: {loss:.4f}')
    # # 冻结 PathEncoder 的参数
    # for param in tree_model.path_encoder.parameters():
    #     param.requires_grad = False



    # 第二阶段：联合训练Predictor
    print_msg("Stage 2: Joint Training with Predictor")
    pbar = tqdm(range(opt.joint_n_epoch))
    for epoch in pbar:
        loss, acc = train_joint(tree_model, predict_model, entity_predictor, tree_optimizer, 
                              predict_optimizer, entity_optimizer, criterion, path_loader, dataset, opt)
        pbar.set_description(f'Joint Loss: {loss:.4f}, Acc: {acc:.4f}')
    
    # 保存模型
    print("Saving models...")
    torch.save(tree_model, exp_path+'/tree_model.pt')
    torch.save(predict_model, exp_path+'/predict_model.pt')
    torch.save(entity_predictor, exp_path + '/entity_model.pt')

def train_contrast(model, optimizer, dataloader, opt):
    """
    通过dropout进行对比学习，每个样本复制一份构建正样本对
    """
    model.train()
    total_loss = 0.0
    batch_num = 0
    
    pbar = tqdm(dataloader, desc='Training Contrast')
    for batch in pbar:
        optimizer.zero_grad()
        inputs = batch[0][:,:,0:-3].to(opt.device)
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]
        path_len = inputs.shape[2]
        # 构建输入矩阵：每个样本复��一份
        input_ids = torch.zeros(2*batch_size, seq_len, path_len, dtype=torch.int64).to(opt.device)
        for i, x in enumerate(inputs):
            input_ids[2*i, :seq_len, :path_len] = x
            input_ids[2*i+1, :seq_len, :path_len] = x  # 直接复制，依赖dropout产生不同表示
        
        # 前向传播计算损失
        loss, embeddings = model(input_ids, contrast=True)  # 返回loss和投影后的表示
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batch_num += 1
        pbar.set_description(f'Training Contrast Loss: {loss.item():.4f}')
    
    return total_loss / batch_num

def train_joint(tree_model, predict_model, entity_predictor, tree_optimizer, predict_optimizer, entity_optimizer, 
                criterion, dataloader, dataset, opt):
    tree_model.train()
    predict_model.train()
    total_loss = 0.0
    batch_num = 0
    total_acc = 0.0
    
    pbar = tqdm(dataloader, desc='Training Joint')
    # 实例化损失函数
    relation_criterion = nn.CrossEntropyLoss()
    entity_criterion = nn.BCEWithLogitsLoss()  # 使用BCEWithLogitsLoss
    
    for batch in pbar:
        tree_optimizer.zero_grad()
        predict_optimizer.zero_grad()
        
        inputs = batch[0][:,:,0:-3].to(opt.device)
        # rel在倒数第二列
        target_rel = batch[0][:,:,-2][:, 0].to(opt.device)
        input_h = batch[0][:,:,-1][:, 0].to(opt.device)

        # 获取batch中每个样本的关系矩阵
        batch_size = input_h.shape[0]
        entity_num = dataset.entity_num # 使用任意关系矩阵获取实体数量
        target_ent = torch.zeros((batch_size, entity_num), device=opt.device)

        for i in range(batch_size):
            rel_idx = target_rel[i].item()
            h_idx = input_h[i].item()
            # 获取关系矩阵
            rel_matrix = dataset.r2mat[dataset.rdict.idx2rel[rel_idx]]
            # 获取从h_idx出发经过rel到达的实体
            reachable = rel_matrix[h_idx].toarray().flatten()
            target_ent[i] = torch.tensor(reachable, device=opt.device)
        # 获取每个样本经过tree路径到达的实体集
        path_num = inputs.shape[2]  # 路径数量
        inputs_entity = torch.zeros((batch_size, path_num, entity_num), device=opt.device)
        
        for i in range(batch_size):
            h_idx = input_h[i].item()
            for j in range(path_num):
                # 初始化结果矩阵为单位矩阵
                result_matrix = sparse.eye(entity_num, format='csr')        
                # 获取当前路径上的所有关系并连乘其矩阵
                for k in range(inputs.shape[1]):  # 遍历路径中的每个关系
                    rel_idx = inputs[i,k,j].item()
                    if rel_idx == 0:  # 遇到padding则停止
                        break
                    rel = dataset.rdict.idx2rel[rel_idx]
                    rel_matrix = dataset.r2mat[rel]
                    result_matrix = result_matrix.dot(rel_matrix)
                
                # 获取从h_idx出发经过整条路径到达的实体
                reachable = result_matrix[h_idx].toarray().flatten()
                inputs_entity[i,j] = torch.tensor(reachable, device=opt.device)

        # 检查每个样本的每条路径是否全0
        path_zero_mask = (inputs_entity == 0).all(dim=2)  # [batch_size, path_num]
        # 如果任何一条路径全0，则将该样本的target设为0
        sample_has_zero_path = path_zero_mask.any(dim=1)  # [batch_size]
        target_ent[sample_has_zero_path] = 0

        # 继续原来的训练流程
        input_emb = tree_model(inputs, inputs_entity, contrast = False)
        pred_relation, pred_entity = predict_model(input_emb, mode = "transformer")
        # pred_entity = entity_predictor(inputs_entity)

        # 计算损失
        pos_mask = (target_ent == 1)
        loss_entity = entity_criterion(
            pred_entity[pos_mask], 
            target_ent[pos_mask]
        )
        loss_relation = relation_criterion(pred_relation, target_rel.reshape(-1))
        # 合并实体预测损失和关系预测损失
        loss = 0.7 * loss_relation + 0.3 * loss_entity

        # 计算实体预测的准确率（可选）
        with torch.no_grad():
            pred_entity_binary = (torch.sigmoid(pred_entity) > 0.9).float()  # 转换为0/1预测
            # 只计算target为1的位置的预测准确率
            target_mask = (target_ent == 1)  # 找出target为1的位置
            if target_mask.sum() > 0:  # 确保有target为1的样本
                entity_acc = (pred_entity_binary[target_mask] == target_ent[target_mask]).float().mean().item()
            else:
                entity_acc = 0.0
            relation_acc = (pred_relation.argmax(dim=1) == target_rel.reshape(-1)).float().mean().item()

        torch.nn.utils.clip_grad_norm_(tree_model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(predict_model.parameters(), max_norm=1.0)
        # torch.nn.utils.clip_grad_norm_(entity_predictor.parameters(), max_norm=1.0)
        # 分别进行反向传播
        loss.backward()  # 使用retain_graph=True保留计算图
        tree_optimizer.step()
        predict_optimizer.step()
        # loss_entity.backward()
        # entity_optimizer.step()
        train_acc = ((pred_relation.argmax(dim=1) == target_rel.reshape(-1)).sum() / pred_relation.shape[0]).cpu().numpy()

        total_acc += train_acc
        batch_num += 1
        
        # 更新进度条描述
        pbar.set_description(
            f'Training Joint Entity Loss: {loss.item():.4f}, '
            # f'Relation Loss: {loss_relation.item():.4f}, '
            f'Entity Acc: {entity_acc:.4f}, '
            f'Relation Acc: {relation_acc:.4f}'
        )
    
    return total_loss / batch_num, total_acc / batch_num

def test(args, opt, dataset):
    # 获取基本设置
    head_rdict = dataset.get_head_relation_dict()
    exp_path = opt.exp_path
    rule_len = opt.learned_path_len
    
    # 加载模型
    tree_model = torch.load(exp_path+'/tree_model.pt')
    predict_model = torch.load(exp_path+'/predict_model.pt')
    tree_model.to(opt.device)
    predict_model.to(opt.device)

    print_msg("开始测试")
    # 设置为评估模式
    tree_model.eval()
    predict_model.eval()
    
    # 初始化变量
    r_num = (head_rdict.__len__() - 1)
    batch_size = 1000
    probs = []
    rule_conf = {}
    candidate_rule = {}
    
    # 生成树路径
    tree_paths = enumerate_tree(r_num, head_rdict, body_len=rule_len, tree_num=args.tree_num)
    body_list = [",".join("|".join(item) for item in path) for path in tree_paths]
    candidate_rule[rule_len] = body_list
    
    # 批处理预测
    n_epoches = math.ceil(float(len(body_list))/ batch_size)
    pbar = tqdm(range(n_epoches))
    for epoches in pbar:
        # 处理当前批次
        start_idx = epoches * batch_size
        end_idx = min((epoches + 1) * batch_size, len(body_list))
        bodies = body_list[start_idx:end_idx]
            
        # 转换为模型输入格式
        body_idx = tree2idx(bodies, head_rdict) 
        inputs = torch.LongTensor(np.array(body_idx)).to(opt.device)
            
        # 模型预测
        with torch.no_grad():
            # TreeRule模型不使用contrast模式进行前向传播
            tree_embeddings = tree_model(inputs, contrast = False)
            # 使用RulePredictor进行预测
            pred_head = predict_model(tree_embeddings,mode = "transformer")
            prob_ = torch.softmax(pred_head, dim=-1)
            probs.append(prob_.detach().cpu())
            
        pbar.set_description(f"已处理 {end_idx}/{len(body_list)} 条规则")
    
    # 合并所有预测结果
    rule_conf[rule_len] = torch.cat(probs, dim=0)
    print(f"规则置信度张量形状: {rule_conf[rule_len].shape}")
    
    return rule_conf, candidate_rule

def get_rule(opt,dataset,rule_conf,candidate_rule):
    print_msg("Generate Rule!")
    head_rdict = dataset.get_head_relation_dict()
    n_rel = head_rdict.__len__()-1
    
    for rule_len in rule_conf:
        rule_path = opt.exp_path+"/rules{}_{}.txt".format(opt.topk, rule_len)
        print("\nrule length:{}".format(rule_len))
        sorted_val, sorted_idx = torch.sort(rule_conf[rule_len],0, descending=True)
        
        n_rules, _ = sorted_val.shape
        
        with open(rule_path, 'w') as g:
            for r in range(n_rel):
                head = head_rdict.idx2rel[r]
                idx = 0
                while idx<opt.topk and idx<n_rules:
                    conf = sorted_val[idx, r]
                    body = candidate_rule[rule_len][sorted_idx[idx, r]]
                    msg = "{:.3f} ({:.3f})\t{} <-- ".format(conf, conf, head)
                    parts = body.split(',')
                    # 再处理每一部分，将其中的 '|' 替换为 ','
                    processed_parts = [part.replace('|', ', ') for part in parts]
                    # 最后，用 ';' 连接这些处理过的部分
                    msg += ';'.join(processed_parts)
                    # 输出或写入文件
                    g.write(msg + '\n')
                    idx+=1
    print("Save rule to {}".format(rule_path))

def get_hits(opt,dataset):
    all_rules = {}
    all_rule_heads = []
    load_rules(opt.exp_path+"/rules{}_{}.txt".format(opt.topk, opt.learned_path_len), all_rules, all_rule_heads)

    for head in all_rules:
        all_rules[head] = all_rules[head][:opt.topk]
        
    fact_rdf, train_rdf, valid_rdf, test_rdf = dataset.fact_rdf, dataset.train_rdf, dataset.valid_rdf, dataset.test_rdf

    msg1 = kg_completion(opt, all_rules, dataset)
    opt.writer.add_text('kg_completion', msg1)



def train(args,opt,dataset):
    # 获取实验路径
    exp_path = opt.exp_path
    
    # 获取头实体-关系字典和所有三元组数据
    head_rdict = dataset.get_head_relation_dict()
    all_rdf = dataset.fact_rdf + dataset.train_rdf + dataset.valid_rdf
    all_id_rdf = rdf2idx(all_rdf,head_rdict)
    
    print_msg("Sampling Tree")
    # 构建实体的后代字典
    entity2desced = construct_descendant(all_rdf)
    max_path_len = args.max_path_len
    
    # 采样树形数据
    train_data = sample_tree_data(opt,2,opt.anchor,all_rdf,entity2desced,head_rdict, sample_ratio= 1.0)
    
    # 获取关系字典和关数量
    rdict = dataset.get_relation_dict()
    relation_num = rdict.__len__()
    
    # 初始化模型
    tree_model = TreeRule(opt.emb_size, relation_num, opt.device, model_type="base").to(opt.device)
    predict_model = RulePredictor(opt.emb_size, relation_num, opt.device).to(opt.device)
   
    # 初始化优化器
    predict_optimizer = torch.optim.AdamW(predict_model.parameters(), lr=opt.predict_lr)
    tree_optimizer = torch.optim.AdamW(tree_model.parameters(), lr=opt.joint_contrast_lr)

    # 定义损失函数
    criterion = nn.BCEWithLogitsLoss()  # 对于实体预测使用BCE损失
    
    # 训练循环
    pbar = tqdm(range(opt.contrast_n_epoch))
    # 准备数据加载器
    path_data = torch.stack(train_data,0)
    rule_data = TensorDataset(path_data)
    path_loader = DataLoader(rule_data,batch_size=opt.batch_size,shuffle=True)
    
    # 设置模型为训练模式
    tree_model.train()
    predict_model.train()
    
    # 开始训练
    for epoch in pbar:
        total_loss = 0.0
        batch_num = 0
        total_acc = 0.0
        batch_pbar = tqdm(path_loader, desc=f'Epoch {epoch+1}/{opt.contrast_n_epoch}', leave=False)
        
        for batch in batch_pbar:
            # 清空梯度
            predict_model.zero_grad()
            tree_model.zero_grad()
            
            # 准备输入数据目标
            inputs =batch[0][:,:,0:-2].to(opt.device)
            target_rel = batch[0][:,:,-1][:, 0].to(opt.device)
            
            # 前向传播
            input_emb = tree_model(inputs,mode='predict')
            pred_head = predict_model(input_emb,opt.predict_mode)
            
            # 计算损失
            loss = criterion(pred_head,target_rel.reshape(-1))
            
            # 反向传播
            loss.backward()
            
            # 计算准确率
            train_acc =((pred_head.argmax(dim=1) == target_rel.reshape(-1)).sum() / pred_head.shape[0]).cpu().numpy()
            total_loss += loss.item()
            total_acc += train_acc
            batch_num += 1
            
            # 新进度条
            batch_pbar.set_postfix(loss=total_loss / batch_num, acc=total_acc / batch_num)
            
            # 更新参数
            predict_optimizer.step()
            tree_optimizer.step()
            
        pbar.update()
        pbar.set_description(f' loss: {total_loss/batch_num}, acc: {total_acc/batch_num}')         
    
    # 保存模型
    print("Saving model...")
    torch.save(tree_model, exp_path+'/tree_model.pt')
    torch.save(predict_model, exp_path+'/predict_model.pt')


