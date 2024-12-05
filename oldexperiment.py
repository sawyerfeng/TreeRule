
from collections import namedtuple
from utils import *
from model import *
import time
import os
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from kg_completion import load_rules,kg_completion
import torch.multiprocessing as mp
import pickle
from sample_path import *
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

def test_train(args,opt,dataset):
    head_rdict = dataset.get_head_relation_dict()
    all_rdf = dataset.fact_rdf + dataset.train_rdf + dataset.valid_rdf
    all_id_rdf = rdf2idx(all_rdf,head_rdict)
    print_msg("Sampling Tree")
    entity2desced = construct_descendant(all_rdf)
    max_path_len = args.max_path_len
    train_data = sample_tree_data(opt,2,opt.anchor,all_rdf,entity2desced,head_rdict)
    rdict = dataset.get_relation_dict()
    relation_num = rdict.__len__()
    # 模型
    contrast_model = TreeRule(opt.emb_size, relation_num, opt.device).to(opt.device)
    predict_model = RulePredictor(opt.emb_size, relation_num, opt.device).to(opt.device)
   
    # 优化器
    predict_optimizer = torch.optim.AdamW(predict_model.parameters(), lr=opt.predict_lr)

    # 损失函数
    predict_criterion = nn.CrossEntropyLoss()
    
    pbar = tqdm(range(opt.contrast_n_epoch))
    path_data = torch.stack(train_data,0)
    path_loader = DataLoader(path_data,batch_size=opt.batch_size,shuffle=True)
    for epoch in pbar:
        total_loss = 0.0
        batch_num = 0
        total_acc = 0.0
        for batch in path_loader:
            predict_model.zero_grad()
            inputs =batch[0][:,0:-2].to(opt.device)
            target = batch[0][:,-1].to(opt.device)
            input_emb = contrast_model(inputs,mode='predict')
            pred_head = predict_model(input_emb,opt.predict_mode)
            loss = predict_criterion(pred_head,target.reshape(-1))
            train_acc =((pred_head.argmax(dim=1) == target.reshape(-1)).sum() / pred_head.shape[0]).cpu().numpy()
            loss.backward()
            predict_optimizer.step()
            total_loss += loss.item()
            total_acc += train_acc
            batch_num += 1
        pbar.update()
        pbar.set_description(f' loss: {total_loss/batch_num}, acc: {total_acc/batch_num}')         
def train(args,opt,dataset):
    writer = opt.writer
    rdict = dataset.get_relation_dict()
    head_rdict = dataset.get_head_relation_dict()
    all_rdf = dataset.fact_rdf + dataset.train_rdf + dataset.valid_rdf
    all_id_rdf = rdf2idx(all_rdf,head_rdict)
    entity2desced = construct_descendant(all_rdf)
    entity2descedid = construct_descendant(all_id_rdf)
    relation_num = rdict.__len__()

    #采样训练数据
    print_msg("Sampling Contrast")
    max_path_len = args.max_path_len
    anchor_num = opt.anchor
    pos_rel_num = (head_rdict.__len__() - 1)/2



    
    ##记录参数
    experiment_options = "<br>".join([f'{field}: {getattr(opt, field)}' for field in opt._fields])
    writer.add_text('experiment_options', experiment_options)

    ##定义模型
    print("Building Model")
    contrast_model = Simrule(opt.emb_size, relation_num, opt.device, opt.drop_prob,opt.tao,opt.bn_hidden_size).to(opt.device)
    predict_model = RulePredictor(opt.emb_size, relation_num, opt.device).to(opt.device)
    ## 定义优化器
    contrast_optimizer = torch.optim.AdamW(contrast_model.parameters(), lr=opt.contrast_lr)
    predict_optimizer = torch.optim.AdamW(predict_model.parameters(), lr=opt.predict_lr)
    ##定义学习率衰减器
    contrast_scheduler = torch.optim.lr_scheduler.ExponentialLR(contrast_optimizer, gamma=opt.gamma)
    predict_scheduler = torch.optim.lr_scheduler.ExponentialLR(predict_optimizer, gamma=opt.gamma)
    ##定义损失函数
    predict_criterion = nn.CrossEntropyLoss()

    """
    1.Contrastive Training,对比学习的无监督训练
    """

    contrast_model.train()

    exp_path = opt.exp_path
    print_msg("Start Traing Contrast Model")

    if os.path.exists(exp_path+'/contrast_model.pt') and opt.load_contrast_model:
        print("Load Contrast Model")
        contrast_model = torch.load(exp_path+'/contrast_model.pt')
    else:
        print("Train Contrast Model")
        if opt.load_contrast_model and opt.contrast_n_epoch == 0:
            pass
        else:
            if os.path.exists(args.data+"/sample_rule_"+str(opt.contrast_ratio)+".pkl") and opt.load_path:
                print("Load Contrast Data")
                with open(args.data+"/sample_rule_"+str(opt.contrast_ratio)+".pkl", 'rb') as f:
                    contrast_data = pickle.load(f)
            else:
                if "kinship" not in args.data:
                    contrast_data = sample_training_data_ratio(opt,max_path_len,opt.contrast_ratio,all_rdf,entity2desced,head_rdict)
                else:
                    contrast_data = sample_training_data(opt,max_path_len,opt.anchor,all_rdf,entity2desced,head_rdict)
                with open(args.data+"/sample_rule_"+str(opt.contrast_ratio)+".pkl", 'wb') as f:
                    pickle.dump(contrast_data, f)
            sample_info = '<br>'.join(['{}: {}'.format(k, v) for k, v in contrast_data['sample_info'].items()])
            writer.add_text("contrast_sample_info",sample_info)
        train_data = contrast_data
        start = time.time()
        for rule_len in opt.body_len_range:
            rule_data = train_data[rule_len]
            rule_num = len(rule_data)
            pbar = tqdm(range(opt.contrast_n_epoch))
            rule_data = torch.stack(rule_data,0)
            rule_data = TensorDataset(rule_data)
            rule_loader = DataLoader(rule_data,batch_size=opt.batch_size,shuffle=True)
            for epoch in pbar:
                total_loss = 0.0
                batch_num = 0
                for batch in rule_loader:
                    contrast_optimizer.zero_grad()
                    inputs =batch[0][:,0:-2].to(opt.device)
                    batch_size = inputs.shape[0]
                    seq_len = inputs.shape[1]
                    input_ids = torch.zeros(2*batch_size,seq_len,dtype=torch.int64).to(opt.device)
                    for i,x in enumerate(inputs):
                        input_ids[2*i,:seq_len] = x
                        # if opt.inverse_negative and random.random() > 0:
                        #     #自反性构建正样本
                        #     input_ids[2*i+1,:seq_len] = torch.where(x.flip(0)>=pos_rel_num, x.flip(0)-pos_rel_num, x.flip(0)+pos_rel_num)
                        # else:
                        #     input_ids[2*i+1,:seq_len] = x
                        if opt.augment == 'drop':
                            input_ids[2*i+1,:seq_len] = x
                        elif  opt.augment == 'delete':
                            #随机删除一个关系   
                            input_ids[2*i+1,:seq_len] = torch.cat((x[:random.randint(0,seq_len-1)],x[random.randint(0,seq_len-1)+1:]))
                        elif opt.augment =='replace':
                            #随机替换一个关系
                            input_ids[2*i+1,:seq_len] = x.clone()
                            input_ids[2*i+1,random.randint(0,seq_len-1)] = random.randint(0,relation_num-1)
                    

                    loss,_ = contrast_model(input_ids)
                    # clip_grad_norm_(contrast_model.parameters(), 1.0)
                    loss.backward()
                    contrast_optimizer.step()
                    total_loss += loss.item()
                    batch_num += 1
                    # writer.add_scalar('contrast_loss_str'+str(rule_len), loss.item(), batch_num)
                pbar.update()
                pbar.set_description(f'rule_len: {rule_len}, loss: {total_loss/batch_num}')
                writer.add_scalar('contrast_loss_str'+str(rule_len), total_loss/batch_num, epoch)    
                # writer.add_scalar('contrast_lr_'+str(rule_len), contrast_optimizer.param_groups[0]['lr'], epoch)
                contrast_scheduler.step()
        end = time.time()
        formatted_time = time.strftime("%H:%M:%S", time.gmtime(end - start))
        writer.add_text("contrast_training_time:","contrast_training_time:"+str(formatted_time))
        print("Time usage: {}".format(formatted_time))

        print("Saving model...")
        torch.save(contrast_model, exp_path+'/contrast_model.pt')
    
    """
    2.Train Rule Predictor，规则预测器的有监督训练，也就是冻结对比学习的参数，只训练规则预测器
    """
    
    print_msg("Start Traing Rule Predictor")
    if os.path.exists(exp_path+'/predict_model.pt') and opt.load_predict_model:
        print("Load Predict Model")
        predict_model = torch.load(exp_path+'/predict_model.pt')
    else:
        if os.path.exists(args.data+"/sample_rule_"+str(opt.predict_ratio)+".pkl") and opt.load_path:
            print("Load Predict Data")
            with open(args.data+"/sample_rule_"+str(opt.predict_ratio)+".pkl", 'rb') as f:
                predict_data = pickle.load(f)
        else:
            if "kinship" not in args.data:
                predict_data = sample_training_data_ratio(opt,max_path_len,opt.predict_ratio,all_rdf,entity2desced,head_rdict)
            else:
                predict_data = sample_training_data(opt,max_path_len,opt.anchor,all_rdf,entity2desced,head_rdict)
            with open(args.data+"/sample_rule_"+str(opt.predict_ratio)+".pkl", 'wb') as f:
                pickle.dump(predict_data, f)
        sample_info = '<br>'.join(['{}: {}'.format(k, v) for k, v in predict_data['sample_info'].items()])
        writer.add_text("predict_sample_info",sample_info)
        train_data = predict_data
        print("Train Predict Model")
        predict_model.train()
        contrast_model.eval()
        for param in contrast_model.parameters():
            param.requires_grad = False
        start = time.time()
        for rule_len in opt.body_len_range:
            rule_data = train_data[rule_len]
            rule_data = torch.stack(rule_data,0)
            rule_data = TensorDataset(rule_data)
            rule_loader = DataLoader(rule_data,batch_size=opt.batch_size,shuffle=True)
            pbar = tqdm(range(opt.predict_n_epoch))
            for epoch in pbar:
                total_loss =0.0
                batch_num = 0
                total_acc = 0.0
                for batch in rule_loader:
                    predict_model.zero_grad()
                    inputs =batch[0][:,0:-2].to(opt.device)
                    target = batch[0][:,-1].to(opt.device)

                    #forward pass
                    input_emb = contrast_model(inputs,mode='predict')
                    pred_head = predict_model(input_emb,opt.predict_mode)
                    loss = predict_criterion(pred_head,target.reshape(-1))
                    train_acc =((pred_head.argmax(dim=1) == target.reshape(-1)).sum() / pred_head.shape[0]).cpu().numpy()
                    loss.backward()
                    predict_optimizer.step()
                    total_loss += loss.item()
                    total_acc += train_acc
                    batch_num += 1
                    # writer.add_scalar('predict_loss_str'+str(rule_len), loss.item(), batch_num)
                    # writer.add_scalar('predict_train_acc'+str(rule_len), train_acc, batch_num)
                pbar.update()
                pbar.set_description(f'rule_len: {rule_len}, loss: {total_loss/batch_num}, acc: {total_acc/batch_num}')
                writer.add_scalar('predict_loss_str'+str(rule_len), total_loss/batch_num, epoch)
                writer.add_scalar('predict_train_acc_'+str(rule_len), total_acc/batch_num, epoch)
                # writer.add_scalar('predict_lr_'+str(rule_len), predict_optimizer.param_groups[0]['lr'], epoch)
                predict_scheduler.step()
        end = time.time()
        formatted_time = time.strftime("%H:%M:%S", time.gmtime(end - start))
        writer.add_text("predict_training_time:","predict_training_time:"+str(formatted_time))
        print("Time usage: {}".format(formatted_time))
        print("Saving model...")
        torch.save(predict_model,exp_path+'/predict_model.pt')

    """
        3.Joint Training，开放对比学习与规则预测器的联合训练，也就是不冻结对比学习的参数，同时训练规则预测器
    """
    if opt.joint_train:
        print("Joint Training")
        contrast_model.train()
        predict_model.train()
        for param in contrast_model.parameters():
            param.requires_grad = True
        start = time.time()
        contrast_optimizer = torch.optim.AdamW(contrast_model.parameters(), lr=opt.joint_contrast_lr)
        predict_optimizer = torch.optim.AdamW(predict_model.parameters(), lr=opt.joint_predict_lr)
        for rule_len in opt.body_len_range:
            rule_data = train_data[rule_len]
            rule_num = len(rule_data)
            pbar = tqdm(range(opt.joint_n_epoch))
            rule_data = torch.stack(rule_data,0)
            rule_data = TensorDataset(rule_data)
            rule_loader = DataLoader(rule_data,batch_size=opt.batch_size,shuffle=True)
            for epoch in pbar:
                total_loss =0.0
                batch_num = 0
                total_acc = 0.0
                for batch in rule_loader:
                    predict_model.zero_grad()
                    contrast_model.zero_grad()
                    inputs =batch[0][:,0:-2].to(opt.device)
                    target = batch[0][:,-1].to(opt.device)
                    input_emb = contrast_model(inputs,mode='joint')
                    pred_head = predict_model(input_emb,opt.predict_mode)
                    loss = predict_criterion(pred_head,target.reshape(-1))
                    loss.backward()
                    train_acc = ((pred_head.argmax(dim=1) == target.reshape(-1)).sum() / pred_head.shape[0]).cpu().numpy()
                    total_loss += loss.item()
                    batch_num += 1
                    total_acc += train_acc
                    predict_optimizer.step()
                    contrast_optimizer.step()
                    # writer.add_scalar('joint_loss_str_'+str(rule_len), loss.item(), batch_num)
                    # writer.add_scalar('joint_train_acc_'+str(rule_len), train_acc, batch_num)
                pbar.update()
                pbar.set_description(f'rule_len: {rule_len}, loss: {total_loss/batch_num}, acc: {total_acc/batch_num}')
                writer.add_scalar('joint_loss_str_'+str(rule_len), total_loss/batch_num, epoch)
                writer.add_scalar('joint_train_acc_'+str(rule_len), total_acc/batch_num, epoch)
                contrast_scheduler.step()
                predict_scheduler.step()
        end = time.time()
        formatted_time = time.strftime("%H:%M:%S", time.gmtime(end - start))
        writer.add_text("joint_training_time:","joint_training_time:"+str(formatted_time))
        print("Time usage: {}".format(formatted_time))
        print("Saving model...")
        torch.save(contrast_model, exp_path+'/joint_contrast_model.pt')
        torch.save(predict_model, exp_path+'/joint_predict_model.pt')

def test(opt,dataset):
    head_rdict = dataset.get_head_relation_dict()
    exp_path = opt.exp_path
    if opt.joint_train:
        contrast_model = torch.load(exp_path+'/joint_contrast_model.pt')
        predict_model = torch.load(exp_path+'/joint_predict_model.pt')
    else:
        contrast_model = torch.load(exp_path+'/contrast_model.pt')
        predict_model = torch.load(exp_path+'/predict_model.pt')
    rule_len = opt.learned_path_len
    contrast_model.to(opt.device)
    predict_model.to(opt.device)

    print_msg("Start Testing")
    contrast_model.eval()
    predict_model.eval()
    r_num = (head_rdict.__len__() - 1)
    batch_size = 3000
    probs = []
    rule_conf = {}
    candidate_rule = {}
    if "fb15k-237" in opt.exp_name:
        pos_only = True
    else:
        pos_only = False
    _, body = enumerate_body(r_num, head_rdict, body_len=rule_len, pos_only=pos_only)
    body_list = ["|".join(b) for b in body]
    candidate_rule[rule_len] = body_list
    n_epoches = math.ceil(float(len(body_list))/ batch_size)
    pbar = tqdm(range(n_epoches))
    for epoches in pbar:
        bodies = body_list[epoches: (epoches+1)*batch_size]
        if epoches == n_epoches-1:
            bodies = body_list[epoches*batch_size:]
        else:
            bodies = body_list[epoches*batch_size: (epoches+1)*batch_size]
            
        body_idx = body2idx(bodies, head_rdict) 
        if torch.cuda.is_available():
            inputs = torch.LongTensor(np.array(body_idx)).to(opt.device)
        else:
            inputs = torch.LongTensor(np.array(body_idx))
        
            
        with torch.no_grad():
            input_embeddings = contrast_model(inputs,mode='eval')
            pred_head = predict_model(input_embeddings,opt.predict_mode) # [batch_size, 2*n_rel+1]
            prob_ = torch.softmax(pred_head, dim=-1)
            probs.append(prob_.detach().cpu())
        pbar.update()
        pbar.set_description("body{}".format((epoches+1)* batch_size))
    rule_conf[rule_len] = torch.cat(probs,dim=0)
    print ("rule_conf",rule_conf[rule_len].shape)
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
                    body = body.split('|')
                    msg += ", ".join(body)
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

    msg1,msg2 = kg_completion(all_rules, dataset,opt)
    opt.writer.add_text('kg_completion', msg1)
