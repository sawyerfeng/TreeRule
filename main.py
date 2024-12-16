import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import *
from utils import *
from experiment import *
import argparse
import json
from opt import get_opt
from datetime import datetime
from tensorboardX import SummaryWriter

if __name__ == "__main__":
    msg = "Tree Rule Mining"
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=512, help="increase output verbosity")
    parser.add_argument("--bn_hidden_size", type=int, default=256, help="increase output verbosity") 
    parser.add_argument("--hidden_size", type=int, default=256, help="increase output verbosity")
    parser.add_argument("--exp_name", type=str, help="increase output verbosity")
    parser.add_argument("--train", action="store_true", help="increase output verbosity")
    parser.add_argument("--test", action="store_true", help="increase output verbosity")
    parser.add_argument("--get_rule", action="store_true", help="increase output verbosity")
    parser.add_argument("--data", default="family", help="increase output verbosity")
    parser.add_argument("--topk", type=int, default=100, help="increase output verbosity")
    parser.add_argument("--gpu", type=int, default=1, help="increase output verbosity")
    parser.add_argument("--model", default="family", help="increase output verbosity")
    parser.add_argument("--model_type", type=str, default="hierarchical",
                      choices=['base', 'hierarchical'],
                      help="选择模型类型：base-基础模型，hierarchical-层次化模型")
    parser.add_argument("--num_heads", type=int, default=4,
                      help="注意力头数")
    parser.add_argument("--max_path_len", type=int, default=100,
                      help="最大路径长度")
    parser.add_argument("--max_tree_paths", type=int, default=100,
                      help="树中最大路径数")
    parser.add_argument("--learned_path_len", type=int, default=3, help="increase output verbosity")
    parser.add_argument("--sparsity", type=float, default=1, help="increase output verbosity")
    parser.add_argument("--anchor", type=int, default=5000, help="increase output verbosity")
    parser.add_argument("--drop", type=float, default=0.1, help="increase output verbosity")
    parser.add_argument("--augment", type=str, default='drop', help="increase output verbosity")
    parser.add_argument("--tao", type=float, default=1, help="increase output verbosity")
    parser.add_argument("--tree_num", type=int, default=2, help="increase output verbosity")
    
    args = parser.parse_args()

    # 设置设备
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    # 设置随机种子
    seed_everything(3407)

    # 加载数据集
    data = args.data
    exp_name = args.exp_name
    inv = False if 'fb15k' in data else True
    dataset = Dataset(data, sparsity=args.sparsity, inv=inv)
    print(f"Dataset: {data}")

    # 获取配置
    opt = get_opt(args, args.exp_name, joint=False)
    opt.writer.add_text("time", str(datetime.now()))

    # 创建实验目录
    if not os.path.exists(opt.exp_path):
        os.makedirs(opt.exp_path)

    # 训练阶段
    if args.train:
        print_msg("Train!")
        train_three_stage(args, opt, dataset)
        # train(args, opt, dataset)

    # 测试阶段
    if args.test:
        print_msg("Test!")
        rule_conf, candidate_rule = test(args, opt, dataset)

    # 获取规则
    if args.get_rule:
        print_msg("Get Rule!")
        get_rule(opt, dataset, rule_conf, candidate_rule)

    # 清理GPU缓存
    torch.cuda.empty_cache()

    # 评估性能
    if opt.get_hits:
        print_msg("Get Hits!")
        get_hits(opt, dataset)
