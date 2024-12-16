import subprocess
from utils import *

# exp_names = ["family","wn18rr","umls","kinship","fb15k-237"]
commands = []
exp_names = ["family/","wn-18rr/","umls/","kinship/"]

# exp_names = ["family/"]
# exp_names = ["family/"]
# exp_names=["fb15k-237/"]
# exp_names=["wn-18rr/"]
# exp_names=["kinship/"]
# exp_names = ["nations/"]
# exp_names =["yago/"]
# exp_names = ["YAGO3-10/"]
# exp_names = ["yago/"]
# exp_names = ["umls/"]
base_path ="/home/fxy/thesis/treeRule_new/"
dataset_path = base_path+"datasets/"
model_path = base_path+"saves/init/"
# model_path = base_path +"saves/abalation/without_contrast"
# model_path = base_path+"saves/nojoint_3_1/"
# model_path = base_path+"saves/nocontrast_3_1/"
# /home/fxy/paper/ContrastRule/saves/abalation/augment/fb15k-237_3_3
path_len = [2]
gpu = 0
# learned_path_len = [2,3]
for name in exp_names: 
    for path in path_len:
        # for learned in learned_path_len:
        learned = path
        command = "/home/fxy/miniconda3/envs/baichuan/bin/python  main.py  --train --test --get_rule --batch_size 256  --hidden_size 512 --bn_hidden_size 64 --exp_name "+name.replace("/","")+"  --data "+dataset_path+name+" --topk 100 --gpu "+str(gpu)+ " --model "+model_path+" --max_path_len "+str(path)+" --learned_path_len "+str(learned)+" --sparsity 1 --anchor 10000"
        commands.append(command)
        print_msg("Running expriment: {} path{} learned_path.".format(name,path,learned))
        print("Running command:{}".format(command))
        process = subprocess.Popen(command, shell=True,executable='/bin/bash')
        process.wait()
        print_msg("Command finished!")
print_msg("All commands completed!")

