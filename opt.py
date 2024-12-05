from experiment import * 

from tensorboardX import SummaryWriter
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def get_opt(args,dataset,joint =True):
    # log_path = "/home/fxy/paper/ContrastRule/logs/abalation/bn_emb/"+args.exp_name+'_'+str(args.bn_hidden_size)+'_'+str(args.max_path_len)+'_'+str(args.learned_path_len)
    # log_path = "/home/fxy/paper/ContrastRule/logs/last/"+args.exp_name+'_'+str(args.max_path_len)+'_'+str(args.learned_path_len)
    # log_path = "/home/fxy/ContrastRule/logs/abalation/drop/"+args.exp_name+'_'+str(args.hidden_size)+'_'+str(args.max_path_len)+'_'+str(args.learned_path_len)
    # log_path = "/home/fxy/ContrastRule/logs/abalation/augment/"+args.exp_name+'_'+str(args.augment)+'_'+str(args.max_path_len)+'_'+str(args.learned_path_len)
    log_path = "/home/fxy/thesis/treeRule/logs/init/"+args.exp_name+'_'+str(args.max_path_len)+'_'+str(args.learned_path_len)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    writer = SummaryWriter(log_path) 
    if not joint:
        if dataset == 'wn-18rr':
        
            opt = training_option(
                exp_name  = args.exp_name,
                #dimention
                batch_size = args.batch_size,
                bn_hidden_size= args.bn_hidden_size,
                emb_size = args.hidden_size,
                anchor = 100000,
                contrast_ratio = 0.5,
                predict_ratio = 0.1,
                none_ratio = 16,
                #switch
                inverse_negative = False,
                load_path = False,
                load_contrast_model = False,
                load_predict_model = False,
                joint_train = False,
                get_hits = True,
                augment = args.augment,
                predict_mode = 'attn',
                #epoch
                contrast_n_epoch = 2, 
                predict_n_epoch = 5,
                joint_n_epoch = 0,
                #learning rate
                contrast_lr = 1e-4,
                predict_lr = 1e-4,
                joint_contrast_lr = 2e-5,
                joint_predict_lr = 2e-5,
                # other
                drop_prob = args.drop if args.augment=='drop' else 0,#0.6 sota
                gamma = 0.9,
                none_drop = 0,
                body_len_range =list(range(args.max_path_len,args.max_path_len+1)),
                device = device,
                writer=writer,
                model = args.model,
                topk = args.topk,
                cpu_num = mp.cpu_count()//2,
                learned_path_len=args.learned_path_len,
                max_path_len = args.max_path_len,
                exp_path = args.model+args.exp_name+'_'+str(args.max_path_len)+'_'+str(args.learned_path_len),
                tao = args.tao,
            )
        elif dataset == 'umls':
            opt = training_option(
                exp_name  = args.exp_name,
                #dimention
                batch_size = args.batch_size,
                bn_hidden_size= args.bn_hidden_size,
                emb_size = args.hidden_size,
                anchor = 100000,
                contrast_ratio = 0.5,
                predict_ratio = 0.1,
                none_ratio = 30,
                #switch
                inverse_negative = False,
                load_path = False,
                load_contrast_model = False,
                load_predict_model = False,
                joint_train = False,
                get_hits = True,
                augment = args.augment,
                predict_mode = 'attn',
                #epoch
                contrast_n_epoch =2, 
                predict_n_epoch = 5,
                joint_n_epoch =0    ,
                #learning rate
                contrast_lr = 1e-4,
                predict_lr = 1e-4,
                joint_contrast_lr = 2e-5,
                joint_predict_lr = 2e-5,
                # other
                drop_prob = args.drop if args.augment=='drop' else 0,#0.6 sota
                gamma = 0.9,
                none_drop = 0,
                body_len_range =list(range(args.max_path_len,args.max_path_len+1)),
                device = device,
                writer=writer,
                model = args.model,
                topk = args.topk,
                cpu_num = mp.cpu_count()//2,
                learned_path_len=args.learned_path_len,
                max_path_len = args.max_path_len,
                exp_path = args.model+args.exp_name+'_'+str(args.max_path_len)+'_'+str(args.learned_path_len),
                tao = args.tao,
            )
        elif dataset == 'family':
            opt = training_option(
                exp_name  = args.exp_name,
                #dimention
                batch_size = args.batch_size,
                emb_size = args.hidden_size,
                bn_hidden_size= args.bn_hidden_size,
                anchor = 5000,
                contrast_ratio = 0.3,
                predict_ratio = 0.1,
                none_ratio = 1,
                #switch
                inverse_negative = False,
                load_path = False,
                load_contrast_model = False,
                load_predict_model = False,
                joint_train = False,
                get_hits = True,
                augment = args.augment,
                predict_mode = 'attn',
                #epoch
                contrast_n_epoch = 1, 
                predict_n_epoch = 5,
                joint_n_epoch = 1,
                #learning rate
                contrast_lr = 1e-5,
                predict_lr = 1e-5,
                joint_contrast_lr = 2e-5,
                joint_predict_lr = 2e-5,
                # other
                drop_prob = args.drop if args.augment=='drop' else 0,#0.6 sota
                gamma = 0.9,
                none_drop = 0,
                body_len_range =list(range(args.max_path_len,args.max_path_len+1)),
                device = device,
                writer=writer,
                model = args.model,
                topk = args.topk,
                cpu_num = mp.cpu_count()//2,
                learned_path_len=args.learned_path_len,
                max_path_len = args.max_path_len,
                exp_path = args.model+args.exp_name+'_'+str(args.max_path_len)+'_'+str(args.learned_path_len),
                tao = args.tao,
            )
        elif dataset == 'kinship':
            opt = training_option(
                exp_name  = args.exp_name,
                #dimention
                batch_size = args.batch_size,
                bn_hidden_size= args.bn_hidden_size,
                emb_size = args.hidden_size,
                anchor = 100000,
                contrast_ratio = 0.5,
                predict_ratio = 0.1,
                none_ratio = 30,
                #switch
                inverse_negative = False,
                load_path = False,
                load_contrast_model = False,
                load_predict_model = False,
                joint_train = False,
                get_hits = True,
                augment = args.augment,
                predict_mode = 'attn',
                #epoch
                contrast_n_epoch = 2, 
                predict_n_epoch = 5,
                joint_n_epoch = 0   ,
                #learning rate
                contrast_lr = 1e-4,
                predict_lr = 1e-4,
                joint_contrast_lr = 2e-5,
                joint_predict_lr = 2e-5,
                # other
                drop_prob = args.drop if args.augment=='drop' else 0,#0.6 sota
                gamma = 0.9,
                none_drop = 0,
                body_len_range =list(range(args.max_path_len,args.max_path_len+1)),
                device = device,
                writer=writer,
                model = args.model,
                topk = args.topk,
                cpu_num = mp.cpu_count()//2,
                learned_path_len=args.learned_path_len,
                max_path_len = args.max_path_len,
                exp_path = args.model+args.exp_name+'_'+str(args.max_path_len)+'_'+str(args.learned_path_len),
                tao = args.tao,
            )
        elif dataset == 'fb15k-237':
            opt = training_option(
                exp_name  = args.exp_name,
                #dimention
                batch_size = args.batch_size,
                bn_hidden_size= args.bn_hidden_size,
                emb_size = args.hidden_size,
                anchor = 100000,
                contrast_ratio = 0.2,
                predict_ratio = 0.1,
                none_ratio = 1.5,
                #switch
                inverse_negative = False,
                load_path = True,
                load_contrast_model = False,
                load_predict_model = False,
                joint_train = False,
                get_hits = True,
                augment = args.augment,
                predict_mode = 'attn',
                #epoch
                contrast_n_epoch = 2, 
                predict_n_epoch = 5,
                joint_n_epoch =5,
                #learning rate
                contrast_lr = 2e-5,
                predict_lr = 1e-4,
                joint_contrast_lr = 2e-5,
                joint_predict_lr = 2e-5,
                # other
                drop_prob = args.drop if args.augment=='drop' else 0,#0.6 sota
                gamma = 0.9,
                none_drop = 0.5,
                body_len_range =list(range(args.max_path_len,args.max_path_len+1)),
                device = device,
                writer=writer,
                model = args.model,
                topk = args.topk,
                cpu_num = mp.cpu_count()//2,
                learned_path_len=args.learned_path_len,
                max_path_len = args.max_path_len,
                exp_path = args.model+args.exp_name+'_'+str(args.max_path_len)+'_'+str(args.learned_path_len),
                tao = args.tao,
            )
        elif dataset == 'yago':
            opt = training_option(
                exp_name  = args.exp_name,
                #dimention
                bn_hidden_size= args.bn_hidden_size,
                batch_size = args.batch_size,
                emb_size = args.hidden_size,
                anchor = 100000,
                contrast_ratio = 0.1,
                predict_ratio = 0.05,
                none_ratio = 1,
                #switch
                inverse_negative = False,
                load_path = True,
                load_contrast_model = False,
                load_predict_model = False,
                joint_train = False,
                get_hits = True,
                augment = args.augment,
                predict_mode = 'attn',
                #epoch
                contrast_n_epoch = 2, 
                predict_n_epoch = 5,
                joint_n_epoch =0,
                #learning rate
                contrast_lr = 1e-4,
                predict_lr = 1e-4,
                joint_contrast_lr = 2e-5,
                joint_predict_lr = 2e-5,
                # other
                drop_prob = args.drop if args.augment=='drop' else 0,#0.6 sota
                gamma = 0.9,
                none_drop = 0.9,
                body_len_range =list(range(args.max_path_len,args.max_path_len+1)),
                device = device,
                writer=writer,
                model = args.model,
                topk = args.topk,
                cpu_num = mp.cpu_count()-1,
                learned_path_len=args.learned_path_len,
                max_path_len = args.max_path_len,
                exp_path = args.model+args.exp_name+'_'+str(args.max_path_len)+'_'+str(args.learned_path_len),
                tao = args.tao,
            )
    else:
        if dataset == 'wn-18rr':
        
            opt = training_option(
                exp_name  = args.exp_name,
                #dimention
                batch_size = args.batch_size,
                bn_hidden_size= args.bn_hidden_size,
                emb_size = args.hidden_size,
                anchor = 100000,
                contrast_ratio = 0.5,
                predict_ratio = 0.1,
                none_ratio = 16,
                #switch
                inverse_negative = False,
                load_path = False,
                load_contrast_model = False,
                load_predict_model = False,
                joint_train = True,
                get_hits = True,
                augment = args.augment,
                predict_mode = 'attn',
                #epoch
                contrast_n_epoch = 2, 
                predict_n_epoch = 0,
                joint_n_epoch = 5,
                #learning rate
                contrast_lr = 2e-5,
                predict_lr = 1e-4,
                joint_contrast_lr = 2e-5,
                joint_predict_lr = 2e-5,
                # other
                drop_prob = args.drop if args.augment=='drop' else 0,#0.6 sota
                gamma = 0.9,
                none_drop = 0,
                body_len_range =list(range(args.max_path_len,args.max_path_len+1)),
                device = device,
                writer=writer,
                model = args.model,
                topk = args.topk,
                cpu_num = mp.cpu_count()//2,
                learned_path_len=args.learned_path_len,
                max_path_len = args.max_path_len,
                exp_path = args.model+args.exp_name+'_'+str(args.max_path_len)+'_'+str(args.learned_path_len),
                tao = args.tao,
            )
        elif dataset == 'umls':
            opt = training_option(
                exp_name  = args.exp_name,
                #dimention
                batch_size = args.batch_size,
                bn_hidden_size= args.bn_hidden_size,
                emb_size = args.hidden_size,
                anchor = 100000,
                contrast_ratio = 0.5,
                predict_ratio = 0.2,
                none_ratio = 30,
                #switch
                inverse_negative = False,
                load_path = False,
                load_contrast_model = False,
                load_predict_model = False,
                joint_train = True,
                get_hits = True,
                augment = args.augment,
                predict_mode = 'attn',
                #epoch
                contrast_n_epoch = 2, 
                predict_n_epoch = 0,
                joint_n_epoch = 5    ,
                #learning rate
                contrast_lr = 2e-5,
                predict_lr = 1e-4,
                joint_contrast_lr = 2e-5,
                joint_predict_lr = 2e-5,
                # other
                drop_prob = args.drop if args.augment=='drop' else 0,#0.6 sota
                gamma = 0.9,
                none_drop = 0,
                body_len_range =list(range(args.max_path_len,args.max_path_len+1)),
                device = device,
                writer=writer,
                model = args.model,
                topk = args.topk,
                cpu_num = mp.cpu_count()//2,
                learned_path_len=args.learned_path_len,
                max_path_len = args.max_path_len,
                exp_path = args.model+args.exp_name+'_'+str(args.max_path_len)+'_'+str(args.learned_path_len),
                tao = 1,
            )
        elif dataset == 'nations':
            opt = training_option(
                exp_name  = args.exp_name,
                #dimention
                batch_size = args.batch_size,
                bn_hidden_size= args.bn_hidden_size,
                emb_size = args.hidden_size,
                anchor = 100000,
                contrast_ratio = 0.5,
                predict_ratio = 0.3,
                none_ratio = 5,
                #switch
                inverse_negative = False,
                load_path = False,
                load_contrast_model = False,
                load_predict_model = False,
                joint_train = True,
                get_hits = True,
                augment = args.augment,
                predict_mode = 'attn',
                #epoch
                contrast_n_epoch = 1, 
                predict_n_epoch = 0,
                joint_n_epoch =10    ,
                #learning rate
                contrast_lr = 2e-5,
                predict_lr = 1e-4,
                joint_contrast_lr = 2e-5,
                joint_predict_lr = 2e-5,
                # other
                drop_prob = args.drop if args.augment=='drop' else 0,#0.6 sota
                gamma = 0.9,
                none_drop = 0,
                body_len_range =list(range(args.max_path_len,args.max_path_len+1)),
                device = device,
                writer=writer,
                model = args.model,
                topk = args.topk,
                cpu_num = mp.cpu_count()//2,
                learned_path_len=args.learned_path_len,
                max_path_len = args.max_path_len,
                exp_path = args.model+args.exp_name+'_'+str(args.max_path_len)+'_'+str(args.learned_path_len),
                tao = 1,
            )
        elif dataset == 'family':
            opt = training_option(
                exp_name  = args.exp_name,
                #dimention
                batch_size = args.batch_size,
                emb_size = args.hidden_size,
                bn_hidden_size= args.bn_hidden_size,
                anchor = 100000,
                contrast_ratio = 0.3,
                predict_ratio = 0.1,
                none_ratio = 1,
                #switch
                inverse_negative = False,
                load_path = False,
                load_contrast_model = False,
                load_predict_model = False,
                joint_train = False,
                get_hits = True,
                augment = args.augment,
                predict_mode = 'attn',
                #epoch
                contrast_n_epoch = 3, 
                predict_n_epoch =0,
                joint_n_epoch = 5,
                #learning rate
                contrast_lr = 2e-5,
                predict_lr = 1e-4,
                joint_contrast_lr = 2e-5,
                joint_predict_lr = 2e-5,
                # other
                drop_prob = args.drop if args.augment=='drop' else 0,#0.6 sota
                gamma = 0.9,
                none_drop = 0,
                body_len_range =list(range(args.max_path_len,args.max_path_len+1)),
                device = device,
                writer=writer,
                model = args.model,
                topk = args.topk,
                cpu_num = mp.cpu_count()//2,
                learned_path_len=args.learned_path_len,
                max_path_len = args.max_path_len,
                exp_path = args.model+args.exp_name+'_'+str(args.max_path_len)+'_'+str(args.learned_path_len),
                tao = args.tao,
            )
        elif dataset == 'kinship':
            opt = training_option(
                exp_name  = args.exp_name,
                #dimention
                batch_size = args.batch_size,
                bn_hidden_size= args.bn_hidden_size,
                emb_size = args.hidden_size,
                anchor = 100000,
                contrast_ratio = 0.5,
                predict_ratio = 0.3,
                none_ratio = 30,
                #switch
                inverse_negative = False,
                load_path = False,
                load_contrast_model = False,
                load_predict_model = False,
                joint_train = True,
                get_hits = True,
                augment = args.augment,
                predict_mode = 'attn',
                #epoch
                contrast_n_epoch = 1, 
                predict_n_epoch = 0,
                joint_n_epoch =10    ,
                #learning rate
                contrast_lr = 2e-5,
                predict_lr = 1e-4,
                joint_contrast_lr = 2e-5,
                joint_predict_lr = 2e-5,
                # other
                drop_prob = args.drop if args.augment=='drop' else 0,#0.6 sota
                gamma = 0.9,
                none_drop = 0,
                body_len_range =list(range(args.max_path_len,args.max_path_len+1)),
                device = device,
                writer=writer,
                model = args.model,
                topk = args.topk,
                cpu_num = mp.cpu_count()//2,
                learned_path_len=args.learned_path_len,
                max_path_len = args.max_path_len,
                exp_path = args.model+args.exp_name+'_'+str(args.max_path_len)+'_'+str(args.learned_path_len),
                tao = args.tao,
            )
        elif dataset == 'fb15k-237':
            opt = training_option(
                exp_name  = args.exp_name,
                #dimention
                batch_size = args.batch_size,
                bn_hidden_size= args.bn_hidden_size,
                emb_size = args.hidden_size,
                anchor = 100000,
                contrast_ratio = 0.2,
                predict_ratio = 0.1,
                none_ratio = 1.5,
                #switch
                inverse_negative = False,
                load_path = True,
                load_contrast_model = False,
                load_predict_model = False,
                joint_train = True,
                get_hits = True,
                augment = args.augment,
                predict_mode = 'mlp',
                #epoch
                contrast_n_epoch = 2, 
                predict_n_epoch = 0,
                joint_n_epoch =5,
                #learning rate
                contrast_lr = 2e-5,
                predict_lr = 1e-4,
                joint_contrast_lr = 2e-5,
                joint_predict_lr = 2e-5,
                # other
                drop_prob = args.drop if args.augment=='drop' else 0,#0.6 sota
                gamma = 0.9,
                none_drop = 0.5,
                body_len_range =list(range(args.max_path_len,args.max_path_len+1)),
                device = device,
                writer=writer,
                model = args.model,
                topk = args.topk,
                cpu_num = mp.cpu_count()//2,
                learned_path_len=args.learned_path_len,
                max_path_len = args.max_path_len,
                exp_path = args.model+args.exp_name+'_'+str(args.max_path_len)+'_'+str(args.learned_path_len),
                tao = args.tao,
            )
        elif dataset == 'YAGO3-10':
            opt = training_option(
                exp_name  = args.exp_name,
                #dimention
                bn_hidden_size= args.bn_hidden_size,
                batch_size = args.batch_size,
                emb_size = args.hidden_size,
                anchor = 100000,
                contrast_ratio = 0.3,
                predict_ratio = 0.1,
                none_ratio =1,
                #switch
                inverse_negative = False,
                load_path = False,
                load_contrast_model = False,
                load_predict_model = False,
                joint_train = True,
                get_hits = True,
                augment = args.augment,
                predict_mode = 'attn',
                #epoch
                contrast_n_epoch = 2, 
                predict_n_epoch = 0,
                joint_n_epoch =5,
                #learning rate
                contrast_lr = 2e-5,
                predict_lr = 1e-4,
                joint_contrast_lr = 2e-5,
                joint_predict_lr = 2e-5,
                # other
                drop_prob = args.drop if args.augment=='drop' else 0,#0.6 sota
                gamma = 0.9,
                none_drop = 0.4,
                body_len_range =list(range(args.max_path_len,args.max_path_len+1)),
                device = device,
                writer=writer,
                model = args.model,
                topk = args.topk,
                cpu_num = mp.cpu_count()-1,
                learned_path_len=args.learned_path_len,
                max_path_len = args.max_path_len,
                exp_path = args.model+args.exp_name+'_'+str(args.max_path_len)+'_'+str(args.learned_path_len),
                tao = args.tao,
            )
    return opt