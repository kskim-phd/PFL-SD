import os
import torch
import numpy as np
from datasets import dataloaders_test
from tqdm import tqdm
import argparse

currentpath = os.path.dirname(os.path.abspath(__file__)).split('/')
rootpath = '/'.join(currentpath[:-1])
root_current = '/'.join(currentpath)

parser = argparse.ArgumentParser()
parser.add_argument("--query_shot",help="number of query shot during evaluation",choices=['query_shot1'])
parser.add_argument("--fold",help="fold number",type=int)
args = parser.parse_args()

if args.query_shot == 'query_shot1': 
    query_shot = 30

fold = args.fold

def get_score(acc_list):

    mean = np.mean(acc_list)
    interval = 1.96*np.sqrt(np.var(acc_list)/len(acc_list))
    return mean,interval


def meta_test(data_path,model,way,shot,pre,transform_type,query_shot=query_shot,trial=10000,return_list=False):
    eval_loader = dataloaders_test.meta_test_dataloader(data_path=data_path,
                                                way=way,
                                                shot=shot,
                                                pre=pre,
                                                transform_type=transform_type,
                                                query_shot=query_shot,
                                                trial=trial)
    
    target = torch.LongTensor([i//query_shot for i in range(query_shot*way)]).cuda()

    acc_list = []
    
    for i, (inp,_) in tqdm(enumerate(eval_loader)):

        inp = inp.cuda()
        pred,max_index = model.meta_test(inp,way=way,shot=shot,query_shot=query_shot)
        
        pred = np.array(pred.cpu())
        max_i = np.array(max_index.cpu())
        if args.query_shot == 'query_shot1':
            save_dir = rootpath+'/evaluation/{}fold/'.format(fold)
            os.makedirs(save_dir,exist_ok=True)
            np.save(save_dir + 'pred1_150sec.npy'.format(fold),pred)
            np.save(save_dir + 'max_index1_150sec.npy'.format(fold),max_i)

        acc = 100*torch.sum(torch.eq(max_index,target)).item()/query_shot/way
        acc_list.append(acc)
   
    if return_list:
        return np.array(acc_list)
    else:
        mean,interval = get_score(acc_list)
        return mean,interval