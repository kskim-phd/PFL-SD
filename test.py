import os
import torch
from models.FRN import FRN
from utils import util
from trainers.eval_test import meta_test
import argparse

currentpath = os.path.dirname(os.path.abspath(__file__)).split('/')
rootpath = '/'.join(currentpath[:-1])
root_current = '/'.join(currentpath)

parser = argparse.ArgumentParser()
# parser.add_argument("--query_shot",help="number of query shot during evaluation",choices=['query_shot1','query_shot2','query_shot3'])
parser.add_argument("--query_shot",help="number of query shot during evaluation",choices=['query_shot1'])
parser.add_argument("--fold",help="fold number",type=int)
args = parser.parse_args()

fold = args.fold

test_path = os.path.join(root_current+'/preprocessed_data/{}fold/logmel_spectrogram/test'.format(fold)) #test data 위치
model_path = root_current+'/trained_model_weights/ver2/2train/logmel_150sec_128size_{}fold.pth'.format(fold) #trained weight 위치

gpu = 0
torch.cuda.set_device(gpu)

model = FRN(resnet=True)
model.cuda()
model.load_state_dict(torch.load(model_path,map_location=util.get_device_map(gpu)),strict=True)
model.eval()

with torch.no_grad():
    way = 2
    for shot in [60]:    #number of support data per class  (ex) 30patch x 2patient = 60)
        mean,interval = meta_test(data_path=test_path,
                                model=model,
                                way=way,
                                shot=shot,
                                pre=False,
                                transform_type=None,
                                trial=1)
        print('%d-way-%d-shot acc: %.3f\t%.3f'%(way,shot,mean,interval))