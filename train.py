import os
import torch
from functools import partial
from trainers import trainer, frn_train
from datasets import dataloaders
from models.FRN import FRN
from utils import util
import os

currentpath = os.path.dirname(os.path.abspath(__file__)).split('/')
rootpath = '/'.join(currentpath[:-1])
root_current = '/'.join(currentpath)

args = trainer.train_parser()
args.opt = 'sgd'     
args.lr = 1e-2        
args.gamma = 1e-1
args.epoch = 500
args.decay_epoch = 50,70,120,180,300,400
args.val_epoch = 20
args.weight_decay = 5e-4
args.nesterov = True
args.no_val = True
args.seed = 42   #42
args.train_transform_type = None
args.test_transform_type = None
args.resnet = True
args.train_shot = 60      #number of support data per class (train)
args.train_way = 2
args.test_shot = [60]      #number of support data per class (test)
args.pre = False
args.gpu = 0

fewshot_path = os.path.join(root_current+"/preprocessed_data/{}fold/logmel_spectrogram/".format(args.fold))   #train data path



pm = trainer.Path_Manager(fewshot_path=fewshot_path,args=args)

train_way = args.train_way
shots = [args.train_shot, args.train_query_shot]

train_loader = dataloaders.meta_train_dataloader(data_path=pm.train,
                                                way=train_way,
                                                shots=shots,
                                                transform_type=args.train_transform_type)

model = FRN(way=train_way,
            shots=[args.train_shot, args.train_query_shot],
            resnet=args.resnet)


pretrained_model_path = root_current+"/pretrained_miniImagenet_model.pth" #pretrained weight load 위치

model.load_state_dict(torch.load(pretrained_model_path,map_location=util.get_device_map(args.gpu)),strict=False)

train_func = partial(frn_train.default_train,train_loader=train_loader)

tm = trainer.Train_Manager(args,path_manager=pm,train_func=train_func)

tm.train(model)

tm.evaluate(model)

save_dir = root_current+'/trained_model_weights/new_weights/2train/'    #weight 저장위치
os.makedirs(save_dir,exist_ok=True)
torch.save(model.state_dict(), save_dir + 'logmel_150sec_128size_{}fold.pth'.format(args.fold))
