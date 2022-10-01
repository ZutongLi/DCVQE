import torch.nn as nn
from scipy import stats
import torch.nn.functional as F
import torch
import numpy as np
import os
import sys
from data_readers import DataReader
from torch.utils.data import Dataset, DataLoader
from DCVQE import DCVQE
from rankingLoss import CorrelationLoss
import shutil
import utils

def Train(model, optimizer, data_loader, criterion, margin):
    '''
    training process
    '''
    L = 0
    model.train()
    cl = CorrelationLoss(margin)
    for i,(features, length, label) in enumerate(data_loader):
        features = features.cuda().float()
        label = label.cuda().float()
         
        optimizer.zero_grad()
        outputs = model(features, length.float())
        outputs = outputs.squeeze()
        loss = 0.5 *criterion(outputs, label) + 0.5 * cl.processCon2(label, outputs)
        loss.backward()
        optimizer.step()
        L = L + loss.item()
    train_loss = L / (i + 1)
    return train_loss

def Eval(model, data_loader, criterion, length_of_eval, scale):
    '''
    eval process
    '''
    L=0
    model.eval()
    y_pred = np.zeros(length_of_eval)
    y_val = np.zeros(length_of_eval)
    y_pred = []
    y_val = []
    with torch.no_grad():
        for i , (features, length, label) in enumerate(data_loader):
            for l in label:
                y_val.append(scale*l.item() )
            features = features.cuda().float()
            label = label.cuda().float()
            outputs = model(features, length.float())
            outputs = outputs.squeeze()

            for pred in outputs:
                y_pred.append(scale*pred.item())
            loss = criterion(outputs, label)
            L = L + loss.item()
    y_val = np.array(y_val)
    y_pred = np.array(y_pred)
    val_loss = L / (i + 1)
    val_PLCC = stats.pearsonr(y_pred, y_val)[0]
    val_SROCC = stats.spearmanr(y_pred, y_val)[0]
    val_RMSE = np.sqrt(((y_pred-y_val) ** 2).mean())
    val_KROCC = stats.stats.kendalltau(y_pred, y_val)[0]
    ret = {'val_loss':val_loss, 'val_PLCC':val_PLCC, 'val_SROCC':val_SROCC, \
            'val_RMSE':val_RMSE, 'val_KROCC':val_KROCC}
    return ret

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("train and eval")
    parser.add_argument('--training_data',type=str, required=True, help="input file")
    parser.add_argument('--experiment_writer', type = str, required= True, help='NA')
    parser.add_argument('--eval_data',type=str, required = True, help = 'eval file')
    parser.add_argument('--o',type=str, required=True, help="output model")
    parser.add_argument('--b',type=int, default=32, help="1 is default, since we don't do resize, so the \
                                                    batchsize must be 1, otherwise it would raise Exception")
    parser.add_argument('--lr',type=float, default=0.0001, help='lr')
    parser.add_argument('--epoch',type=int, default=3, help='epoch')
    parser.add_argument('--weight_decay', type=float, default=0.0,help='weight decay (default: 0.0)')
    parser.add_argument('--gpu',type=str, default="-1", help='epoch')
    parser.add_argument('--max_size',type=int, default=608, help='epoch')
    parser.add_argument('--decay_ratio', type=float, default=0.8,help='weight decay (default: 0.0)')
    parser.add_argument('--scale', type=float, default=4.64,help='weight decay (default: 0.0)')
    parser.add_argument('--reduced_size', type=int, default=128, help='NA')
    parser.add_argument('--max_len', type = int, default = 240, help = 'max_len')
    parser.add_argument('--feature_size', type=int, default = 4096 , help = 'NA') 
    parser.add_argument('--activate_leng',type=int,default=6,help="NA")
    parser.add_argument('--margin',type=float,default=0.0,help="NA")

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print('[ DEBUG ] :: __main__ args {}'.format(args))

    decay_interval = int(args.epoch/ 10)
    
    train_reader = DataReader(args.training_data, max_len = args.max_len, \
                                    scale = args.scale, feat_dim = args.feature_size)
    eval_reader = DataReader(args.eval_data, max_len = args.max_len, \
                                    scale = args.scale, feat_dim = args.feature_size)
    train_loader = DataLoader(train_reader, batch_size=args.b, shuffle=True, num_workers=5)
    eval_loader = DataLoader(eval_reader, batch_size=args.b, shuffle=True, num_workers=3)
    
    model = DCVQE(input_size = args.feature_size, \
                  reduced_size=args.reduced_size,\
                  max_len = args.max_len, \
                  activate_leng=args.activate_leng)

    model = nn.DataParallel(model)
    
    model = model.cuda()
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_interval, gamma=args.decay_ratio)
    best_val_criterion = 60000
    best_dct = {}
    for epo in range(args.epoch):
        Train(model, optimizer, train_loader, criterion, args.margin)
        val_dct = Eval(model, eval_loader, criterion, len(eval_reader), args.scale)
        if val_dct['val_loss'] < best_val_criterion:
            print('[ DEBUG ] update best model using best_val_criterion in epoch {}'.format(epo))
            print("[ DEBUG ] Val result, val loss {:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".
                    format(val_dct['val_loss'], val_dct['val_SROCC'], val_dct['val_KROCC'], \
                            val_dct['val_PLCC'], val_dct['val_RMSE']   ))
            torch.save(model.state_dict(), args.o)
            best_val_criterion = val_dct['val_loss']   #val_dct['val_loss']
            best_dct = val_dct
    handle = open(args.experiment_writer,'a')
    handle.write('loss:{}\tSROCC:{}\tKROCC:{}\tPLCC:{}\tRMSE:{}\n'.
                    format(best_dct['val_loss'], best_dct['val_SROCC'], best_dct['val_KROCC'],\
                           best_dct['val_PLCC'], best_dct['val_RMSE'] ))
    handle.flush()
    '''
    max_srocc = utils.find_best_performance(args.experiment_writer)
    print('[ DEBUG ] current best performance loss: {}'.format(max_srocc) )
    if float(best_dct['val_SROCC']) >= max_srocc:
        shutil.copy(args.training_data,'bestDataSplit/') 
        shutil.copy(args.eval_data, 'bestDataSplit/')
        shutil.copy(args.o, 'bestDataSplit/' + args.o)
    '''
