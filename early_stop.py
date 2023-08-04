import numpy as np
import torch
import os
from queue import Queue
import copy
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, patience=7, verbose=False, delta=0):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path

        self.patience = patience
        self.verbose = verbose
        self.counter = 0

        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_model = None
        self.best_acc = None
      
    def __call__(self, acc, model):
        if self.best_acc is None:
            self.best_acc = acc
            self.save_checkpoint(model)
            self.best_model = copy.deepcopy(model)
        elif acc == 1:
            self.early_stop = True
            self.best_acc = acc
            self.save_checkpoint(model)
            self.best_model = copy.deepcopy(model)
        elif acc <= self.best_acc + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_acc = acc
            self.save_checkpoint(model)
            self.best_model = copy.deepcopy(model)
            self.counter = 0
        return self.counter
    
    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.save_path)	# 这里会存储迄今最优模型的参数

class EarlyStopping_2: # 这个是别人写的工具类，大家可以把它放到别的地方
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, save_path, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path


    def __call__(self, val_loss, model):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return self.counter

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.save_path)	# 这里会存储迄今最优模型的参数


        

