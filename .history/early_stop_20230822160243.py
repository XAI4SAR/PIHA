import numpy as np
import torch
import copy
class EarlyStopping:
    def __init__(self, save_path, patience=7, verbose=False, delta=0):

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
