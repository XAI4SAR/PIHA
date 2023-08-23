from model.MSNet import *
from model.Aconvnet import *
from model.Densenet121 import *
import torch
from torch import nn, optim
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
import warnings
import os
from tensorboardX import SummaryWriter
from Dataset import*
from tqdm import tqdm
from early_stop import EarlyStopping
from torchsummary import summary

warnings.simplefilter(action='ignore', category=FutureWarning)

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():    
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])
        return correct
    
def validate(val_loader, model):
    
    # switch to evaluate mode
    F = nn.CrossEntropyLoss()
    model.eval()
    sum = 0
    val_loss = 0
    with torch.no_grad():
        print('Validate:')

        for i, (images, ASC_part, target,_) in enumerate(tqdm(val_loader)):

            images = images.float().to(device)
            target = target.to(device)
            ASC_part = ASC_part.float().to(device)
            ASC_part = transforms.CenterCrop(64)(ASC_part).float().to(device)
            output = model(images, ASC_part)
            val_loss += F(output,target)
            result = accuracy(output, target)
            sum += result.sum()

        acc = sum/len(val_loader.dataset)
        val_loss = val_loss/len(val_loader)
    return acc, val_loss


def parameter_setting(args):
    config = {}
    config['arch'] = args.arch
    config['datatxt_train'] = args.datatxt_train
    config['datatxt_val'] = args.datatxt_val
    config['datatxt_OFA1'] = args.datatxt_OFA1
    config['datatxt_OFA2'] = args.datatxt_OFA2
    config['datatxt_OFA3'] = args.datatxt_OFA3
    config['cate_num'] = args.cate_num
    config['batch_size'] = args.batch_size
    config['num_epochs'] = args.num_epochs
    config['save_path'] = args.save_path
    config['pretrain'] = args.pretrain
    config['part_num'] = args.part_num
    config['patience'] = args.patience
    config['attention_setting'] = args.attention_setting
    config['device'] = args.device
    config['train_num'] = args.train_num
    return config

 

def get_dataloader(config, data_transforms):

    dataset_train = Mstar_ASC_part_2(config['datatxt_train'], transform=data_transforms)
# 
    dataloader = {}
    dataloader['train'] = DataLoader(dataset_train,
                                    batch_size=config['batch_size'],
                                    shuffle=True,
                                    drop_last=False,
                                    num_workers=4,
                                    )       
    
    dataset_test = Mstar_ASC_part_2(config['datatxt_val'], transform=data_transforms)

    dataloader['val'] = DataLoader(dataset_test,
                                    batch_size=config['batch_size'],
                                    shuffle=False,
                                    drop_last=False,
                                    num_workers=4,
                                    )
    
    dataset_test = Mstar_ASC_part_2(config['datatxt_test_SOC10'], transform=data_transforms)

    dataloader['SOC10'] = DataLoader(dataset_test,
                                    batch_size=config['batch_size'],
                                    shuffle=False,
                                    drop_last=False,
                                    num_workers=4,
                                    )
    
    dataset_test = Mstar_ASC_part_2(config['datatxt_test_SOC14'], transform=data_transforms)

    dataloader['SOC14'] = DataLoader(dataset_test,
                                    batch_size=config['batch_size'],
                                    shuffle=False,
                                    drop_last=False,
                                    num_workers=4,
                                    )
    
    dataset_test = Mstar_ASC_part_2(config['datatxt_test_EOCdepression'], transform=data_transforms)

    dataloader['EOCdepression'] = DataLoader(dataset_test,
                                    batch_size=config['batch_size'],
                                    shuffle=False,
                                    drop_last=False,
                                    num_workers=1,
                                    )

    assert(next(iter(dataloader['train']))[1].size()[1]==config['part_num'])
    assert(next(iter(dataloader['val']))[1].size()[1]==config['part_num'])
    assert(next(iter(dataloader['SOC10']))[1].size()[1]==config['part_num'])
    assert(next(iter(dataloader['SOC14']))[1].size()[1]==config['part_num'])
    assert(next(iter(dataloader['EOCdepression']))[1].size()[1]==config['part_num'])

    return dataloader
def load_pretrained_model(path, model):
    
    pretrain_dict = torch.load(path)
    model_dict = {}
    state_dict = model.state_dict()
    print('already loaded：')
    for k in pretrain_dict.keys():   
        if k in model.state_dict().keys():
            model_dict[k] = pretrain_dict[k]
            print(k)
    state_dict.update(model_dict)
    model.load_state_dict(state_dict)
    return model


def train(config, i):
    os.environ["CUDA_VISIBLE_DEVICES"] = config['device']
    if not os.path.exists(config['save_path']):
        os.makedirs(config['save_path'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    early_stopping = EarlyStopping(os.path.join(config['save_path'], '{}.pth'.format(i)), config['patience'])

    save_dir = config['save_path']
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    log_path = os.path.join(save_dir, 'log{}'.format(i))
    if not os.path.exists(log_path):  
        os.makedirs(log_path)

    print(torch.cuda.get_device_name(),device)
    if config['arch'] == 'MSNet':
        init_lr = 0.0005
        data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(64),      
    ])
    model_CNN = eval(config['arch'])(config['cate_num'], config['part_num'], 100, config['attention_setting'])
        
    if config['arch'] == 'Aconvnet':
        init_lr = 0.005
        data_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])
    model_CNN = eval(config['arch'])(config['cate_num'], config['part_num'], config['attention_setting'])

    if config['arch'] == 'Densenet121':
        init_lr = 0.0005
        data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(64),      
    ])
    model_CNN = eval(config['arch'])(config['cate_num'], config['part_num'], config['attention_setting'])
   
    if config['pretrain']:
        model_CNN = load_pretrained_model(config['pretrain'], model_CNN)  
    model_CNN.to(device)
    
    optimizer =  optim.SGD(filter(lambda p: p.requires_grad, model_CNN.parameters()), init_lr, momentum=0.9, weight_decay=1e-3, nesterov=True)

    dataloader = get_dataloader(config, data_transforms)

    loss_func = nn.CrossEntropyLoss()

    writer = SummaryWriter(log_path)

    for epoch in range(config['num_epochs']): 
        loss_sum = 0
        model_CNN.train()
        print('Epochs: ', epoch)
        for data, ASC_part, labels, _ in tqdm(dataloader['train']):
            
            data = data.float().to(device)
            labels = labels.to(device)
            ASC_part = transforms.CenterCrop(64)(ASC_part).float().to(device)
            output = model_CNN(data, ASC_part)
            loss2 = loss_func(output, labels)
            loss_sum += loss2
            loss = loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()      
        val_loss = 0.0
        acc1, val_loss = validate(dataloader['val'], model_CNN)
        writer.add_scalar('accuracy', acc1, (epoch+1))
        writer.add_scalars('loss', {'cls_train': loss_sum.item()/len(dataloader['train']),  
                                    'cls_val': val_loss.item(),}, epoch + 1) 
        print('{}准确率:{}'.format(epoch+1, acc1.item()))
        conuter = early_stopping(acc1, model_CNN)
        writer.add_scalar('conuter', conuter, (epoch+1))
    
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    model_CNN = load_pretrained_model(early_stopping.save_path, model_CNN)

    acc_val, _ = validate(dataloader['val'], model_CNN)
    acc_SOC10, _ = validate(dataloader['SOC10'], model_CNN)
    acc_SOC14, _ = validate(dataloader['SOC14'], model_CNN)
    acc_EOCdepression, _ = validate(dataloader['EOCdepression'], model_CNN)
    print('*******************************************************')    
    return acc_val, acc_SOC10, acc_SOC14, acc_EOCdepression, epoch

if __name__ == '__main__': 


    # os.environ ["CUDA_VISIBLE_DEVICES"] = '3'
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # save_root = './result/'
    # datatxt = '/STAT/wc/Experiment/dataset/10CLASS_15/list'
    # exp_num = '20230707_12_'     
    # train_name = 'train_10.txt'  
    # val_name = 'val_10.txt'
    # cate_num = 10
    # part_num = 4
    # arch = 'MSNet' # 'MS-Net', 'Aconvnet', 'DenseNet'
    # datatxt_train = os.path.join(datatxt, train_name)
    # datatxt_val = os.path.join(datatxt, val_name)
    # datatxt_test_SOC10 = '/STAT/wc/Experiment/dataset/10CLASS_15/list/test10.txt'
    # datatxt_test_SOC14 = '/STAT/wc/Experiment/dataset/10CLASS_15/list/test14.txt'
    # datatxt_test_EOCdepression = '/STAT/wc/Experiment/dataset/EOC_depression_6/list/test.txt'
    # attention = True
    # save_path = os.path.join(save_root, exp_num)
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
   


    # train_nums = 5
    
    # pretrain = None #'/home/hzl/STAT/code_wu/Experiment/Experiment5/pretrain_mod el/aconvnet_opensar_pretrained.pth' #'/home/hzl/STAT/code_wu/Experiment/Experiment5/pretrain_model/resnet18_opensar_pretrain.pth'

    # patience = 200
    # information = '\n'.join([ 
    #     '__file__:{}'.format(__file__),
    #     'datatxt_train:{}'.format(datatxt_train),
    #     'datatxt_test_SOC10:{}'.format(datatxt_test_SOC10),
    #     'datatxt_test_SOC14:{}'.format(datatxt_test_SOC14),
    #     'datatxt_test_EOCdepression:{}'.format(datatxt_test_EOCdepression),
    #     'arch:{}'.format(arch),
    #     'cate_num:{}'.format(cate_num),
    #     'pretrain:{}'.format(pretrain),
    #     'init_lr:{}'.format(init_lr),
    #     'part_num:{}'.format(part_num),
    #     'attention_setting:{}'.format(attention_setting),
    #     'patience:{}'.format(patience),
    # ])


    # with open(os.path.join(save_path, 'information.txt'), 'w') as f:
    #     f.write(information) 
    # dataset setting
    parser = argparse.ArgumentParser(prog='CNN_training')
    parser.add_argument('--arch', default='MSNet')
    parser.add_argument('--datatxt_train', default='data/Train/list/train_90.txt')
    parser.add_argument('--datatxt_OFA1', default='data/OFA1_2/list/OFA1.txt')
    parser.add_argument('--datatxt_OFA2', default='data/OFA1_2/list/OFA1.txt')
    parser.add_argument('--datatxt_OFA3', default='data/OFA3/list/OFA3.txt')
    parser.add_argument('--datatxt_val', default='data/Train/list/val_90.txt')
    parser.add_argument('--cate_num', type=int, default=10)
    parser.add_argument('--save_path', default='result/') 
    parser.add_argument('--pretrain', default=None)
    # training setting
    parser.add_argument('--train_num', default=1)
    parser.add_argument('--batch_size', type=int, nargs='+', default=32)

    parser.add_argument('--num_epochs', type=int, default=1000)

    parser.add_argument('--part_num', type=int, default=4)

    parser.add_argument('--patience', type=int, default=200)
    parser.add_argument('--attention_setting', default=True)
    parser.add_argument('--device', default='0')

    args = parser.parse_args()
    config = parameter_setting(args)
    val_list = []
    SOC10_list = []
    SOC14_list = []
    EOCdepression_list = []
    stop_epoch_list = []
    
    for i in range(config['train_nums']):
        val_acc, SOC10_acc, SOC14_acc, EOCdepression_acc, stop_epoch = train(config, i)
        val_list.append(str(val_acc.item()))
        SOC10_list.append(str(SOC10_acc.item())) 
        SOC14_list.append(str(SOC14_acc.item())) 
        EOCdepression_list.append(str(EOCdepression_acc.item()))
        
        stop_epoch_list.append(str(stop_epoch))

        val_result = 'val:' + '\t'.join(val_list) + '\n'
        SOC10_result = 'OFA1:' + '\t'.join(SOC10_list) + '\n'
        SOC14_result = 'OFA2:' + '\t'.join(SOC14_list) + '\n'
        EOCdepression_result = 'OFA3:' + '\t'.join(EOCdepression_list) + '\n'
        stop_epoch_result = 'stop_epoch:' + '\t'.join(stop_epoch_list) + '\n'
        
        with open(os.path.join(config['save_path'], 'result.txt'), 'w') as f:
            f.write(val_result)
            f.write(SOC10_result)
            f.write(SOC14_result)
            f.write(EOCdepression_result)
            f.write(stop_epoch_result)

        


 


