import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import os
import shutil
import argparse
import numpy as np

from helper import *
import models

from utils import cal_param_size, cal_multi_adds, adjust_lr, AverageMeter, save_dict_to_json, accuracy, correct_num
from dataloader.dataloaders import load_dataset

import time


parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
# Setup Parameters
parser.add_argument('--gpu-id', type=str, default='2')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=128, help='batch size')
parser.add_argument('--num-workers', type=int, default=8, help='number of workers')

# Data Parameters
parser.add_argument('--dataset', default='cifar100', type=str, help='Dataset')
parser.add_argument('--data', default='./data/', type=str, help='Dataset directory')
parser.add_argument('--data-aug', default='None', type=str, help='extra data augmentation')

# Model Parameters
parser.add_argument('--arch', default='CIFAR_ResNet18', type=str, help='network architecture')
parser.add_argument('--embed_dim', default=128, type=int, help='Embedding dimensionality of the network. Note: dim = 64, 128 or 512 is used in most papers, depending on the architecture.')

# Optimization Parameters
parser.add_argument('--optimizer', default='sgd', help='Optimizer setting')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--lr-type', default='multistep', type=str, help='learning rate strategy: multistep or cosine')
parser.add_argument('--init-lr',  type=float, default=0.1, help='learning rate')
parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
parser.add_argument('--milestones', default=[100, 150], type=list, help='milestones for lr-multistep')

parser.add_argument('--method', default='Smooth', type=str, help='method')
parser.add_argument('--warmup-epoch', default=5, type=int, help='warmup epoch')
parser.add_argument('--print-freq', type=float, default=50, help='print frequency')
parser.add_argument('--exp_id', type=str, default='1', help='experiment id')

# weight for losses
parser.add_argument('--weight-cls', type=float, default=1, help='weight for cross-entropy loss')
parser.add_argument('--weight-div', type=float, default=0.8, help='weight for adaptive self-distillation loss')
parser.add_argument('--weight-con', type=float, default=0.01, help='weight for class-wise matching loss')
# KD temperature
# parser.add_argument('--kd_T', type=float, default=4, help='temperature for Knowledge distillation')
parser.add_argument('--con_temp', type=float, default=0.3, help='temperature for contrastive learning')

parser.add_argument('--manual_seed', type=int, default=0)
parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=str, help='saved checkpoint directory')
parser.add_argument('--eval-checkpoint', default='./checkpoint/resnet18_best.pth', type=str,
                    help='evaluate checkpoint directory')
parser.add_argument('--resume-checkpoint', default='./checkpoint/resnet18.pth', type=str,
                    help='resume checkpoint directory')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--evaluate', '-e', action='store_true', help='evaluate model')
parser.add_argument('--trick', action='store_true', help='select pre_prediction ways')

# global hyperparameter set
args = parser.parse_args()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

info = str(os.path.basename(__file__).split('.')[0]) \
       + '_Dataset_' + args.dataset \
       + '_Arch_' + args.arch \
       + '_Method_' + args.method \
       + '_W_div_' + str(args.weight_div) \
       + '_W_con_' + str(args.weight_con) \
       + '_T_con_' + str(args.con_temp) \
       + '_' + args.exp_id
update_freq = 0.8
args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset)
args.checkpoint_dir = os.path.join(args.checkpoint_dir, str(os.path.basename(__file__).split('.')[0]))
args.checkpoint_dir = os.path.join(args.checkpoint_dir, info)
if not os.path.isdir(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)
args.log_txt = os.path.join(args.checkpoint_dir, info + '.txt')
args.configs = os.path.join(args.checkpoint_dir, 'test_best_metrics' + '.json')
print('dir for checkpoint:', args.checkpoint_dir)

np.random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)

trainloader, valloader = load_dataset(args=args)

print('-----------------------------------------')
print('Dataset: ' + args.dataset)
if args.dataset.startswith('CIFAR'):
    num_classes = len(set(trainloader.dataset.targets))
else:
    num_classes = len(set(trainloader.dataset.classes))

print('Number of train dataset: ', len(trainloader.dataset))
print('Number of validation dataset: ', len(valloader.dataset))
print('Number of classes: ', num_classes)
print('-----------------------------------------')
C, H, W = trainloader.dataset[0][0][0].size() if isinstance(trainloader.dataset[0][0], list) is True else \
trainloader.dataset[0][0].size()
# --------------------------------------------------------------------------------------------

# Model
print('==> Building model..')
model = getattr(models, args.arch)
net = model(num_classes=num_classes).eval()

print('Arch: %s, Params: %.2fM, Multi-adds: %.2fG'
      % (args.arch, cal_param_size(net) / 1e6, cal_multi_adds(net, (1, C, H, W)) / 1e9))

data = torch.zeros((1, C, H, W))
_, f_dim = net.forward(data, embedding=True)
args.f_dim = f_dim.size(1)

del (net)
net = model(num_classes=num_classes).cuda()
net = torch.nn.DataParallel(net)
cudnn.benchmark = True

Color = Colorer.instance()

# Training
def train(epoch, criterion_list, optimizer):
    train_loss = AverageMeter('train_loss', ':.4e')
    train_loss_cls = AverageMeter('train_loss_cls', ':.4e')
    train_loss_con = AverageMeter('train_loss_con', ':.4e')
    top1 = AverageMeter('train_top1', ':.4e')
    top5 = AverageMeter('train_top5', ':.4e')

    n_batch = len(trainloader)
    emb_dim = args.f_dim
    epoch_proto_feature = torch.zeros(num_classes, emb_dim).cuda()
    num_proto = torch.zeros(num_classes)

    if epoch >= args.warmup_epoch:
        lr = adjust_lr(optimizer, epoch, args)
    start_time = time.time()
    criterion_cls = criterion_list[0]

    if epoch == 0:
        pre_proto_feat = None
        coslogits = torch.zeros(len(trainloader.dataset), num_classes, dtype=torch.float32)
        prelogits = torch.zeros(len(trainloader.dataset), num_classes, dtype=torch.float32)
    else:
        pre_proto_feat = torch.load(os.path.join(args.checkpoint_dir, 'proto_predictions.pth.tar'), map_location=torch.device('cpu'))['proto_feats']
        coslogits = torch.load(os.path.join(args.checkpoint_dir, 'proto_predictions.pth.tar'), map_location=torch.device('cpu'))['coslogits']
        prelogits = torch.load(os.path.join(args.checkpoint_dir, 'proto_predictions.pth.tar'), map_location=torch.device('cpu'))['prelogits']

    net.train()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        batch_start_time = time.time()

        if isinstance(inputs, list) is False:
            inputs = inputs.cuda()
            batch_size = inputs.size(0)
        else:
            batch_size = inputs[0].size(0)

        if isinstance(targets, list) is False:
            targets = targets.cuda()
        else:
            input_indices = targets[1].cuda()
            targets = targets[0].cuda()

        if epoch < args.warmup_epoch:
            lr = adjust_lr(optimizer, epoch, args, batch_idx, len(trainloader))

        logit, feature = net(inputs, embedding=True)

        batch_proto_feature = torch.zeros(num_classes, emb_dim).cuda()
        batch_class = torch.zeros(num_classes)
        feature = F.normalize(feature, p=2, dim=1)


        for i in range(batch_size):
            class_index = targets[i]
            batch_proto_feature[class_index] = batch_proto_feature[class_index] + feature[i]
            epoch_proto_feature[class_index] = epoch_proto_feature[class_index] + feature[i]
            num_proto[class_index] += 1
            batch_class[class_index] += 1

        sample_index = np.unique(targets.cpu().detach())
        for index in sample_index:
            batch_proto_feature[index] = batch_proto_feature[index] / batch_class[index]

        loss, loss_cls, loss_con, coslogits, prelogits = IPASD(feature, logit, input_indices, targets, pre_proto_feat, criterion_cls, num_classes, coslogits, prelogits, epoch, args)

        # ===================Losses=====================
        train_loss.update(loss.item(), batch_size)
        train_loss_cls.update(loss_cls.item(), batch_size)
        train_loss_con.update(loss_con.item(), batch_size)

        # ===================Metrics=====================
        acc_top1, acc_top5 = accuracy(logit, targets, topk=(1, 5))
        top1.update(acc_top1.item(), inputs.size(0))
        top5.update(acc_top5.item(), inputs.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.print_freq == 0:
            print('Train Epoch:{}, batch_idx:{}/{}, lr:{:.5f}, Acc:{:.4f}, Duration:{:.2f}'
                  .format(epoch, batch_idx, n_batch, lr,
                          top1.avg, time.time() - batch_start_time
                          ))

    for i in range(num_classes):
        epoch_proto_feature[i] /= num_proto[i]


    train_info = 'Epoch:{}\t lr:{:.5f}\t duration:{:.3f}' \
                 '\n[Train] train_loss:{:.5f}\t' \
                 ' train_loss_cls:{:.5f}\t' \
                 ' train_loss_con:{:.5f}\n' \
                 '\t\t train Top1_acc: {:.4f}\t '\
                  ' train Top5_acc:{:.4f}' \
        .format(epoch, lr, time.time() - start_time,
                train_loss.avg, train_loss_cls.avg, train_loss_con.avg,
                top1.avg, top5.avg)
    print(Color.green(train_info))
    print(Color.green("[Update] proto feature!"))
    update_proto_feat = epoch_proto_feature

    # Store proto_feats and predictions
    proto_state = {
        'coslogits': coslogits.cpu(),
        'prelogits': prelogits.cpu(),
        'proto_feats': update_proto_feat.cpu()
    }
    torch.save(proto_state, os.path.join(args.checkpoint_dir, 'proto_predictions.pth.tar'))



def test(epoch, criterion_list):
    test_loss = AverageMeter('test_loss', ':.4e')
    top1 = AverageMeter('test_top1', ':.4e')
    top5 = AverageMeter('test_top5', ':.4e')

    criterion_cls = criterion_list[0]
    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs = inputs.cuda()
            targets = targets.cuda()

            logit = net(inputs)

            if isinstance(logit, list) or isinstance(logit, tuple):
                logit = logit[0]
            loss_cls = criterion_cls(logit, targets)

            test_loss.update(loss_cls.item(), inputs.size(0))

            acc_top1, acc_top5 = accuracy(logit, targets, topk=(1, 5))
            top1.update(acc_top1.item(), inputs.size(0))
            top5.update(acc_top5.item(), inputs.size(0))

            if batch_idx % args.print_freq == 0:
                print('Test Epoch:{}, batch_idx:{}/{}, Acc:{:.4f}'.format(epoch, batch_idx, len(valloader), top1.avg))

    test_info = '[Test] test_loss:{:.5f}\t test Top1_acc:{:.4f}\t test Top5_acc:{:.4f}\n' \
        .format(test_loss.avg, top1.avg, top5.avg)
    print(Color.green(test_info))

    return top1.avg


if __name__ == '__main__':
    best_acc = 0.  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    start_epoch_time = time.time()

    criterion_cls = nn.CrossEntropyLoss()

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.cuda()

    if args.evaluate:
        print('load trained weights from ' + args.eval_checkpoint)
        checkpoint = torch.load(args.eval_checkpoint_dir,
                                map_location=torch.device('cpu'))
        net.module.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        top1_acc = test(start_epoch, criterion_list)
    else:
        trainable_list = nn.ModuleList([])
        trainable_list.append(net)
        optimizer = optim.SGD(trainable_list.parameters(), lr=args.init_lr, momentum=args.momentum,
                              weight_decay=args.weight_decay, nesterov=True)

        if args.resume:
            args.resume_checkpoint = os.path.join(args.checkpoint_dir, model.__name__ + '.pth.tar')
            print(Color.green('[!] Resume from ' + args.resume_checkpoint))
            checkpoint = torch.load(args.resume_checkpoint, map_location=torch.device('cpu'))
            net.module.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch'] + 1

        for epoch in range(start_epoch, args.epochs):
            train(epoch, criterion_list, optimizer)
            acc = test(epoch, criterion_list)

            state = {
                'net': net.module.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, os.path.join(args.checkpoint_dir, model.__name__ + '.pth.tar'))

            is_best = False
            if best_acc < acc:
                best_acc = acc
                best_epoch = epoch
                is_best = True

            test_merics = {
                'epoch': epoch,
                'test_acc': acc,
                'best_acc': best_acc,
                }
            save_dict_to_json(test_merics, args.configs)

            if is_best:
                shutil.copyfile(os.path.join(args.checkpoint_dir, model.__name__ + '.pth.tar'),
                                os.path.join(args.checkpoint_dir, model.__name__ + '_best.pth.tar'))

        print(Color.yellow('Evaluate the best model:'))
        args.evaluate = True
        checkpoint = torch.load(args.checkpoint_dir + '/' + model.__name__ + '_best.pth.tar',
                                map_location=torch.device('cpu'))
        net.module.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        top1_acc = test(start_epoch, criterion_list)

