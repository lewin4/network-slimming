from __future__ import print_function
import os
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import models
from dataset.sewage_loader import get_loaders
from torch.utils.tensorboard import SummaryWriter
from training import get_learning_rate_scheduler, get_uncompressed_model
import time
from sklearn.metrics import classification_report
from utils.model_size import compute_model_nbits, bits_to_kb
from tqdm import tqdm


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=["cifar10", "cifar100", "sewage", "miniimagenet"],
                        help='training dataset (default: cifar10)')
    parser.add_argument('--num_classes', type=int, default=100,
                        help='training dataset (default: cifar100)')
    parser.add_argument('--image_dir', type=str, default=r'E:\LY\data\classification_aug',
                        help='training dataset path')
    parser.add_argument('--image_shape', type=list, default=[32, 32],
                        help='the shape feed to network')
    parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true', default=False,
                        help='train with channel sparsity regularization')
    parser.add_argument('--s', type=float, default=0.0001,
                        help='scale sparse rate (default: 0.0001)')
    parser.add_argument('--refine',
                        default=r'E:\LY\network-slimming\logs\vgg_cifar100_output\prune\0.6pruned.pth.tar', type=str,
                        metavar='PATH',
                        help='path to the pruned model to be fine tuned')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--epochs', type=int, default=160, metavar='N',
                        help='number of epochs to train (default: 160)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--resume', default=r"", type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrain', default='', type=str,
                        metavar='PATH',
                        help='path to pretrain model (default: none)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=10086, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save', default='./logs/vgg_cifar100_output/refine', type=str, metavar='PATH',
                        help='path to save prune model (default: current directory)')
    parser.add_argument('--arch', default='vgg', type=str,
                        help='architecture to use')
    parser.add_argument('--depth', default=101, type=int,
                        help='depth of the neural network')
    parser.add_argument('--log_interval', default=50, type=int,
                        help='interval of log train items')

    localtime = time.strftime("%m-%d-%Hh%Mm", time.localtime())

    args = parser.parse_args()
    summary_writer = SummaryWriter(args.save + "/{}/tensorboard/".format(localtime))

    if summary_writer is not None:
        summary_writer.add_text("config", str(args))
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    kwargs = {'num_workers': 2, 'pin_memory': True}
    if args.dataset == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data.cifar10', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.Pad(4),
                                 transforms.RandomCrop(32),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                             ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'cifar100':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data.cifar100', train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.Pad(4),
                                  transforms.RandomCrop(32),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                              ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'sewage':
        train_loader, test_loader, _ = get_loaders(image_dir=args.image_dir,
                                                   batch_size=args.batch_size,
                                                   img_shape=args.image_shape,
                                                   radio=[0.7, 0.2, 0.1],
                                                   **kwargs)
    elif args.dataset == "miniimagenet":
        from MLclf import MLclf
        # Download the original mini-imagenet data:
        # only need to run this line before you download the mini-imagenet dataset for the first time.
        # MLclf.miniimagenet_download(Download=False)
        # Transform the original data into the format that fits the task for classification:
        train_transform = transforms.Compose([
                                  transforms.Pad(4),
                                  transforms.RandomCrop(32),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                              ])
        # train_transform = transforms.Compose(
        #     [transforms.RandomHorizontalFlip(),
        #      transforms.Resize((32, 32)),
        #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        val_transform = transforms.Compose([
                                  # transforms.Pad(4),
                                  # transforms.RandomCrop(32),
                                  # transforms.RandomHorizontalFlip(),
                                  # transforms.ToTensor(),
                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                              ])
        # val_transform = transforms.Compose(
        #     [transforms.Resize((32, 32)),
        #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset, validation_dataset, test_dataset = MLclf.miniimagenet_clf_dataset(
            ratio_train=0.6, ratio_val=0.2,
            seed_value=None, shuffle=True,
            transform=None,
            save_clf_data=True)

        # The dataset can be transformed to dataloader via torch:
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs)
        test_loader = torch.utils.data.DataLoader(
            dataset=validation_dataset,
            batch_size=args.test_batch_size,
            shuffle=True,
            **kwargs)
    else:
        raise ValueError("No valid dataset is given.")

    if args.refine:
        checkpoint = torch.load(args.refine)
        cfg = checkpoint.get('cfg', None)
        # model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=checkpoint['cfg'])
        model = get_uncompressed_model(args.arch,
                                       pretrained=False,
                                       num_classes=args.num_classes,
                                       dataset=args.dataset,
                                       aux_logits=True,
                                       cfg=cfg)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> refine from '{}'".format(args.refine))
        del checkpoint
    else:
        # model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)
        model = get_uncompressed_model(
            args.arch,
            pretrained=False,
            dataset=args.dataset,
            num_classes=args.num_classes,
            aux_logits=True)

    print(model)

    summary_writer.add_text("model", str(model))

    compressed_model_size_bits = compute_model_nbits(model)
    model_size_kbs = bits_to_kb(compressed_model_size_bits)
    print("Model size(KB):    {}".format(model_size_kbs))

    if args.no_cuda:
        model.cpu()
    else:
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for param_group in optimizer.param_groups:
        param_group["initial_lr"] = args.lr

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            # torch.save(checkpoint["state_dict"], "state_dice.pth")
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
                  .format(args.resume, checkpoint['epoch'], best_prec1))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.pretrain:
        if os.path.isfile(args.pretrain):
            print("=> loading model '{}'".format(args.pretrain))
            checkpoint = torch.load(args.pretrain)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded model '{}' ".format(args.pretrain))
        else:
            raise Exception("=> no model found at '{}'".format(args.pretrain))

    lr_scheduler = get_learning_rate_scheduler({"lr_scheduler": {"type": "multistep", "last_epoch": args.start_epoch,
                                                                 "milestones": [80, 120], "factor": 0.1}},
                                               optimizer, args.epochs, len(train_loader))

    # additional subgradient descent on the sparsity-induced penalty term
    def updateBN() -> None:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.grad.data.add_(args.s * torch.sign(m.weight.data))  # L1

    def train(epoch: int):
        model.train()
        train_loss = 0
        pred_list = torch.Tensor()
        true_list = torch.Tensor()
        batch_num = len(train_loader)
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            if args.cuda:
                # data, target = train_transform(data.cuda()), target.cuda()
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target.long())
            optimizer.zero_grad()
            output = model(data)
            if args.arch == "googlenet":
                loss0 = F.cross_entropy(output[0], target)
                loss1 = F.cross_entropy(output[1], target)
                loss2 = F.cross_entropy(output[2], target)
                loss = 0.6*loss0 + 0.2*loss1 + 0.2*loss2
            else:
                loss = F.cross_entropy(output, target)
            if summary_writer is not None:
                summary_writer.add_scalar("Train batch loss",
                                          loss.item(),
                                          (epoch-args.start_epoch)*batch_num+(batch_idx))
            train_loss += loss.item()
            # pred = output.data.max(1, keepdim=True)[1]
            loss.backward()
            if args.arch == "googlenet":
                pred = output[0].data.max(1, keepdim=True)[1]
            else:
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            pred_list = torch.cat((pred_list, pred.squeeze().cpu()), 0)
            true_list = torch.cat((true_list, target.cpu()), 0)
            if args.sr:
                updateBN()
            optimizer.step()
            if lr_scheduler.step_batch():
                lr_scheduler.step()
            if (batch_idx % args.log_interval) == 0:
                print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss))

        report = classification_report(true_list, pred_list, labels=range(args.num_classes), digits=4, output_dict=True)
        acc = report["accuracy"]
        train_loss = train_loss / len(train_loader.dataset)
        if summary_writer is not None:
            summary_writer.add_scalar("Train epoch loss", train_loss, epoch)
            summary_writer.add_scalar("Train acc", acc, epoch)

    def test(epoch):
        model.eval()
        test_loss = 0
        correct = 0
        pred_list = torch.Tensor()
        true_list = torch.Tensor()
        for data, target in test_loader:
            if args.cuda:
                # data, target = train_transform(data.cuda()), target.long().cuda()
                data, target = data.cuda(), target.long().cuda()
            with torch.no_grad():
                output = model(data)
                test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            pred_list = torch.cat((pred_list, pred.squeeze().cpu()), 0)
            true_list = torch.cat((true_list, target.cpu()), 0)
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()  # get target have the same shape of pred

        report = classification_report(true_list, pred_list, labels=range(args.num_classes), digits=4, output_dict=True)
        acc = report["accuracy"]
        precision = report["macro avg"]["precision"]
        recall = report["macro avg"]["recall"]
        f1_score = report["macro avg"]['f1-score']
        test_loss /= len(test_loader.dataset)
        print(
            '\nTest set: Average loss: {:.4f},\nAccuracy: {:.4f},\nPrecision: {:.4f},\nRecall: {:.4f},\nf1-score: {:.4f}'.format(
                test_loss, acc, precision, recall, f1_score))
        if summary_writer is not None:
            summary_writer.add_scalar("test loss", test_loss, epoch)
            summary_writer.add_scalar("test acc", acc, epoch)
        return acc, report

    def test_fps(start_epoch, epochs):
        model.eval()
        from thop import profile, clever_format
        input = torch.randn(1, 3, 32, 32).cuda()
        macs, params = profile(model, inputs=(input,))
        macs, params = clever_format([macs, params], "%.3f")
        print(macs)
        print("="*40)
        print(params)
        total_times = 0
        print("{} epochs will be test.".format(epochs - start_epoch))
        model.cpu()
        for epoch in range(start_epoch, epochs):
            print("{} epoch start......".format(epoch))
            times = 0
            epoch_time = time.time()
            for data, target in test_loader:
                # if args.cuda:
                #     data, target = data.cuda(), target.cuda()
                start_time = time.time()
                with torch.no_grad():
                    output = model(data)
                stop_time = time.time()
                times += (stop_time - start_time)
            total_times += times
            print("{} epoch finish. time: {}. Pure inference time: {}".format(epoch, time.time() - epoch_time, times))
        num = len(test_loader.dataset) * (epochs - start_epoch)
        print(
            "\nAll time: {}, \nImage num: {}, \nTime per image: {}".format(total_times, num, total_times / float(num)))

    def save_checkpoint(state, is_best, refine, filepath):
        if refine:
            state["cfg"] = cfg
        torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
        if is_best:
            del state["optimizer"]
            torch.save(state, os.path.join(filepath, 'model_best.pth.tar'))
            # shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))

    # test_fps(args.start_epoch, args.epochs)

    # acc, report = test(0)
    # print(acc)
    # print(report)

    best_prec1 = 0.
    best_precision = 0.
    best_recall = 0.
    best_f1_score = 0.
    for epoch in range(args.start_epoch, args.epochs):
        # if epoch in [args.epochs * 0.5, args.epochs * 0.75]:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] *= 0.1
        print("Training start epoch: {}/{}".format(epoch, args.epochs))
        train(epoch)
        prec1, report = test(epoch)
        if lr_scheduler.step_epoch():
            # last_acc is between 0 and 100. We need between 0 and 1
            lr_scheduler.step(prec1)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if is_best:
            best_precision = report["macro avg"]["precision"]
            best_recall = report["macro avg"]["recall"]
            best_f1_score = report["macro avg"]['f1-score']
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.refine, filepath=args.save)

        print("Best accuracy: " + str(best_prec1))
        print("Best precision:" + str(best_precision))
        print("Best recall:" + str(best_recall))
        print("Best f1-score:" + str(best_f1_score))

    print("Best accuracy: " + str(best_prec1))
    print("Best precision:" + str(best_precision))
    print("Best recall:" + str(best_recall))
    print("Best f1-score:" + str(best_f1_score))


if __name__ == "__main__":
    main()
