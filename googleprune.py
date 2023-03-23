import os
import argparse
from torch.autograd import Variable
from torchvision import datasets, transforms
from models import GoogLeNet
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
from typing import Union, Tuple

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=19,
                    help='depth of the vgg')
parser.add_argument('--percent', type=float, default=0.2,
                    help='scale sparse rate (default: 0.5)')
parser.add_argument('--model', default='logs/googlenet_cifar10_output/model_best.pth.tar', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='./logs/googlenet_cifar10_output/prune', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)


# simple test model after Pre-processing prune (simple set BN scales to zeros)
def test(test_model):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'cifar10':
        dataset = datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
        test_loader = DataLoader(
            dataset,
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'cifar100':
        dataset = datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
        test_loader = DataLoader(
            dataset,
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    else:
        raise ValueError("No valid dataset is given.")
    test_model.eval()
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = test_model(data)
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(dataset), 100. * correct / len(dataset)))
    return correct / float(len(dataset))


def get_inception_branch_name(layer_name: str) -> Tuple[Union[str, None], Union[str, None]]:
    name_list = layer_name.split(".")
    inception = None
    branch = None
    for word in name_list:
        if "inception" in word:
            inception = word
        if "branch" in word:
            branch = word
    return inception, branch


if __name__ == "__main__":
    if args.dataset == "cifar10":
        num_classes = 10
    elif args.dataset == "sewage":
        num_classes = 6
    elif args.dataset == "miniimagenet":
        num_classes = 100
    else:
        raise Exception
    model = GoogLeNet(num_classes=num_classes, aux_logits=True)
    if args.cuda:
        model.cuda()

    if args.model:
        if os.path.isfile(args.model):
            print("=> loading checkpoint '{}'".format(args.model))
            checkpoint = torch.load(args.model)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
                  .format(args.model, checkpoint['epoch'], best_prec1))
        else:
            raise Exception("=> no checkpoint found at '{}'".format(args.model))

    # print(model)
    # 剪枝前的准确率
    print("剪枝前的准确率:")
    acc = test(model)

    # 总通道数
    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]

    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index + size)] = m.weight.data.abs().clone()
            index += size

    y, i = torch.sort(bn)
    thre_index = int(total * args.percent)
    thre = y[thre_index]

    pruned = 0
    cfg = []
    cfg_mask = []
    for k, m in enumerate(model.named_modules()):
        name = m[0]
        m = m[1]
        if isinstance(m, nn.BatchNorm2d):
            if "aux" in name:
                continue
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre).float().cuda()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                  format(k, mask.shape[0], int(torch.sum(mask))))

    pruned_ratio = pruned / total

    print('Pre-processing Successful!')
    # 预处理后的准确率
    print("预处理后的准确率:")
    acc = test(model)

    # Make real prune
    print(cfg)
    newmodel = GoogLeNet(num_classes=num_classes, aux_logits=True, cfg=cfg)
    if args.cuda:
        newmodel.cuda()

    # 剪枝后空模型的准确率
    print("剪枝后空模型的准确率:")
    acc = test(newmodel)

    num_parameters = sum([param.nelement() for param in newmodel.parameters()])
    savepath = os.path.join(args.save, str(args.percent) + "prune.txt")
    with open(savepath, "w") as fp:
        fp.write("Configuration: \n" + str(cfg) + "\n")
        fp.write("Number of parameters: \n" + str(num_parameters) + "\n")
        fp.write("Test accuracy: \n" + str(acc) + "\n")
        fp.write("cfg: \n" + str(cfg))

    # 转移参数
    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    inception_name_start_mask = {}
    branch_list = []
    inception_in_channel = torch.Tensor(0)
    for [m0, m1] in zip(model.named_modules(), newmodel.named_modules()):
        m0name = m0[0]
        m0 = m0[1]
        m1name = m1[0]
        m1 = m1[1]
        if isinstance(m0, nn.BatchNorm2d):
            if "aux" not in m0name:
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                m1.weight.data = m0.weight.data[idx1.tolist()].clone()
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                m1.running_mean = m0.running_mean[idx1.tolist()].clone()
                m1.running_var = m0.running_var[idx1.tolist()].clone()
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                    end_mask = cfg_mask[layer_id_in_cfg]

        elif isinstance(m0, nn.Conv2d):
            inception_name, branch_name = get_inception_branch_name(m0name)

            if "aux1" in m0name:
                idx0 = np.squeeze(np.argwhere(np.asarray(inception_name_start_mask["inception4b"].cpu().numpy())))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                m1.weight.data = w1.clone()
            elif "aux2" in m0name:
                idx0 = np.squeeze(np.argwhere(np.asarray(inception_name_start_mask["inception4e"].cpu().numpy())))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                m1.weight.data = w1.clone()
            else:
                # 新进入一个branch
                if branch_name is not None and branch_name not in branch_list:
                    # 新进入一个inception
                    if inception_name is not None and inception_name not in inception_name_start_mask:
                        if inception_name == "inception3a":
                            inception_name_start_mask[inception_name] = start_mask.clone()
                        else:
                            inception_name_start_mask[inception_name] = inception_in_channel.clone()
                            inception_in_channel = torch.Tensor(0)
                    else:
                        inception_in_channel = torch.cat((inception_in_channel, start_mask.cpu()), 0)
                    branch_list.append(branch_name)
                    start_mask = inception_name_start_mask[inception_name]

                if branch_name == "branch4":
                    inception_in_channel = torch.cat((inception_in_channel, end_mask.cpu()), 0)
                    branch_list.clear()  # 清空分支列表

                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.weight.data = w1.clone()
        elif isinstance(m0, nn.Linear):
            if "aux" not in m0name:
                idx0 = np.squeeze(np.argwhere(np.asarray(inception_in_channel.cpu().numpy())))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                m1.weight.data = m0.weight.data[:, idx0].clone()
                m1.bias.data = m0.bias.data.clone()

    torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()},
               os.path.join(args.save, str(args.percent) + 'pruned.pth.tar'))

    print(newmodel)

    # 剪枝后的准确率
    print("剪枝后的准确率:")
    test(newmodel)
