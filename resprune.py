import os
import argparse
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from models import *
from dataset import get_loaders, PM_dataset
from training import get_uncompressed_model

def main():
    # Prune settings
    parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='training dataset (default: cifar10)')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--image_dir', type=str, default=r'E:\LY\data\classification_aug',
                            help='training dataset path')
    parser.add_argument('--image_shape', type=list, default=[192, 256],
                        help='the shape feed to network')
    parser.add_argument('--test-batch-size', type=int, default=8, metavar='N',
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--depth', type=int, default=20,
                        help='depth of the resnet')
    parser.add_argument('--percent', type=float, default=0.8,
                        help='scale sparse rate (default: 0.5)')
    parser.add_argument('--model', default=r'E:\LY\network-slimming\logs\resnet34_cifar10_output\04-23-17h40m_0.8684\model_best.pth.tar', type=str, metavar='PATH',
                        help='path to the model (default: none)')
    parser.add_argument('--save', default='logs/resnet34_cifar10_output/prune', type=str, metavar='PATH',
                        help='path to save pruned model (default: none)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # model = resnet(depth=args.depth, dataset=args.dataset)
    model = get_uncompressed_model("resnet34", pretrained=False, num_classes=args.num_classes)

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
            # print("=> no checkpoint found at '{}'".format(args.model))

    def test(model):
        kwargs = {'num_workers': 4, 'pin_memory': True}
        if args.dataset == 'cifar10':
            test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
                batch_size=args.test_batch_size, shuffle=False, **kwargs)
        elif args.dataset == 'cifar100':
            test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
                batch_size=args.test_batch_size, shuffle=False, **kwargs)
        elif args.dataset == 'sewage':
            _, test_loader, _ = get_loaders(image_dir=args.image_dir,
                                            batch_size=args.test_batch_size,
                                            img_shape=args.image_shape,
                                            **kwargs)
        elif args.dataset == "PM":
            test_dataset = PM_dataset(
                "PM/PALM-Training400",
                "PM/PALM-Validation400",
                train=False,
                transform=transforms.Compose([
                    transforms.Resize((1024, 1024)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ]),
            )
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=args.test_batch_size,
                shuffle=True,
                **kwargs,
            )
        else:
            raise ValueError("No valid dataset is given.")
        model.eval()
        correct = 0
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            with torch.no_grad():
                output = model(data)
                pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
            correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
        return correct / float(len(test_loader.dataset))

    acc = test(model)

    total = 0

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]

    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
            index += size

    y, i = torch.sort(bn)
    thre_index = int(total * args.percent)
    thre = y[thre_index]


    cfg = []
    cfg_mask = []
    model_modules = list(model.modules())
    for layer_id in range(len(model_modules)):
        m = model_modules[layer_id]
        if isinstance(m, nn.BatchNorm2d) and (isinstance(model_modules[layer_id+1], channel_selection) or isinstance(model_modules[layer_id+1], nn.Conv2d)):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre).float().cuda()
            if torch.sum(mask) == 0:
                bn_max = weight_copy.max()
                mask = weight_copy.ge(bn_max).float()
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            print('layer index: {:d}\t\ttotal channel: {:d}\t\tremaining channel: {:d}'.
                format(layer_id, mask.shape[0], int(torch.sum(mask))))
        # elif isinstance(m, nn.MaxPool2d):
        #     cfg.append('M')


    print('Pre-processing Successful!')

    # simple test model after Pre-processing prune (simple set BN scales to zeros)


    acc = test(model)

    print("Cfg:")
    print(cfg)

    newmodel = get_uncompressed_model("resnet34", pretrained=False, num_classes=args.num_classes, cfg=cfg)
    if args.cuda:
        newmodel.cuda()

    num_parameters = sum([param.nelement() for param in model.parameters()])
    pruned_num_parameters = sum([param.nelement() for param in newmodel.parameters()])
    pruned_ratio = (pruned_num_parameters / num_parameters)

    savepath = os.path.join(args.save, "{}prune.txt".format(args.percent))

    old_modules = list(model.modules())
    new_modules = list(newmodel.modules())
    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    conv_count = 0

    for layer_id in range(len(old_modules)):
        m0 = old_modules[layer_id]
        m1 = new_modules[layer_id]
        if isinstance(m0, nn.BatchNorm2d) and (isinstance(model_modules[layer_id+1], channel_selection) or isinstance(model_modules[layer_id+1], nn.Conv2d)):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))

            if isinstance(old_modules[layer_id + 1], channel_selection):
                # If the next layer is the channel selection layer, then the current batchnorm 2d layer won't be pruned.
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()

                # We need to set the channel selection layer.
                m2 = new_modules[layer_id + 1]
                m2.indexes.data.zero_()
                m2.indexes.data[idx1.tolist()] = 1.0

                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):
                    end_mask = cfg_mask[layer_id_in_cfg]
            else:
                m1.weight.data = m0.weight.data[idx1.tolist()].clone()
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                m1.running_mean = m0.running_mean[idx1.tolist()].clone()
                m1.running_var = m0.running_var[idx1.tolist()].clone()
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                    end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d):
            if conv_count == 0:
                m1.weight.data = m0.weight.data.clone()
                conv_count += 1
                continue
            if isinstance(old_modules[layer_id-1], channel_selection) or isinstance(old_modules[layer_id-1], nn.BatchNorm2d):
                # This convers the convolutions in the residual block.
                # The convolutions are either after the channel selection layer or after the batch normalization layer.
                conv_count += 1
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()

                # If the current convolution is not the last convolution in the residual block, then we can change the
                # number of output channels. Currently we use `conv_count` to detect whether it is such convolution.
                if conv_count % 2 != 1:
                    w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.weight.data = w1.clone()
                continue

            # We need to consider the case where there are downsampling convolutions.
            # For these convolutions, we just copy the weights.
            m1.weight.data = m0.weight.data.clone()
        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))

            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()

    torch.save(
        {'cfg': cfg, 'state_dict': newmodel.state_dict()},
        os.path.join(args.save, '{}pruned.pth.tar'.format(args.percent))
    )

    # print(newmodel)
    model = newmodel
    pruned_acc = test(model)
    with open(savepath, "w") as fp:
        fp.write("Configuration: \n" + str(cfg) + "\n")
        fp.write("Number of parameters: \n" + str(num_parameters) + "\n")
        fp.write("Test accuracy: \n" + str(acc) + "\n")
        fp.write("Number of new parameters: \n" + str(pruned_num_parameters) + "\n")
        fp.write("Test accuracy: \n" + str(pruned_acc) + "\n")
        fp.write("Prune radio: \n" + str(pruned_ratio) + "\n")


if __name__ == "__main__":
    main()