import torch
import torch.optim as optim
import torch.nn as nn
import os, shutil
import json

from tools.MvcnnDataset import SingleImgDataset, MultiviewImgDataset
from tools.Trainer import ModelNetTrainer
from models.GVCNN import SVCNN, GVCNN

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="GVCNN")
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage",
                    default=8)  # it will be *12 images in each batch for mvcnn
parser.add_argument("-lr", type=float, help="learning rate", default=5e-5)
parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.001)
parser.add_argument("-cnn_name", "--cnn_name", type=str, help="cnn model name", default="inception")
parser.add_argument("-num_views", type=int, help="number of views", default=12)
# parser.add_argument("-train_path", type=str, default="./train")
# parser.add_argument("-val_path", type=str, default="./val")
parser.set_defaults(train=False)

args = parser.parse_args()
args.single_train_path = 'train_single_3d.json'
args.single_test_path = 'test_single_3d.json'
args.multi_train_path = 'train_3d.json'
args.multi_test_path = 'test_3d.json'
args.pretraining = True
pretraining = args.pretraining


def create_folder(log_dir):
    # make summary folder
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        print('WARNING: summary folder already exist!!')
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)
    return log_dir


def train_3d_single():
    # STAGE 1
    print('Stage_1 begin:')
    
    log_dir = args.name + '_stage_1'
    create_folder(log_dir)

    svcnn = SVCNN(args.name, nclasses=21, pretraining=True, cnn_name=args.cnn_name)
    optimizer = optim.Adam(svcnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_file = open(args.single_train_path)
    train_list = json.load(train_file)
    train_dataset = SingleImgDataset(train_list, scale_aug=False, rot_aug=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

    test_file = open(args.single_test_path)
    test_list = json.load(test_file)
    val_dataset = SingleImgDataset(test_list, scale_aug=False, rot_aug=False, test_mode=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    print('num_train_files: ' + str(len(train_dataset.data_list)))
    print('num_val_files: ' + str(len(val_dataset.data_list)))
    trainer = ModelNetTrainer(svcnn, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'svcnn', log_dir,
                              num_views=1)
    trainer.train(60)
    return svcnn


def train_3d_multi(svcnn):
    print('Stage_2 begin:')
    log_dir = args.name + '_stage_2'
    create_folder(log_dir)
    gvcnn = GVCNN(args.name, svcnn, nclasses=21, num_views=args.num_views)
    del svcnn

    optimizer = optim.Adam(gvcnn.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))

    train_file = open(args.multi_train_path)
    train_list = json.load(train_file)
    train_dataset = MultiviewImgDataset(train_list, scale_aug=False, rot_aug=False,
                                        num_views=args.num_views)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True,
                                               num_workers=0)

    test_file = open(args.multi_test_path)
    test_list = json.load(test_file)
    val_dataset = MultiviewImgDataset(test_list, scale_aug=False, rot_aug=False, num_views=args.num_views)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False, num_workers=0)
    print('num_train_files: ' + str(len(train_dataset.data_list)))
    print('num_val_files: ' + str(len(val_dataset.data_list)))
    trainer = ModelNetTrainer(gvcnn, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'gvcnn', log_dir,
                              num_views=args.num_views)
    trainer.train(300)


if __name__ == '__main__':
    log_dir = args.name
    create_folder(args.name)

    config_f = open(os.path.join(log_dir, 'config.json'), 'w')
    json.dump(vars(args), config_f)
    config_f.close()

    svcnn = train_3d_single()
    train_3d_multi(svcnn)
