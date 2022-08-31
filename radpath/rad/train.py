import os
import csv
import argparse
import json
import numpy as np
import pandas as pd
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.utils as utils
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.networks.nets import DenseNet121, DenseNet169, Classifier

from utilities import metrics_all
from utils import EarlyStopping, Accuracy_Logger
from data import data_create



def parse_args():
    parser = argparse.ArgumentParser(description="Rad_densenet")

    # model config
    parser.add_argument("-b", "--batch_size", type=int, default=10, help="batch size")
    parser.add_argument("-e", "--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="initial learning rate")
    parser.add_argument("--weighted_loss", action='store_true', help='turn on weighted loss') 
    parser.add_argument("--model", type=str, default='densenet121', choices=['densenet121', 'densenet169'])
    parser.add_argument("-m", "--modality", type=str, choices=['t1', 't2', 't1ce', 'flair', 'all'], default = 'all')

    # regularizer
    parser.add_argument("--dropout", type=float, default=0, help='dropout rate for densenets')
    parser.add_argument("--l1", action='store_true', help='whether use l1 loss or not')
    parser.add_argument("--early_stopping", action='store_true', default=False, help='enable early stopping')

    # data config
    parser.add_argument("-f", "--fold", type=int, default=0, help="cross validation fold number, [0-4]")
    parser.add_argument("--machine", type=str, default = 'cedar', choices = ['solar', 'cedar', 'mial'])
    parser.add_argument("-x", "--exp_number", type=str, default = 'exp_temp', help = 'experiment folder for logging')
    
    args = parser.parse_args()
    return args


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def train(args, data_path, exp_path, label_path):

    torch.cuda.empty_cache()

    # set the training and validation files
    train_csv = os.path.join(label_path, 'split_{}_train.csv'.format(args.fold))
    val_csv = os.path.join(label_path, 'split_{}_val.csv'.format(args.fold))
    test_csv = os.path.join(label_path, 'split_{}_test.csv'.format(args.fold))

    # load the data to the dataloaders
    print('\n data loaders... \n')
    train_loader, val_loader, test_loader = data_create(data_root = data_path, label_root = label_path, batch_size=args.batch_size, fold = args.fold, modality=args.modality)
    
    # load network
    if args.modality == 'all':
        num_modalities = 4
        print('\n See the transforms- AddChanneld')
        raise NotImplementedError
    else:
        num_modalities = 1
    
    net = None
    if args.model == 'densenet121':
        print('\n Loading DenseNet121')
        net = DenseNet121(spatial_dims=3, in_channels=num_modalities, out_channels=3, dropout_prob=args.dropout)
    elif args.model == 'densenet169':
        print('\n Loading DenseNet169')
        net = DenseNet169(spatial_dims=3, in_channels=num_modalities, out_channels=3, dropout_prob=args.dropout)
    else:
        print('\n ERROR: Network architecture not specified!')
        raise NotImplementedError

    print('Parameter count =', count_parameters(net))
    print('\n Done loading the model')

    # move to GPU
    print('\n moving models to GPU ...\n')
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        model = net.to(device)
        device_ids = list(range(n_gpu))
        if len(device_ids) > 1:
            raise NotImplementedError 
            model = torch.nn.DataParallel(model, device_ids=device_ids)
            print('\n Using data parallel with n_gpus=',n_gpu)
        print('Device =', device, 'n_gpu = ', n_gpu)
    else:
        print('\n GPU not available')
        raise NotImplementedError

    # loss weights
    if args.weighted_loss:
        raise NotImplementedError
        print('\n Using weighted loss')
        weighted_loss = torch.tensor([0.12, 0.54, 0.33]).to(device)
    else:
        print('\n Using unweighted loss')
        weighted_loss = None

    # loss type
    print('\n Using cross entropy loss')
    criterion = nn.CrossEntropyLoss(weight=weighted_loss)
    criterion.to(device)

    # regularization
    if args.l1:
        raise NotImplementedError
        print('\n Adding L1 loss')
        print('\n Check what exactly is happening lol')

    # Optimizer
    print('\n Setting the optimizer')
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Setting up early stopping
    print('\n Setting up early stopping')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=50, stop_epoch=500, verbose=True)
    else:
        early_stopping = None

    # Summary writer
    writer = SummaryWriter(os.path.join(exp_path, 'tensorboard_fold_{}'.format(args.fold)))

    # Training
    print('\n===Start training for fold {}, lr={} ===\n'.format(args.fold, args.lr))
    for epoch in range(args.epochs):
        print("\n TRAINING: Epoch %d learning rate %f \n" % (epoch, optimizer.param_groups[0]['lr']))
        model = train_loop(model, train_loader, epoch, optimizer, criterion, writer)
        print("\n VALIDATION")
        stop = validate(model, val_loader, args.fold, epoch, criterion, writer, early_stopping, exp_path, val_csv)
        if stop:
            break
        print("\n xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(exp_path, "fold_{}_checkpoint.pt".format(args.fold))))
    else:
        torch.save(model.state_dict(), os.path.join(exp_path, "fold_{}_checkpoint.pt".format(args.fold)))

    print("\n TESTING")
    final_model_performance(model, test_loader, criterion, exp_path, args.fold, test_csv)
    writer.close()



def train_loop(model, train_loader, epoch, optimizer, criterion, writer):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loss = 0
    train_error = 0
    acc_logger = Accuracy_Logger(n_classes=3)
    model.train()

    for i, data in enumerate(train_loader, 0):    
        model.zero_grad()
        optimizer.zero_grad()
        inputs, labels, radpathID = data['image'], data['gt'], data['radpath_ID']
        inputs, labels = inputs.to(device), labels.to(device)
        pred = model(inputs)
        loss = criterion(pred, labels)
        
        loss.backward()
        optimizer.step()

        predict = torch.argmax(pred, 1)
        acc_logger.log_batch(predict, labels)
        error = 1 - predict.float().eq(labels.float()).float().mean().item()
        train_loss += loss.item()
        train_error += error

    
    train_loss /= 165
    train_error /= 165

    print("Epoch: %d, train_loss: %.4f, train_accuracy: %.4f" % (epoch, train_loss, 1-train_error))

    for i in range(3):
        acc, correct, count = acc_logger.get_summary(i)
        print("Class %d: acc %.4f, correct %d/%d" % (i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        print('\n Writing to tensorboard')
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/accuracy', 1-train_error, epoch)

    return model



def validate(model, val_loader, fold, epoch, criterion, writer, early_stopping, exp_path, val_csv):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    total = 0
    correct = 0
    loss_val = 0 
    acc_logger = Accuracy_Logger(n_classes=3)

    with torch.no_grad():
        val_result_csv = os.path.join(exp_path, 'val_results_fold_{}.csv'.format(fold))
        with open(val_result_csv, 'wt', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            for data in val_loader:
                images_val, labels_val, radpathID = data['image'], data['gt'], data['radpath_ID']
                images_val, labels_val = images_val.to(device), labels_val.to(device)
                pred_val = model.forward(images_val)
                predict_val = torch.argmax(pred_val, 1)
                acc_logger.log_batch(predict_val, labels_val)

                loss = criterion(pred_val, labels_val)
                loss_val += loss.item()
                total += labels_val.size(0)
                # record val predicted responses
                responses = F.softmax(pred_val, dim=1).cpu().numpy()
                responses = [responses[i] for i in range(responses.shape[0])]
                for r in responses:
                    csv_writer.writerow(r)

    label = 'class'
    label_dict = {'G':0, 'O':1, 'A':2}
    precision, recall, cm, f1_micro_val, bal_acc_val, kappa_val = metrics_all(val_csv, val_result_csv, label, label_dict)
    loss_val /= 28

    if writer:
        writer.add_scalar('val/loss', loss_val, epoch)
        writer.add_scalar('val/f1_micro', f1_micro_val, epoch)
        writer.add_scalar('val/bal_acc', bal_acc_val, epoch)
        writer.add_scalar('val/kappa', kappa_val, epoch)
        #writer.add_scalar('val/avg_precision', np.mean(precision), epoch)
        #writer.add_scalar('val/avg_recall', np.mean(recall), epoch)

    print("\n[Fold %d epoch %d] val result: \navg_precision %.2f%% avg_recall %.2f%%\n" %
        (fold, epoch, 100*np.mean(precision), 100*np.mean(recall)))
    print('f1:', f1_micro_val)
    print('kappa:', kappa_val)
    print('bal_acc:', bal_acc_val)
    print('confusion matrix: \n', cm)

    for i in range(3):
        acc, correct, count = acc_logger.get_summary(i)
        print("Class %d: acc %.4f, correct %d/%d" % (i, acc, correct, count))
        if writer:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)

    if early_stopping:
        assert exp_path
        early_stopping(epoch, loss_val, model, ckpt_name=os.path.join(exp_path, "fold_{}_checkpoint.pt".format(fold)))
        if early_stopping.early_stop:
            print("Early stopping")

            # val_results last epoch
            val_results = dict()
            val_results['precision'] = precision.tolist()
            val_results['recall'] = recall.tolist()
            val_results['cm'] = cm.tolist()
            val_results['f1'] = f1_micro_val.tolist()
            val_results['kappa'] = kappa_val.tolist()
            val_results['bal_acc'] = bal_acc_val.tolist()
            val_results['loss'] = loss_val

            acc0, _, _ = acc_logger.get_summary(0)
            acc1, _, _ = acc_logger.get_summary(1)
            acc2, _, _ = acc_logger.get_summary(2)
            val_results['acc_class0'] = acc0
            val_results['acc_class1'] = acc1
            val_results['acc_class2'] = acc2
            with open(os.path.join(exp_path, 'val_metrics_fold_{}.json'.format(fold)), 'w') as js:
                json.dump(val_results, js)

            return True

    return False



def final_model_performance(model, loader, criterion, exp_path, fold, test_csv):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('\n Final testing the model... \n')

    model.eval()
    total = 0
    correct = 0
    test_loss = 0 
    acc_logger = Accuracy_Logger(n_classes=3)

    with torch.no_grad():
        test_result_csv = os.path.join(exp_path, 'test_results_fold_{}.csv'.format(fold))
        with open(test_result_csv, 'wt', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            for data in loader:
                images_test, labels_test, radpathID = data['image'], data['gt'], data['radpath_ID']
                images_test, labels_test = images_test.to(device), labels_test.to(device)
                pred_test = model.forward(images_test)
                predict_test = torch.argmax(pred_test, 1)
                acc_logger.log_batch(predict_test, labels_test)

                test_loss += criterion(pred_test, labels_test)
                total += labels_test.size(0)
                # record test predicted responses
                responses = F.softmax(pred_test, dim=1).cpu().numpy()
                responses = [responses[i] for i in range(responses.shape[0])]
                for r in responses:
                    csv_writer.writerow(r)

    label = 'class'
    label_dict = {'G':0, 'O':1, 'A':2}
    precision_test, recall_test, cm_test, f1_micro_test, bal_acc_test, kappa_test = metrics_all(test_csv, test_result_csv, label, label_dict)
    
    print("\n[Fold %d] Test result: \navg_precision %.2f%% avg_recall %.2f%%\n" %
        (fold, 100*np.mean(precision_test), 100*np.mean(recall_test)))
    print('f1:', f1_micro_test)
    print('kappa:', kappa_test)
    print('bal_acc:', bal_acc_test)
    print('precision', precision_test)
    print('recall', recall_test)
    print('confusion matrix: \n', cm_test)
    for i in range(3):
        acc, correct, count = acc_logger.get_summary(i)
        print("Class %d: acc %.4f, correct %d/%d" % (i, acc, correct, count))
    
    test_results = dict()
    test_results['precision'] = precision_test.tolist()
    test_results['recall'] = recall_test.tolist()
    test_results['cm'] = cm_test.tolist()
    test_results['f1'] = f1_micro_test.tolist()
    test_results['kappa'] = kappa_test.tolist()
    test_results['bal_acc'] = bal_acc_test.tolist()
    test_results['loss'] = test_loss.item()/28

    acc0, _, _ = acc_logger.get_summary(0)
    acc1, _, _ = acc_logger.get_summary(1)
    acc2, _, _ = acc_logger.get_summary(2)
    test_results['acc_class0'] = acc0
    test_results['acc_class1'] = acc1
    test_results['acc_class2'] = acc2

    with open(os.path.join(exp_path, 'test_metrics_fold_{}.json'.format(fold)), 'w') as js:
        json.dump(test_results, js)

    # Save these numbers in a csv file
    save_results(exp_path, fold, f1_micro_test, kappa_test, bal_acc_test)



def save_results(exp_path, fold, f1, kappa, bacc):
    results_file = os.path.join(exp_path, 'test_results_summary.csv')
    if os.path.isfile(results_file):
        # read csv using pd and update and save (first remove mean and std rows)
        df = pd.read_csv(results_file)
        df = df[df['fold'].isin(['mean', 'std'])==False]
        new_row = {'fold':fold, 'f1_micro':f1, 'cohens_kappa':kappa, 'bal_acc':bacc}
        df = df.append(new_row, ignore_index=True)
    else:
        # create a df and update and save
        df= pd.DataFrame(columns=['fold', 'f1_micro', 'cohens_kappa', 'bal_acc'])
        new_row = {'fold':fold, 'f1_micro':f1, 'cohens_kappa':kappa, 'bal_acc':bacc}
        df = df.append(new_row, ignore_index=True)
    
    # Calculate mean and std
    new_row = {'fold':'mean', 'f1_micro':df.mean(axis=0)[-3], 'cohens_kappa':df.mean(axis=0)[-2], 'bal_acc':df.mean(axis=0)[-1]}
    df1 = df.append(new_row, ignore_index=True)
    new_row = {'fold':'std', 'f1_micro':df.std(axis=0, ddof=0)[-3], 'cohens_kappa':df.std(axis=0, ddof=0)[-2], 'bal_acc':df.std(axis=0, ddof=0)[-1]}
    df2 = df1.append(new_row, ignore_index=True)
    df2.to_csv(results_file, index=False)



def main(args):
    # cuda setup
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

    # Setting the paths
    data_path = '/path/to/data/folder'
    label_path = '/path/to/label/folder'
    exp_path = '/path/to/result/folder'
        
    print('Data_path:', data_path, '\nLabel_path:', label_path, '\nExp_path:', exp_path)
    
    if not os.path.isdir(exp_path):
        os.mkdir(exp_path)

    # save the experiment's arguments in json file
    with open(os.path.join(exp_path, 'experiment_args_fold_{}.json'.format(args.fold)), 'w') as js:
        json.dump(args.__dict__, js, indent = 2)

    train(args, data_path, exp_path, label_path)



if __name__ == "__main__":
    args = parse_args()
    main(args)
    print('\n\n\n ==============================END OF SCRIPT==============================\n\n\n')


