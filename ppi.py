import torch
import torch.nn as nn
from torch import utils
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import argparse
import matplotlib.pyplot as plt
import logging
import random
from pandas.core.common import SettingWithCopyWarning
import warnings
import math
import time
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

parser = argparse.ArgumentParser(description='模型参数设置')
parser.add_argument('--weight_decay', type=float, default=1e-2, choices=[3e-4], help="weight_decay")
parser.add_argument('--lr', type=float, default=5 * 1e-4, choices=[5e-4], help="learning rate")
parser.add_argument('--epoch', type=int, default=40, choices=[40], help="epochs_number")
parser.add_argument('--batch_size', type=int, default=8192, choices=[8192], help="batch_size")
parser.add_argument('--kgDimension', type=int, default=256, choices=[64], help="7kDimension")
parser.add_argument('--ddiDimension', type=int, default=1024, choices=[512], help="ddiDimension")
parser.add_argument('--sample_number', type=int, default=20, choices=[7], help="sample_number")
parser.add_argument('--dropout', type=float, default=0.3, choices=[0.3], help="dropout")
parser.add_argument('--pLayers', type=int, default=1, choices=[1], help="pLayers")
parser.add_argument('--n_splits', type=int, default=5, choices=[5], help="n_splits")
net_args = parser.parse_args()

logName = 'test'


class GNNLayer(nn.Module):
    def __init__(self, dimension, sampleSize, DKG, pLayers, layer, ddi=False):
        super().__init__()
        self.DKG = DKG.long()
        self.sampleSize = sampleSize
        self.layer = layer
        self.drugNumber = len(torch.unique(DKG[:, 0]))
        self.relationNumber = len(torch.unique(DKG[:, 2]))
        self.tailNumber = len(torch.unique(DKG[:, 1]))
        if ddi:
            self.drugNumber = 1649
            self.tailNumber = 1649
        self.dimension = dimension
        self.drugEmbeding = nn.Embedding(num_embeddings=self.drugNumber, embedding_dim=dimension)
        self.relationEmbeding = nn.Embedding(num_embeddings=self.relationNumber, embedding_dim=dimension)
        self.tailEmbeding = nn.Embedding(num_embeddings=self.tailNumber, embedding_dim=dimension)
        fullConnectionLayers = []
        for i in range(pLayers):
            if (i < pLayers - 1):
                fullConnectionLayers.append(nn.Linear(dimension, dimension))
                fullConnectionLayers.append(nn.Sigmoid())
            else:
                fullConnectionLayers.append(nn.Linear(dimension, dimension))
        self.fullConnectionLayer = nn.Sequential(*fullConnectionLayers)
        self.fullConnectionLayer2 = nn.Sequential(nn.Linear(dimension * 2, dimension), nn.BatchNorm1d(dimension))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, arguments):
        if self.layer == 0:
            X = arguments
        elif self.layer == 1:
            embedding1, X = arguments
        elif self.layer == 2:
            embedding1, embedding2, X = arguments
        elif self.layer == 3:
            embedding1, embedding2, embedding3, X = arguments

        hadamardProduct = self.drugEmbeding(self.DKG[:, 0]) * self.relationEmbeding(self.DKG[:, 2])
        semanticsFeatureScore = torch.sum(self.fullConnectionLayer(hadamardProduct), dim=1).reshape((-1, 1))
        tempEmbedding = semanticsFeatureScore * self.tailEmbeding(self.DKG[:, 1])
        neighborhoodEmbedding = torch.zeros(self.drugNumber, self.dimension).to('cuda')

        for i in range(self.drugNumber):
            # 采样
            length = torch.sum(self.DKG[:, 0] == i)
            if length == 0:
                continue
            if length >= self.sampleSize:
                index = list(utils.data.WeightedRandomSampler(self.DKG[:, 0] == i, self.sampleSize, replacement=False))
                neighborhoodEmbedding[i] = torch.sum(tempEmbedding[index], dim=0)
            else:
                neighborhoodEmbedding[i] = torch.sum(tempEmbedding[self.DKG[:, 0] == i], dim=0)
                # index = list(
                #     utils.data.WeightedRandomSampler(self.DKG[:, 0] == i, int(self.sampleSize - length),
                #                                      replacement=True))
                # neighborhoodEmbedding[i] = neighborhoodEmbedding[i] + torch.sum(tempEmbedding[index], dim=0)

        concatenate = torch.cat([self.drugEmbeding.weight, neighborhoodEmbedding], 1)
        if self.layer == 0:
            return self.fullConnectionLayer2(concatenate), X
        elif self.layer == 1:
            return embedding1, self.fullConnectionLayer2(concatenate), X
        elif self.layer == 2:
            return embedding1, embedding2, self.fullConnectionLayer2(concatenate), X
        elif self.layer == 3:
            return embedding1, embedding2, embedding3, self.fullConnectionLayer2(concatenate), X


class FusionLayer(nn.Module):
    def __init__(self, dimension, dropout, GNNlayers):
        super().__init__()
        self.GNNlayers = GNNlayers
        self.fullConnectionLayer = nn.Sequential(
            nn.Linear(dimension * 2, dimension),
            nn.BatchNorm1d(dimension),
            nn.Softmax(dim=1),
            nn.Dropout(dropout),
            nn.Linear(dimension, int(dimension / 2)),
            nn.BatchNorm1d(int(dimension / 2)),
            nn.Softmax(dim=1),
            # nn.Dropout(dropout),
            nn.Linear(int(dimension / 2), 1),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, arguments):
        if self.GNNlayers == 1:
            embedding1, X = arguments
            Embedding = embedding1
        elif self.GNNlayers == 2:
            embedding1, embedding2, X = arguments
            Embedding = torch.cat([embedding1, embedding2], 1)
        elif self.GNNlayers == 3:
            embedding1, embedding2, embedding3, X = arguments
            Embedding = torch.cat([embedding1, embedding2, embedding3], 1)
        elif self.GNNlayers == 4:
            embedding1, embedding2, embedding3, embedding4, X = arguments
            Embedding = torch.cat([embedding1, embedding2, embedding3, embedding4], 1)
        X = X.long()
        drugA = X[:, 0]
        drugB = X[:, 1]
        finalEmbedding = torch.cat([Embedding[drugA], Embedding[drugB]], 1).float()
        return self.fullConnectionLayer(finalEmbedding)


def train(net, train_iter, num_epochs, lr, wd, X_test, y_test, fold=0, device=torch.device(f'cuda')):
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    loss = nn.BCELoss()
    train_acc_list = []
    test_acc_list = []
    max_test_acc = 0
    max_test_sen = 0
    max_test_pre = 0
    max_test_auc = 0
    max_test_aupr = 0
    max_test_mcc = 0
    for epoch in range(num_epochs):
        train_result = []
        net.train()
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y.float())
            l.backward()
            optimizer.step()
            train_result += torch.flatten(torch.round(y_hat).int() == y.int()).tolist()

        net.eval()
        with torch.no_grad():
            y_hat = net(X_test)
            tn, fp, fn, tp = confusion_matrix(y_test, torch.round(y_hat).int().cpu()).ravel()  #tn, fp, fn, tp
            fpr, tpr, thresholds = metrics.roc_curve(y_test, y_hat.cpu(), pos_label=1)
            precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_hat.cpu())
            train_acc = sum(train_result) / len(train_result)
            test_acc = (tn + tp) / (fp + tp + fn + tn)
            test_sen = tp / (tp + fn)  # sensitivity = TP / (TP + FN)
            test_pre = tp / (tp + fp)
            test_auc = metrics.auc(fpr, tpr)
            test_aupr = metrics.auc(recall, precision)
            test_mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            if test_acc == max(test_acc_list):
                max_test_acc = test_acc
                max_test_sen = test_sen
                max_test_pre = test_pre
                max_test_auc = test_auc
                max_test_aupr = test_aupr
                max_test_mcc = test_mcc
            print(f'tn:{tn}, fp:{fp}, fn:{fn}, tp:{tp}')
            print(f'train acc {train_acc:.4f}')
            print(f'test acc {test_acc:.4f}')
            print(f'test sen {test_sen:.4f}')
            print(f'test pre {test_pre:.4f}')
            print(f'test auc {test_auc:.4f}')
            print(f'test aupr {test_aupr:.4f}')
            print(f'test mcc {test_mcc:.4f}')
            print(f'max test acc {max_test_acc:.4f}')
            print(f'max test sen {max_test_sen:.4f}')
            print(f'max test pre {max_test_pre:.4f}')
            print(f'max test auc {max_test_auc:.4f}')
            print(f'max test aupr {max_test_aupr:.4f}')
            print(f'max test mcc {max_test_mcc:.4f}')
            print("epoch:{}...".format(epoch))
            print("fold:{}...".format(fold))
            print("----------------------------------------------")
            logging.info(f'train acc {train_acc:.4f}')
            logging.info(f'test acc {test_acc:.4f}')
            logging.info(f'test sen {test_sen:.4f}')
            logging.info(f'test pre {test_pre:.4f}')
            logging.info(f'test auc {test_auc:.4f}')
            logging.info(f'test aupr {test_aupr:.4f}')
            logging.info(f'test mcc {test_mcc:.4f}')
            logging.info(f'max test acc {max_test_acc:.4f}')
            logging.info(f'max test sen {max_test_sen:.4f}')
            logging.info(f'max test pre {max_test_pre:.4f}')
            logging.info(f'max test auc {max_test_auc:.4f}')
            logging.info(f'max test aupr {max_test_aupr:.4f}')
            logging.info(f'max test mcc {max_test_mcc:.4f}')
            logging.info("epoch:{}...".format(epoch))
            logging.info("fold:{}...".format(fold))
            logging.info("----------------------------------------------")
    x = range(len(train_acc_list))
    y1 = train_acc_list
    y2 = test_acc_list
    plt.plot(x, y1, color='r', label="train_acc")  # s-:方形
    plt.plot(x, y2, color='g', label="test_acc")  # o-:圆形
    plt.xlabel("epoch")  # 横坐标名字
    plt.ylabel("accuracy")  # 纵坐标名字
    plt.legend(loc="best")  # 图例
    plt.savefig('./data/' + logName + '.png')
    plt.show()
    return max_test_acc, max_test_sen, max_test_pre, max_test_auc, max_test_aupr, max_test_mcc


def train_KFold(lr, wd, KG1, data, n_splits, num_epochs, batch_size):
    test_acc_list = []
    test_sen_list = []
    test_pre_list = []
    test_auc_list = []
    test_aupr_list = []
    test_mcc_list = []
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    features = data.iloc[:, 0:2]
    labels = data.iloc[:, 2:]
    for i, (train_index, test_index) in enumerate(kf.split(features, labels)):
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

        X_train2 = {'0': list(X_train.iloc[:, 1]), '1': list(X_train.iloc[:, 0])}
        X_train2 = pd.DataFrame(data=X_train2)

        X_train = pd.concat([X_train, X_train2], axis=0).reset_index(drop=True)
        y_train = pd.concat([y_train, y_train], axis=0).reset_index(drop=True)

        KG2 = torch.tensor(pd.concat([X_train, y_train], axis=1).to_numpy()).to('cuda')

        X_train = torch.tensor(X_train.to_numpy())
        y_train = torch.tensor(y_train.to_numpy())
        X_test = torch.tensor(X_test.to_numpy())
        y_test = torch.tensor(y_test.to_numpy())

        dataset = utils.data.TensorDataset(X_train, y_train)
        train_iter = utils.data.DataLoader(dataset, batch_size, shuffle=True)
        # net.load_state_dict(torch.load("./data/model_parameter.pkl"))

        net = nn.Sequential(
            GNNLayer(net_args.kgDimension, net_args.sample_number, KG1, net_args.pLayers, 0),
            GNNLayer(net_args.ddiDimension, net_args.sample_number, KG2, net_args.pLayers, 1, True),
            FusionLayer(net_args.kgDimension + net_args.ddiDimension, net_args.dropout, 2))

        test_acc, test_sen, test_pre, test_auc, test_aupr, test_mcc = \
            train(net, train_iter, num_epochs, lr, wd, X_test, y_test, i)
        # train(net, train_iter, num_epochs, lr, wd, X_test, y_test, fold=0, device=torch.device(f'cuda'))
        test_acc_list.append(test_acc)
        test_sen_list.append(test_sen)
        test_pre_list.append(test_pre)
        test_auc_list.append(test_auc)
        test_aupr_list.append(test_aupr)
        test_mcc_list.append(test_mcc)
        print(f'fold {i},  max_test_acc:{test_acc:.4f}')
        print(f'fold {i},  max_test_sen:{test_sen:.4f}')
        print(f'fold {i},  max_test_pre:{test_pre:.4f}')
        print(f'fold {i},  max_test_auc:{test_auc:.4f}')
        print(f'fold {i},  max_test_aupr:{test_aupr:.4f}')
        print(f'fold {i},  max_test_mcc:{test_mcc:.4f}')
        print("---------------------------------------")
        logging.info(f'fold {i},  max_test_acc:{test_acc:.4f}')
        logging.info(f'fold {i},  max_test_sen:{test_sen:.4f}')
        logging.info(f'fold {i},  max_test_pre:{test_pre:.4f}')
        logging.info(f'fold {i},  max_test_auc:{test_auc:.4f}')
        logging.info(f'fold {i},  max_test_aupr:{test_aupr:.4f}')
        logging.info(f'fold {i},  max_test_mcc:{test_mcc:.4f}')
        logging.info("---------------------------------------")
    print(f'avg test acc {sum(test_acc_list) / n_splits:.4f}')
    print(f'avg test sen {sum(test_sen_list) / n_splits:.4f}')
    print(f'avg test pre {sum(test_pre_list) / n_splits:.4f}')
    print(f'avg test auc {sum(test_auc_list) / n_splits:.4f}')
    print(f'avg test aupr {sum(test_aupr_list) / n_splits:.4f}')
    print(f'avg test mcc {sum(test_mcc_list) / n_splits:.4f}')
    logging.info(f'avg test acc {sum(test_acc_list) / n_splits:.4f}')
    logging.info(f'avg test sen {sum(test_sen_list) / n_splits:.4f}')
    logging.info(f'avg test pre {sum(test_pre_list) / n_splits:.4f}')
    logging.info(f'avg test auc {sum(test_auc_list) / n_splits:.4f}')
    logging.info(f'avg test aupr {sum(test_aupr_list) / n_splits:.4f}')
    logging.info(f'avg test mcc {sum(test_mcc_list) / n_splits:.4f}')


def data_preprocessing(data):
    # proteinmap = {}
    # proteinlist = np.array(data.iloc[:, 0]).tolist()
    # for protein in proteinlist:
    #     if protein not in proteinmap:
    #         proteinmap[protein] = len(proteinmap)

    protein_number = pd.read_csv("ProteinNumber.csv")
    key_list = list(protein_number.iloc[:, 0])
    value_list = list(protein_number.iloc[:, 1])
    protein_map = dict(zip(key_list, value_list))


    entitymap = {}
    entitylist = np.array(data.iloc[:, 1]).tolist()
    for entity in entitylist:
        if entity not in entitymap:
            entitymap[entity] = len(entitymap)

    relationmap = {}
    relationlist = np.array(data.iloc[:, 2]).tolist()
    for relation in relationlist:
        if relation not in relationmap:
            relationmap[relation] = len(relationmap)

    data.iloc[:, 0] = data.iloc[:, 0].map(protein_map)
    data.iloc[:, 1] = data.iloc[:, 1].map(entitymap)
    data.iloc[:, 2] = data.iloc[:, 2].map(relationmap)

    # pd.DataFrame(list(proteinmap.items())).to_csv('ProteinNumber.csv', index=False)

    return data


def NegativeGenerate(KG, PPIlist):
    NegativeSamplelist = []
    NegativeSampleCounter = 0
    while NegativeSampleCounter < len(PPIlist):
        PPInumber1 = random.randint(0, KG['0'].nunique() - 1)
        PPInumber2 = random.randint(0, KG['0'].nunique() - 1)
        if PPInumber1 == PPInumber2:
            continue
        PPIpair = []
        PPIpair.append(PPInumber1)
        PPIpair.append(PPInumber2)
        flag = 0
        for pair in PPIlist:
            if PPIpair == pair:
                flag = 1
                break
        if flag == 1:
            continue
        for pair in NegativeSamplelist:
            if PPIpair == pair:
                flag = 1
                break
        if flag == 1:
            continue
        if flag == 0:
            NegativeSamplelist.append(PPIpair)
            NegativeSampleCounter = NegativeSampleCounter + 1
            print(f'NegativeGenerate:{len(PPIlist) - NegativeSampleCounter}')
    return pd.DataFrame(NegativeSamplelist)


if __name__ == '__main__':
    start_time = time.time()
    logging.basicConfig(level=logging.INFO, filename='./data/' + logName + '.log', filemode='w',
                        format="%(message)s")

    datadf = pd.read_csv("KnowledgeGraph.csv")
    KG1 = data_preprocessing(datadf)
    KG1 = torch.tensor(KG1.to_numpy()).to('cuda')
    PPI = pd.read_csv("event_encode.csv")

    train_KFold(net_args.lr, net_args.weight_decay, KG1, PPI, net_args.n_splits, net_args.epoch, net_args.batch_size)
    end_time = time.time()
    print("代码运行了 {:.2f} 秒".format(end_time - start_time))
    logging.info("代码运行了 {:.2f} 秒".format(end_time - start_time))