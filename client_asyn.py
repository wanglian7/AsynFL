import torch.nn as nn
import torch
import pickle  # for pkl file reading
import os
import sys
import numpy as np
import time
import copy
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
# from PIL import Image
import torchvision.transforms as transforms
from config import SERVER_ADDR, SERVER_PORT, read_data, read_options
from utils import recv_msg, send_msg
import socket
import struct
import scipy.io

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def update_weights(global_weight, local_weight):
    """
    Returns the average of the weights.
    """
    w_new = copy.deepcopy(local_weight)
    device = next(iter(local_weight.values())).device
    for key in w_new.keys():
        # for i in range(1, len(w)):
        global_weight[key] = global_weight[key].to(device)

        w_new[key] = w_new[key] - global_weight[key]
        w_new[key] = torch.div(w_new[key], n_nodes)
    return w_new


# Model for MQTT_IOT_IDS dataset
class Logistic(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Logistic, self).__init__()
        self.layer = nn.Linear(in_dim, out_dim)
        self.sm = nn.Sigmoid().to(device)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer(x)
        logit = self.sm(x)
        return logit


###################################################################################
# Import the data set

# socket
sock = socket.socket()
sock.connect((SERVER_ADDR, SERVER_PORT))
print('---------------------------------------------------------------------------')
try:
    msg = recv_msg(sock, 'MSG_INIT_SERVER_TO_CLIENT')
    options = msg[1]
    n_nodes = options['num_clients']
    cid = msg[2]
    # first step: set the optimizer & criterion
    lr_rate = options['lr']
    gamma = options['decay']
    num_straggle = options['num_straggle']
    is_iid = options['iid']
    sleep_secs = options['sleep_secs']
    print("Receive the Learning rate:", lr_rate, "decay:", gamma)
    model = Logistic(784, 10).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)
    step = 0
    criterion = torch.nn.CrossEntropyLoss().to(device)
    non_iid_degree = options['non_iid_degree']
    data_set = options['data_set']
    # choose the dataset
    print("read data")
    if data_set == 'mnist':
        if is_iid:
            train_data = dsets.MNIST(root='./mnist', train=True, transform=transforms.ToTensor(), download=True)
            print("succ read mnist iid!!!!")
        else:

            file_path = './mnist/niid' + str(non_iid_degree) + '/mnist' + str(cid) + '.pkl'
            train_data = read_data(file_path)
            print("succ read mnist non-iid!!!!")
    else:
        if is_iid:
            train_data = dsets.MNIST(root='./traffic', train=True, transform=transforms.ToTensor(), download=True)
            print("succ read traffic iid!!!!")
        else:
            file_path = './traffic/niid' + str(non_iid_degree) + '/traffic' + str(cid) + '.pkl'
            train_data = read_data(file_path)
            print("succ read traffic niid!!!!!")
    # dataset_size_per_user=len(train_data)//n_nodes
    # indices=list(range(len(train_data)))
    # train_indices=indices[cid*dataset_size_per_user:(cid+1)*dataset_size_per_user]
    # train_sampler=SubsetRandomSampler(train_indices)
    train_time = []
    while True:
        print('---------------------------------------------------------------------------')

        msg = recv_msg(sock, 'MSG_WEIGHT_TAU_SERVER_TO_CLIENT')
        is_last_round = msg[1]
        global_model_weights = msg[2]
        num_epoch = msg[3]
        n = msg[4]

        print('Epoches: ', num_epoch, ' Number of data samples: ', n)
        # make the data loader
        if is_last_round:
            saveTitle = 'local_' + 'straggler_' + str(options['num_straggle']) + 'T' + str(
                options['num_round']) + 'E' + str(options['E']) + 'B' + str(
                options['n']) + '_K' + str(options['clients_per_round'])
            scipy.io.savemat(saveTitle + '_time' + '.mat', mdict={saveTitle + '_time': train_time})
            break
        train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                   batch_size=n,
                                                   shuffle=True)
        print('Make dataloader successfully')

        start = time.time()
        model.load_state_dict(global_model_weights)

        # num_epoch = options['num_epoch']  # Local epoches
        # batch_size = 400  # Data sample for training per comm. round
        model.train()
        # time.sleep(0.00000134*n*num_epoch)   #模拟训练时间
        x, y = next(iter(train_loader))
        x, y = x.to(device),y.to(device)
        # print('Round:', round)
        for i in range(num_epoch):
            if is_iid:
                x = Variable(x.view(-1, 28 * 28).to(device))
                y = Variable(y.to(device))
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(optimizer.state_dict()['param_groups'][0]['lr'])

        # acc, loss = local_test(model=model, test_dataloader=test_loader)
        step += num_epoch
        upload_weight = update_weights(global_model_weights, model.state_dict())
        if cid < num_straggle:
            print("I am straggle sleep for", sleep_secs, " s")
            time.sleep(sleep_secs)  # 如果是拖延者，睡眠10s
        print("upload the updating model para from client: ", cid)
        # print(upload_weight)
        msg = ['MSG_WEIGHT_TIME_SIZE_CLIENT_TO_SERVER', upload_weight, step]
        send_msg(sock, msg)
        end = time.time()
        train_time.append(end - start)
    # print("loss:", loss, "    acc:", acc)


except (struct.error, socket.error):
    print('Server has stopped')
    pass
