import os
import copy
import time
import pickle
import numpy as np
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import scipy.io
from config import SERVER_ADDR, SERVER_PORT, read_options, read_data
import importlib
import socket
import errno
import fcntl
import os
import select
import sys

from utils import recv_msg, send_msg
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

def setNonBlocking(fd):
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)


def nonBlockingWrite(fd, data):
    try:
        nw = os.write(fd, data)
        return nw
    except OSError as e:
        if e.errno == errno.EWOULDBLOCK:
            return -1


def test_inference(model, testloader):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    # device = 'cuda' if args['gpu'] else 'cpu'
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # testloader = DataLoader(test_dataset, batch_size=128,
    #                         shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        # images = Variable(images.view(-1, 28 * 28))
        # labels = Variable(labels)
        # Inference
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct / total
    loss = loss / total
    return accuracy, loss


# change
def average_weights(w):   #没有用到的函数
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def update_weights(global_weight, local_weight):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(global_weight)
    for key in w_avg.keys():
        # for i in range(1, len(w)):
        w_avg[key] += local_weight[key]
        # w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


# Model
class Logistic(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Logistic, self).__init__()
        self.layer = nn.Linear(in_dim, out_dim)
        self.sm = nn.Sigmoid().to(device)

    def forward(self, x):
        x = x.view(x.size(0), -1).to(device)
        x = self.layer(x)
        logit = self.sm(x)
        return logit


# change
def select_clients():
    num_clients = min(options['clients_per_round'], n_nodes)
    return np.random.choice(range(0, len(client_sock_all)), num_clients, replace=False).tolist()

if __name__ == '__main__':
    options = read_options()
    listening_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listening_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listening_sock.bind((SERVER_ADDR, SERVER_PORT))
    client_sock_all = []

    num_rounds = options['num_round']
    n_nodes = options['num_clients']
    print("Total Clients: ", n_nodes, "run ", num_rounds, "num_rounds")
    # Establish connections to each client, up to n_nodes clients, setup for clients
    while len(client_sock_all) < n_nodes:
        listening_sock.listen(5)
        print("Waiting for incoming connections...")
        (client_sock, (ip, port)) = listening_sock.accept()
        print('Got connection from ', (ip, port))

        # Add the accepted socket to the list of client sockets
        client_sock_all.append([ip, port, client_sock])

    # first: send some initial info
    for i in range(0, n_nodes):
        msg = ['MSG_INIT_SERVER_TO_CLIENT', options, i]
        send_msg(client_sock_all[i][2], msg)

    # mnist
    global_model = Logistic(784, 10).to(device)
    global_model.train()

    global_weights = global_model.state_dict()

    print('All clients connected')

    # Load test dataset
    if options['data_set'] == 'mnist':
        test_data = read_data('./mnist/MNIST_test.pkl')
    else:
        test_data = read_data('./traffic/traffic_test.pkl')
    test_loader = DataLoader(dataset=test_data, batch_size=128, shuffle=True)

    # Initialize lists for tracking results
    cv_loss, cv_acc, cv_step, cv_time, cv_cid = [], [], [], [], []

    # Initial parameters
    E_train = options['E']
    n_train = options['n']
    aggregation_count = 0
    local_weights, local_losses = [], []
    print(f'\n | Global Training Round : {aggregation_count} |\n')

    # Send initial weights to each client
    is_last_round = False
    start = time.time()
    for i in range(0, n_nodes):
        sock = client_sock_all[i][2]
        msg = ['MSG_WEIGHT_TAU_SERVER_TO_CLIENT', is_last_round, global_weights, E_train, n_train]
        send_msg(sock, msg)

    is_last_round = False
    print('---------------------------------------------------------------------------')
    aggregation_count += 1

    # Receive model weights from clients
    train_number = 0
    done = False
    input_sockets = [client[2] for client in client_sock_all]
    while not done:
        readable, _, _ = select.select(input_sockets, [], [], 10)  # Wait up to 10 seconds
        for sock in readable:
            try:
                msg = recv_msg(sock, 'MSG_WEIGHT_TIME_SIZE_CLIENT_TO_SERVER')
                if msg:
                    w = msg[1]  # Get local model parameters
                    step = msg[2]  # Get step count
                    # local_weights = {k: v.to(device) for k, v in w.items()}
                    local_weights = copy.deepcopy(w)

                    # Update global model parameters
                    global_weights = update_weights(global_model.state_dict(), local_weights)
                    global_model.load_state_dict(global_weights)

                    # Perform inference on the test set
                    test_acc, test_loss = test_inference(global_model, test_loader)
                    print(test_acc, test_loss)

                    # Save test results
                    cv_acc.append(test_acc)
                    cv_loss.append(test_loss)
                    cv_step.append(step)
                    cv_cid.append(i)
                    latency = time.time() - start
                    print("total time:", latency)
                    cv_time.append(latency)
                    train_number += 1

                    # Check if the stopping condition is met
                    if test_loss < 0.0136:
                        is_last_round = True
                        done = True
                        num_straggle = options['num_straggle']
                        if options['iid']:
                            saveTitle = 'ASYN_iid_slow' + str(num_straggle) + '_' + str(E_train) + '_' + str(n_train)
                        else:
                            saveTitle = 'ASYN_niid_slow' + str(num_straggle) + '_' + str(E_train) + '_' + str(n_train)
                        print("训练总轮数：" + str(train_number))
                        # Save test results
                        scipy.io.savemat(saveTitle + '_acc.mat', mdict={saveTitle + '_acc': cv_acc})
                        scipy.io.savemat(saveTitle + '_loss.mat', mdict={saveTitle + '_loss': cv_loss})
                        scipy.io.savemat(saveTitle + '_cid.mat', mdict={saveTitle + '_cid': cv_cid})
                        scipy.io.savemat(saveTitle + '_step.mat', mdict={saveTitle + '_step': cv_step})
                        scipy.io.savemat(saveTitle + '_time.mat', mdict={saveTitle + '_time': cv_time})

                        # Send final round message to all clients
                        for j in range(0, n_nodes):
                            sock = client_sock_all[j][2]
                            msg = ['MSG_WEIGHT_TAU_SERVER_TO_CLIENT', is_last_round, global_weights, E_train, n_train]
                            send_msg(sock, msg)
                        exit(0)  # Exit program

                    # If not the last round, send updated global model parameters to all clients
                    msg = ['MSG_WEIGHT_TAU_SERVER_TO_CLIENT', is_last_round, global_weights, E_train, n_train]
                    send_msg(sock, msg)
            except Exception as e:
                print(f"Error processing message from client: {e}")
                continue

    # Clean up resources
    listening_sock.close()
    for sock in client_sock_all:
        sock[2].close()