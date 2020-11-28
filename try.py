# import numpy as np
from utils import *
from modules import *

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.optim import lr_scheduler
import math
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
from sklearn import preprocessing


class DeepSurv(nn.Module):
    def __init__(self, D_in, D_h, D_out, drop_prob):
        super().__init__()
        self.fc1 = nn.Linear(D_in, D_h)
        self.fc2 = nn.Linear(D_h, D_h)
        self.dr1 = nn.Dropout(drop_prob)
        self.fc3 = nn.Linear(D_h, D_out)

    def forward(self, x):
        z1_relu = F.relu(self.fc1(x))
        z2_relu = F.relu(self.fc2(z1_relu))
        z_pred = self.fc3(self.dr1(z2_relu))

        return z_pred

def NegativeLogLikelihood(E, y_true, y_pred):
  hazard_ratio = torch.exp(y_pred)
  log_risk = torch.log(torch.cumsum(hazard_ratio, 0))
  log_risk = torch.reshape(log_risk, (list(log_risk.size())[0],)) #discuss
  uncensored_likelihood = y_pred.t() - log_risk
  censored_likelihood = uncensored_likelihood * E
  neg_likelihood = -torch.sum(censored_likelihood)
  return neg_likelihood

def createGraph():
    G = nx.Graph()
    corr_matrix = np.load('corr_matrix.npy')
    print(corr_matrix)
    G.add_nodes_from([1,2,3,4,5,6,7,8,9,10])
    added=[]

    for i in range(len(corr_matrix)):
    	added.append([])
    	for j in range(len(corr_matrix[0])):
    		added[i].append(0)

    for i in range(len(corr_matrix)):
    	for j in range(len(corr_matrix[0])):
    		if(corr_matrix[i][j]!=0 and added[i][j]==0):
    			G.add_edge(i+1, j+1, weight=corr_matrix[i][j])
    			added[i][j]=1; added[j][i]=1


    return G

ground_truth_G = createGraph()
df = pd.read_csv("data.csv")
x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, label, time = [], [], [], [], [], [], [], [], [], [], [], []
c = 5498
for ind in df.index:

    if df['label'][ind] == 4:
        x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, time_, label_ = df['x1'][ind], df['x2'][ind], df['x3'][ind], df['x4'][ind], df['x5'][ind],  df['x6'][ind], df['x7'][ind], df['x8'][ind], df['x9'][ind], df['x10'][ind], df['time'][ind], 1
        x1.append(x_1); x2.append(x_2); x3.append(x_3); x4.append(x_4); x5.append(x_5)
        x6.append(x_6); x7.append(x_7); x8.append(x_8); x9.append(x_9); x10.append(x_10);
        time.append(time_)
        label.append(label_)

    elif c >= 0 and df['label'][ind] == 0:
        x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, time_, label_ = df['x1'][ind], df['x2'][ind], df['x3'][ind], df['x4'][ind], df['x5'][ind],  df['x6'][ind], df['x7'][ind], df['x8'][ind], df['x9'][ind], df['x10'][ind], df['time'][ind], 0
        c -= 1
        x1.append(x_1); x2.append(x_2); x3.append(x_3); x4.append(x_4); x5.append(x_5)
        x6.append(x_6); x7.append(x_7); x8.append(x_8); x9.append(x_9); x10.append(x_10)
        time.append(time_)
        label.append(label_)

data = {"x1":x1, "x2":x2, "x3":x3, "x4":x4, "x5":x5, "x6":x6, "x7":x7, "x8":x8, "x9":x9, "x10":x10, "time":time, "label":label}
dummy = pd.DataFrame(data, columns = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "time", "label"])

X = np.array(dummy[["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10"]])
Y = np.array(dummy["time"])
E = np.array(dummy["label"])

X_train,X_val,Y_train,Y_val=train_test_split(X,Y,test_size=0.25, random_state=0)
X_train,X_val,E_train,E_val=train_test_split(X,E,test_size=0.25, random_state=0)

train = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(X_train), torch.from_numpy(Y_train), torch.from_numpy(E_train))
trainloader = DataLoader(train, batch_size = 64, shuffle = False)

#DEFINE
data_variable_size = 10
x_dims = 1
z_dims = 1
encoder_hidden = 64
decoder_hidden = 64
batch_size = 64
encoder_dropout = 0
decoder_dropout = 0
factor = True
lr = 1e-4
lr_decay = 200
survival_hidden = 32
survival_out = 1
survival_dropout = 0.5
gamma= 1
c_A = 1
EPOCH = 800
tau_A = 0.0
lambda_A = 0.
graph_threshold = 0.3

# Generate off-diagonal interaction graph
off_diag = np.ones([data_variable_size, data_variable_size]) - np.eye(data_variable_size)
rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float64)
rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float64)
rel_rec = torch.DoubleTensor(rel_rec)
rel_send = torch.DoubleTensor(rel_send)

# add adjacency matrix A
adj_A = np.zeros((data_variable_size, data_variable_size))

encoder = MLPEncoder(data_variable_size * x_dims, x_dims, encoder_hidden,
                     z_dims, adj_A,
                     batch_size = batch_size,
                     do_prob = encoder_dropout, factor = factor).double()

decoder = MLPDecoder(data_variable_size * x_dims,
                     z_dims, x_dims, encoder,
                     data_variable_size = data_variable_size,
                     batch_size = batch_size,
                     n_hid = decoder_hidden,
                     do_prob = decoder_dropout).double()

survival = DeepSurv(data_variable_size, survival_hidden, survival_out, survival_dropout)

#=======================================
#set up training parameters
#=======================================

optimizer = optim.Adam(list(encoder.parameters())+list(decoder.parameters())+list(survival.parameters()), lr = lr)

scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay,gamma=gamma)

triu_indices = get_triu_offdiag_indices(data_variable_size)
tril_indices = get_tril_offdiag_indices(data_variable_size)

rel_rec = Variable(rel_rec)
rel_send = Variable(rel_send)


def _h_A(A, m):
    expm_A = matrix_poly(A*A, m)
    h_A = torch.trace(expm_A) - m
    return h_A

prox_plus = torch.nn.Threshold(0.,0.)

def stau(w, tau):
    w1 = prox_plus(torch.abs(w)-tau)
    return torch.sign(w)*w1

def update_optimizer(optimizer, original_lr, c_A):
    '''related LR to c_A, whenever c_A gets big, reduce LR proportionally'''
    MAX_LR = 1e-2
    MIN_LR = 1e-4

    estimated_lr = original_lr / (math.log10(c_A) + 1e-10)
    if estimated_lr > MAX_LR:
        lr = MAX_LR
    elif estimated_lr < MIN_LR:
        lr = MIN_LR
    else:
        lr = estimated_lr

    # set LR
    for parame_group in optimizer.param_groups:
        parame_group['lr'] = lr

    return optimizer, lr

#===================================
# training:
#===================================

def train(epoch, best_val_loss, ground_truth_G, lambda_A, c_A, optimizer):
    #DEFINE
    data_variable_size = 10
    x_dims = 1
    z_dims = 1
    encoder_hidden = 64
    decoder_hidden = 64
    batch_size = 64
    encoder_dropout = 0
    decoder_dropout = 0
    factor = True
    lr = 1e-4
    lr_decay = 200
    survival_hidden = 32
    survival_out = 1
    survival_dropout = 0.5
    gamma= 1
    c_A = 1
    # EPOCH = 50
    tau_A = 0.0
    lambda_A = 0.
    graph_threshold = 0.3

    nll1_train = []
    kl_train = []
    mse_train = []
    shd_trian = []
    nll2_train = []

    encoder.train()
    decoder.train()
    survival.train()
    scheduler.step()

    # optimizer, lr = update_optimizer(optimizer, lr, c_A)

    #64x10 1x64
    for i, data in enumerate(trainloader):
        inputs ,relations, time, events = data
        inputs, relations, time, events = Variable(inputs).double(), Variable(relations).double(), Variable(time), Variable(events)

        # reshape data
        # relations = relations.resize_((list(relations.size())[0],10,1))
        inputs = inputs.unsqueeze(2)
        # inputs = inputs.float()
        optimizer.zero_grad()
        # print(inputs.type())
        # print("hey")
        enc_x, logits, origin_A, adj_A_tilt_encoder, z_gap, z_positive, myA, Wa = encoder(inputs, rel_rec, rel_send)
        edges = logits

        flat = edges.view(-1,data_variable_size)
        flat = flat.float()
        d_out = survival(flat)

        dec_x, output, adj_A_tilt_decoder = decoder(inputs, edges, data_variable_size * x_dims, rel_rec, rel_send, origin_A, adj_A_tilt_encoder, Wa)

        if torch.sum(output != output):
            print('nan error\n')

        target = inputs
        preds = output
        variance = 0

        #reconstruction accuracy loss
        loss_nll1 = nll_gaussian(preds, target, variance)

        loss_kl = kl_gaussian_sem(logits)

        loss_nll2 = NegativeLogLikelihood(events, time, d_out)

        loss = loss_kl + loss_nll1 + loss_nll2

        one_adj_A = origin_A
        sparse_loss = tau_A*torch.sum(torch.abs(one_adj_A))

        h_A = _h_A(origin_A, data_variable_size)
        loss += lambda_A * h_A + 0.5 * c_A * h_A * h_A + 100. * torch.trace(origin_A*origin_A) + sparse_loss

        loss.backward()
        loss = optimizer.step()

        myA.data = stau(myA.data, tau_A*lr)

        if torch.sum(origin_A != origin_A):
            print('nan error\n')

        # compute metric
        graph = origin_A.data.clone().numpy()
        graph[np.abs(graph) < graph_threshold] = 0

        fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.Graph(graph))

        mse_train.append(F.mse_loss(preds, target).item())
        nll1_train.append(loss_nll1.item())
        nll2_train.append(loss_nll2.item())
        kl_train.append(loss_kl.item())
        shd_trian.append(shd)

    return np.mean(np.mean(kl_train)  + np.mean(nll1_train)), np.mean(nll1_train), np.mean(mse_train), graph, origin_A, np.mean(nll2_train), d_out

#===================================
# main
#===================================

best_ELBO_loss = np.inf
best_NLL1_loss = np.inf
best_MSE_loss = np.inf
best_NLL2_loss = np.inf
best_epoch = 0
best_ELBO_graph = []
best_NLL1_graph = []
best_MSE_graph = []


h_A_new = torch.tensor(1.)
h_tol = 1e-8
k_max_iter = 1
h_A_old = np.inf

for step_k in range(k_max_iter):
    while c_A < 1e+20:
        for epoch in range(EPOCH):
            print("iter: "+str(step_k)+" Epoch: "+str(epoch))
            ELBO_loss, NLL1_loss, MSE_loss, graph, origin_A, NLL2_loss, surv_out = train(epoch, best_ELBO_loss, ground_truth_G, lambda_A, c_A, optimizer)
            if ELBO_loss < best_ELBO_loss:
                best_ELBO_loss = ELBO_loss
                best_epoch = epoch
                best_ELBO_graph = graph

            if NLL1_loss < best_NLL1_loss:
                best_NLL1_loss = NLL1_loss
                best_epoch = epoch
                best_NLL1_graph = graph

            if MSE_loss < best_MSE_loss:
                best_MSE_loss = MSE_loss
                best_epoch = epoch
                best_MSE_graph = graph

            if NLL2_loss < best_NLL2_loss:
                best_NLL2_loss = NLL2_loss
                print(NLL2_loss)

        print("Optimization Finished!")
        print("Best Epoch: {:04d}".format(best_epoch))
        if ELBO_loss > 2 * best_ELBO_loss:
            break

        # update parameters
        A_new = origin_A.data.clone()
        h_A_new = _h_A(A_new, data_variable_size)
        if h_A_new.item() > 0.25 * h_A_old:
            c_A*=10
        else:
            break

        # update parameters
        # h_A, adj_A are computed in loss anyway, so no need to store
    h_A_old = h_A_new.item()
    lambda_A += c_A * h_A_new.item()

    if h_A_new.item() <= h_tol:
        break

print("Best Epoch: {:04d}".format(best_epoch))

# test()
#Metric - Concordance index
X_tr = ((torch.Tensor(X_train)).unsqueeze(2)).double()
enc_x, logits, origin_A, adj_A_tilt_encoder, z_gap, z_positive, myA, Wa = encoder(X_tr, rel_rec, rel_send)
edges = logits

flat = edges.view(-1,data_variable_size).float()
hr_pred = survival(flat)
hr_pred = (torch.exp(hr_pred)).detach().numpy()
ci_train=concordance_index(Y_train,-hr_pred,E_train)

X_te = ((torch.Tensor(X_val)).unsqueeze(2)).double()
enc_x, logits, origin_A, adj_A_tilt_encoder, z_gap, z_positive, myA, Wa = encoder(X_te, rel_rec, rel_send)
edges = logits

flat = edges.view(-1,data_variable_size).float()
hr_pred2 = survival(flat)
hr_pred2 = (torch.exp(hr_pred2)).detach().numpy()
ci_test = concordance_index(Y_val,-hr_pred2,E_val)

print('Concordance Index for training dataset:', ci_train)
print('Concordance Index for test dataset:', ci_test)
#End of Survival metric

print("Best NLL2 loss: ",best_NLL2_loss)

print (best_ELBO_graph)
print(nx.to_numpy_array(ground_truth_G))
fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.Graph(best_ELBO_graph))
print('Best ELBO Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

print(best_NLL1_graph)
print(nx.to_numpy_array(ground_truth_G))
fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.Graph(best_NLL1_graph))
print('Best NLL Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)


print (best_MSE_graph)
print(nx.to_numpy_array(ground_truth_G))
fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.Graph(best_MSE_graph))
print('Best MSE Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

graph = origin_A.data.clone().numpy()
graph[np.abs(graph) < 0.1] = 0
# print(graph)
fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.Graph(graph))
print('threshold 0.1, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

graph[np.abs(graph) < 0.2] = 0
# print(graph)
fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.Graph(graph))
print('threshold 0.2, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)

graph[np.abs(graph) < 0.3] = 0
# print(graph)
fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.Graph(graph))
print('threshold 0.3, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz)
