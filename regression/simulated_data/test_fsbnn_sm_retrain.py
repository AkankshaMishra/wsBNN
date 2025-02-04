# Modified: 23 Aug 2024
# Author: Akanksha Mishra
# Apply FSBNN-ws on Friedman1 dataset

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import datetime
import time
import sys
sys.path.insert(0, '/home/akanksha19231201/projects/wsbnn-sbnn/regression')

from tools import sigmoid
from sklearn.model_selection import train_test_split
from sparse_bnn_vhd import FeatureSelectionBNN
from sklearn.datasets import make_friedman1
from scipy.stats import norm

torch.set_default_dtype(torch.float64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

curr_dir = "/home/akanksha19231201/projects/wsbnn-sbnn/regression/friedman1"
code = "fsbnn_sm"

# ------------------------------------------------------------------------------------------------------
data_size = 5000
test_size = 1000
data_dim = 100
sigma_noise = 1.

trainsets = []
x, y_data = make_friedman1(n_samples = data_size, n_features = data_dim)

selected_features = np.load(f"{curr_dir}/{code}/indices_sorted_run1.npy")
selected_indices = selected_features[0:10]
print(selected_indices)
x_data = x[:, selected_indices]
data_dim = len(selected_indices) 
print(x_data.shape)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.20)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.90)

# Convert train, test and valid data to tensor
x_train = torch.Tensor(x_train).to(device)
y_train = torch.LongTensor(y_train).to(device)
x_test = torch.Tensor(x_test).to(device)
y_test = torch.LongTensor(y_test).to(device)
x_val = torch.Tensor(x_val).to(device)
y_val = torch.LongTensor(y_val).to(device)

trainsets = [[x_train, y_train]]
print(x_train.shape, x_test.shape)
# ------------------------------------------------------------------------------------------------------
batch_size = x_train.shape[0]
num_batches = data_size / batch_size
learning_rate = torch.tensor(1e-5)
epochs = 100000
hidden_dim = [16, 8, 4]
L = 3
total = (data_dim+1) * hidden_dim[0] + (hidden_dim[0]+1) * hidden_dim[1] + (hidden_dim[1]+1) * hidden_dim[2] + (hidden_dim[2]+1) * 1
a = np.log(total) + 0.1*np.log(hidden_dim[0]) + 0.1*np.log(hidden_dim[1]) + 0.1*np.log(hidden_dim[2]) + np.log(np.sqrt(data_size)*data_dim)
lm = 1/np.exp(a)
phi_prior = torch.tensor(lm)
temp = 0.5

test_MSEs = []
test_biases = []
test_variances = []
data_size = x_train.shape[0]

print('Train size:{}, Test size:{}, Total Features:{}, Epochs:{}, Hidden Layers:{}, Hidden Dims:{}'.format(data_size, test_size, data_dim, epochs, L, hidden_dim))

# Prepare the header for the metrics table
metrics_header = "Run\tTest MSE\tTest Bias\tTest Variance\n"
metrics_rows = []

start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
start_time_dt = datetime.datetime.now()

for k in range(11, 12):

    # Set seed for each run
    np.random.seed(k)
    torch.manual_seed(k)
    print('------------ round {} ------------'.format(k))

    # create FS sparse BNN
    net = FeatureSelectionBNN(data_dim=data_dim, hidden_dim = hidden_dim, device=device).to(device)
    # Wrap model in DataParallel
    net = nn.DataParallel(net)
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, foreach=False)
    x_train = trainsets[0][0]
    y_train = trainsets[0][1]
    training_loss = []
    validation_loss = []
    train_accuracy = []
    test_accuracy = []
    val_accuracy = []

    for epoch in range(epochs):  # loop over the dataset multiple times
        train_losses = []
        log_likelihoods = []
        kls = []
        permutation = torch.randperm(x_train.size(0)).to(device)  # Ensure permutation tensor is on device

        for i in range(0, data_size, batch_size):
            optimizer.zero_grad()
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = x_train[indices], y_train[indices]
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            loss, nll_MC,kl_MC, _ = net.module.sample_elbo(batch_x, batch_y, 1, temp, phi_prior, num_batches)
            if torch.isnan(loss):
                break
            train_losses.append(loss.item())
            log_likelihoods.append(nll_MC.item())
            kls.append(kl_MC.item())
            loss.backward()
            optimizer.step()
        
        training_loss.append(np.mean(train_losses))

        # Validation step
        net.eval()  # Set the model to evaluation mode
        valid_losses = []
        with torch.no_grad():
            _, _, _, val_pred = net.module.sample_elbo(x_val.to(device), y_val.to(device), 30, temp, phi_prior, num_batches)
            val_pred = torch.mode(val_pred, dim=0).values
            val_loss = torch.sum(val_pred != y_val.to(device)) / y_val.shape[0]
            valid_losses.append(val_loss.item())
            val_accuracy.append(torch.sum(val_pred == y_val.to(device)) / y_val.shape[0])
        validation_loss.append(np.mean(valid_losses))
        net.train()  # Set the model back to training mode

        _, _, _, pred = net.module.sample_elbo(x_train.to(device), y_train.to(device), 30, temp, phi_prior, num_batches)
        pred = torch.mode(pred, dim=0).values
        train_accuracy.append(torch.sum(pred == y_train) / y_train.shape[0])

        _, _, _, pred2 = net.module.sample_elbo(x_test.to(device), y_test.to(device), 30, temp, phi_prior, num_batches)
        pred2 = torch.mode(pred2, dim=0).values
        test_accuracy.append(torch.sum(pred2 == y_test) / y_test.shape[0])

        if epoch % 1000 == 0:
            one1_w = (net.module.l1.w != 0).float()
            one1_b = (net.module.l1.b != 0).float()
            sparsity = torch.sum(one1_w) + torch.sum(one1_b)
            print('Epoch {}, Train_Loss: {}, Log_likelihood: {},KL: {}, phi_prior: {}, sparsity: {}'.format(epoch, np.mean(train_losses), np.mean(log_likelihoods), np.mean(kls), phi_prior,
                                                                                 sparsity))
    
    print('Epoch {}, Train_Loss: {}'.format(epoch, np.mean(train_losses)))
    print('Finished Training')

    torch.save(net.state_dict(), f"{curr_dir}/{code}/model_run{k}.pth")

    print("\n", "----------- Network Sparsity -----------")
    one1_w = (sigmoid(net.module.l1.w_theta)).float()
    one1_b = (sigmoid(net.module.l1.b_theta) > 0.5).float()

    print('l1 Overall w sparsity: {}'.format(torch.mean(one1_w)))
    print('l1 w Edges: {}'.format(one1_w))
    # p = torch.mean(one1_w, axis=1)
    sorted, indices = torch.sort(one1_w, 0, descending=True) #write one1_w in place of p for weight sharing
    print('features selected in the first layer: {}'.format(indices[0:10]))
    torch.save(sorted, f"{curr_dir}/{code}/weights_sorted_run{k}.pt")
    indices_cpu = indices.cpu().numpy()
    np.save(f"{curr_dir}/{code}/indices_sorted_run{k}", indices_cpu)

    print('l1 Overall b sparsity: {}'.format(torch.mean(one1_b)))
    print('l1 b Edges: {}'.format(one1_b))

    # prediction
    _, _, _, pred = net.module.sample_elbo(x_train, y_train, 30, temp, phi_prior, num_batches)
    pred = pred.mean(dim=0)
    train_mse = torch.mean((pred - y_train) ** 2)
    # train_mse = torch.sqrt(torch.mean((pred - y_train) ** 2))

    print("----------- Training -----------")
    print('y_train: {}'.format(y_train[0:20]))
    print('pred_train: {}'.format(pred[0:20]))
    print('MSE_train: {}'.format(train_mse))

    print("\n", "----------- Testing -----------")
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    _, _, _, pred2 = net.module.sample_elbo(x_test, y_test, 30, temp, phi_prior, num_batches)
    pred2 = pred2.mean(dim=0)
    test_mse = torch.mean((pred2 - y_test) ** 2)
    # test_mse = torch.sqrt(torch.mean((pred2 - y_test) ** 2))
    test_MSEs.append(test_mse.data)

    print('y_test: {}'.format(y_test[0:20]))
    print('pred_test: {}'.format(pred2[0:20]))
    print('MSE_test: {}'.format(test_mse))
    print("\n")

    # Calculate Bias and Variance
    bias = torch.mean(pred2 - y_test)
    # bias = torch.mean((pred2 - y_test) ** 2)
    variance = torch.var(pred2)

    test_biases.append(bias.data)
    test_variances.append(variance.data)

    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics_row = f"{k}\t{test_mse.item()}\t{bias.item()}\t{variance.item()}\n"
    metrics_rows.append(metrics_row)

end_time_dt = datetime.datetime.now()
total_execution_time = end_time_dt - start_time_dt

# Write metrics to file
with open(f"{curr_dir}/{code}/retrain_result.txt", "a") as f:
    f.write(f"Total execution time: {total_execution_time}\n")
    f.write(f"Start time: {start_time}\n")
    f.write(metrics_header)
    f.writelines(metrics_rows)
    f.write(f"End time: {end_time}\n")

train_MSE = torch.tensor(test_MSEs)
biases = torch.tensor(test_biases)
variances = torch.tensor(test_variances)

print("\n", "----------- Summary -----------")
print('MSE_test: {}'.format(torch.mean(train_MSE)))
print('MSE_test_sd: {}'.format(torch.std(train_MSE)))
print('Bias: {}'.format(torch.mean(biases)))
print('Variance: {}'.format(torch.mean(variances)))

print('MSE_test all: {}'.format(train_MSE))
print('Bias all: {}'.format(biases))
print('Variance all: {}'.format(variances))