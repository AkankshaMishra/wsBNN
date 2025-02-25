# Modified: 20 Feb 2025
# Author: Akanksha Mishra
# Apply FSBNN-ws on Friedman1 dataset

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import datetime
import sys
sys.path.insert(0, '/Users/Akanksha Mishra/Documents/genomics/code/wsBNN/regression')

from tools import sigmoid
from sklearn.model_selection import train_test_split
from sparse_bnn_vhd import FeatureSelectionBNN
from sklearn.datasets import make_friedman1
from sklearn.preprocessing import StandardScaler
from mlxtend.evaluate import bias_variance_decomp

torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

curr_dir = "/Users/Akanksha Mishra/Documents/genomics/code/wsBNN/regression/simulated_data"
code = "wsbnn"

# ------------------------------------------------------------------------------------------------------
data_size = 5000
test_size = 1000
data_dim = 100

trainsets = []
x_data, y = make_friedman1(n_samples = data_size, n_features = data_dim)
scaler_y = StandardScaler()
y_data = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.20)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10)

# Convert train, test and valid data to tensor
x_train = torch.Tensor(x_train).to(device)
y_train = torch.LongTensor(y_train).to(device)
x_test = torch.Tensor(x_test).to(device)
y_test = torch.LongTensor(y_test).to(device)
x_val = torch.Tensor(x_val).to(device)
y_val = torch.LongTensor(y_val).to(device)

trainsets = [[x_train, y_train]]
# ------------------------------------------------------------------------------------------------------
batch_size = 128
num_batches = data_size / batch_size
learning_rate = torch.tensor(1e-3)
epochs = 1000
hidden_dim = [16, 8, 4]
L = 3
total = (data_dim+1) * hidden_dim[0] + (hidden_dim[0]+1) * hidden_dim[1] + (hidden_dim[1]+1) * hidden_dim[2] + (hidden_dim[2]+1) * 1
a = np.log(total) + 0.1*np.log(hidden_dim[0]) + 0.1*np.log(hidden_dim[1]) + 0.1*np.log(hidden_dim[2]) + np.log(np.sqrt(data_size)*data_dim)
lm = 1/np.exp(a)
phi_prior = torch.tensor(lm)
temp = 0.5
n_MC_samples = 30
data_size = x_train.shape[0]

# Prepare for training
print('Train size:{}, Test size:{}, Total Features:{}, Epochs:{}, Hidden Layers:{}, Hidden Dims:{}'.format(x_train.shape[0], test_size, data_dim, epochs, L, hidden_dim))

# Prepare the header for the metrics table
metrics_header = "Run\tTest MSE\tTest Bias\tTest Variance\n"
metrics_rows = []

start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
start_time_dt = datetime.datetime.now()

# Wrap the FSBNN model in a scikit-learn compatible wrapper
class FSBNNWrapper:
    def __init__(self, model, device, temp, phi_prior, num_batches):
        self.model = model
        self.device = device
        self.temp = temp
        self.phi_prior = phi_prior
        self.num_batches = num_batches

    def fit(self, X_train, y_train):
        # Convert data to tensors
        X_train = torch.Tensor(X_train).to(self.device)
        y_train = torch.Tensor(y_train).to(self.device).long()  # Ensure labels are long type
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)  # Adjust learning rate
        
        self.model.train()
        for epoch in range(epochs):  # Train for a few epochs
            optimizer.zero_grad()
            loss, _, _, _ = self.model.module.sample_elbo(X_train, y_train, n_MC_samples, self.temp, self.phi_prior, self.num_batches)
            loss.backward()
            optimizer.step()

        return self  # Return the trained model

    def predict(self, X):
        X = torch.Tensor(X).to(self.device)
        _, _, _, preds = self.model.module.sample_elbo(
            X, torch.zeros(X.shape[0], dtype=torch.long).to(self.device), 
            n_MC_samples, self.temp, self.phi_prior, self.num_batches
        )
        return preds.mean(dim=0).cpu().detach().numpy().reshape(-1)

for k in range(1, 2):

    # Set seed for each run
    np.random.seed(k)
    torch.manual_seed(k)
    print('------------ round {} ------------'.format(k))
    net = FeatureSelectionBNN(data_dim=data_dim, hidden_dim = hidden_dim, device=device).to(device)
    net = nn.DataParallel(net)
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, foreach=False)
    x_train = trainsets[0][0]
    y_train = trainsets[0][1]
    training_loss = []
    validation_loss = []

    for epoch in range(epochs):  # loop over the dataset multiple times
        train_losses = []
        valid_losses = []
        permutation = torch.randperm(x_train.size(0)).to(device)  # Ensure permutation tensor is on device

        for i in range(0, data_size, batch_size):
            optimizer.zero_grad()
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = x_train[indices], y_train[indices]
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            loss, nll_MC,kl_MC, _ = net.module.sample_elbo(batch_x, batch_y, n_MC_samples, temp, phi_prior, num_batches)
            if torch.isnan(loss):
                break
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
        
        training_loss.append(np.mean(train_losses))

        # Validation step
        net.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            val_loss, _, _, _ = net.module.sample_elbo(x_val.to(device), y_val.to(device), n_MC_samples, temp, phi_prior, num_batches)
            valid_losses.append(val_loss.item())
        validation_loss.append(np.mean(valid_losses))
        net.train()  # Set the model back to training mode

        print('Epoch {}, Train_Loss: {}, Valid _Loss: {}, phi_prior: {}'.format(epoch, np.mean(train_losses), np.mean(valid_losses), phi_prior))
    
    print('Epoch {}, Train_Loss: {}'.format(epoch, np.mean(train_losses)))
    print('Finished Training')
    torch.save(net.state_dict(), f"{curr_dir}/{code}/model_run{k}.pth")

    print("\n", "----------- Network Sparsity -----------")
    one1_w = (sigmoid(net.module.l1.w_theta)).float()
    # print('l1 Overall w sparsity: {}'.format(torch.mean(one1_w)))
    # print('l1 w Edges: {}'.format(one1_w))
    # p = torch.mean(one1_w, axis=1)
    sorted, indices = torch.sort(one1_w, 0, descending=True) #write one1_w in place of p for weight sharing
    print('features selected in the first layer: {}'.format(indices[0:10]))
    torch.save(sorted, f"{curr_dir}/{code}/weights_sorted_run{k}.pt")
    indices_cpu = indices.cpu().numpy()
    np.save(f"{curr_dir}/{code}/indices_sorted_run{k}", indices_cpu)

    # Prepare for bias-variance decomposition
    x_train_cpu = x_train.cpu().numpy()
    y_train_cpu = scaler_y.inverse_transform(y_train.cpu().reshape(-1, 1)).flatten()
    x_test_cpu = x_test.cpu().numpy()
    y_test_cpu = scaler_y.inverse_transform(y_test.cpu().reshape(-1, 1)).flatten()

    # Wrap trained model for bias-variance decomposition
    fsbnn_wrapper = FSBNNWrapper(net, device, temp, phi_prior, num_batches)
    mse2, bias2, var2 = bias_variance_decomp(estimator=fsbnn_wrapper, 
                                            X_train=x_train_cpu, y_train=y_train_cpu, 
                                            X_test=x_test_cpu, y_test=y_test_cpu, 
                                            loss='mse', num_rounds=30, random_seed=42)

    # Print the results
    print(f"Test MSE (mlxtend): {mse2}")
    print(f"Test Bias (mlxtend): {bias2}")
    print(f"Test Variance (mlxtend): {var2}")

    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics_row = f"{k}\t{mse2}\t{bias2}\t{var2}\n"
    metrics_rows.append(metrics_row)

    # Plotting the training and validation loss curves for the current run
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), np.log(training_loss), label='Training Loss')
    plt.plot(range(epochs), np.log(validation_loss), label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss Curves for Run {k}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{curr_dir}/{code}/loss_curves_run{k}.png")
    plt.close()

end_time_dt = datetime.datetime.now()
total_execution_time = end_time_dt - start_time_dt

# Write metrics to file
with open(f"{curr_dir}/{code}/result.txt", "a") as f:
    f.write(f"Total execution time: {total_execution_time}\n")
    f.write(f"Start time: {start_time}\n")
    f.write(metrics_header)
    f.writelines(metrics_rows)
    f.write(f"End time: {end_time}\n")