#Modified: 16 Sep 2024
#Author: Akanksha Mishra
# Apply SBNN on GISETTE dataset
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time
import sys
sys.path.insert(0, '/workspace/classification')

import scipy.io
from tools import sigmoid
from sparse_bnn_classification_vhd import SparseBNNClassification
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, classification_report

start_time = time.time()
torch.set_default_dtype(torch.float64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

curr_dir = "/workspace/classification/dataset-stg-4"
code = "sbnn_samplemean"

start_date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#------------------------------------------------------------------------------------------------------
# Create a simple dataset
mat = scipy.io.loadmat(f'{curr_dir}/gisette.mat')
X = mat['X']
Y = mat['Y'].reshape((X.shape[0]))
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)
label_encoder_name_mapping = dict(zip(label_encoder.classes_,
                                         label_encoder.transform(label_encoder.classes_)))
print("Mapping of Label Encoded Classes", label_encoder_name_mapping, sep="\n")

X = torch.Tensor(preprocessing.normalize(X,axis=0))
Y = torch.LongTensor(Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.90)

# Convert train, test and valid data to tensor
x_train = torch.Tensor(x_train).to(device)
y_train = torch.LongTensor(y_train).to(device)
x_test = torch.Tensor(x_test).to(device)
y_test = torch.LongTensor(y_test).to(device)
x_val = torch.Tensor(x_val).to(device)
y_val = torch.LongTensor(y_val).to(device)

# Generate target function
data_size = X.shape[0]
data_dim = x_train.shape[1]
# target_classes = list(label_encoder_name_mapping.keys())
# Correct the target_classes to be strings
target_classes = [str(label) for label in list(label_encoder_name_mapping.keys())]
no_of_classes = len(target_classes)
sigma_noise = 1.
rep = 1

trainsets = [[x_train,y_train]]
print(x_train.shape, x_test.shape)

# ------------------------------------------------------------------------------------------------------
batch_size = x_train.shape[0]
num_batches = data_size / batch_size
learning_rate = torch.tensor(1e-3)
epochs = 10000
hidden_dim = [16,8,4]
L = 3
# Is to updated for feature selection model (consider only first layer)
# total = (data_dim+1) * hidden_dim + (hidden_dim+1) * hidden_dim + (hidden_dim+1) * hidden_dim + (hidden_dim+1) * 1
# a = np.log(total) + 0.1*((L+1)*np.log(hidden_dim) + np.log(np.sqrt(data_size)*data_dim))
total = (data_dim+1) * hidden_dim[0] + (hidden_dim[0]+1) * hidden_dim[1] + (hidden_dim[1]+1) * hidden_dim[2] + (hidden_dim[2]+1) * 1
a = np.log(total) + 0.1*(np.log(hidden_dim[0]) + 0.1*np.log(hidden_dim[1]) + 0.1*np.log(hidden_dim[2]) + np.log(np.sqrt(data_size)*data_dim))
lm = 1/np.exp(a)
phi_prior = torch.tensor(lm)
temp = 0.5


train_Loss = []
test_Loss = []
sparse_overalls = []
sparse_overalls2 = []
FNRs = []
FPRs = []
no_of_features_selected = []
training_loss = []
validation_loss = []
data_size = x_train.shape[0]

print('Train size:{}, Test size:{}, Total Features:{}, Epochs:{}, Hidden Layers:{}, Hidden Dims:{}'.format(data_size, x_test.shape[0], data_dim, epochs, L, hidden_dim))

# Prepare the header for the metrics table
metrics_header = "Run\tTest Accuracy\tWeighted Precision\tWeighted Recall\tWeighted F1 Score\n"
metrics_rows = []

for k in range(1, 2):
    # Set seed for each run
    np.random.seed(k)
    torch.manual_seed(k)
    print('------------ round {} ------------'.format(k))

    # create sparse BNN
    net = SparseBNNClassification(data_dim, hidden_dim = hidden_dim, target_dim = no_of_classes, device = device).to(device)
    # Wrap model in DataParallel
    net = nn.DataParallel(net)
    net.to(device)
    print(net.parameters())

    optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate, foreach=False)
    x_train = trainsets[0][0]
    y_train = trainsets[0][1]
    training_loss = []
    validation_loss = []
    train_accuracy = []
    test_accuracy = []
    val_accuracy = []

    for epoch in range(epochs): 
        train_losses = []
        permutation = torch.randperm(data_size)

        for i in range(0, data_size, batch_size):
            optimizer.zero_grad() # Sets gradients of all model parameters to zero.
            indices = permutation[i : i + batch_size]
            batch_x, batch_y = x_train[indices], y_train[indices]
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            loss, _ = net.module.sample_elbo(batch_x, batch_y, 1, temp, phi_prior, num_batches)
            if torch.isnan(loss):
                break
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        training_loss.append(np.mean(train_losses))

        # one1_w = (sigmoid(net.l1.w_theta) > 0.5).float()
        # p = torch.sum(one1_w, axis=1)
        # no_of_features_selected.append(torch.sum(p>=1))

        # Validation step
        net.eval()  # Set the model to evaluation mode
        valid_losses = []
        with torch.no_grad():
            _, val_pred = net.module.sample_elbo(x_val.to(device), y_val.to(device), 30, temp, phi_prior, num_batches)
            val_pred = torch.mode(val_pred, dim=0).values
            val_loss = torch.sum(val_pred != y_val.to(device)) / y_val.shape[0]
            valid_losses.append(val_loss.item())
            val_accuracy.append(torch.sum(val_pred == y_val.to(device)) / y_val.shape[0])
        validation_loss.append(np.mean(valid_losses))
        net.train()  # Set the model back to training mode

        _, pred = net.module.sample_elbo(x_train.to(device), y_train.to(device), 30, temp, phi_prior, num_batches)
        pred = torch.mode(pred, dim=0).values
        train_accuracy.append(torch.sum(pred == y_train) / y_train.shape[0])

        _, pred2 = net.module.sample_elbo(x_test.to(device), y_test.to(device), 30, temp, phi_prior, num_batches)
        pred2 = torch.mode(pred2, dim=0).values
        test_accuracy.append(torch.sum(pred2 == y_test) / y_test.shape[0])

        if epoch % 1000 == 0:
            one1_w = (net.module.l1.w != 0).float()
            one1_b = (net.module.l1.b != 0).float()
            one2_w = (net.module.l2.w != 0).float()
            one2_b = (net.module.l2.b != 0).float()
            one3_w = (net.module.l3.w != 0).float()
            one3_b = (net.module.l3.b != 0).float()
            one4_w = (net.module.l4.w != 0).float()
            one4_b = (net.module.l4.b != 0).float()
            sparsity = (torch.sum(one1_w) + torch.sum(one2_w) + torch.sum(one3_w) + torch.sum(one4_w) + 
                        torch.sum(one1_b) + torch.sum(one2_b) + torch.sum(one3_b) + torch.sum(one4_b)) / total
            print('Epoch {}, Train_Loss: {}, phi_prior: {}, sparsity: {}'.format(
                epoch, np.mean(train_losses), phi_prior, sparsity))
    print('Epoch {}, Train_Loss: {}'.format(epoch, np.mean(train_losses)))
    print("Finished Training")

    torch.save(net.state_dict(), f"{curr_dir}/{code}/model_run{k}.pth")

    # sparsity level
    print('l1.w_theta final: {}'.format(net.module.l1.w_theta))
    one1_w = (sigmoid(net.module.l1.w_theta)).float()
    one1_b = (sigmoid(net.module.l1.b_theta) > 0.5).float()
    one2_w = (sigmoid(net.module.l2.w_theta) > 0.5).float()
    one2_b = (sigmoid(net.module.l2.b_theta) > 0.5).float()
    one3_w = (sigmoid(net.module.l3.w_theta) > 0.5).float()
    one3_b = (sigmoid(net.module.l3.b_theta) > 0.5).float()
    one4_w = (sigmoid(net.module.l4.w_theta) > 0.5).float()
    one4_b = (sigmoid(net.module.l4.b_theta) > 0.5).float()
    sparse_overall = (torch.sum(one1_w) + torch.sum(one2_w) + torch.sum(one3_w) + torch.sum(one4_w) +
                        torch.sum(one1_b) + torch.sum(one2_b) + torch.sum(one3_b) + torch.sum(one4_b)) / total
    sparse_overalls.append(sparse_overall)
    sparse_overall2 = (torch.sum(sigmoid(net.module.l1.w_theta)) + torch.sum(sigmoid(net.module.l1.b_theta)) +
                       torch.sum(sigmoid(net.module.l2.w_theta)) + torch.sum(sigmoid(net.module.l2.b_theta)) +
                       torch.sum(sigmoid(net.module.l3.w_theta)) + torch.sum(sigmoid(net.module.l3.b_theta)))/total
    sparse_overalls2.append(sparse_overall2)
    torch.set_printoptions(profile="full")

    print("\n", "----------- Network Sparsity -----------")
    print('l1 Overall w sparsity: {}'.format(torch.mean(one1_w)))
    print('l1 w Edges: {}'.format(one1_w))
    p = torch.mean(one1_w, axis=1)
    sorted, indices = torch.sort(p,0, descending=True)
    print('features selected in the first layer: {}'.format(indices[0:10]))
    torch.save(sorted, f"{curr_dir}/{code}/weights_sorted_run{k}.pt")
    # Move indices to CPU before saving
    indices_cpu = indices.cpu().numpy()
    np.save(f"{curr_dir}/{code}/indices_sorted_run{k}", indices_cpu)

    print('l1 Overall b sparsity: {}'.format(torch.mean(one1_b)))
    print('l1 b Edges: {}'.format(one1_b))

    # prediction
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    _, pred = net.module.sample_elbo(x_train, y_train, 30,
                                temp, phi_prior, num_batches)
    print("shape of output -> ", pred.shape)

    pred = torch.mode(pred, dim=0).values
    train_loss = torch.sum(pred != y_train) / y_train.shape[0]
    train_Loss.append(train_loss)

    print("----------- Training -----------")
    print('y_train: {}'.format(y_train[0:20]))
    print('pred_train: {}'.format(pred[0:20]))
    print('binary_loss_train: {}'.format(train_loss))

    # Metrics
    train_acc = accuracy_score(y_train.cpu().numpy(), pred.cpu().numpy())
    train_precision = precision_score(y_train.cpu().numpy(), pred.cpu().numpy(), average='weighted')
    train_recall = recall_score(y_train.cpu().numpy(), pred.cpu().numpy(), average='weighted')
    train_f1 = f1_score(y_train.cpu().numpy(), pred.cpu().numpy(), average='weighted')
    
    print(f"Train Accuracy: {train_acc}")
    print(f"Weighted Precision: {train_precision}")
    print(f"Weighted Recall: {train_recall}")
    print(f"Weighted F1 Score: {train_f1}")

    # testing
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    _, pred2 = net.module.sample_elbo(x_test, y_test, 30, temp, phi_prior, num_batches)
    print("shape of output -> ", pred2.shape)
    
    pred2 = torch.mode(pred2, dim=0).values
    test_loss = torch.sum(pred2 == y_test) / y_test.shape[0]
    test_Loss.append(test_loss)

    print('y_test: {}'.format(y_test[0:20]))
    print('pred_test: {}'.format(pred2[0:20]))
    print('binary_loss_test: {}'.format(test_loss))

    # Metrics
    test_acc = accuracy_score(y_test.cpu().numpy(), pred2.cpu().numpy())
    test_precision = precision_score(y_test.cpu().numpy(), pred2.cpu().numpy(), average='weighted')
    test_recall = recall_score(y_test.cpu().numpy(), pred2.cpu().numpy(), average='weighted')
    test_f1 = f1_score(y_test.cpu().numpy(), pred2.cpu().numpy(), average='weighted')

    print(f"Test Accuracy: {test_acc}")
    print(f"Weighted Precision: {test_precision}")
    print(f"Weighted Recall: {test_recall}")
    print(f"Weighted F1 Score: {test_f1}")

    metrics_row = f"{k}\t{test_acc}\t{test_precision}\t{test_recall}\t{test_f1}\n"
    metrics_rows.append(metrics_row)

    print(classification_report(y_test.cpu().numpy(), pred2.cpu().numpy(), target_names=target_classes))
    print(confusion_matrix(y_test.cpu().numpy(), pred2.cpu().numpy()))
    cm = confusion_matrix(y_test.cpu().numpy(), pred2.cpu().numpy())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig(f"{curr_dir}/{code}/cm_run{k}.png")
    plt.close()
    print("Confusion Matrix saved")

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
    print(f"Loss curves for run {k} saved")

# Save metrics to a text file
metrics_table = metrics_header + ''.join(metrics_rows)
end_time = time.time()
execution_time = end_time - start_time
current_date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

with open(f"{curr_dir}/{code}/metrics.txt", "a") as metrics_file:
    metrics_file.write(f"Total Execution Time: {execution_time} seconds\n")
    metrics_file.write(f"Start Time: {start_date_time}\n\n")
    metrics_file.write(metrics_table)
    metrics_file.write(f"\nEnd Time: {current_date_time}\n")

print("\nTotal Execution Time: {} seconds".format(execution_time))
print(start_date_time)
print(current_date_time)