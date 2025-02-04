# Modified: 5 July 2024
# Author: Akanksha Mishra
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import sys
sys.path.insert(0, '/workspace/classification')

from tools import sigmoid
from sklearn.model_selection import train_test_split
from sparse_bnn_classification_vhd import FeatureSelectionBNNClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from scipy.special import expit

start_time = time.time()
torch.set_default_dtype(torch.float64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

curr_dir = "/workspace/classification/friedman"
code = "fsbnn_samplemean"

start_date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ------------------------------------------------------------------------------------------------------
# Create a simple dataset
def generate_friedman_classification_dataset(n, p):
    data = np.random.uniform(0, 1, (n, p))
    Y = 10 * np.sin(np.square(data[:, 0] * data[:, 1])) + 20 * np.square(data[:, 2]) + 10 * np.sign(data[:, 3] * data[:, 4] - 0.2) + np.random.normal(0, 1)
    y = np.where(expit(Y) > 0.5, 1, 0)
    return data, y

data_size = 5000
test_size = 1000
data_dim = 100
no_of_classes = 2
target_classes = ['Class 1', 'Class 2']
sigma_noise = 1.
rep = 10

trainsets = []
x_data, y_data = generate_friedman_classification_dataset(data_size, data_dim)
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
learning_rate = torch.tensor(1e-3).to(device)
epochs = 1
hidden_dim = [16, 8, 4]
L = 3
total = (data_dim + 1) * hidden_dim[0] + (hidden_dim[0] + 1) * hidden_dim[1] + (hidden_dim[1] + 1) * hidden_dim[2] + (hidden_dim[2] + 1) * 1
a = np.log(total) + 0.1 * (np.log(hidden_dim[0]) + 0.1 * np.log(hidden_dim[1]) + 0.1 * np.log(hidden_dim[2]) + np.log(np.sqrt(data_size) * data_dim))
lm = 1 / np.exp(a)
phi_prior = torch.tensor(lm).to(device)
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
data_size = 3600

# Prepare the header for the metrics table
metrics_header = "Run\tTest Accuracy\tWeighted Precision\tWeighted Recall\tWeighted F1 Score\n"
metrics_rows = []

for k in range(12, 13):
    # Set seed for each run
    np.random.seed(k)
    torch.manual_seed(k)
    print('------------ round {} ------------'.format(k))

    # create FS sparse BNN
    net = FeatureSelectionBNNClassification(data_dim, hidden_dim=hidden_dim, target_dim=no_of_classes, device=device).to(device)
    # Wrap model in DataParallel
    net = nn.DataParallel(net)
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, foreach=False)  # <- Set foreach to False
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
            optimizer.zero_grad()  # Sets gradients of all model parameters to zero.
            indices = permutation[i: i + batch_size]
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

        if epoch % 1000 == 0 or epoch == epochs - 1:
            one1_w = (net.module.l1.w != 0).float()
            one1_b = (net.module.l1.b != 0).float()
            sparsity = (torch.sum(one1_w) + torch.sum(one1_b)) / total
            print('Epoch {}, Train_Loss: {}, phi_prior: {}, sparsity: {}'.format(epoch, np.mean(train_losses), phi_prior, sparsity))

    print('Epoch {}, Train_Loss: {}'.format(epoch, np.mean(train_losses)))
    print("Finished Training")

    torch.save(net.state_dict(), f"{curr_dir}/{code}/model_run{k}.pth")

    # sparsity level
    print('l1.w_theta final: {}'.format(net.module.l1.w_theta))
    one1_w = (sigmoid(net.module.l1.w_theta)).float()
    one1_b = (sigmoid(net.module.l1.b_theta) > 0.5).float()
    sparse_overall = (torch.sum(one1_w) + torch.sum(one1_b)) / total
    sparse_overalls.append(sparse_overall)
    sparse_overall2 = (torch.sum(sigmoid(net.module.l1.w_theta)) + torch.sum(sigmoid(net.module.l1.b_theta))) / total
    sparse_overalls2.append(sparse_overall2)
    torch.set_printoptions(profile="full")

    print("\n", "----------- Network Sparsity -----------")
    print('l1 Overall w sparsity: {}'.format(torch.mean(one1_w)))
    print('l1 w Edges: {}'.format(one1_w))
    sorted, indices = torch.sort(one1_w, 0, descending=True)
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
    _, pred = net.module.sample_elbo(x_train, y_train, 30, temp, phi_prior, num_batches)
    print("shape of output -> ", pred.shape)

    pred = torch.mode(pred, dim=0).values
    train_loss = torch.sum(pred != y_train) / y_train.shape[0]
    train_Loss.append(train_loss)

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
    test_loss = torch.sum(pred2 != y_test) / y_test.shape[0]
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


###############################################################

# #Modified: 3 July 2024
# #Author: Akanksha Mishra
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import datetime
# import time
# import sys
# sys.path.insert(0, '/workspace/classification')

# from tools import sigmoid
# from sklearn.model_selection import train_test_split
# from sparse_bnn_classification_vhd import FeatureSelectionBNNClassification
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
# from scipy.special import expit

# start_time = time.time()
# torch.set_default_dtype(torch.float64)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

# curr_dir = "/workspace/classification/friedman"
# code = "fsbnn_samplemean"

# start_date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# # ------------------------------------------------------------------------------------------------------
# # Create a simple dataset
# def generate_friedman_classification_dataset(n, p):
#     data = np.random.uniform(0, 1, (n, p))
#     Y = 10 * np.sin(np.square(data[:, 0] * data[:, 1])) + 20 * np.square(data[:, 2]) + 10 * np.sign(data[:, 3] * data[:, 4] - 0.2) + np.random.normal(0, 1)
#     y = np.where(expit(Y) > 0.5, 1, 0)
#     return data, y

# data_size = 5000
# test_size = 1000
# data_dim = 100
# no_of_classes = 2
# target_classes = ['Class 1', 'Class 2']
# sigma_noise = 1.
# rep = 10

# trainsets = []
# x_data, y_data = generate_friedman_classification_dataset(data_size, data_dim)
# x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.20)
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.90)

# # Convert train, test and valid data to tensor
# x_train = torch.Tensor(x_train)
# y_train = torch.LongTensor(y_train)
# x_test = torch.Tensor(x_test)
# y_test = torch.LongTensor(y_test)
# x_val = torch.Tensor(x_val)
# y_val = torch.LongTensor(y_val)

# trainsets = [[x_train, y_train]]
# print(x_train.shape, x_test.shape)

# # ------------------------------------------------------------------------------------------------------
# batch_size = x_train.shape[0]
# num_batches = data_size / batch_size
# learning_rate = torch.tensor(1e-3)
# epochs = 50000
# hidden_dim = [16, 8, 4]
# L = 3
# total = (data_dim + 1) * hidden_dim[0] + (hidden_dim[0] + 1) * hidden_dim[1] + (hidden_dim[1] + 1) * hidden_dim[2] + (hidden_dim[2] + 1) * 1
# a = np.log(total) + 0.1 * (np.log(hidden_dim[0]) + 0.1 * np.log(hidden_dim[1]) + 0.1 * np.log(hidden_dim[2]) + np.log(np.sqrt(data_size) * data_dim))
# lm = 1 / np.exp(a)
# phi_prior = torch.tensor(lm)
# temp = 0.5

# train_Loss = []
# test_Loss = []
# sparse_overalls = []
# sparse_overalls2 = []
# FNRs = []
# FPRs = []
# no_of_features_selected = []
# training_loss = []
# validation_loss = []
# data_size = 3600

# # Prepare the header for the metrics table
# metrics_header = "Run\tTest Accuracy\tPrecision\tRecall\tF1 Score\n"
# metrics_rows = []

# for k in range(5, 6):
#     # Set seed for each run
#     np.random.seed(k)
#     torch.manual_seed(k)
#     print('------------ round {} ------------'.format(k))

#     # create FS sparse BNN
#     net = FeatureSelectionBNNClassification(data_dim, hidden_dim=hidden_dim, target_dim=no_of_classes, device=device).to(device)
#     optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
#     x_train = trainsets[0][0]
#     y_train = trainsets[0][1]
#     training_loss = []
#     validation_loss = []
#     train_accuracy = []
#     test_accuracy = []
#     val_accuracy = []

#     for epoch in range(epochs):
#         train_losses = []
#         permutation = torch.randperm(data_size)

#         for i in range(0, data_size, batch_size):
#             optimizer.zero_grad()  # Sets gradients of all model parameters to zero.
#             indices = permutation[i: i + batch_size]
#             batch_x, batch_y = x_train[indices], y_train[indices]
#             batch_x = batch_x.to(device)
#             batch_y = batch_y.to(device)
#             loss, _ = net.sample_elbo(batch_x, batch_y, 1, temp, phi_prior, num_batches)
#             if torch.isnan(loss):
#                 break
#             train_losses.append(loss.item())
#             loss.backward()
#             optimizer.step()

#         training_loss.append(np.mean(train_losses))

#         # Validation step
#         net.eval()  # Set the model to evaluation mode
#         valid_losses = []
#         with torch.no_grad():
#             _, val_pred = net.sample_elbo(x_val.to(device), y_val.to(device), 30, temp, phi_prior, num_batches)
#             val_pred = torch.mode(val_pred, dim=0).values
#             val_loss = torch.sum(val_pred != y_val.to(device)) / y_val.shape[0]
#             valid_losses.append(val_loss.item())
#             val_accuracy.append(torch.sum(val_pred == y_val.to(device)) / y_val.shape[0])
#         validation_loss.append(np.mean(valid_losses))
#         net.train()  # Set the model back to training mode

#         _, pred = net.sample_elbo(x_train.to(device), y_train.to(device), 30, temp, phi_prior, num_batches)
#         pred = torch.mode(pred, dim=0).values
#         train_accuracy.append(torch.sum(pred == y_train) / y_train.shape[0])

#         _, pred2 = net.sample_elbo(x_test.to(device), y_test.to(device), 30, temp, phi_prior, num_batches)
#         pred2 = torch.mode(pred2, dim=0).values
#         test_accuracy.append(torch.sum(pred2 == y_test) / y_test.shape[0])

#         if epoch % 1000 == 0 or epoch == epochs - 1:
#             one1_w = (net.l1.w != 0).float()
#             one1_b = (net.l1.b != 0).float()
#             sparsity = (torch.sum(one1_w) + torch.sum(one1_b)) / total
#             print('Epoch {}, Train_Loss: {}, phi_prior: {}, sparsity: {}'.format(epoch, np.mean(train_losses), phi_prior, sparsity))

#     print('Epoch {}, Train_Loss: {}'.format(epoch, np.mean(train_losses)))
#     print("Finished Training")

#     torch.save(net.state_dict(), f"{curr_dir}/{code}/model_run{k}.pth")

#     # sparsity level
#     print('l1.w_theta final: {}'.format(net.l1.w_theta))
#     one1_w = (sigmoid(net.l1.w_theta)).float()
#     one1_b = (sigmoid(net.l1.b_theta) > 0.5).float()
#     sparse_overall = (torch.sum(one1_w) + torch.sum(one1_b)) / total
#     sparse_overalls.append(sparse_overall)
#     sparse_overall2 = (torch.sum(sigmoid(net.l1.w_theta)) + torch.sum(sigmoid(net.l1.b_theta))) / total
#     sparse_overalls2.append(sparse_overall2)
#     torch.set_printoptions(profile="full")

#     print("\n", "----------- Network Sparsity -----------")
#     print('l1 Overall w sparsity: {}'.format(torch.mean(one1_w)))
#     print('l1 w Edges: {}'.format(one1_w))
#     sorted, indices = torch.sort(one1_w, 0, descending=True)
#     print('features selected in the first layer: {}'.format(indices[0:10]))
#     torch.save(sorted, f"{curr_dir}/{code}/weights_sorted_run{k}.pt")
#     np.save(f"{curr_dir}/{code}/indices_sorted_run{k}", indices)

#     print('l1 Overall b sparsity: {}'.format(torch.mean(one1_b)))
#     print('l1 b Edges: {}'.format(one1_b))

#     # prediction
#     x_train = x_train.to(device)
#     y_train = y_train.to(device)
#     _, pred = net.sample_elbo(x_train, y_train, 30, temp, phi_prior, num_batches)
#     print("shape of output -> ", pred.shape)

#     pred = torch.mode(pred, dim=0).values
#     train_loss = torch.sum(pred != y_train) / y_train.shape[0]
#     train_Loss.append(train_loss)

#     print("----------- Training -----------")
#     print('y_train: {}'.format(y_train[0:20]))
#     print('pred_train: {}'.format(pred[0:20]))
#     print('binary_loss_train: {}'.format(train_loss))

#     # ------------------------------------------------------------------------------------------------------
#     print("\n", "----------- Testing -----------")
#     # testing
#     x_test = x_test.to(device)
#     y_test = y_test.to(device)
#     _, pred2 = net.sample_elbo(x_test, y_test, 30, temp, phi_prior, num_batches)
#     pred2 = torch.mode(pred2, dim=0).values
#     test_loss = torch.sum(pred2 != y_test) / y_test.shape[0]
#     test_Loss.append(test_loss)
#     print('test_loss: {}'.format(test_loss))
#     print('y_test: {}'.format(y_test[0:20]))
#     print('pred_test: {}'.format(pred2[0:20]))

#     train_accuracy.append(accuracy_score(y_train.cpu(), pred.cpu()))
#     test_accuracy.append(accuracy_score(y_test.cpu(), pred2.cpu()))

#     print("\n", "----------- Confusion Matrix -----------")
#     cm = confusion_matrix(y_test.cpu(), pred2.cpu())
#     cm_display = ConfusionMatrixDisplay(cm, display_labels=target_classes).plot()
#     plt.show()
#     plt.savefig(f"{curr_dir}/{code}/confusion_matrix_run{k}.png")
#     torch.save(cm, f"{curr_dir}/{code}/confusion_matrix_run{k}.pt")

#     print("\n", "----------- Metrics -----------")
#     acc = accuracy_score(y_test.cpu(), pred2.cpu())
#     precision = precision_score(y_test.cpu(), pred2.cpu())
#     recall = recall_score(y_test.cpu(), pred2.cpu())
#     f1 = f1_score(y_test.cpu(), pred2.cpu())

#     print('Accuracy: {:.4f}'.format(acc))
#     print('Precision: {:.4f}'.format(precision))
#     print('Recall: {:.4f}'.format(recall))
#     print('F1 Score: {:.4f}'.format(f1))

#     # Save metrics to a row for the table
#     metrics_rows.append(f"Run{k + 1}\t{acc:.4f}\t{precision:.4f}\t{recall:.4f}\t{f1:.4f}\n")

#     # Plotting metrics
#     epochs_range = list(range(1, epochs + 1))
#     plt.figure(figsize=(12, 8))
#     plt.plot(epochs_range, training_loss, label='Training Loss')
#     plt.plot(epochs_range, validation_loss, label='Validation Loss')
#     plt.plot(epochs_range, train_accuracy[:epochs], label='Training Accuracy')
#     plt.plot(epochs_range, test_accuracy[:epochs], label='Testing Accuracy')
#     plt.xlabel('Epochs')
#     plt.ylabel('Value')
#     plt.title('Training and Validation Loss and Accuracy')
#     plt.legend(loc='best')
#     plt.grid(True)
#     plt.savefig(f"{curr_dir}/{code}/training_validation_metrics_run{k}.png")
#     plt.show()

# end_time = time.time()
# end_date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# # Write the full report to a file
# metrics_file = f"{curr_dir}/{code}/metrics_report.txt"
# with open(metrics_file, "w") as f:
#     f.write(f"Date: {start_date_time}\n")
#     f.write(f"Start Time: {start_time}\n")
#     f.write(metrics_header)
#     f.writelines(metrics_rows)
#     f.write(f"End Time: {end_date_time}\n")

# print("Elapsed time: ", end_time - start_time)


############################################################

# # Modified: 3 July 2024
# # Author: Akanksha Mishra
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import time
# import sys
# sys.path.insert(0,'/workspace/classification')

# from tools import sigmoid
# from sklearn.model_selection import train_test_split
# from sparse_bnn_classification_vhd import FeatureSelectionBNNClassification
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
# from scipy.special import expit

# start_time = time.time()
# torch.set_default_dtype(torch.float64)
# # if (torch.cuda.is_available()):
# #     device = torch.device('cuda')
# # else:
# #     device = torch.device('cpu')
# device = torch.device('cpu')
# print(device)

# curr_dir = "/workspace/classification/friedman"
# code = "fsbnn_samplemean"

# # np.random.seed(123)
# # torch.manual_seed(456)

# #------------------------------------------------------------------------------------------------------
# # Create a simple dataset
# def generate_friedman_classification_dataset(n, p):
#     data = np.random.uniform(0, 1, (n, p))
#     Y = 10 * np.sin(np.square(data[:,0] * data[:,1])) + 20 * np.square(data[:,2]) + 10 * np.sign(data[:,3] * data[:,4] - 0.2) + np.random.normal(0,1)
#     y = np.where(expit(Y) > 0.5, 1, 0)
#     return data, y

# data_size = 5000
# test_size = 1000
# data_dim = 100
# no_of_classes = 2
# target_classes = ['Class 1', 'Class 2']
# sigma_noise = 1.
# rep = 1

# trainsets = []

# x_data, y_data = generate_friedman_classification_dataset(data_size, data_dim)
# x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.20)
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.90)

# # Convert train, test and valid data to tensor
# x_train = torch.Tensor(x_train)
# y_train = torch.LongTensor(y_train)

# x_test = torch.Tensor(x_test)
# y_test = torch.LongTensor(y_test)

# x_val = torch.Tensor(x_val)
# y_val = torch.LongTensor(y_val)

# trainsets = [[x_train,y_train]]
# print(x_train.shape, x_test.shape)


# # ------------------------------------------------------------------------------------------------------
# batch_size = x_train.shape[0]
# num_batches = data_size / batch_size
# learning_rate = torch.tensor(1e-3)
# epochs = 10
# hidden_dim = [16,8,4]
# L = 3
# # total = (data_dim+1) * hidden_dim + (L-1)*((hidden_dim+1) * hidden_dim) + (hidden_dim+1) * 1
# # a = np.log(total) + 0.1*((L+1)*np.log(hidden_dim) + np.log(np.sqrt(data_size)*data_dim))
# total = (data_dim+1) * hidden_dim[0] + (hidden_dim[0]+1) * hidden_dim[1] + (hidden_dim[1]+1) * hidden_dim[2] + (hidden_dim[2]+1) * 1
# a = np.log(total) + 0.1*(np.log(hidden_dim[0]) + 0.1*np.log(hidden_dim[1]) + 0.1*np.log(hidden_dim[2]) + np.log(np.sqrt(data_size)*data_dim))
# lm = 1/np.exp(a)
# phi_prior = torch.tensor(lm)
# temp = 0.5

# train_Loss = []
# test_Loss = []
# sparse_overalls = []
# sparse_overalls2 = []
# FNRs = []
# FPRs = []
# no_of_features_selected = []
# training_loss = []
# validation_loss = []
# train_accuracy = []
# test_accuracy = []
# val_accuracy = []
# # l1_wtheta = np.zeros(shape=(epochs, data_dim, hidden_dim))
# data_size = 3600

# for k in range(rep):
#     print('------------ round {} ------------'.format(k))
#     # create FS sparse BNN
#     net = FeatureSelectionBNNClassification(data_dim, hidden_dim = hidden_dim, target_dim = no_of_classes, device = device).to(device)
#     optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)
#     x_train = trainsets[k][0]
#     y_train = trainsets[k][1]
#     training_loss = []
#     for epoch in range(epochs): 
#         train_losses = []
#         permutation = torch.randperm(data_size)

#         for i in range(0, data_size, batch_size):
#             optimizer.zero_grad() # Sets gradients of all model parameters to zero.
#             indices = permutation[i : i + batch_size]
#             batch_x, batch_y = x_train[indices], y_train[indices]
#             batch_x = batch_x.to(device)
#             batch_y = batch_y.to(device)
#             loss, _ = net.sample_elbo(batch_x, batch_y, 1, temp, phi_prior, num_batches)
#             if torch.isnan(loss):
#                 break
#             train_losses.append(loss.item())
#             loss.backward()
#             optimizer.step()

#         training_loss.append(np.mean(train_losses))
#         # wtheta = np.array(net.l1.w_theta.detach().numpy())
#         # l1_wtheta[epoch] = wtheta
#         # print('Epoch {}, l1.w_theta: {}'.format(epoch, wtheta))
#         # one1_w = (sigmoid(net.l1.w_theta) > 0.5).float() 
#         # p = torch.sum(one1_w, axis=1)
#         # no_of_features_selected.append(torch.sum(p>=1))

#         # Validation step
#         net.eval()  # Set the model to evaluation mode
#         valid_losses = []
#         with torch.no_grad():
#             _, val_pred = net.sample_elbo(x_val.to(device), y_val.to(device), 30, temp, phi_prior, num_batches)
#             val_pred = torch.mode(val_pred, dim=0).values
#             val_loss = torch.sum(val_pred != y_val.to(device)) / y_val.shape[0]
#             valid_losses.append(val_loss.item())
#             val_accuracy.append(torch.sum(val_pred == y_val.to(device)) / y_val.shape[0])
#         validation_loss.append(np.mean(valid_losses))
#         net.train()  # Set the model back to training mode

#         _, pred = net.sample_elbo(x_train.to(device), y_train.to(device), 30,
#                                 temp, phi_prior, num_batches)
#         pred = torch.mode(pred, dim=0).values
#         train_accuracy.append(torch.sum(pred == y_train) / y_train.shape[0])

#         _, pred2 = net.sample_elbo(x_test.to(device), y_test.to(device), 30,
#                                 temp, phi_prior, num_batches)
#         pred2 = torch.mode(pred2, dim=0).values
#         test_accuracy.append(torch.sum(pred2 == y_test) / y_test.shape[0])

#         if epoch % 1000 == 0 or epoch == epochs-1:
#             one1_w = (net.l1.w != 0).float()
#             one1_b = (net.l1.b != 0).float()
#             sparsity = ( torch.sum(one1_w) + torch.sum(one1_b) ) / total 
#             print('Epoch {}, Train_Loss: {}, phi_prior: {}, sparsity: {}'.format(epoch, np.mean(train_losses), phi_prior,
#                                                                                 sparsity))
#     print('Epoch {}, Train_Loss: {}'.format(epoch, np.mean(train_losses)))
#     print("Finished Training")

#     torch.save(net.state_dict(), f"{curr_dir}/{code}/model_run2.pth")

#     # torch.onnx.export(net, x_test, temp, phi_prior, 'moon.onnx')
#     # torch.onnx.export(model, batch.text, 'rnn.onnx', input_names=input_names, output_names=output_names)
#     # make_dot(net(x_train, temp, phi_prior), params=dict(list(net.named_parameters())), show_attrs=True, show_saved=True).render(f"{curr_dir}/{code}/fsbnn_torchviz", format = "png")

#     # sparsity level
#     print('l1.w_theta final: {}'.format(net.l1.w_theta))
#     one1_w = (sigmoid(net.l1.w_theta)).float()  
#     one1_b = (sigmoid(net.l1.b_theta) > 0.5).float()
#     sparse_overall = ( torch.sum(one1_w) + torch.sum(one1_b) ) / total
#     sparse_overalls.append(sparse_overall)
#     sparse_overall2 = ( torch.sum(sigmoid(net.l1.w_theta)) + torch.sum(sigmoid(net.l1.b_theta)) ) / total
#     sparse_overalls2.append(sparse_overall2)
#     torch.set_printoptions(profile="full")

#     print("\n", "----------- Network Sparsity -----------")
#     print('l1 Overall w sparsity: {}'.format(torch.mean(one1_w)))
#     print('l1 w Edges: {}'.format(one1_w))
#     # p = torch.mean(one1_w, axis=1) #Commented for weight sharing
#     sorted, indices = torch.sort(one1_w,0, descending=True) #write one1_w instead of p for weight sharing
#     print('features selected in the first layer: {}'.format(indices[0:10]))
#     torch.save(sorted, f"{curr_dir}/{code}/weights_sorted_run2.pt")
#     # torch.save(indices, f"{curr_dir}/{code}/ws_fsbnn_indices_run11.pt")
#     np.save(f"{curr_dir}/{code}/indices_sorted_run2",indices)

#     print('l1 Overall b sparsity: {}'.format(torch.mean(one1_b)))
#     print('l1 b Edges: {}'.format(one1_b))

#     # prediction
#     x_train = x_train.to(device)
#     y_train = y_train.to(device)
#     _, pred = net.sample_elbo(x_train, y_train, 30,
#                                 temp, phi_prior, num_batches)
#     print("shape of output -> ", pred.shape)

#     pred = torch.mode(pred, dim=0).values
#     train_loss = torch.sum(pred != y_train) / y_train.shape[0]
#     train_Loss.append(train_loss)

#     print("----------- Training -----------")
#     print('y_train: {}'.format(y_train[0:20]))
#     print('pred_train: {}'.format(pred[0:20]))
#     print('binary_loss_train: {}'.format(train_loss))

#     # ------------------------------------------------------------------------------------------------------
#     print("\n", "----------- Testing -----------")
#     # testing
#     # prediction

#     x_test = x_test.to(device)
#     y_test = y_test.to(device)
#     _, pred2 = net.sample_elbo(
#         x_test, y_test, 30, temp, phi_prior, num_batches)
#     pred2 = torch.mode(pred2, dim=0).values
#     test_loss = torch.sum(pred2 != y_test) / y_test.shape[0]
#     test_Loss.append(test_loss)

#     print('y_test: {}'.format(y_test[0:20]))
#     print('pred_test: {}'.format(pred2[0:20]))
#     print('binary_loss_test: {}'.format(test_loss))
#     print("\n")

# train_LOSS = torch.tensor(train_Loss)
# test_LOSS = torch.tensor(test_Loss)
# sparse_overalls = torch.tensor(sparse_overalls)
# sparse_overalls2 = torch.tensor(sparse_overalls2)
# FNRs = torch.tensor(FNRs)
# FPRs = torch.tensor(FPRs)

# print("\n", "----------- Summary -----------")
# print('binary_loss_MEAN_train: {}'.format(torch.mean(train_LOSS)))
# print('binary_loss_std_train: {}'.format(torch.std(train_LOSS)))
# print('binary_loss_MEAN_test: {}'.format(torch.mean(test_LOSS)))
# print('binary_loss_std_test: {}'.format(torch.std(test_LOSS)))
# print('sparsity: {}'.format(torch.mean(sparse_overalls)))
# print('sparsity2: {}'.format(torch.mean(sparse_overalls2)))
# print('FNR: {}'.format(torch.mean(FNRs)))
# print('FNR sd: {}'.format(torch.std(FNRs)))
# print('FPR: {}'.format(torch.mean(FPRs)))
# print('FPR sd: {}'.format(torch.std(FPRs)))

# print('sparsity all: {}'.format(sparse_overalls))
# print('sparsity all 2: {}'.format(sparse_overalls2))
# print('binary_loss_train all: {}'.format(train_LOSS))
# print('binary_loss_test all: {}'.format(test_LOSS))
# print('FNRs: {}'.format(FNRs))
# print('FPRs: {}'.format(FPRs))

# y_train = y_train.cpu()
# pred = pred.cpu()

# print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_train, pred)))

# print('Micro Precision: {:.2f}'.format(precision_score(y_train, pred, average='micro')))
# print('Micro Recall: {:.2f}'.format(recall_score(y_train, pred, average='micro')))
# print('Micro F1-score: {:.2f}\n'.format(f1_score(y_train, pred, average='micro')))

# print('Macro Precision: {:.2f}'.format(precision_score(y_train, pred, average='macro')))
# print('Macro Recall: {:.2f}'.format(recall_score(y_train, pred, average='macro')))
# print('Macro F1-score: {:.2f}\n'.format(f1_score(y_train, pred, average='macro')))

# print('Weighted Precision: {:.2f}'.format(precision_score(y_train, pred, average='weighted')))
# print('Weighted Recall: {:.2f}'.format(recall_score(y_train, pred, average='weighted')))
# print('Weighted F1-score: {:.2f}'.format(f1_score(y_train, pred, average='weighted')))

# print('\nClassification Report\n')
# print(classification_report(y_train, pred, target_names = target_classes))


# y_test = y_test.cpu()
# pred2 = pred2.cpu()

# print('\nTest Accuracy: {:.2f}\n'.format(accuracy_score(y_test, pred2)))

# print('Micro Precision: {:.2f}'.format(precision_score(y_test, pred2, average='micro')))
# print('Micro Recall: {:.2f}'.format(recall_score(y_test, pred2, average='micro')))
# print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, pred2, average='micro')))

# print('Macro Precision: {:.2f}'.format(precision_score(y_test, pred2, average='macro')))
# print('Macro Recall: {:.2f}'.format(recall_score(y_test, pred2, average='macro')))
# print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, pred2, average='macro')))

# print('Weighted Precision: {:.2f}'.format(precision_score(y_test, pred2, average='weighted')))
# print('Weighted Recall: {:.2f}'.format(recall_score(y_test, pred2, average='weighted')))
# print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, pred2, average='weighted')))

# print('\nClassification Report\n')
# print(classification_report(y_test, pred2, target_names = target_classes))

# # Generate confusion matrix
# cm = confusion_matrix(y_test, pred2)

# # Calculate and display Type I errors for each class
# type1_errors = np.sum(cm, axis=0) - np.diag(cm)
# print("\nType I Errors (False Positives) for each class:")
# for i, error in enumerate(type1_errors):
#     print(f"Class {i}: {error}")

# # Calculate and display Type II errors for each class
# type2_errors = np.sum(cm, axis=1) - np.diag(cm)
# print("\nType II Errors (False Negatives) for each class:")
# for i, error in enumerate(type2_errors):
#     print(f"Class {i}: {error}")

# # plt.plot(training_loss)
# # plt.xlabel("epochs \n Final Loss: {:.2f}".format(training_loss[epochs-1]))
# # plt.ylabel("Training Loss")
# # plt.tight_layout()
# # plt.savefig(f"{curr_dir}/{code}/train_loss_run11")
# # plt.show()

# plt.figure()
# plt.plot(np.log(training_loss), label='train')
# plt.plot(np.log(validation_loss), label='valid')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss')
# plt.legend()
# plt.savefig(f"{curr_dir}/{code}/train_valid_loss_run2")
# plt.show()

# plt.figure()
# plt.plot(train_accuracy, label = 'train_accuracy')
# plt.plot(test_accuracy, label = 'test_accuracy')
# plt.xlabel("epochs \n Final Train Accuracy : {:.3f}, Final Test Accuracy: {:.3f}".format(train_accuracy[epochs-1], test_accuracy[epochs-1]))
# plt.ylabel("accuracy")
# plt.title('Training and Test Accuracy')
# plt.legend()
# plt.tight_layout()
# plt.savefig(f"{curr_dir}/{code}/train_test_acc_run2")
# plt.show()

# # plt.plot(no_of_features_selected)
# # plt.xlabel("epochs \n Total no of Features selected: {}".format(no_of_features_selected[epochs-1]))
# # plt.ylabel("no of Features selected")
# # plt.tight_layout()
# # plt.savefig(f"{curr_dir}/{code}/No_Of_Features_Selected")
# # plt.show()

# fig, ax = plt.subplots(figsize = (8,8))
# cm = confusion_matrix(y_train, pred)
# cm_display = ConfusionMatrixDisplay(cm)
# cm_display.plot(ax = ax)
# plt.title("Confusion matrix for training data", fontdict = {'fontsize': 18, 'color': 'teal'}, pad = 15)
# plt.savefig(f"{curr_dir}/{code}/cm_train_run2")
# plt.show()

# fig, ax = plt.subplots(figsize = (8,8))
# cm = confusion_matrix(y_test, pred2)
# cm_display = ConfusionMatrixDisplay(cm)
# cm_display.plot(ax = ax)
# plt.title("Confusion matrix for testing data", fontdict = {'fontsize': 18, 'color': 'teal'}, pad = 15)
# plt.savefig(f"{curr_dir}/{code}/cm_test_run2")
# plt.show()


# print("Total Time Consumed --> ", time.time() - start_time)