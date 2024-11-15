import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import torch.optim as optim
from sklearn.model_selection import train_test_split
import copy

Hidden = [17, 256, 256, 256, 2]
Lr = 0.01
Batch = 4000
num_epochs = 1000

# MLP Model
class MLP(nn.Module):
    def __init__(self, hidden):
        super(MLP, self).__init__()
        self.hidden = copy.deepcopy(hidden)
        self.S = nn.Sequential(nn.Linear(self.hidden[0], self.hidden[1]), nn.ReLU(),
                                         nn.Linear(self.hidden[1], self.hidden[2]), nn.ReLU(),
                                         nn.Linear(self.hidden[2], self.hidden[3]), nn.ReLU(),
                                         nn.Linear(self.hidden[3], self.hidden[4]))
        self.ac = nn.Softmax()
    def forward(self, x):
        return self.ac(self.S(x))

# Data Loader: Loads data from a CSV file, handles missing values, and normalizes data
def data_loader(file):
    data_pd = pd.read_csv(file)
    data_pd.fillna(-1, inplace=True)
    data_np = np.array(data_pd)
    data_input = np.zeros((data_np.shape[0], 17))
    label = np.zeros(data_np.shape[1])
    for i in range(data_np.shape[1]):
        data_row = data_np[:, i]
        data_row[data_row == "NAN"] = -1
        data_row[data_row == "Other"] = -2
        data_row[data_row == "null"] = -2
        data_row[data_row == "Returning_Visitor"] = 0
        data_row[data_row == "New_Visitor"] = 1
        data_row[data_row == "FALSE"] = 0
        data_row[data_row == "TRUE"] = 1
        if i == 17: label = data_row.astype(float)
        else: data_input[:, i] = data_row / (np.max(data_row) - np.min(data_row))
    return data_input, label

# Training the Model
torch.seed()
torch.random.seed()
file = "train.csv"
data_train, label = data_loader(file)
np.expand_dims(label, axis=0)
data_input_tr, data_input_va, label_tr, label_va = train_test_split(data_train, label, test_size=0.2, random_state=0)

Net = MLP(Hidden)
loss_fn = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
optimizer = optim.Adam(Net.parameters(), lr=Lr)

Ac = 0
Ac_best = 0
for epoch in range(num_epochs):
    # Forward propagation
    batch_index = np.random.choice(range(data_input_tr.shape[0]), Batch)
    inputs = torch.FloatTensor(data_input_tr[batch_index])
    targets = torch.FloatTensor(np.vstack((abs(1-label_tr[batch_index]), label_tr[batch_index]))).T
    outputs = Net(inputs)
    loss = criterion(outputs, targets)

    # Backward propagation and Optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    va_out = Net(torch.FloatTensor(data_input_va)).detach().squeeze(1).numpy()
    pre = np.argmax(va_out, axis=1)
    Ac = 1 - sum(abs(pre - label_va)) / label_va.shape[0]
    if Ac > Ac_best:
        Ac_best = Ac
        torch.save(Net.state_dict(), "parameters.pth")
        print(Ac)

#test
file = "test.csv"
data_test, label = data_loader(file)
Net = MLP(Hidden)
Net.load_state_dict(torch.load("parameters.pth"))
result = Net(torch.FloatTensor(data_test)).detach().numpy()
result_pd = pd.DataFrame(np.argmax(result, axis=1))
result_pd.to_csv("results.csv", encoding = 'utf-8', index=False , header=False)

# ROC Curve and AUC Calculation
def plot_roc_curve(true_labels, predicted_probs):
    auc_score = roc_auc_score(true_labels, predicted_probs)
    fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

