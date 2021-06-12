import torch
import sys
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import numpy as np
import pandas as pd

np.set_printoptions(suppress=False)


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.linear(x)
        return self.sigmoid(out)


data_train = pd.read_csv("data_train.csv")
data_test  = pd.read_csv("data_test.csv")
FEATURES   =  ['age','hypertension','heart_disease','ever_married', 'avg_glucose_level', 'bmi']

x_train = data_train[FEATURES].astype(np.float32)
y_train = data_train['stroke'].astype(np.float32)

x_test = data_test[FEATURES].astype(np.float32)
y_test = data_test['stroke'].astype(np.float32)

fTrain = torch.from_numpy(x_train.values)
tTrain = torch.from_numpy(y_train.values.reshape(2945,1))

fTest= torch.from_numpy(x_test.values)
tTest = torch.from_numpy(y_test.values)

batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 16
num_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
learning_rate = 0.001
input_dim = 6
output_dim = 1

model = LogisticRegressionModel(input_dim, output_dim)

criterion = torch.nn.BCELoss(reduction='mean') 
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

for epoch in range(num_epochs):
    # print ("Epoch #",epoch)
    model.train()
    optimizer.zero_grad()
    # Forward pass
    y_pred = model(fTrain)
    # Compute Loss
    loss = criterion(y_pred, tTrain)
    # print(loss.item())
    # Backward pass
    loss.backward()
    optimizer.step()
y_pred = model(fTest)


rmse = mean_squared_error(tTest, y_pred.detach().numpy())
acc = accuracy_score(tTest, np.argmax(y_pred.detach().numpy(), axis=1))

with open("metrics.txt", 'w') as outfile:
        outfile.write("Accuracy: " + str(acc) + "\n")
        outfile.write("RMSE: " + str(rmse) + "\n")
        outfile.write(str(classification_report(tTest, y_pred.detach().numpy().round())) + "\n")