import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, num_layers):
        super(ConvNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2))
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5))
        
        self.lstm1 = torch.nn.LSTM(
            input_size= 768,
            hidden_size=128,
            num_layers=num_layers,
        )
        self.do = nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(128, 63)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(-1)
        
        x,_ = self.lstm1(x.view(1,1,x.shape[0]))
        x = x[:, -1, :]
        x = self.do(x)
        x = self.fc2(x)
        return (x)

def validate_model(smh_test, sml_test, model, loss_func):
    loss_avg = 0
    for i in np.arange(smh_test.shape[0]-2):
    
        data = smh_test[i]
        label = sml_test[i+1]
        
        row = np.array(data)
        output = model(torch.tensor(row).view(1,1,row.shape[0], row.shape[1]).type(torch.FloatTensor))
        loss = loss_func(output,torch.tensor(label.flatten()).type(torch.FloatTensor).view(1,-1))
        loss_avg += loss.item()
    return loss_avg


smh = np.load("preprocess/result_speed_matrix_high.npy")
sml = np.load("preprocess/result_speed_matrix_low.npy")

print(smh.shape)
print(np.max(smh))
print(sml.shape)
print(np.max(sml))

smh = smh/np.max(smh)
sml = sml/np.max(sml)


smh_tr = smh[:int(0.9*smh.shape[0])]
smh_vl = smh[-int(0.9*smh.shape[0]):]
sml_tr = sml[:int(0.9*sml.shape[0])]
sml_vl = sml[-int(0.9*sml.shape[0]):]

n_layers = 2
model = ConvNet(n_layers)
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
epochs = 10
loss_acc = []
loss_train=[]

for epoch in np.arange(epochs):
    loss_avg = 0
    for i in np.arange(smh_tr.shape[0]-2):
    
        data = smh[i]
        label = sml[i+1]
        optimizer.zero_grad()
        row = np.array(data)

        output = model(torch.tensor(row).view(1,1,row.shape[0], row.shape[1]).type(torch.FloatTensor))

        loss = loss_func(output,torch.tensor(label.flatten()).type(torch.FloatTensor).view(1,-1))
        loss_avg += loss.item()
    
        loss.backward()
        optimizer.step()

    loss_vl = validate_model(smh_vl, sml_vl, model, loss_func)
    print("LOSS VAL: {}".format(loss_vl))
    loss_acc.append(validate_model(smh_vl, sml_vl, model, loss_func))
    loss_train.append(loss_avg)
    print("LOSS TRAIN: {}".format(loss_avg))
    print('------------------')

