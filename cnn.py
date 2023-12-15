import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import numpy as np 
from tqdm import tqdm

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # defining convolution layers
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        # find out what the shape looks like after feeding data through cnn
        # create random dataset to feed in
        x = torch.randn(50,50).view(-1,1,50,50)
        self._to_linear = None
        self.convs(x)

        # defining fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 512)
        # 2 classes for data (dog/cat) 
        self.fc2 = nn.Linear(512, 2)

    def convs(self, x):
        # convolutional operations 
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))

        print(x[0].shape)
        
        # set _to_linear to calulated size if none
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x
    
    # define forward pass
    def forward(self, x):
        # perform convolutional operations 
        x = self.convs(x)
        # reshape output to a 1D tensor using view
        x = x.view(-1, self._to_linear)
        # apply ReLu activation to first fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # activation layer
        return F.softmax(x, dim=1)
    
net = Net()

# load in data
training_data = np.load("training_data.npy", allow_pickle = True)

optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
X = X/255.0
y = torch.Tensor([i[1] for i in training_data])

VAL_PCT = 0.1
val_size = int(len(X)*VAL_PCT)
print(val_size)

train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]

print(len(train_X))
print(len(test_X))

# if memory error, make smaller
BATCH_SIZE = 100
EPOCHS = 1

#training model = really slow
for epoch in range(EPOCHS):
    for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
        #print(i, i+BATCH_SIZE)
        batch_X = train_X[i:i+BATCH_SIZE].view(-1,1,50,50)
        batch_y = train_y[i:i+BATCH_SIZE]

        #optimizer.zero_grad()
        net.zero_grad()

        outputs = net(batch_X)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()

print(loss)

# calc accuracy 
correct = 0
total = 0
with torch.no_grad():
    for i in tqdm(range(len(test_X))):
        real_class = torch.argmax(test_y[i])
        net_out = net(test_X[i].view(-1,1,50,50))[0]
        predicted_class = torch.argmax(net_out)
        if predicted_class == real_class:
            correct += 1
        total += 1

print("Accuracy:", round(correct/total, 3))