import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import numpy as np 
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from matplotlib import style

# Global variables

# if memory error, make batch size smaller
BATCH_SIZE = 100
# number of loops you want to train model
EPOCHS = 3

# load in data
training_data = np.load("training_data.npy", allow_pickle = True)

X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
X = X/255.0
y = torch.Tensor([i[1] for i in training_data])

VAL_PCT = 0.1
val_size = int(len(X)*VAL_PCT)

# test dataset using the last elements in data past val_size
test_X = X[-val_size:]
test_y = y[-val_size:]

MODEL_NAME = f"model-{int(time.time())}"  # gives a dynamic model name, to just help with things getting messy over time
print(MODEL_NAME)

class Net(nn.Module):
    def __init__(self):
        super().__init__() # run the init of parent class (nn.Module)
        # defining convolution layers
        self.conv1 = nn.Conv2d(1, 32, 5) # input is 1 image, 32 output channels, 5x5 kernel/window
        self.conv2 = nn.Conv2d(32, 64, 5) 
        self.conv3 = nn.Conv2d(64, 128, 5)

        # find out what the shape looks like after feeding data through cnn
        # create random dataset to feed in
        x = torch.randn(50,50).view(-1,1,50,50)
        self._to_linear = None
        self.convs(x)

        # defining fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 512) # flattening
        # 2 classes for data (dog/cat) 
        self.fc2 = nn.Linear(512, 2) # 512 in, 2 out 

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
        # output layer = no activation
        x = self.fc2(x)
        # activation layer
        return F.softmax(x, dim=1)

net = Net()
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

# training model is really slow     
def train(net):
    # training dataset using first elements in data up to val_size
    train_X = X[:-val_size]
    train_y = y[:-val_size]
    print(len(train_X))

    with open("model.log", "a") as f:
        for epoch in range(EPOCHS):
            for i in tqdm(range(0, len(train_X), BATCH_SIZE)): 
                # from 0 to the len of x, stepping batch size at a time [:50] 
                #print(i, i+BATCH_SIZE)
                batch_X = train_X[i:i+BATCH_SIZE].view(-1,1,50,50)
                batch_y = train_y[i:i+BATCH_SIZE]

                # calc in sample accuracy and loss
                acc, loss = fwd_pass(batch_X, batch_y, train=True)

                #print(f"Acc: {round(float(acc),2)}  Loss: {round(float(loss),4)}")
                #f.write(f"{MODEL_NAME},{round(time.time(),3)},in_sample,{round(float(acc),2)},{round(float(loss),4)}\n")
                # just to show the above working, and then get out:
                if i % 50 == 0:
                    val_acc, val_loss = test(size=100)
                    f.write(f"{MODEL_NAME},{round(time.time(),3)},{round(float(acc),2)},{round(float(loss), 4)},{round(float(val_acc),2)},{round(float(val_loss),4)}\n")

# calc accuracy 
# def test(net):
#     correct = 0
#     total = 0
#     #test_X = X[-val_size:]
#     #test_y = y[-val_size:]
#     print(len(test_X))

#     with torch.no_grad():
#         for i in tqdm(range(len(test_X))):
#             real_class = torch.argmax(test_y[i])
#             net_out = net(test_X[i].view(-1,1,50,50))[0] # returns a list
#             predicted_class = torch.argmax(net_out)
#             if predicted_class == real_class:
#                 correct += 1
#             total += 1

#     print("Accuracy:", round(correct/total, 3))

def fwd_pass(X, y, train=False):

    if train:
        net.zero_grad()
    outputs = net(X)
    matches  = [torch.argmax(i)==torch.argmax(j) for i, j in zip(outputs, y)]
    acc = matches.count(True)/len(matches)
    loss = loss_function(outputs, y)

    if train:
        loss.backward()
        optimizer.step()

    return acc, loss

def test(size=32):
    X, y = test_X[:size], test_y[:size]
    # val_loss = out of sample lost
    val_acc, val_loss = fwd_pass(X.view(-1, 1, 50, 50), y)
    return val_acc, val_loss

train(net)

val_acc, val_loss = test(size=100)
print(val_acc, val_loss)

style.use("ggplot")

model_name = MODEL_NAME # grab whichever model name you want here. We could also just reference the MODEL_NAME if you're in a notebook still.

def create_acc_loss_graph(model_name):
    contents = open("model.log", "r").read().split("\n")

    times = []
    accuracies = []
    losses = []

    val_accs = []
    val_losses = []

    for c in contents:
        if model_name in c:
            name, timestamp, acc, loss, val_acc, val_loss = c.split(",")

            times.append(float(timestamp))
            accuracies.append(float(acc))
            losses.append(float(loss))

            val_accs.append(float(val_acc))
            val_losses.append(float(val_loss))


    fig = plt.figure()

    ax1 = plt.subplot2grid((2,1), (0,0))
    ax2 = plt.subplot2grid((2,1), (1,0), sharex=ax1)


    ax1.plot(times, accuracies, label="acc")
    ax1.plot(times, val_accs, label="val_acc")
    ax1.legend(loc=2)
    ax2.plot(times,losses, label="loss")
    ax2.plot(times,val_losses, label="val_loss")
    ax2.legend(loc=2)
    plt.show()

create_acc_loss_graph(model_name)