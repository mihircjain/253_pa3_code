from torchvision import utils
from basic_fcn import *
from dataloader import *
# from utils import *
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time
from tqdm import tqdm, tqdm_notebook
# import sys

train_dataset = CityScapesDataset(csv_file='train.csv')
val_dataset = CityScapesDataset(csv_file='val.csv')
test_dataset = CityScapesDataset(csv_file='test.csv')
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=4,
                          num_workers=0,
                          shuffle=True)
val_loader = DataLoader(dataset=val_dataset,
                          batch_size=2,
                          num_workers=0,
                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                          batch_size=1,
                          num_workers=0,
                          shuffle=True)

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
#         torch.nn.init.xavier_uniform(m.bias.data, 0)
        nn.init.constant_(m.bias, 0)
        
n_class = 34
epochs = 100
criterion = nn.CrossEntropyLoss() # Choose an appropriate loss function from https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html
fcn_model = FCN(n_class=n_class)
fcn_model.apply(init_weights)
#fcn_model = torch.load('best_model')
optimizer = optim.Adam(fcn_model.parameters(), lr=5e-3)

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("GPU is available")
    fcn_model = fcn_model.cuda()
    
def train():
    for epoch in range(epochs):
        ts = time.time()
        for iter, (X, tar, Y) in tqdm(enumerate(train_loader), desc ="Iteration num: "): # X=input_images, tar=one-hot labelled, y=segmentated
            optimizer.zero_grad()
            Y = Y.long()
            if use_gpu:
                inputs = X.cuda() # Move your inputs onto the gpu
                labels = Y.cuda() # Move your labels onto the gpu

            else:
                inputs, labels =  X, tar # Unpack variables into inputs and labels

            outputs = fcn_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
        
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        torch.save(fcn_model, 'best_model')

        val(epoch)
        fcn_model.train()


def val(epoch):
    fcn_model.eval()
    #Complete this function - Calculate loss, accuracy and IoU for every epoch
    # Make sure to include a softmax after the output from your model
    # Evaluate
    total = 0
    correct = 0
    for iter, (X, tar, Y) in tqdm(enumerate(val_loader)):
        inputs, targets, Y = X, tar, Y.long()
        if use_gpu:
            inputs, Y = inputs.cuda(), Y.cuda()

        outputs = fcn_model(inputs)
        
        
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(Y.data).cpu().sum()

    print('Epoch : %d Test Acc : %.3f' % (epoch + 1, 100.*correct/total))
    print('--------------------------------------------------------------')
    net.train()

def test():
    pass
    #Complete this function - Calculate accuracy and IoU 
    # Make sure to include a softmax after the output from your model
    
if __name__ == "__main__":
    val(0)  # show the accuracy before training
    # train()

tic = 0
for iter, (X, tar, Y) in enumerate(train_loader):
    print(iter,": ", time.time()-tic)
    tic = time.time()

