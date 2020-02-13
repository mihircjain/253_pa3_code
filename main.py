from torchvision import utils
from basic_fcn import *
from dataloader import *
from utils import *
# from utils import *
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time
from tqdm import tqdm, tqdm_notebook
# import sys


train_dataset = CityScapesDataset(csv_file='train.csv', transforms=transforms.RandomCrop(512,1024))
val_dataset = CityScapesDataset(csv_file='val.csv')
test_dataset = CityScapesDataset(csv_file='test.csv')
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=4,
                          num_workers=4,
                          shuffle=True, 
                         )
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
    '********  losses for all epochs  ********'
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        '******** running loss for one epoch  ********'
        running_loss = 0.0
        
        ts = time.time()
        for iter, (X, Y) in tqdm(enumerate(train_loader), desc ="Iteration num: "): # X=input_images, tar=one-hot labelled, y=segmentated
            optimizer.zero_grad()
            Y = Y.long()
            if use_gpu:
                inputs = X.cuda() # Move your inputs onto the gpu
                labels = Y.cuda() # Move your labels onto the gpu

            else:
                inputs, labels =  X, Y # Unpack variables into inputs and labels

            outputs = fcn_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            '******** accumulate running_loss for each batch ********'
            running_loss += loss.item() * inputs.size(0)

            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
        
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        torch.save(fcn_model, 'best_model')
        
        '******** save average training loss for each epoch  ********'
        train_losses.append(running_loss/len(train_loader))
        '******** save average validation loss for each epoch  ********'
        val_loss = val(epoch)
        val_losses.append(val_loss)
        
        fcn_model.train()
        
    x = [i for i in range(epochs)]
    plt.title("Plot showing training and validation loss against number of epochs")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.plot(x, train_losses, color='r', label='training loss')
    plt.plot(x, val_losses, color = 'b', label = 'validation loss')
    
    plt.legend()
    plt.show()
    plt.plot()



def val(epoch):
    fcn_model.eval()
#     TP = [0 for i in range(34)]
#     TP = torch.FloatTensor(TP)
#     FN = [0 for i in range(34)]
#     FN = torch.FloatTensor(FN)
#     FP = [0 for i in range(34)]
#     FP = torch.FloatTensor(FP)
    #Complete this function - Calculate loss, accuracy and IoU for every epoch
    # Make sure to include a softmax after the output from your model
    # Evaluate
    total = 0
    correct = 0
    '******** running loss for one epoch  ********'
    running_loss = 0.0
    
    for iter, (X, tar, Y) in tqdm(enumerate(val_loader)):
        inputs, targets, Y = X, tar, Y.long()
        if use_gpu:
            inputs, labels = inputs.cuda(), Y.cuda()

        outputs = fcn_model(inputs)
        '******** caluculate validation loss  ********'
        loss = criterion(outputs, labels)
        '******** accumulate running_loss for each batch ********'
        running_loss += loss.item() * inputs.size(0)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()
        
       
#         tp = list()
#         tp = torch.FloatTensor(tp)
#         fp = list()
#         fp = torch.FloatTensor(fp)
#         fn = list()
#         fn = torch.FloatTensor(fn)
#         tp, fp, fn = iou(outputs, targets)
#         TP = [sum(x) for x in zip(TP,tp)]
#         FP = [sum(x) for x in zip(FP,fp)]
#         FN = [sum(x) for x in zip(FN,fn)]

#     union = TP
#     intersection = [i+j+k for i,j,k in zip(TP,FP,FN)]
#     class_iou = [u/i for u, i in zip(union, intersection)]
#     av_iou = sum(class_iou)/len(class_iou)
    
    print('Epoch : %d Validation Pixel Acc : %.3f' % (epoch + 1, 100.*correct/total))
    #print('--------------------------------------------------------------')
    #print('Epoch : %d Validation Avg IOU : %.3f' % (epoch + 1, av_iou))
    #print('--------------------------------------------------------------')
    #print('Epoch : %d Each class IOU : %.3f' % (epoch + 1, class_iou))
    return (running_loss/len(val_loader))

def test():
    fcn_model.eval()
#     TP = [0 for i in range(34)]
#     TP = torch.FloatTensor(TP)
#     FN = [0 for i in range(34)]
#     FN = torch.FloatTensor(FN)
#     FP = [0 for i in range(34)]
#     FP = torch.FloatTensor(FP)
    #Complete this function - Calculate loss, accuracy and IoU for every epoch
    # Make sure to include a softmax after the output from your model
    # Evaluate
    total = 0
    correct = 0
    for iter, (X, tar, Y) in tqdm(enumerate(val_loader)):
        inputs, targets, Y = X, tar, Y.long()
        if use_gpu:
            inputs, labels = inputs.cuda(), Y.cuda()

        outputs = fcn_model(inputs)
        
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()
        
       
#         tp = list()
#         tp = torch.FloatTensor(tp)
#         fp = list()
#         fp = torch.FloatTensor(fp)
#         fn = list()
#         fn = torch.FloatTensor(fn)
#         tp, fp, fn = iou(outputs, targets)
#         TP = [sum(x) for x in zip(TP,tp)]
#         FP = [sum(x) for x in zip(FP,fp)]
#         FN = [sum(x) for x in zip(FN,fn)]

#     union = TP
#     intersection = [i+j+k for i,j,k in zip(TP,FP,FN)]
#     class_iou = [u/i for u, i in zip(union, intersection)]
#     av_iou = sum(class_iou)/len(class_iou)
    
    print('Test Pixel Acc : %.3f' %  100.*correct/total)
    print('--------------------------------------------------------------')
#     print('Test Avg IOU : %.3f' % av_iou)
#     print('--------------------------------------------------------------')
#     print('Each class IOU : %.3f' % class_iou)
    

    #Complete this function - Calculate accuracy and IoU 
    # Make sure to include a softmax after the output from your model
    
if __name__ == "__main__":
#     val(0)  # show the accuracy before training
    train()
