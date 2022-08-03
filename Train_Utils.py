# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 11:17:51 2021

@author: Narmin Ghaffari Laleh
"""

import os
import time
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

##############################################################################

def Train_model(model, trainLoaders, valLoaders = [], criterion = None, optimizer = None, num_epochs = 25,
                results_dir = '', patience = 10, stop_epoch = 30, resultFolder = '', fold = 0, trainFull = False):
    
    since = time.time()

    train_acc_history = []
    train_loss_history = []

    val_acc_history = []
    val_loss_history = []
    
    early_stopping = EarlyStopping(patience = patience, stop_epoch = stop_epoch, verbose = True, resultFolder = resultFolder, fold = fold, trainFull = trainFull)
    
    for epoch in range(num_epochs):
        phase = 'train'
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train() 
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in tqdm(trainLoaders):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(trainLoaders.dataset)
        epoch_acc = running_corrects.double() / len(trainLoaders.dataset)
        
        train_acc_history.append(epoch_acc)
        train_loss_history.append(epoch_loss)
        
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        print()
        
        if valLoaders:
            phase = 'val'
    
            model.eval()   # Set model to evaluate mode
        
            running_loss = 0.0
            running_corrects = 0
            predList = []
            
            # Iterate over data.
            for inputs, labels in tqdm(valLoaders):
                inputs = inputs.to(device)
                labels = labels.to(device)
        
                with torch.set_grad_enabled(phase == 'train'):            
                    #outputs = model(inputs)
                    outputs = nn.Softmax(dim=1)(model(inputs)) 
                    loss = criterion(outputs, labels)
        
                    _, preds = torch.max(outputs, 1)
                    predList = predList + outputs.tolist()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        
            val_loss = running_loss / len(valLoaders.dataset)
            val_acc = running_corrects.double() / len(valLoaders.dataset)
            
            val_acc_history.append(val_acc)
            val_loss_history.append(val_loss)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, val_loss, val_acc))
            
            early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "checkpoint.pt"))
            if early_stopping.early_stop:
                print("Early stopping")
                break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    return model, train_loss_history, train_acc_history, val_acc_history, val_loss_history 

##############################################################################    
    
def Validate_model(model, dataloaders, criterion, hasLabels):
    
    phase = 'test'

    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0
    predList = []
    
    if hasLabels:
    # Iterate over data.
        for inputs, labels in tqdm(dataloaders):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            with torch.set_grad_enabled(phase == 'train'):            
                #outputs = model(inputs)
                outputs = nn.Softmax(dim=1)(model(inputs)) 
                loss = criterion(outputs, labels)
    
                _, preds = torch.max(outputs, 1)
                predList = predList + outputs.tolist()
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders.dataset)
        epoch_acc = running_corrects.double() / len(dataloaders.dataset)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

        return epoch_loss, epoch_acc, predList 
    
    else:
        
        for inputs in tqdm(dataloaders):
            inputs = inputs.to(device)
        
            with torch.set_grad_enabled(phase == 'train'):            
                outputs = nn.Softmax(dim=1)(model(inputs))
                predList = predList + outputs.tolist()
                
        return predList
                
                
                
                
##############################################################################
    
class EarlyStopping:
    def __init__(self, patience = 20, stop_epoch = 50, verbose=False, resultFolder = '', fold = 0, trainFull = False):

        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.resultFolder = resultFolder
        self.trainFull = trainFull
        self.fold = fold

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint'):

        score = -val_loss
        if self.trainFull:
            ckpt_name = os.path.join(self.resultFolder, 'bestModelFull')
        else:
            ckpt_name = os.path.join(self.resultFolder, 'bestModelFold_' + str(self.fold))

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss    
