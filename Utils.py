# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 11:00:34 2021

@author: Lars Hilgers
"""

import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from pytorch_pretrained_vit import ViT
#from efficientnet_pytorch import EfficientNet
import pandas as pd
import sklearn.metrics as metrics
import os
import json
import warnings
import shutil
import matplotlib.pyplot as plt
import seaborn as sbn

###############################################################################

class DatasetLoader(torch.utils.data.Dataset):

    def __init__(self, imgs, labels, transform = None, target_patch_size = -1, hasLabels = True):
        'Initialization'
        self.labels = labels
        self.imgs = imgs
        self.target_patch_size = target_patch_size
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.hasLabels = hasLabels

        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.imgs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        X = Image.open(self.imgs[index])
        width, height = X.size
        max_img_size = max(X.size)
        left_pad = int((max_img_size-width)/2)
        right_pad = int(max_img_size-width-left_pad)
        top_pad = int((max_img_size-height)/2)
        bottom_pad = int(max_img_size-height-top_pad)
        if self.target_patch_size is not None:
            X = transforms.Pad([left_pad, top_pad, right_pad, bottom_pad])(X)
            X = X.resize((self.target_patch_size, self.target_patch_size))
            X = np.array(X)
        if self.transform is not None:
            X = self.transform(X)
        if self.hasLabels:
            y = self.labels[index]
            return X, y
        else: 
            return X
###############################################################################
        

def Initialize_model(model_name, num_classes, use_pretrained = True):

    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained = use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 256
        
        #input_size = 512
     
    elif model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained = use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        
    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
        
    elif model_name == "vgg16":
        """ VGG11_bn
        """
        model_ft = models.vgg16_bn(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
        
    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299
        
    elif model_name == "vit":
        
        model_ft = ViT('B_32_imagenet1k', pretrained = True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 384
    elif model_name == 'efficient':
        model_ft = EfficientNet.from_pretrained('efficientnet-b7')
        num_ftrs = model_ft._fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

###############################################################################

def CalculateTotalROC(resultsPath, results, target_labelDict, resultGraphs):
    
    totalData = []
    returnList = []
    
    for item in results:
        data = pd.read_csv(os.path.join(resultsPath, item))
        totalData.append(data)
    totalData = pd.concat(totalData)
    y_true = list(totalData['CLASSIFICATION'])
    keys = list(target_labelDict.keys())
    
    for key in keys:
        y_pred = totalData[str(key)]
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label = target_labelDict[key])
        print('TOTAL AUC FOR target {} IN THIS DATASET IS : {} '.format(key, np.round(metrics.auc(fpr, tpr), 3)))
        auc_values = []
        nsamples = 1000
        rng = np.random.RandomState(666)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        for i in range(nsamples):
            indices = rng.randint(0, len(y_pred), len(y_pred))
            if len(np.unique(y_pred[indices])) < 2 or np.sum(y_true[indices]) == 0:
                continue    
            fpr, tpr, thresholds = metrics.roc_curve(y_true[indices], y_pred[indices], pos_label = target_labelDict[key])
            auc_values.append(metrics.auc(fpr, tpr))
        
        auc_values = np.array(auc_values)
        auc_values.sort()
        
        returnList.append('TOTAL AUC For Target {} In This Dataset Is : {} '.format(key, np.round(metrics.auc(fpr, tpr), 3)))
        returnList.append('Lower Confidence Interval For Target {}: {}'.format(key, np.round(auc_values[int(0.025 * len(auc_values))], 3)))
        returnList.append('Higher Confidence Interval For Target {} : {}'.format(key, np.round(auc_values[int(0.975 * len(auc_values))], 3)))
        
        print('Lower Confidence Interval For Target {}: {}'.format(key, np.round(auc_values[int(0.025 * len(auc_values))], 3)))        
        print('Higher Confidence Interval For Target {} : {}'.format(key, np.round(auc_values[int(0.975 * len(auc_values))], 3)))
        
        fig, ax = plt.subplots()
        plt.title('Receiver Operating Characteristic', fontsize = 15)
        plt.plot(fpr, tpr, color = "black")
        plt.plot([0, 1], color = "navy", ls="--")
        plt.ylabel('TPR (sensitivity', fontsize = 15)
        plt.xlabel('FPR (1-specificity)', fontsize = 15)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.tick_params(axis='both', which='minor', labelsize=15)
        ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
            
        plt.savefig(resultGraphs + str(key) + ".jpg")
        
    totalData.to_csv(os.path.join(resultsPath, 'TEST_RESULTS_PATIENT_SCORES_TOTAL.csv'), index = False)
    return returnList, totalData

###############################################################################

def MergeResultCSV(resultsPath, results, returnList):
    
    totalData = []    
    for item in results:
        data = pd.read_csv(os.path.join(resultsPath, item))
        totalData.append(data)
    totalData = pd.concat(totalData)
    pd.DataFrame(returnList).to_csv(os.path.join(resultsPath, 'Evaluations.csv'), index = False)
    totalData.to_csv(os.path.join(resultsPath, 'TEST_RESULT_TILE_SCORES_TOTAL.csv'), index = False)

###############################################################################
    
def Summarize_Classic(args, labels, reportFile):

    if args.target_labels is not None:
        print("label column: {}".format(args.target_labels))
        reportFile.write("label column: {}".format(args.target_labels) + '\n')
    
    print("label dictionary: {}".format(args.labelsDict) + "\n")
    reportFile.write("label dictionary: {}".format(args.labelsDict) + '\n\n')
    
    
    for i in range(args.num_classes):
        print('Patient-LVL; Number of samples registered in class %d: %d' % (i, labels.count(i)))
        reportFile.write('Patient-LVL; Number of samples registered in class %d: %d' % (i, labels.count(i)) + '\n')
    
        
    print('\n##############################################################\n')
    reportFile.write('\n**********************************************************************'+ '\n')

###############################################################################

def ReadExperimentFile(args, deploy=False):
    with open(args.ExpFile) as json_file:
        data = json.load(json_file)

    args.csv_name = 'CLEANED_DATA'
    args.project_name = args.ExpFile.split('\\')[-1].replace('.txt', '')

    try:
        args.thumbData = data["thumbData"]
    except:
        raise NameError("NO THUMB DATA!")

    try:
        args.target_labels = data["target_labels"]
    except:
        warnings.warn("NO TARGET LABELS! \ DEFAULT VALUE WILL BE USED: None")
        args.target_labels = None

    try:
        args.modelAdr = data["modelAdr"]
    except:
        warnings.warn("MODEL ADRESS IS NOT DEFINED! \n DEFAULT VALUE WILL BE USED: None")
        args.modelAdr = None

    try:
        args.batch_size = data["batch_size"]
    except:
        warnings.warn("BATCH SIZE IS NOT DEFINED! \n DEFAULT VALUE WILL BE USED: 64")
        args.batch_size = 64

    try:
        args.prediction_threshold = float(data["prediction_threshold"])
    except:
        warnings.warn("PREDICTION THRESHOLD IS NOT DEFINED! \n DEFAULT VALUE WILL BE USED: 0.90")
        args.prediction_threshold = 0.90

    try:
        args.max_epochs = int(data['max_epochs'])
    except:
        warnings.warn('EPOCH NUMBER IS NOT DEFINED! \n DEFAULT VALUE WILL BE USED : 50')
        args.max_epochs = 50

    try:
        args.seed = int(data['seed'])
    except:
        warnings.warn('SEED IS NOT DEFINED! \n DEFAULT VALUE WILL BE USED : 1')
        args.seed = 1

    try:
        args.model_name = data['model_name']
    except:
        warnings.warn('MODEL NAME IS NOT DEFINED! \n DEFAULT VALUE WILL BE USED : resnet')
        args.model_name = 'resnet'

    try:
        args.hasLabels = bool(data['hasLabels'])
    except:
        warnings.warn('LABEL STATUS IS NOT DEFINED! \n DEFAULT VALUE WILL BE USED : True')
        args.hasLabels = True

    try:
        args.sortThumbs = bool(data['sortThumbs'])
    except:
        warnings.warn('NO DECISION IF THUMBS SHOULD BE SORTED! \n DEFAULT VALUE WILL BE USED : False')
        args.sortThumbs = False

    try:
        args.trainFull = bool(data['trainFull'])
    except:
        warnings.warn('NO DECISION IF FULL TRAINING! \n DEFAULT VALUE WILL BE USED : False')
        args.trainFull = False

    try:
        args.foldNumber = int(data['foldNumber'])
    except:
        warnings.warn('Number of folds not defined!! \n DEFAULT VALUE WILL BE USED : 3')
        args.foldNumber = 3

    try:
        args.lr = float(data['lr'])
    except:
        warnings.warn('Learning rate not defined!! \n DEFAULT VALUE WILL BE USED : 0.0001')
        args.lr = 0.0001

    try:
        args.wd = float(data['wd'])
    except:
        warnings.warn('Weight decay not defined!! \n DEFAULT VALUE WILL BE USED : 0.0001')
        args.wd = 0.0001

    try:
        args.patience = int(data['patience'])
    except:
        warnings.warn('Patience not defined!! \n DEFAULT VALUE WILL BE USED : 10')
        args.patience = 10

    try:
        args.stopEpoch = int(data['stopEpoch'])
    except:
        warnings.warn('Stop epoch not defined!! \n DEFAULT VALUE WILL BE USED : 20')
        args.stopEpoch = 20
    
    try:
        args.freezeRatio = float(data['freezeRatio'])
    except:
        warnings.warn('Freeze ratio not defined!! \n DEFAULT VALUE WILL BE USED : 0.5')
        args.freezeRatio = 0.5
    
    try:
        args.SSL = bool(data['SSL'])
    except:
        warnings.warn('SSL not defined!! \n DEFAULT VALUE WILL BE USED : False')
        args.SSL = False

    try:
        args.cm_xlabels = data["cm_xlabels"]
    except:
        warnings.warn('X-Labels for confusion matrix not defined!! \n Target labels will be used.')
        args.cm_xlabels = args.target_labels + ["Undecided"]

    try:
        args.cm_ylabels = data["cm_ylabels"]
    except:
        warnings.warn('Y-Labels for confusion matrix not defined!! \n Unique cohort labels will be used.')
        args.cm_ylabels = []

    try:
        args.cm_sortlabels = data["cm_sortlabels"]
    except:
        warnings.warn('Sorting labels for confusion matrix not defined!! \n Target labels will be used.')
        args.cm_sortlabels = args.target_labels

    return args


##############################################################################

def CreateProjectFolder(ExName, ExAdr):
    ExName = ExName
    outputPath = ExAdr.split('\\')
    outputPath = outputPath[:-1]
    outputPath[0] = outputPath[0] + '\\' 
    outputPath_root = os.path.join(*outputPath)
    outputPath = os.path.join(outputPath_root, ExName)

    return outputPath

##############################################################################

def CheckForTargetType(labelsList):
    if len(set(labelsList)) >= 5:
        labelList_temp = [str(i) for i in labelsList]
        checkList1 = [s for s in labelList_temp if isfloat(s)]
        checkList2 = [s for s in labelList_temp if isint(s)]
        if not len(checkList1) == 0 or not len(checkList2):
            med = np.median(labelsList)
            labelsList = [1 if i > med else 0 for i in labelsList]
        else:
            raise NameError('IT IS NOT POSSIBLE TO BINARIZE THE NOT NUMERIC TARGET LIST!')
    return labelsList

##############################################################################

def SortThumbs(results, result_directory, pred_threshold, hasLabels):
    
    if hasLabels:
        
        pred = results.iloc[:, 2:9].idxmax(axis=1)
        pred_values = results.iloc[:,2:9].max(axis=1)
        
    else:
        
        pred = results.iloc[:, 1:8].idxmax(axis=1)
        pred_values = results.iloc[:,1:8].max(axis=1)

    for label in set(pred):
        sort_dir = os.path.join(result_directory, label)
        os.mkdir(sort_dir)

    unsure_dir = os.path.join(result_directory, "other_unsure")
    os.mkdir(unsure_dir)


    for path, prediction, value in zip(results.PATH, pred, pred_values):
        if value > pred_threshold:
            shutil.copy(path, os.path.join(result_directory, prediction))
        else:
            shutil.copy(path, unsure_dir)
            
##############################################################################  
            
def ConfusionMatrix(true_y, pred_y, result, target_labels):


    cfm = metrics.confusion_matrix(true_y, pred_y, target_labels)

    #normalized cfm
    #cmn = cfm.astype("float")/cfm.sum(axis=1)[:,np.newaxis]
    
    #fig, ax = plt.subplots(figsize=(10,10))
    sbn.heatmap(cfm, annot=True, fmt='.2f', xticklabels=target_labels, yticklabels=target_labels)
    plt.title("Confusion Matrix")
    #plt.ylabel('Actual')
    #plt.xlabel('Predicted')
    plt.tick_params(labelsize = 10)
    plt.savefig(os.path.join(result, "Confusion Matrix.jpg"))

def CalculateAccuracy(results_df):

    results_classified = results_df.drop(results_df[results_df["model_prediction"] == "Unclassified"].index)
    ground_truth = list(results_classified["CLASSIFICATION"])
    model_prediction = list(results_classified["model_prediction"])

    correct = 0

    if len(ground_truth) == 0:
        acc = -1

    else:
        for i in range(len(ground_truth)):

            if ground_truth[i] == model_prediction[i]:

                correct = correct + 1

        acc = correct/len(ground_truth)

    return(acc)


def CalculateTotalAUROC(y_true, scores, lab):
    #y_true = results["CLASSIFICATION"]

    # for row in range(len(results)):
    #   results.iloc[row, [5,8]] = results.iloc[row, [5,8]] / sum(results.iloc[row, [5,8]])

    #y_pred = results.iloc[:, 8]

    print(scores)

    auc_total = metrics.roc_auc_score(y_true, scores, labels = lab, multi_class = "ovo")
    auc_values = []
    nsamples = 1000
    rng = np.random.RandomState(666)
    y_true = np.array(y_true)
    scores = np.array(scores)
    for i in range(nsamples):
        indices = rng.randint(0, len(scores), len(scores))
        if len(np.unique(scores[indices])) < 2:
            continue
        auc = metrics.roc_auc_score(y_true[indices], scores[indices], labels=lab, multi_class="ovo")
        auc_values.append(auc)

    auc_values = np.array(auc_values)
    auc_values.sort()

    auc_lower = np.round(auc_values[int(0.025 * len(auc_values))], 10)
    auc_upper = np.round(auc_values[int(0.975 * len(auc_values))], 10)

    print("Macro-averaged AUC for this data set: {}".format(auc_total))
    print("Lower Confidence Interval for AUC for this data set: {}".format(auc_lower))
    print("Higher Confidence Interval for AUC for this data set: {}".format(auc_upper))

    return(auc_total, auc_lower, auc_upper)

def ConfusionMatrix_new(result_df, result, target_labels_x, target_labels_y, sort_labels):

    true_y = result_df["CLASSIFICATION"]
    pred_y = result_df["model_prediction"]

    cfm = metrics.confusion_matrix(true_y, pred_y, labels = sort_labels)
    #cfm = np.delete(cfm,(-1,-2,-3,-4,-5),0)
    cfm = cfm[~np.all(cfm == 0, axis = 1)]

    print(cfm)

    #normalized cfm
    cmn = cfm.astype("float")/cfm.sum(axis=1)[:,np.newaxis]

    colors = sbn.color_palette("rocket")
    sbn.heatmap(cmn, annot=cfm, annot_kws={"size": 12}, fmt='d', xticklabels=target_labels_x, yticklabels=target_labels_y, cmap = colors, vmin = 0, vmax = 1.0)
    #plt.title("Confusion Matrix")
    plt.tick_params(labelsize = 12)
    plt.yticks(rotation = 0)
    plt.tight_layout()
    plt.savefig(os.path.join(result, "Confusion Matrix.jpg"))
    plt.close()