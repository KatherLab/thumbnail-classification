# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 10:32:34 2021

@author: Lars Hilgers
"""

import Utils as utils
import os
import torch
import random
import argparse
import torchvision
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from sklearn import preprocessing
import sklearn.metrics as metrics
from sklearn.model_selection import StratifiedKFold
from Utils import DatasetLoader, Initialize_model
from Train_Utils import Train_model, Validate_model
from Utils import CalculateTotalROC, MergeResultCSV, Summarize_Classic, ConfusionMatrix, ReadExperimentFile, CreateProjectFolder, CalculateAccuracy


##############################################################################

parser = argparse.ArgumentParser(description='Main Script to Run Training')
parser.add_argument('--ExpFile', type = str, default = r"C:\Users\lhilgers\sciebo\Lars\Experiments\Thumb_Classifier\Experiments\Classic\Thresholds\Test\CR07_FullTrain_Classic_Test.txt", help = 'Adress to the experiment File')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

##############################################################################

if __name__ == '__main__':


        args = ReadExperimentFile(args, deploy=True)

        args.projectFolder = CreateProjectFolder(args.project_name, args.ExpFile)

        os.mkdir(args.projectFolder)

        # Load Ground Truth Data:

        print('\n*** LOAD THE DATASET FOR TRAINING ***\n')

        thumb_data = pd.read_excel(args.thumbData)
        thumb_data = thumb_data[thumb_data["CLASSIFICATION"].notna()]


        images = list(thumb_data["PATH"])
        labels = list(thumb_data["CLASSIFICATION"])

        le = preprocessing.LabelEncoder()
        labels = le.fit_transform(labels)

        args.num_classes = len(set(labels))
        args.labelsDict = dict(zip(le.classes_, range(len(le.classes_))))
        args.target_labels = list(le.classes_)


        reportFile = open(os.path.join(args.projectFolder,'Report.txt'), 'a', encoding="utf-8")
        reportFile.write('**********************************************************************' + '\n')
        reportFile.write(str(args))
        reportFile.write('\n' + '**********************************************************************' + '\n')

        Summarize_Classic(args, list(labels), reportFile)

        args.result = os.path.join(args.projectFolder, 'RESULTS')
        os.makedirs(args.result, exist_ok = True)
        args.split_dir = os.path.join(args.projectFolder, 'SPLITS')
        os.makedirs(args.split_dir, exist_ok = True)


        if args.trainFull:

            print('\n***THIS IS A FULL TRAINING!***\n')
            reportFile.write('\n***THIS IS A FULL TRAINING!***\n')

            train_index = list(range(0,len(labels)))

            val_index = random.choices(train_index, k=int(len(train_index) * 0.05))

            val_x = [images[i] for i in val_index]
            val_y = [labels[i] for i in val_index]

            train_index = [i for i in train_index if i not in val_index]

            train_x = [images[i] for i in train_index]
            train_y = [labels[i] for i in train_index]

            train_df = pd.DataFrame(zip(train_x,train_y))

            train_subset = pd.DataFrame()

            for tissue in set(train_df[1]):
                train_subset =  train_subset.append(train_df[train_df[1]==tissue].sample(n))


            train_x = list(train_subset[0])
            train_y = list(train_subset[1])


            df = pd.DataFrame(list(zip(train_x, train_y)), columns=['PATH', 'CLASSIFICATION'])
            df.to_csv(os.path.join(args.split_dir, 'FULL_TRAIN_TRAIN_SET' + '.csv'), index=False)

            df = pd.DataFrame(list(zip(val_x, val_y)), columns=['PATH', 'CLASSIFICATION'])
            df.to_csv(os.path.join(args.split_dir, 'FULL_TRAIN_VALIDATION_SET' + '.csv'), index=False)

            model_ft, input_size = Initialize_model(args.model_name, args.num_classes, use_pretrained=True)

            params = {'batch_size': args.batch_size,
                      'shuffle': True,
                      'num_workers': 16,
                      'pin_memory': False}

            train_set = DatasetLoader(train_x, train_y, transform = torchvision.transforms.ToTensor, target_patch_size = input_size)
            train_generator = torch.utils.data.DataLoader(train_set, **params)

            if 'val_index' in locals():
                val_set = DatasetLoader(val_x, val_y, transform=torchvision.transforms.ToTensor,
                                        target_patch_size=input_size)
                val_generator = torch.utils.data.DataLoader(val_set, **params)

            noOfLayers = 0
            for name, child in model_ft.named_children():
                noOfLayers += 1
            cut = int(args.freezeRatio * noOfLayers)
            ct = 0
            for name, child in model_ft.named_children():
                ct += 1
                if ct < cut:
                    for name2, params in child.named_parameters():
                        params.requires_grad = False

            model_ft = model_ft.to(device)


           # Optimizer

            print('\n*** INITIALIZE THE  OPTIMIZER***\n', end=' ')
            reportFile.write('\n*** INITIALIZE THE  OPTIMIZER***\n')
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_ft.parameters()) , lr = args.lr, weight_decay = args.wd)

            criterion = nn.CrossEntropyLoss()

           # Start of training

            print('\n*** START TRAINING ***', end=' ')

            model, train_loss_history, train_acc_history, val_acc_history, val_loss_history = Train_model(model=model_ft,
                                                                                                          trainLoaders=train_generator,
                                                                                                          valLoaders=val_generator,
                                                                                                          criterion=criterion,
                                                                                                          optimizer=optimizer,
                                                                                                          num_epochs=args.max_epochs,
                                                                                                          patience=args.patience,
                                                                                                          stop_epoch=args.stopEpoch, trainFull = True, resultFolder = args.result)

            torch.save(model.state_dict(), os.path.join(args.projectFolder, 'RESULTS', 'MODEL_Full_Final'))

            df = pd.DataFrame(list(zip(train_loss_history, train_acc_history)),
                              columns=['train_loss_history', 'train_acc_history'])

            df.to_csv(os.path.join(args.result, 'TRAIN_HISTORY_full' + '.csv'))

            reportFile.close()


        else:
                print('**********************************************************************')
                print('START OF CROSS VALIDATION! THIS RUN USES {} % OF THE DATASET!'.format(100))
                print('**********************************************************************')

                reportFile.write('**********************************************************************\n')
                reportFile.write('START OF CROSS VALIDATION\n')
                reportFile.write('**********************************************************************\n')




                folds = args.foldNumber
                kf = StratifiedKFold(n_splits=folds, random_state=args.seed, shuffle=True)
                kf.get_n_splits(images, labels)

                fold = 1

                for train_index, test_index in kf.split(images, labels):
                    images = np.array(images)
                    labels = np.array(labels)

                    test_x = images[test_index]
                    test_y = labels[test_index]

                    if not len(images)<50:
                        val_index = random.choices(train_index, k = int(len(train_index) * 0.05))

                        val_x = images[val_index]
                        val_y = labels[val_index]

                    train_index = [i for i in train_index if i not in val_index]

                    train_x = images[train_index]
                    train_y = labels[train_index]

                    #train_df = pd.DataFrame(zip(train_x,train_y))

                    #train_subset = pd.DataFrame()
                    #print(train_df)
                    #for tissue in set(train_df[1]):
                    #   train_subset =  train_subset.append(train_df[train_df[1]==tissue].sample(n))
                    #rain_subset = pd.DataFrame(train_subset)

                    #print(train_subset)
                    #train_x = list(train_subset[0])

                    #train_y = list(train_subset[1])

                    #print(train_x, train_y)


                    df = pd.DataFrame(list(zip(train_x, train_y)), columns =['PATH', 'CLASSIFICATION'])
                    df.to_csv(os.path.join(args.split_dir, 'Train_' + str(fold)+ '.csv'), index = False)

                    df = pd.DataFrame(list(zip(test_x, test_y)), columns =['PATH', 'CLASSIFICATION'])
                    df.to_csv(os.path.join(args.split_dir, 'Test_' + str(fold)+ '.csv'), index = False)

                    df = pd.DataFrame(list(zip(val_x, val_y)), columns =['PATH', 'CLASSIFICATION'])
                    df.to_csv(os.path.join(args.split_dir, 'Validation_' + str(fold)+ '.csv'), index = False)

                    model_ft, input_size = Initialize_model(args.model_name, args.num_classes, use_pretrained = True)

                    if args.SSL:
                        model_ft.load_state_dict(torch.load(args.modelAdr),strict = False)

                    params = {'batch_size': args.batch_size,
                              'shuffle': True,
                              'num_workers': 16,
                              'pin_memory' : False}

                    train_set = DatasetLoader(train_x, train_y, transform = torchvision.transforms.ToTensor, target_patch_size = input_size)
                    train_generator = torch.utils.data.DataLoader(train_set, **params)

                    if 'val_index' in locals():
                        val_set = DatasetLoader(val_x, val_y, transform = torchvision.transforms.ToTensor, target_patch_size = input_size)
                        val_generator = torch.utils.data.DataLoader(val_set, **params)

                    params = {'batch_size': args.batch_size,
                              'shuffle': False,
                              'num_workers': 16,
                              'pin_memory' : False}

                    test_set = DatasetLoader(test_x, test_y, transform = torchvision.transforms.ToTensor, target_patch_size = input_size)
                    test_generator = torch.utils.data.DataLoader(test_set, **params)

                    noOfLayers = 0
                    for name, child in model_ft.named_children():
                        noOfLayers += 1
                    cut = int(args.freezeRatio * noOfLayers)
                    ct = 0
                    for name, child in model_ft.named_children():
                        ct += 1
                        if ct < cut:
                            for name2, params in child.named_parameters():
                                params.requires_grad = False


                    model_ft = model_ft.to(device)


                    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_ft.parameters()) , lr = args.lr, weight_decay = args.wd)
                    criterion = nn.CrossEntropyLoss()

                    model, train_loss_history, train_acc_history, val_acc_history, val_loss_history = Train_model(model = model_ft, trainLoaders = train_generator,valLoaders = val_generator, criterion = criterion,
                                                                                                                  optimizer = optimizer, num_epochs = args.max_epochs, patience = args.patience, stop_epoch = args.stopEpoch,
                                                                                                                  trainFull = False, resultFolder = args.result, fold = fold)
                    torch.save(model.state_dict(), os.path.join(args.result,'MODEL_FOLD_FINAL' + str(fold)))
                    df = pd.DataFrame(list(zip(train_loss_history, train_acc_history, val_loss_history, val_acc_history)),
                                                      columns =['train_loss_history', 'train_acc_history', 'val_loss_history', 'val_acc_history'])

                    df.to_csv(os.path.join(args.result, 'TRAIN_HISTORY_FOLD_' + str(fold) + '.csv'), index = False)

                    epoch_loss, epoch_acc, predList  = Validate_model(model, test_generator, criterion, args.hasLabels)

                    predictions = {}
                    for index, key in enumerate(list(args.labelsDict.keys())):
                        predictions[key] = []
                        for item in predList:
                            predictions[key].append(item[index])

                    predictions = pd.DataFrame.from_dict(predictions)

                    results_df = pd.DataFrame(list(zip(test_x, test_y)), columns =['PATH', 'CLASSIFICATION'])
                    results_df = pd.concat([results_df, predictions], axis=1)

                    results_df.to_csv(os.path.join(args.result, 'TEST_RESULT_SCORES_' + str(fold) + '.csv'), index = False)

                    keys = list(args.labelsDict.keys())
                    for index, key in enumerate(keys):
                        fpr, tpr, thresholds = metrics.roc_curve(test_y, results_df[key], pos_label = args.labelsDict[key])
                        print('AUC FOR TARGET {} IN THIS DATA SET IN FOLD {} IS: {} '.format(key, fold, metrics.auc(fpr, tpr)))
                        reportFile.write('AUC FOR TARGET {} IN THIS DATA SET IN FOLD {} IS: {} '.format(key, fold, metrics.auc(fpr, tpr)) + '\n')


                    fold += 1

                reportFile.write('**********************************************************************' + '\n')
                test_scores = []
                for i in range(args.foldNumber):
                    test_scores.append('TEST_RESULT_SCORES_' + str(i+1) + '.csv')

                graph_dir = (os.path.join(args.result, "Graphs"))
                os.mkdir(graph_dir)

                ######################
                results_df = []

                for item in test_scores:
                    data = pd.read_csv(os.path.join(args.result, item))
                    results_df.append(data)
                results_df = pd.concat(results_df)

                results_df["model_prediction"] = results_df.iloc[:, 2:].idxmax(axis=1)

                keys = list(args.labelsDict.keys())
                true_y = results_df["CLASSIFICATION"]

                returnList = []


                for key in keys:
                    pred_y = results_df[str(key)]
                    fpr, tpr, thresholds = metrics.roc_curve(true_y, pred_y, pos_label=args.labelsDict[key])

                    print('AUC FOR TARGET {} IN THIS DATA SET IS: {} '.format(key, metrics.auc(fpr, tpr)))
                    # reportFile.write('AUC FOR TARGET {} IN THIS DATA SET IS: {}\n '.format(key, metrics.auc(fpr, tpr)))
                    returnList.append('AUC FOR TARGET {} IN THIS DATA SET IS: {}\n '.format(key, metrics.auc(fpr, tpr)))

                    auc_values = []
                    nsamples = 1000
                    rng = np.random.RandomState(666)
                    y_true = np.array(true_y)
                    y_pred = np.array(pred_y)

                    for i in range(nsamples):
                        indices = rng.randint(0, len(y_pred), len(y_pred))
                        if len(np.unique(y_pred[indices])) < 2 or np.sum(y_true[indices]) == 0:
                            continue
                        fpr, tpr, thresholds = metrics.roc_curve(y_true[indices], y_pred[indices],
                                                                 pos_label=args.labelsDict[key])
                        auc_values.append(metrics.auc(fpr, tpr))

                    auc_values = np.array(auc_values)
                    auc_values.sort()

                    print('Lower Confidence Interval For Target {}: {}'.format(key, np.round(
                        auc_values[int(0.025 * len(auc_values))], 10)))
                    # reportFile.write('Lower Confidence Interval For Target {}: {}'.format(key, np.round(
                    # auc_values[int(0.025 * len(auc_values))], 3)))
                    returnList.append('Lower Confidence Interval For Target {}: {}'.format(key, np.round(
                        auc_values[int(0.025 * len(auc_values))], 10)))

                    print('Higher Confidence Interval For Target {} : {}'.format(key, np.round(
                        auc_values[int(0.975 * len(auc_values))], 10)))
                    # reportFile.write('Higher Confidence Interval For Target {} : {}'.format(key, np.round(
                    # auc_values[int(0.975 * len(auc_values))], 3)))
                    returnList.append('Higher Confidence Interval For Target {} : {} \n'.format(key, np.round(
                        auc_values[int(0.975 * len(auc_values))], 10)))

                    fig, ax = plt.subplots()
                    plt.title('Receiver Operating Characteristic', fontsize=15)
                    plt.plot(fpr, tpr, color="black")
                    plt.plot([0, 1], color="navy", ls="--")
                    plt.ylabel('TPR (sensitivity', fontsize=15)
                    plt.xlabel('FPR (1-specificity)', fontsize=15)
                    ax.tick_params(axis='both', which='major', labelsize=15)
                    ax.tick_params(axis='both', which='minor', labelsize=15)
                    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')

                    plt.savefig(os.path.join(graph_dir, str(key) + ".jpg"))

                    plt.clf()


                inv_labelsDict = {v: k for k, v in args.labelsDict.items()}
                true_y = true_y.replace(inv_labelsDict)



                results_df["CLASSIFICATION"] = true_y
                results_df.to_csv(os.path.join(args.result, 'THUMB_CLASSIFICATION_RESULTS.csv'), index=False)

                accuracy = utils.CalculateAccuracy(results_df)

                scores = results_df.iloc[:,2:(len(results_df.columns)-1)]
                true_y = results_df["CLASSIFICATION"]


                auc_total, auc_lower, auc_upper = utils.CalculateTotalAUROC(true_y, scores, args.target_labels)

                returnList.append("Macro-averaged AUC for this data set: {}".format(auc_total))
                returnList.append("Lower Confidence Interval for AUC for this data set: {}".format(auc_lower))
                returnList.append("Higher Confidence Interval for AUC for this data set: {}".format(auc_upper))

                print("Total Accuracy is: {:.4f}".format(accuracy))
                # reportFile.write("Total Accuracy is: {:.4f}".format(accuracy))
                returnList.append("Total Accuracy is: {:.4f}".format(accuracy))

                utils.ConfusionMatrix_new(results_df, graph_dir, args.cm_xlabels, args.cm_ylabels, args.cm_sortlabels)

                ##############################

                for item in returnList:
                    reportFile.write(item + '\n')
                reportFile.write('**********************************************************************' + '\n')

                MergeResultCSV(args.result, test_scores, returnList)

                reportFile.close()

###############################################################################















