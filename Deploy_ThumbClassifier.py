"""
Created on Wed Dec 15 15:55 2021

@author: Lars Hilgers
"""

import Utils as utils
from Train_Utils import Validate_model
import torch.nn as nn
import torchvision
import pandas as pd
import argparse
import torch
import os
import random
from sklearn import preprocessing
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np


##############################################################################

parser = argparse.ArgumentParser(description = 'Main Script to Run Training')
parser.add_argument('--ExpFile', type = str, default = r"C:\Users\lhilgers\sciebo\Lars\Experiments\Thumb_Classifier\Experiments\Classic\Thresholds\Test\FOXTROT_InternalValidation_Classic_Threshold.txt", help = 'Adress to the experiment File')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


##############################################################################

if __name__ == '__main__':

            args = utils.ReadExperimentFile(args, deploy = True)
            model_targets = args.target_labels

            thumb_data = pd.read_excel(args.thumbData)
            random.seed(args.seed)

            args.projectFolder = utils.CreateProjectFolder(args.project_name, args.ExpFile)

            os.mkdir(args.projectFolder)

            reportFile  = open(os.path.join(args.projectFolder,'Report.txt'), 'a', encoding="utf-8")

            reportFile.write('**********************************************************************'+ '\n')
            reportFile.write('DEPLOYING...')
            reportFile.write('\n' + '**********************************************************************'+ '\n')


            print('\n*** Load the DataSet ***\n')

            images = list(thumb_data["PATH"])

            args.num_classes = len(args.target_labels)

            args.result = os.path.join(args.projectFolder, 'RESULTS')
            os.makedirs(args.result, exist_ok = True)

            args.split_dir = os.path.join(args.projectFolder, 'DATA')
            os.makedirs(args.split_dir, exist_ok = True)

            if args.hasLabels:
                labels = list(thumb_data["CLASSIFICATION"])

                le = preprocessing.LabelEncoder()
                le.fit(args.target_labels)
                labelsList = le.transform(labels)
                args.labelsDict = dict(zip(le.classes_, range(len(le.classes_))))
                args.target_labels = list(set(labels))

                if args.cm_ylabels == []:
                    args.cm_ylabels = args.target_labels

                utils.Summarize_Classic(args, list(labelsList), reportFile)

                test_x = images
                test_y = labelsList


                df = pd.DataFrame(list(zip(test_x, test_y)), columns =['PATH', 'CLASSIFICATION'])
                df.to_csv(os.path.join(args.split_dir, 'THUMB_DATA'+ '.csv'), index = False)

                model, input_size = utils.Initialize_model(args.model_name, args.num_classes, use_pretrained = True)

                test_set = utils.DatasetLoader(test_x, test_y, transform = torchvision.transforms.ToTensor, target_patch_size = input_size)

                params = {'batch_size': args.batch_size,
                      'shuffle': False,
                      'num_workers': 16}

                test_generator = torch.utils.data.DataLoader(test_set, **params)

            else:

                #utils.Summarize_Classic(args, list(labelsList), reportFile)

                args.labelsDict = dict(zip(args.target_labels, range(len(args.target_labels))))

                test_x = images

                df = pd.DataFrame(list(test_x))
                df.to_csv(os.path.join(args.split_dir, 'THUMB_DATA'+ '.csv'), index = False)

                model, input_size = utils.Initialize_model(args.model_name, args.num_classes, use_pretrained = True)

                test_set = utils.DatasetLoader(test_x, labels = [],transform = torchvision.transforms.ToTensor, target_patch_size = input_size, hasLabels = False)

                params = {'batch_size': args.batch_size,
                      'shuffle': False,
                      'num_workers': 16}

                test_generator = torch.utils.data.DataLoader(test_set, **params)


            # load the pretrained thumb classifier model
            model.load_state_dict(torch.load(args.modelAdr))
            model = model.to(device)
            criterion = nn.CrossEntropyLoss()

            print('\n*** START DEPLOYING ***\n')

            if args.hasLabels:

                epoch_loss, epoch_acc, predList  = Validate_model(model, test_generator, criterion, args.hasLabels)

            else:

                predList = Validate_model(model, test_generator, criterion, args.hasLabels)

            scores = {}
            for index, key in enumerate(list(args.labelsDict.keys())):
                scores[key] = []
                for item in predList:
                    scores[key].append(item[index])

            scores = pd.DataFrame.from_dict(scores)
            pred_values = scores.max(axis=1)
            prediction_y = scores.idxmax(axis=1)

            for i in range(len(pred_values)):
                if pred_values[i] < args.prediction_threshold:
                    prediction_y[i] = "Unclassified"


            prediction_y.name = "model_prediction"


            if args.hasLabels:

                df = pd.DataFrame(list(zip(test_x, test_y)), columns =['PATH', 'CLASSIFICATION'])
                results_df = pd.concat([df, scores, prediction_y], axis=1)

                results_df.to_csv(os.path.join(args.result, 'THUMB_CLASSIFICATION_RESULTS.csv'), index = False)

                graph_dir = (os.path.join(args.result, "Graphs"))
                os.mkdir(graph_dir)

                #returnList, results_df = utils.CalculateTotalROC(resultsPath = args.result, results = results_df, labelsDict =  labelsDict, resultGraphs=graph_dir)

                keys = list(args.labelsDict.keys())
                true_y = results_df["CLASSIFICATION"]

                returnList = []

                for key in keys:
                    pred_y = results_df[str(key)]
                    fpr, tpr, thresholds = metrics.roc_curve(true_y, pred_y, pos_label= args.labelsDict[key])

                    print('AUC FOR TARGET {} IN THIS DATA SET IS: {} '.format(key, metrics.auc(fpr, tpr)))
                    #reportFile.write('AUC FOR TARGET {} IN THIS DATA SET IS: {}\n '.format(key, metrics.auc(fpr, tpr)))
                    returnList.append('AUC FOR TARGET {} IN THIS DATA SET IS: {}\n '.format(key, metrics.auc(fpr, tpr)))

                    auc_values = []
                    nsamples = 1000
                    rng = np.random.RandomState(666)
                    y_true = np.array(true_y)
                    y_pred = np.array(pred_y)

                    for i in range(nsamples):
                        indices = rng.randint(0, len(pred_y), len(pred_y))
                        if len(np.unique(pred_y[indices])) < 2 or np.sum(true_y[indices]) == 0:
                            continue
                        fpr, tpr, thresholds = metrics.roc_curve(true_y[indices], pred_y[indices],pos_label=args.labelsDict[key])
                        auc_values.append(metrics.auc(fpr, tpr))

                    auc_values = np.array(auc_values)
                    auc_values.sort()


                    print('Lower Confidence Interval For Target {}: {}'.format(key, np.round(
                            auc_values[int(0.025 * len(auc_values))], 10)))
                    #reportFile.write('Lower Confidence Interval For Target {}: {}'.format(key, np.round(
                            #auc_values[int(0.025 * len(auc_values))], 3)))
                    returnList.append('Lower Confidence Interval For Target {}: {}'.format(key, np.round(
                            auc_values[int(0.025 * len(auc_values))], 10)))

                    print('Higher Confidence Interval For Target {} : {}'.format(key, np.round(
                            auc_values[int(0.975 * len(auc_values))], 10)))
                    #reportFile.write('Higher Confidence Interval For Target {} : {}'.format(key, np.round(
                            #auc_values[int(0.975 * len(auc_values))], 3)))
                    returnList.append('Higher Confidence Interval For Target {} : {} \n'.format(key, np.round(
                            auc_values[int(0.975 * len(auc_values))], 10)))


                    fig, ax = plt.subplots()
                    plt.title('Receiver Operating Characteristic', fontsize = 15)
                    plt.plot(fpr, tpr, color = "black")
                    plt.plot([0, 1], color = "navy", ls="--")
                    plt.ylabel('TPR (sensitivity', fontsize = 15)
                    plt.xlabel('FPR (1-specificity)', fontsize = 15)
                    ax.tick_params(axis='both', which='major', labelsize=15)
                    ax.tick_params(axis='both', which='minor', labelsize=15)
                    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')

                    plt.savefig(os.path.join(graph_dir, str(key) + ".jpg"))

                    plt.clf()

                inv_labelsDict = {v: k for k, v in args.labelsDict.items()}
                true_y = true_y.replace(inv_labelsDict)

                results_df["CLASSIFICATION"] = true_y
                results_df.to_csv(os.path.join(args.result, 'THUMB_CLASSIFICATION_RESULTS.csv'), index=False)

                accuracy = utils.CalculateAccuracy(results_df)

                auc_total, auc_lower, auc_upper = utils.CalculateTotalAUROC(true_y, scores, model_targets)

                returnList.append("Macro-averaged AUC for this data set: {}".format(auc_total))
                returnList.append("Lower Confidence Interval for AUC for this data set: {}".format(auc_lower))
                returnList.append("Higher Confidence Interval for AUC for this data set: {}".format(auc_upper))

                print("Total Accuracy is: {:.4f}".format(accuracy))
                #reportFile.write("Total Accuracy is: {:.4f}".format(accuracy))
                returnList.append("Total Accuracy is: {:.4f}".format(accuracy))

                utils.ConfusionMatrix_new(results_df, graph_dir, args.cm_xlabels, args.cm_ylabels, args.cm_sortlabels)



                for item in returnList:
                    reportFile.write(item + '\n')
                reportFile.write('**********************************************************************' + '\n')

                pd.DataFrame(returnList).to_csv(os.path.join(args.result, "Evaluation.csv"), index = False)

            else:

                df = pd.DataFrame(list(test_x), columns = ['PATH'])
                results_df = pd.concat([df, scores, prediction_y], axis=1)

                results_df.to_csv(os.path.join(args.result, 'THUMB_CLASSIFICATION_RESULTS.csv'), index = False)

            if args.sortThumbs:
                utils.SortThumbs(results_df, args.result, args.prediction_threshold, args.hasLabels)

            reportFile.close()