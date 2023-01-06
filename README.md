# thumbnail-classification
This repository contains python scripts for classification of colorectal cancer datasets consisting of histopathological images such as whole tissue resections containing tumor or only healthy tissue, biopsies, lymph node resections, IHCs etc. Simple low-res thumbnails are used as inputs for a CNN classifier based on a Resnet-18 in order to reduce computation time while remaining highly accurate. Our workflows are described in the following paper:
# how to use these scripts
If you want to train your own models, use Train_ThumbClassifier.py. If you simply want to deploy one of our models on a data set use Deploy_ThumbClassifier.py
For both scripts you only need to provide a text file ("experiment file") through which you can set parameters for training or deployment.
Here is an overview of all the parameters:
- thumbData: path to the folder containing your thumbnails
- target_labels: labels/classes that you want to use for training a new classifier
- modelAdr: path to a pre-trained model you want to use for deployment
- batch_size: defines batch size for training a new model
- prediction_threshold: defines a threshold for classification confidence. if classification confidence lies below the threshold the case is classified as "Undecided".
- max_epochs: define maximum number of training epochs
- seed: defines a seed for the random number generator
- model_name: defines which pre-trained model is used for transfer learning. possible models: 
- hasLabels: defines if your data is labeled or not when deploying a model.
- sortThumbs: defines if your thumbnails should be sorted in to folders according to their predicted label.
- trainFull: defines if the whole data set should be used for training the model (except for a small validation set).
- foldNumber: defines the number of folds used for cross-validation
- lr: defines learning rate
- wd: defines weight decay
- patience: defines how many epochs should be gone through while the validation_loss increases before stopping the training
- stopEpoch: defines earliest epoch after which training will be stopped
- freezeRatio: freezes a proportion of the neural network during training
- SSL: toggles self supervised learning
- cm_xlabels: defines labels for the x-axis labels of the confusion matrix
- cm_ylabels: defines labels for the y-axis labels of the confusion matrix
- cm_sortlabels: defines sorting order for confusion matrix labels
# results
# models
The models used in our paper were trained on data from the FOXTROT trial containing the following classes of tissue: tumor resection, healthy tissue, lymph node, biopsy, fat, IHC & TMA. These models were shown to produce accurate classification results on multiple external cohorts. 
