# LUTA
Label Universal Targetted Attacks


## Software Requirements
* Python 3.5.2
* TensorFlow 1.9.0
* Keras 2.2.4
* Foolbox 1.8.0 (https://github.com/bethgelab/foolbox)

All the development/testing was done on Ubuntu 16.04. Once the software requirements are met. We need to download the data and 
arrange it in the desired hiearchy. 

## Data Downloading
Download the ImageNet ILSVRC2012 dataset from http://image-net.org/download (training and validation dataset). One needs to 
register before downloading.In order to setup the data for our code, we will create directories named by the wnids and inside
each directory there will be two folders training and testing. The training images will be inside training folder and the
validation images will become part of the testing folder.

The testing images will come from the imagenet validation dataset, which provides 50 samples for each class. The training images will
come from the imageNet training data. One also needs to create a directory for the images to be used for inhibition of leakage as 
discussed in the paper. This dataset can be created by picking 50 samples from each class (wnid) which get correctly predicted 
by VGG16 by confidence greater than 60%. Once the data is in this shape, you can open the code folder and and find the 
file __config.ini__, this file has to be modified to include the relevant paths.

Below, there is a description of each field present in the *config.ini*. 

| S/N | Field         | Field Description  |
| ----|-------------| ------------------|
|1    | saveDir       | Path of the directory where the results will be saved. The results are saved as “AdversarialAttackResults.db” |
|2    | datasetDir    | Path of the directory where the dataset will be present, folders named by wnids and inside each folder we should have testing and training folder.  |
|3    | inhibitionDatsetDir      | Path of the directory where samples used for inhibition will be present. Should include samples from all the classes.  |
|4    | imageNetValidationDir    | Path of the directory where imagenet validation images can be found. There are 50000 images. |
|5    | imageNet2012ValidationGroundTruthFile | Path of the file “ILSVRC2012_validation_ground_truth.txt”. This comes with ImageNet2012 validation dataset. |
|6    | imageNet2012LabelMapFile  | Path of the file “imagenet_2012_challenge_label_map_proto.pbtxt”. This comes with the imageNet2012 validation dataset. |
|7    | sourceIdentities   | It is a comma separated Wnids that will be taken as source classes. Note the data will be picked based on these wnids and the path of the dataset set in datasetDir. |
|8    | targetIdentities   | It is comma separated Wnids that will be taken as target classes.|
|9    | attackModels       | Comma separated attack Model Ids. It represents the deep model for launching the target attack. You can find the table below to select it. |
|10   | etas               | Comma separated values of eta for each algorithm id.|
|11   | algorithmId        | Comma separated Algorithm IDs. These algorithms will be launched one by one on each deep Models that you have select on each pair of source and target Identities. Please see the table below to find the algorithm ids. |

The algorithmIds can be selected from table below.

| S/N | Algorithm Description   | Algorithm ID |
| ----|-------------------------| -------------|
|1    | LinfinityBounded        | 3            |
|2    | L2Bounded               | 4            |

and the AttackModels can be selected from below table

| S/N | AttackModel Description | Attack Model ID |
| ----|-------------------------| ----------------|
|1    | VGG16                   | 1               |
|2    | ResNet50                | 2               |
|3    | InceptionV3             | 3               |
|4    | MobileNetV2             | 4               |


Once you have setup the config.ini file, you run the code by running the script as 
 ```
 python MainScriptToRunAttacksWithInhibition.py 
 
 ```
 to run attacks with inhibition as described in the paper.
 
One can also run 

```
python MainScriptToRunAttacksWithoutInhibition.py 
```

to run the attacks without inhibition. 

The results are saved in the database. One can check the tables *attacktrainingperformance* and *attacktestingperformance* 
to find the training and testing accuracy. The perturbations are saved in *attack* table, in the column *perturbedimage*. 





