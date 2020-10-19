import os
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import roc_curve,auc

WorkPath = "D:/TestAndAnalysis"
DataMidPath = "/Data/"
AnalysisMidPath = "/RelocalizationAnalysis/"
RelocalizationDataMidPath = "/RelocalizationData/"

DataFileName = "Data.csv"
ROCFileName = "ROC.png"
LearningCurveFileName = "LearningCurve.png"
RefPoseFileName = "RefPose.json"
PredictPoseFileName = "PredictPose.json"
EigenVectorFileName = "EigenVector.json"

MoveThreshold = 5
RotateThreshole = 5

# Original dataset
EigenNames = []
EigenSpace = []
Lable = []

# Train dataset
TrainEigenSpace = []
TrainEigenSpaceNormalized = []
TrainLable = []

# Test dataset
TestEigenSpace = []
TestEigenSpaceNormalized = []
TestLable = []

# Parameters
RatioInNormalization = 1000
TrainMeanInNormalization = []
TrainStdInNormalization = []

# Classifier
Clf = SVC()

# Something to do
def Run():
    LoadData()
    WriteDataFile()
    SplitAndNormalize()
    GridSearchParam()
    LearningCurve()
    ROCCurve()

def SetWorkPath(workPath):
    global WorkPath
    WorkPath = workPath

def LoadData():
    global WorkPath
    global DataMidPath
    global RelocalizationDataMidPath
    global RefPoseFileName
    global PredictPoseFileName
    global EigenVectorFileName
    global MoveThreshold
    global RotateThreshole
    global EigenNames
    global EigenSpace
    global Lable
    Data = os.listdir(WorkPath + DataMidPath)
    for Scene in Data:
        RelocalizationData = os.listdir(WorkPath + DataMidPath + Scene + RelocalizationDataMidPath)
        for Position in RelocalizationData:
            PositionData = os.listdir(WorkPath + DataMidPath + Scene + RelocalizationDataMidPath + Position + "/")
            RefPoseData = {}
            PredictPoseData = {}
            for JsonFile in PositionData:
                if JsonFile == RefPoseFileName:
                    with open(WorkPath + DataMidPath + Scene + RelocalizationDataMidPath + Position + "/" + JsonFile,'r') as RefPoseRead:
                        RefPoseData = json.load(RefPoseRead)
                    RefPoseRead.close()

                if JsonFile == PredictPoseFileName:
                    with open(WorkPath + DataMidPath + Scene + RelocalizationDataMidPath + Position + "/" + JsonFile,'r') as PredictPoseRead:
                        PredictPoseData = json.load(PredictPoseRead)
                    PredictPoseRead.close()

                if JsonFile == EigenVectorFileName:
                    with open(WorkPath + DataMidPath + Scene + RelocalizationDataMidPath + Position + "/" + JsonFile,'r') as EigenVectorRead:
                        EigenVectorData = json.load(EigenVectorRead)
                        if len(EigenNames) == 0:
                            EigenNames = list(EigenVectorData.keys())
                        EigenVector = []
                        for EigenItem in EigenNames:
                            EigenVector.append(EigenVectorData[EigenItem])
                        EigenSpace.append(EigenVector)
                    EigenVectorRead.close()
                    
            if(abs(RefPoseData['x']-PredictPoseData['x'])<MoveThreshold and
                abs(RefPoseData['y']-PredictPoseData['y'])<MoveThreshold and
                abs(RefPoseData['phi']-PredictPoseData['phi'])<RotateThreshole):
                Lable.append(1)
            else:
                Lable.append(0)

def WriteDataFile():
    global WorkPath
    global AnalysisMidPath
    global DataFileName
    global EigenNames
    global EigenSpace
    try:
        DataFile=open(WorkPath + AnalysisMidPath+DataFileName,'w')
        for EigenItem in EigenNames:
            DataFile.write(EigenItem)
            DataFile.write(",")
        DataFile.write("Lable")
        DataFile.write(",\n")

        cnt = 0
        for EigenVector in EigenSpace:
            for EigenItem in EigenVector:
                DataFile.write(str(EigenItem))
                DataFile.write(",")
            DataFile.write(str(Lable[cnt]))
            DataFile.write(",\n")
            cnt = cnt + 1
    except Exception :
        print("Write DataFile Fail!!!")
    finally:
        DataFile.close();

def Normalize(DataSpace):
    global RatioInNormalization
    global TrainMeanInNormalization
    global TrainStdInNormalization
    Array = np.array(DataSpace)
    [row,col] = np.shape(DataSpace)
    for i in range(row):
        for j in range(col):
            Array[i,j] = ((Array[i,j] - TrainMeanInNormalization[j])/TrainStdInNormalization[j])*RatioInNormalization
    return Array.tolist()

def SplitAndNormalize():
    global EigenSpace
    global Lable
    global TrainEigenSpace
    global TrainEigenSpaceNormalized
    global TrainLable
    global TestEigenSpace
    global TestEigenSpaceNormalized
    global TestLable
    global TrainMeanInNormalization
    global TrainStdInNormalization
    TrainEigenSpace,TestEigenSpace,TrainLable,TestLable=train_test_split(EigenSpace,Lable,test_size=0.2,random_state=1)
    TrainMeanInNormalization = np.mean(TrainEigenSpace,axis=0)
    TrainStdInNormalization = np.std(TrainEigenSpace,axis=0)
    TrainEigenSpaceNormalized = Normalize(TrainEigenSpace)
    TestEigenSpaceNormalized = Normalize(TestEigenSpace)

def GridSearchParam():
    global Clf
    C_range = np.logspace(-4, 5, 10)
    gamma_range = np.logspace(-9, 3, 13)
    kernel = ['rbf']
    ParamGrid = dict(gamma = gamma_range, C = C_range, kernel = kernel)
    Cv = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 1)
    Svc = GridSearchCV(SVC(), param_grid = ParamGrid, cv = Cv)
    Svc.fit(TrainEigenSpaceNormalized, TrainLable)
    Clf = Svc.best_estimator_

def LearningCurve():
    global Clf
    global TrainEigenSpaceNormalized
    global TrainLable
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=1)
    train_sizes,train_scores,test_scores=learning_curve(Clf,TrainEigenSpaceNormalized,TrainLable,train_sizes=np.linspace(0.1,1.0,10),cv=cv)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Output .png file
    plt.figure()
    plt.title("Learning Curve")
    plt.ylim(0.8,1.05)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="navy")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="navy", label="Cross-validation score")
    plt.legend(loc="best")
    plt.savefig(WorkPath+AnalysisMidPath+LearningCurveFileName)

def ROCCurve():
    global Clf
    global TrainEigenSpaceNormalized
    global TrainLable
    TrainPredictLable = Clf.predict(TrainEigenSpaceNormalized)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = roc_curve(TrainLable, TrainPredictLable)
    roc_auc = auc(fpr, tpr)
    
    # Output .png file
    plt.figure()
    plt.plot(fpr, tpr, color='r', label='ROC curve (area = %0.6f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(WorkPath+AnalysisMidPath+ROCFileName)
    
def GetEigenNames():
    global EigenNames
    return EigenNames

def GetEigenSpace():
    global EigenSpace
    return EigenSpace

def GetLable():
    global Lable
    return Lable

def GetTrainEigenSpace():
    global TrainEigenSpace
    return TrainEigenSpace

def GetTrainEigenSpaceNormalized():
    global TrainEigenSpaceNormalized
    return TrainEigenSpaceNormalized

def GetTrainLable():
    global TrainLable
    return TrainLable

def GetTestEigenSpace():
    global TestEigenSpace
    return TestEigenSpace

def GetTestEigenSpaceNormalized():
    global TestEigenSpaceNormalized
    return TestEigenSpaceNormalized

def GetTestLable():
    global TestLable
    return TestLable

def GetTrainMeanAndStdInNormalization():
    global TrainMeanInNormalization
    global TrainStdInNormalization
    list_value = np.vstack((TrainMeanInNormalization,TrainStdInNormalization))
    return list_value.tolist()

def GetRatioInNormalization():
    global RatioInNormalization
    return RatioInNormalization

def GetSVCParams():
    global Clf
    return [Clf.C, Clf.cache_size, Clf.degree, Clf.gamma, Clf.tol]
