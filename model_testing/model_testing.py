import numpy as np
import pandas as pd
import sys, os, pickle
from hyperopt.pyll import scope
from hyperopt import fmin, tpe, hp, Trials
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

#Imports and parses the dataset by separating the order rank column into 'values'. Order rank is the target column to be predicted.
def importDataset(filename, trinary):
    dataset = pd.read_excel(filename)
    dataset = dataset.dropna(subset=["order_rank"]).dropna()
    dataset['order_rank'] =  dataset['order_rank'].apply(lambda x: 0 if x == 0  or x == 1  else 1) if trinary == 1 else dataset['order_rank']
    dataset['order_rank'] =  dataset['order_rank'].apply(lambda x: 0 if x == 0 else 1) if trinary == 2 else dataset['order_rank']
    points = dataset.drop(columns=['order_rank'])
    points = encode(points)
    values = dataset[['order_rank']]
    return points, values

#One hot encodes the data.
def encode(points):
    r = OneHotEncoder()
    preprocessor = ColumnTransformer(transformers=[('cat', r, ["hashed_manager_on_duty"])], remainder='passthrough')
    return preprocessor.fit_transform(points)

#Generates a matrix showing the average predictions by the model for each test point of each class.
def getAverageProbabilityDistributionDiagonalSum(predictions, test_values):
    class_totals = {}
    class_counts = {}
    for prediction_list, test_value in zip(predictions, test_values):
        if test_value not in class_totals:
            class_totals[test_value] = [0] * len(prediction_list)
            class_counts[test_value] = 0
        class_totals[test_value] = [total + pred for total, pred in zip(class_totals[test_value], prediction_list)]
        class_counts[test_value] += 1

    diagonalSum = 0
    for test_value in sorted(class_totals.keys()):
        total_list = class_totals[test_value]
        average_list = [total / class_counts[test_value] if class_counts[test_value] > 0 else 0 for total in total_list]
        diagonalSum += average_list[test_value]

    return diagonalSum

class BatchTrain:
    def __init__(self, models, spaces, objectives, labels, points, values, maxEvaluations, oversampling, state, metric):
        self.models = models
        self.spaces = spaces
        self.objectives = objectives
        self.labels = labels
        self.points = points
        self.values = values
        self.maxEvaluations = maxEvaluations
        self.oversampling = oversampling
        self.state = state
        self.metric = metric
        self.averageAccuracy = 0
        self.results = {}

    #Prints a matrix showing the average predictions by the model for each test point of each class.
    def printProbabilityDistribution(self, predictions, test_values):
        class_totals = {}
        class_counts = {}
        for prediction_list, test_value in zip(predictions, test_values):
            if test_value not in class_totals:
                class_totals[test_value] = [0] * len(prediction_list)
                class_counts[test_value] = 0
            class_totals[test_value] = [total + pred for total, pred in zip(class_totals[test_value], prediction_list)]
            class_counts[test_value] += 1

        total_count = len(predictions)

        diagonalSum = 0
        for test_value in sorted(class_totals.keys()):
            total_list = class_totals[test_value]
            average_list = [total / class_counts[test_value] if class_counts[test_value] > 0 else 0 for total in total_list]
            proportion = class_counts[test_value] / total_count
            print(f"Class {test_value} ({proportion:.2%}): {[round(avg, 4) for avg in average_list]}")
            diagonalSum += average_list[test_value]
        
        return diagonalSum

    #Calulates the accuracy and prediction probabilities of a model on the data.        
    def getAccuracy(self, model, trainingPoints, trainingValues, testPoints, testValues):
        originalStdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        trainingValues = np.ravel(trainingValues)
        try:
            model.fit(X=trainingPoints, y=trainingValues, verbose=0)
        except:
            model.fit(X=trainingPoints, y=trainingValues)
        predictionProbabilities = model.predict_proba(testPoints)
        testValues = np.ravel(testValues)
        predictions = np.argmax(predictionProbabilities, axis=1)
        accuracy = accuracy_score(testValues, predictions)
        sys.stdout = originalStdout
        return accuracy, predictionProbabilities, testValues

    #Trains each model and tunes it using Hyperopt. Saves the results so they can be printed and saves the models.
    def train(self):
        trainingPoints, testPoints, trainingValues, testValues = train_test_split(self.points, self.values, test_size=0.2, random_state=self.state)
        trainingPoints, validationPoints, trainingValues, validationValues = train_test_split(trainingPoints, trainingValues, test_size=0.2, random_state=self.state * 2)
        if self.oversampling:
            try:
                smote = SMOTE(sampling_strategy=0.7, k_neighbors=20)
                trainingPoints, trainingValues = smote.fit_resample(points, values)
            except Exception as e:
                print("Oversampling Failed: " + str(e))
        for model, space, objective, label, max_evals in zip(self.models, self.spaces, self.objectives, self.labels, self.maxEvaluations):
            print("Training " + label)
            trials = Trials()
            objective = objective(trainingPoints, trainingValues, validationPoints, validationValues, self.getAccuracy)
            hyperparameters = fmin(fn = objective,
                                   space = space,
                                   algo = tpe.suggest,
                                   max_evals = max_evals,
                                   trials = trials)
            hyperparameters = {key: int(value) if isinstance(value, float) and value.is_integer() else value for key, value in hyperparameters.items()}
            tunedModel = model(**hyperparameters)
            accuracy, predictionProbabilities, testValues = self.getAccuracy(tunedModel, trainingPoints, trainingValues, testPoints, testValues)
            distributionSum = self.printProbabilityDistribution(predictionProbabilities, testValues)
            print(label + ": " + str(accuracy) + r'% accuracy')
            key = label + " " + str(hyperparameters) + " :"
            if key in self.results:
                self.results[key] += (accuracy, distributionSum)
            else:
                self.results[key] = (accuracy, distributionSum)
            with open((label + '.pkl'), 'wb') as file:
                pickle.dump(model, file)

    #Sorts the models by performance on the passed in metric and prints them along with their accuracy and distribution (diagonal) sum.
    def printResults(self):
        print("------------------------------------------------")
        sortedModels = sorted(self.results.items(), key=lambda x: (x[1][0], x[1][1]) if self.metric == "acc" else (x[1][1], x[1][0]), reverse=True)
        for i, ((model), (accuracy, distributionSum)) in enumerate(sortedModels, start=1):
            print(f"{i}. {model} Accuracy {accuracy:.4f} Distribution Sum {distributionSum:.4f}")
            print()
        print("------------------------------------------------")
        
class XgBoost():
    space={'max_depth': scope.int(hp.quniform("max_depth", 3, 18, 1)),
           'gamma': hp.uniform ('gamma', 1,9),
           'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
           'reg_lambda' : hp.uniform('reg_lambda', 0,1),
           'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
           'min_child_weight' : scope.int(hp.quniform('min_child_weight', 0, 10, 1)),
           'n_estimators': 180
          }
    constructor = XGBClassifier
    label = "XGBoost"
    maxEvals = 2
    metric = "acc"
    def objective(self, trainingPoints, trainingValues, testPoints, testValues, getAccuracy):
        def objective(space):
                            model=XGBClassifier(
                                n_estimators =space['n_estimators'], max_depth = int(space['max_depth']), gamma = space['gamma'],
                                reg_alpha = int(space['reg_alpha']),min_child_weight=int(space['min_child_weight']),
                                colsample_bytree=int(space['colsample_bytree']))   
                            accuracy, predictionProbabilities, values = getAccuracy(model, trainingPoints, trainingValues, testPoints, testValues)
                            distributionSum = getAverageProbabilityDistributionDiagonalSum(predictionProbabilities, values)
                            loss = -1 * accuracy if self.metric == "acc" else -1 * distributionSum
                            return loss
        return objective
    
class CatBoost():
    space = {
        'depth': hp.quniform('depth', 3, 12, 1),
        'l2_leaf_reg': hp.quniform('l2_leaf_reg', 40, 180, 1),
        'iterations': hp.quniform('iterations', 150, 500, 25),
        'learning_rate': hp.loguniform('learning_rate', -5, 0),
        'colsample_bylevel': hp.uniform('colsample_bylevel', 0.5, 1),
        'min_child_samples': scope.int(hp.quniform('min_child_samples', 1, 10, 1)),
    }
    constructor = CatBoostClassifier
    label = "CatBoost"
    maxEvals = 1
    metric = "acc"
    def objective(self, trainingPoints, trainingValues, testPoints, testValues, getAccuracy):
        def objective(space):
            model = CatBoostClassifier(
                iterations=space['iterations'],
                depth=int(space['depth']),
                l2_leaf_reg=int(space['l2_leaf_reg']),
                learning_rate=space['learning_rate'],
                colsample_bylevel=space['colsample_bylevel'],
                min_child_samples=int(space['min_child_samples']),
            )

            accuracy, predictionProbabilities, values = getAccuracy(model, trainingPoints, trainingValues, testPoints, testValues)
            distributionSum = getAverageProbabilityDistributionDiagonalSum(predictionProbabilities, values)
            loss = -1 * accuracy if self.metric == "acc" else -1 * distributionSum
            return loss

        return objective
                            
class RandomForest():
    space = {
        'n_estimators': hp.quniform('n_estimators', 75, 120, 15),
        'max_depth': hp.quniform('max_depth', 3, 5, 1),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 10, 1)),
        'max_features': hp.uniform('max_features', 0.1, 1),
    }
    constructor = RandomForestClassifier
    label = "Random Forest"
    maxEvals = 2
    metric = "acc"
    def objective(self, trainingPoints, trainingValues, testPoints, testValues, getAccuracy):
        def objective(space):
            model = RandomForestClassifier(
                n_estimators=int(space['n_estimators']),
                max_depth=int(space['max_depth']),
                min_samples_split=int(space['min_samples_split']),
                min_samples_leaf=int(space['min_samples_leaf']),
                max_features=space['max_features'],
            )

            accuracy, predictionProbabilities, values = getAccuracy(model, trainingPoints, trainingValues, testPoints, testValues)
            distributionSum = getAverageProbabilityDistributionDiagonalSum(predictionProbabilities, values)
            loss = -1 * accuracy if self.metric == "acc" else -1 * distributionSum
            return loss

        return objective
    
class DecisionTree():
    space = {
        'max_depth': hp.quniform('max_depth', 3, 7, 1),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 20, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 10, 1)),
        'max_features': hp.uniform('max_features', 0.1, 1),
    }
    constructor = DecisionTreeClassifier
    label = "Decision Tree"
    maxEvals = 10
    metric = "acc"
    def objective(self, trainingPoints, trainingValues, testPoints, testValues, getAccuracy):
        def objective(space):
            model = DecisionTreeClassifier(
                max_depth=int(space['max_depth']),
                min_samples_split=int(space['min_samples_split']),
                min_samples_leaf=int(space['min_samples_leaf']),
                max_features=space['max_features'],
            )

            accuracy, predictionProbabilities, values = getAccuracy(model, trainingPoints, trainingValues, testPoints, testValues)
            distributionSum = getAverageProbabilityDistributionDiagonalSum(predictionProbabilities, values)
            loss = -1 * accuracy if self.metric == "acc" else -1 * distributionSum
            return loss

        return objective


class DummyClassifierMostFrequent():
    def __init__(self):
        self.model = DummyClassifier(strategy="most_frequent")
    
    def fit(self, X, y):
        return self.model.fit(X, y)
    
    def predict_proba(self, testPoints):
        return self.model.predict_proba(testPoints)

class MajorityClassClassifier():
    space = {'dummy' : 'dummy'}
    constructor = DummyClassifierMostFrequent
    label = "Majority Class Classifier"
    maxEvals = 1
    metric = "acc"
    def objective(self, trainingPoints, trainingValues, testPoints, testValues, getAccuracy):
        def objective(space):
            model = DummyClassifierMostFrequent().model
            
            accuracy, predictionProbabilities, values = getAccuracy(model, trainingPoints, trainingValues, testPoints, testValues)
            distributionSum = getAverageProbabilityDistributionDiagonalSum(predictionProbabilities, values)
            loss = -1 * accuracy if self.metric == "acc" else -1 * distributionSum
            return loss

        return objective
    

class LightGBM:
    space = {
        'max_depth': hp.quniform('max_depth', 6, 16, 1),
        'reg_lambda': hp.quniform('reg_lambda', 40, 180, 1),
        'iterations': hp.quniform('iterations', 150, 500, 25),
        'learning_rate': hp.loguniform('learning_rate', -5, 0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
        'min_child_samples': hp.quniform('min_child_samples', 1, 10, 1),
        'num_leaves': hp.quniform('num_leaves', 20, 200, 1)
    }
    constructor = LGBMClassifier
    label = "LightGBM"
    maxEvals = 20
    metric = "acc"
    def objective(self, trainingPoints, trainingValues, testPoints, testValues, getAccuracy):
        def objective(space):
            model = LGBMClassifier(
                iterations=int(space['iterations']),
                max_depth=int(space['max_depth']),
                reg_lambda=int(space['reg_lambda']),
                learning_rate=space['learning_rate'],
                colsample_bytree=space['colsample_bytree'],
                min_child_samples=int(space['min_child_samples']),
                num_leaves=int(space["num_leaves"])
            )

            accuracy, predictionProbabilities, values = getAccuracy(model, trainingPoints, trainingValues, testPoints, testValues)
            distributionSum = getAverageProbabilityDistributionDiagonalSum(predictionProbabilities, values)
            loss = -1 * accuracy if self.metric == "acc" else -1 * distributionSum
            return loss

        return objective

class AdaBoost:
    space = {
        'n_estimators': hp.quniform('n_estimators', 20, 50, 1),
        'learning_rate': hp.loguniform('learning_rate', -4, 0),
    }
    constructor = AdaBoostClassifier
    label = "AdaBoost"
    maxEvals = 1
    metric = "acc"
    def objective(self, trainingPoints, trainingValues, testPoints, testValues, getAccuracy):
        def objective(space):
            model = AdaBoostClassifier(
                n_estimators=int(space['n_estimators']),
                learning_rate=space['learning_rate']
            )

            accuracy, predictionProbabilities, values = getAccuracy(model, trainingPoints, trainingValues, testPoints, testValues)
            distributionSum = getAverageProbabilityDistributionDiagonalSum(predictionProbabilities, values)
            loss = -1 * accuracy if self.metric == "acc" else -1 * distributionSum
            return loss

        return objective

#Populates each matrix necessary for batch train.
def populateBatchTrainFields(modelClasses, models=[], spaces=[], objectives=[], labels=[], evaluations=[]):
    for modelClass in modelClasses:
        models.append(modelClass.constructor)
        spaces.append(modelClass.space)
        objectives.append(modelClass.objective)
        labels.append(modelClass.label)
        evaluations.append(modelClass.maxEvals)
    return models, spaces, objectives, labels, evaluations

dataset = "order-processed-large-hashed.xlsx"
state = 50
oversampling = True
#Set to anything other than "acc" to use the diagonal sum of the probability averages per class as the loss for hyperparameter tuning.
metric="distro"
#0 for multiclass, 1 to combine solo orders and orders sent out first, 2 to just separate solo orders from MODs
trinary = 0
oversampling = False if trinary == 0 else oversampling

#Edit this to control which models are ran.
modelClasses = [MajorityClassClassifier(), LightGBM(), CatBoost(), XgBoost()]

for model in modelClasses:
    model.metric = metric

models, spaces, objectives, labels, evaluations = populateBatchTrainFields(modelClasses)

points, values = importDataset(dataset,  trinary)

batchTrain = BatchTrain(models, spaces, objectives, labels, points, values, evaluations, oversampling, state, metric)
batchTrain.train()
batchTrain.printResults()