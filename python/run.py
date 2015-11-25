import numpy as np
import os
import sys

from math import sqrt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from utils import featurizer, getTargets, outputPredictions

missingFeaturesString = "NA"

trainFeatPath = "/Users/mihaileric/Documents/Stats202/Kaggle/data/training_features.csv" #
trainTargetPath = "/Users/mihaileric/Documents/Stats202/Kaggle/data/training_target.csv"
valFeatPath = "/Users/mihaileric/Documents/Stats202/Kaggle/data/validation_features.csv"
valTargetPath = "/Users/mihaileric/Documents/Stats202/Kaggle/data/validation_target.csv"
leaderboardFeatPath = "/Users/mihaileric/Documents/Stats202/Kaggle/data/leaderboard_features.csv"


def buildModel():
    """Defines a series of transforms to be applied to model."""
    trainFeatures = featurizer(trainFeatPath)
    trainLabels = getTargets(trainTargetPath)

    valFeatures = featurizer(valFeatPath)
    valLabels = getTargets(valTargetPath)


    imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
    clfPipe = Pipeline([('dictVector', DictVectorizer(sparse=False)), ('imputer', imputer),
                        ('featureReduction', PCA(n_components=4)),
                      ('clf', RandomForestClassifier(criterion='entropy'))])


    clfPipe.fit(trainFeatures, trainLabels)

    modelPredictions = clfPipe.predict(valFeatures)

    error = sqrt(mean_squared_error(modelPredictions, valLabels))

    print "Computed error: %f" %(error)


if __name__ == "__main__":
    buildModel()