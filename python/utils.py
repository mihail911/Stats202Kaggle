""" Utilities for reading in data and outputting
"""

import csv
import numpy as np

def featurizer(filePath):
    """Returns features for given file."""
    features = []
    with open(filePath, "r") as f:
        reader = csv.DictReader(f)
        for example in reader:
            examplesDict = dict(example.items())
            cleanDict = {}
            for featureName, featureValue in examplesDict.iteritems():
                if featureValue == "NA":
                    cleanDict[featureName] = np.nan
                else:
                    cleanDict[featureName] = float(featureValue)
            features.append(cleanDict)

    return features


def getTargets(filePath):
    """
    Returns list of targets for all examples in given data path.
    :param filePath:
    """
    targets = []
    with open(filePath, "r") as f:
        reader = csv.DictReader(f)
        for example in reader:
            targets.append(float(example["ALSFRS_slope"]))

    return targets


def outputPredictions(predictList, fileName):
    """
    Output predictions in necessary format given a list of (subject.id, slope) pairs
    """
    rowNames = ["subject.id", "ALSFRS_slope"]
    with open(fileName, "w") as f:
        writer = csv.DictWriter(f, fieldnames=rowNames)
        writer.writeheader()
        for example in predictList:
            writer.writerow({"subject.id": example[0], "ALSFRS_slope": example[1]})


if __name__ == "__main__":
    #featurizer(trainFeatPath)
    testOutput = [(23, 0.4), (25, 0.2), (67, 0.1)]
    outputPredictions(testOutput, "testFile.csv")

