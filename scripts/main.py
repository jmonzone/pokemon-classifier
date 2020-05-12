# It is required to run the featureextraction.py code first!
# This is so that a text file of feature vectors can be created
# in order for this file to run.

from sklearn.svm import LinearSVC
import numpy as np
from modules.imagefeatures import *
import cv2
from imutils import paths
import os
from skimage import io
import random

training_ids = random.sample(range(1, 4046), 809)
data = []
labels = []

f = open("data/pokemonfeatures.txt");

for line in f:
    values = line.split()
    linenum = values[0]
    id = values[1]

    if int(linenum) not in training_ids:
        continue;

    name = values[2]
    type1 = values[3]
    type2 = values[4]
    features = np.asarray(values[5:], "float32")
    data.append(features)
    labels.append(type1)

f.close()

model = LinearSVC(class_weight="balanced", max_iter=2000)
model.fit(data, labels)

correct = 0
type_count = {}
type_correct = {}

f = open("data/pokemonfeatures.txt");

for line in f:
    values = line.split()
    linenum = values[0]
    id = values[1]

    if int(linenum) in training_ids:
        continue;

    name = values[2]
    type1 = values[3]

    if type1 in type_count:
        type_count[type1] += 1
    else:
        type_count[type1] = 1
        type_correct[type1] = 0

    type2 = values[4]
    features = np.asarray(values[5:], "float32")
    features = np.asarray(features)
    prediction = model.predict(features.reshape(1,-1))

    print(name, prediction)
    if prediction == type1:
        correct += 1
        type_correct[type1] += 1

f.close()


accuracy = (correct / 3236) * 100
print(accuracy)
print(type_correct)
print(type_count)
