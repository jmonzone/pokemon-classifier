import numpy as np
from modules.imagefeatures import *
from imutils import paths
import os
from skimage import io

def getFeatures(image):
    features = []

    for reso in gaussPyramid(image):
        box_counts = getBoxCounts(reso);
        features.extend(box_counts)

        hists = colorHist(reso)
        hists = hists.flatten()
        features.extend(hists)

    (lbp, hists, _) = lbp3x3(image,3)
    features.extend(hists)

    (lbp, hists, _) = lbp9x9(image,3)
    features.extend(hists)

    return features


all_pokemon = {}

f = open('data/pokemondata.txt');

for line in f:
    values = line.split(',')
    index = str(values[0])

    if index in all_pokemon : continue

    type2 = values[3] if values[3] else "-"

    pokemon = {
        'name' : values[1].replace(" ", "") ,
        'type1' : values[2],
        'type2' : type2
    }

    all_pokemon[index] = pokemon

f.close()


file = open("data/pokemonfeatures.txt","w")

id = 0
for imagePath in paths.list_images("../images/centered-sprites"):
    id += 1
    pokedex_number = str(imagePath.split(os.path.sep)[-1])[:-4]
    pokemon = all_pokemon[pokedex_number]
    print("Analyzing " + pokemon['name'])
    image = cv2.imread(imagePath)
    features = [id, pokedex_number, pokemon['name'], pokemon['type1'], pokemon['type2']]
    features += getFeatures(image)
    text = " ".join(str(x) for x in features)
    file.write(text + "\n")

file.close()
