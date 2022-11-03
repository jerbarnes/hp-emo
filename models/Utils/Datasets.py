import os
import numpy as np


class Emotion_Dataset(object):
    def __init__(self,
                 DIR,
                 emotion="anger"):
        self.open_data(DIR, emotion)

    def open_data(self, DIR, emotion):
        data = {}
        for split in ["train", "dev", "test"]:
            data[split] = {}
            textfile = os.path.join(DIR, split, "{0}.txt".format(emotion))
            labelfile = os.path.join(DIR, split, "{0}_labels.txt".format(emotion))
            data[split]["text"] = [" ".join(l.strip().split()) for l in open(textfile)]
            data[split]["labels"] = [float(l.strip()) for l in open(labelfile)]
        self.data = data

class Emotion_Testset(object):
    def __init__(self,
                 DIR,
                 emotion="anger"):
        self.open_data(DIR, emotion)

    def open_data(self, DIR, emotion):
        data = {}
        for split in ["test"]:
            data[split] = {}
            textfile = os.path.join(DIR, split, "{0}.txt".format(emotion))
            labelfile = os.path.join(DIR, split, "{0}_labels.txt".format(emotion))
            data[split]["text"] = [" ".join(l.strip().split()) for l in open(textfile)]
            data[split]["labels"] = [float(l.strip()) for l in open(labelfile)]
        self.data = data
