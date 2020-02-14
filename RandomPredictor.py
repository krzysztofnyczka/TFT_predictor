import pandas as pd
import numpy as np

class RandomPredictor:
    def __init__(self, df):
        self.df = df
        class_probs = df.target.value_counts()
        l = len(df)
        self.classes = [k for k in class_probs.keys()]
        self.probs = [v/l for k, v in class_probs.items()]
        
    def classify(self, row):
        return np.random.choice(self.classes, p=self.probs)