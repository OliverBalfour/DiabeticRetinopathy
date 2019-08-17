
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD
from models.model_utils import load_xy
from models.stacked.ann import train

X, Y = load_xy('densenet121')
train(X,Y,'densenet121')
