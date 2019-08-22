
from models.model_utils import load_xy
from models.stacked.linreg import train

# import all 10 algorithms, train on both vectors, and save acc to txt/csv

model = ['mobilenet', 'densenet121'][1]

X, Y = load_xy(model)
train(X, Y, model)
