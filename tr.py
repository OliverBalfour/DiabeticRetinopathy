
from models.model_utils import load_xy
from models.stacked.linreg import train

model = ['mobilenet', 'densenet121'][1]

X, Y = load_xy(model)
train(X, Y, model)
