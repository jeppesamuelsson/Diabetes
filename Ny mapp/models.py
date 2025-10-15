from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.compose import ColumnTransformer
from config import FEATURE_NAMES

def make_v01():
    # Iteration 1: StandardScaler + LinearRegression
    pre = ColumnTransformer([("scale", StandardScaler(), FEATURE_NAMES)], remainder="drop")
    reg = LinearRegression()
    return Pipeline([("pre", pre), ("reg", reg)])

def make_v02():
    # Iteration 2: StandardScaler + RidgeCV
    pre = ColumnTransformer([("scale", StandardScaler(), FEATURE_NAMES)], remainder="drop")
    reg = RidgeCV(alphas=[0.1, 1.0, 10.0])
    return Pipeline([("pre", pre), ("reg", reg)])
