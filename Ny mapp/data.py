from sklearn.datasets import load_diabetes

def load_data(as_frame: bool = True):
    """
    Loads the open scikit-learn Diabetes dataset exactly as the assignment specifies.
    y acts as a progression index: higher = worse.
    """
    Xy = load_diabetes(as_frame=as_frame)
    X = Xy.frame.drop(columns=["target"])
    y = Xy.frame["target"]  # acts as a "progression index" (higher = worse)
    return X, y
