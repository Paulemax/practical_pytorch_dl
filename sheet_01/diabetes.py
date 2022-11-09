from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

def prepare_data():
    tmp=load_diabetes()
    X=tmp["data"]
    y=tmp["target"]
    return train_test_split(X, y, test_size=0.25, random_state=42) #X_train, X_test, y_train, y_test
