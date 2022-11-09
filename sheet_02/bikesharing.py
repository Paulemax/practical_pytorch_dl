#download data from https://archive.ics.uci.edu/ml/machine-learning-databases/00275/

import io
import pandas as pd
from datetime import datetime
from sklearn.model_selection  import train_test_split

def prepare_bike_sharing(path="./Bike-Sharing-Dataset/day.csv", ):
    df_data = pd.read_csv(path,sep=",")
    df_data["dteday"]=df_data["dteday"].apply(lambda x:(datetime.strptime(x,"%Y-%m-%d").timetuple().tm_yday))
    df_data.drop(["instant",  "registered", "casual", "atemp"],axis=1,inplace=True)
    df_train, df_test = train_test_split(df_data,test_size=0.1, random_state=44)
    return df_train, df_test