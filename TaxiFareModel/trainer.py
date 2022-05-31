from bleach import clean
from sklearn.feature_selection import SelectFpr
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from TaxiFareModel.data import clean_data, get_data
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.encoders import *

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
                # create preprocessing pipeline
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())])

        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))])

        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])], remainder="drop")

        self.pipe = Pipeline([
             ('preproc', preproc_pipe),
             ('linear_model', LinearRegression())])
        return self.pipe

    def run(self,X_train,y_train):
        """set and train the pipeline"""
        self.pipeline = self.pipe.fit(X_train,y_train)
        return self.pipeline


    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        print(rmse)
        return rmse



if __name__ == "__main__":
    # get data
    df = get_data()
    # clean data
    df = clean_data(df)
    # set X and y
    y = df["fare_amount"]
    X = df.drop(columns="fare_amount")
    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    # train
    pipe = Trainer(X,y)
    pipe.set_pipeline()
    pipe.run(X_train, y_train)
    # evaluate
    pipe.evaluate(X_val, y_val)
