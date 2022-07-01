import pandas as pd
import numpy as np
import pickle
import os
import mlflow

# ---------------------------------------
LOG_DATA_PKL = "data.pkl"
LOG_MODEL_PKL = "model.pkl"
LOG_METRIC_PKL = "metrics.pkl"

YEAR_DIC = {1990: 1, 1991: 2, 1992: 3, 1993: 4, 1994: 5,
            1995: 6, 1996: 7, 1997: 8, 1998: 9, 1999: 10,
            2000: 11, 2001: 12, 2002: 13, 2003: 14, 2004: 15,
            2005: 16, 2006: 17, 2007: 18, 2008: 19, 2009: 20,
            2010: 21, 2011: 22, 2012: 23, 2013: 24, 2014: 25,
            2015: 26, 2016: 27, 2017: 28}

TRANSMISSION_DICT = {'manual': 1, 'automatic': 2,
                     'automated_manual': 3, 'direct_drive': 4}
# ---------------------------------------


class PricePrediction:
    """Production class for predicting the salary of car based on features"""

    # ===========================================
    # ********* Initialization Function *********
    # ===========================================

    # Constructor
    def __init__(self, mlflow_uri, run_id):
        self.tracking_uri = mlflow_uri
        self.run_id = run_id

        mlflow_objs = self.load_mlflow_objs()
        self.model = mlflow_objs[0]
        self.features_names = mlflow_objs[1]

    def load_mlflow_objs(self):
        """ Load objects from mlflow run """
        # Initialize client and experiment
        mlflow.set_tracking_uri(self.tracking_uri)
        run = mlflow.get_run(self.run_id)
        artifacts_path = run.info.artifact_uri

        # Load data pickle
        data_path = os.path.join(artifacts_path, LOG_DATA_PKL)
        with open(data_path, 'rb') as data_file:
            data_pkl = pickle.load(data_file)

        # Load model pickle
        model_path = os.path.join(artifacts_path, LOG_MODEL_PKL)
        with open(model_path, 'rb') as model_file:
            model_pkl = pickle.load(model_file)

        # Return data and model objects
        return model_pkl["model_object"], data_pkl['feature_names']

    # ===========================================
    # ***************** Getters *****************
    # ===========================================

    def get_features(self):
        return self.features_names

    # ===========================================
    # *********** Prediction Function ***********
    # ===========================================

    def create_features_array(self, sample):
        """ Create the features array from a list of car features"""
        sample_df = pd.DataFrame(self.features_names, columns=['features'])
        sample_df['value'] = sample_df['features'].isin(sample)

        # Changing the values in sample dataframe
        sample_df.iloc[[0], [1]] = YEAR_DIC[sample[1]]
        sample_df.iloc[[1], [1]] = sample[2]
        sample_df.iloc[[2], [1]] = sample[3]
        sample_df.iloc[[3], [1]] = TRANSMISSION_DICT[sample[4]]
        sample_df.iloc[[4], [1]] = (sample[5] + sample[6]) / 2

        sample_df.replace({False: 0, True: 1}, inplace=True)
        sample_df.set_index("features", inplace=True)
        sample_df = sample_df.transpose()

        return sample_df

    def predict_car_price(self, car_features):
        """ Return the price of the car based on features"""

        # Create the features array
        features_array = self.create_features_array(car_features)

        # Predict and format
        predictions = self.model.predict(features_array.values)
        car_price = round(np.expm1(predictions)[0], 3)

        return car_price
