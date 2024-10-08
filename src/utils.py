import os
import sys
import dill
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from src.exception import CustomException


def save_object(file_path, obj):
    try :
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open (file_path, "wb") as file_obj :
            dill.dump(obj, file_obj)
            
    except Exception as e :
        raise CustomException(e, sys)
    
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try :
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]
            
            # gridsearch_cv = GridSearchCV(model, para, cv=cv, n_jobs=n, verbose=verbose, refit=refit)
            gridsearch_cv = GridSearchCV(model, para, cv=3)
            gridsearch_cv.fit(X_train, y_train)
            
            # Train model
            model.set_params(**gridsearch_cv.best_params_)
            model.fit(X_train, y_train)

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Evaluation by getting r2 score
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            # Insert to report dictionary
            report[list(models.keys())[i]] = test_model_score
            
        return report
        
    except Exception as e :
        raise CustomException(e, sys)
    
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)