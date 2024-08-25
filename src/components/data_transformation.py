import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig :
    """ 
    Configuration class to store the path for saving the preprocessor object
    """
    preprocesser_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

    
class DataTransformation :
    """
    Data transformation class to handle data preprocessing
    """
    def __init__(self) :
        # Initialize the config for storing the preprocessor path
        self.data_transformation_config = DataTransformationConfig()
        
        
    def get_transformer_object(self):
        """
        This function is respondsible for data transformation pipeline
        """
        try :
            # define numerical column
            numerical_columns = ["writing_score", "reading_score"]
            # define categorical column
            categorical_columns = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]
        
            num_pipeline = Pipeline(
                steps = [
                    # Handling missing value data with mean
                    ("imputer", SimpleImputer(strategy = "median")),
                    # Standard scaling
                    ("scalar", StandardScaler())
                ])
            
            cat_pipeline = Pipeline(
                steps = [
                    # Handling missing value data with mean
                    ("imputer", SimpleImputer(strategy = "most_frequent")),
                    # One hot encoding
                    ("one_hot_encoder", OneHotEncoder(sparse_output=False)),
                    # Standard scaling
                    ("scalar", StandardScaler(with_mean=False))              
                ])
            
            logging.info(f"Numerical columns encoding completed : {numerical_columns}")
            logging.info(f"Categorical columns encoding completed : {categorical_columns}")
            
            # Combine both pipelines
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
        
    def initiate_data_transformation(self, train_path, test_path) :
        """
        This function reads train and test data, applies the transformation pipeline,
        and saves the preprocessor object.
        """
        try :
            # Read train and test dataset
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train test data completed")
            
            # Get the transformation pipeline
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_transformer_object()
            
            # define target column
            target_column_name = "math_score"
            # define numerical column
            numerical_columns = ["writing_score", "reading_score"]
            
            input_feature_train_df = train_df.drop(columns = [target_column_name], axis = 1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns = [target_column_name], axis = 1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")
            
            # Transform the train anstest data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.fit_transform(input_feature_test_df)
            
            # Combine transformed input features and target features
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            # Save the preprocessor object
            logging.info(f"Saved preprocessing object.")
            save_object(
                file_path = self.data_transformation_config.preprocesser_obj_file_path,
                obj = preprocessing_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocesser_obj_file_path
            )
            
        except Exception as e :
            raise CustomException(e, sys)

