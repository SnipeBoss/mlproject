import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestioConfig :
    """
    Configuration class to store paths for saving data.
    """
    train_data_path : str = os.path.join('artifacts', "train.csv")
    test_data_path : str = os.path.join('artifacts', "test.csv")
    raw_data_path : str = os.path.join('artifacts', "data.csv")


class DataIngestion:
    """
    Data Ingestion class to handle reading, splitting, amd saving data.
    """
    def __init__(self):
        # Initialize the config for storing paths
        self.ingestion_config = DataIngestioConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        
        try :
            # Read datasource
            df = pd.read_csv("notebook\data\stud.csv")
            logging.info("read the dataset as dataframe -> data ingestion")
            
            # Create folder to store data 
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok = True)
            
            # Change raw datasource to csv
            df.to_csv(self.ingestion_config.raw_data_path, index = True, header = True)
            logging.info("Train/Test Split initiated")
            
            # Perform Train Test Split
            train_set, test_set = train_test_split(df, 
                                                   test_size = 0.2, 
                                                   random_state = 42)
            
            # Save the tain and test dataset to respective paths
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)
            
            logging.info("Ingestion of the data is completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
            
        except Exception as e:
            raise CustomException(e, sys)
        
        
if __name__ == "__main__":
    # Create an instance of DataIngestion and perform and perform data ingestion
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    # Perform data transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
    
    # Train model
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_array=train_arr, test_array=test_arr))
        

