import os
import sys
import pandas as pd
from dataclasses import dataclass
from pathlib import Path

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_ensemble import ModelEnsemble

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train_data.pkl')
    test_data_path: str = os.path.join('artifacts', 'test_data.pkl')
    raw_data_path: str = os.path.join('artifacts', 'rice_plant_diseases')
    curated_split_path: str = os.path.join('artifacts', 'all.py')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def load_curated_split(self):
        """
        Load the curated train/test split from all.py file
        Had to curate rather than train test split on the whole data set as there was duplicate images
        and chance of duplicate images spilling in train and test to get high accuracy
        """
        try:
            logging.info("Loading curated split from all.py")
            
            # Execute the all.py file to get the variables
            split_vars = {}
            with open(self.ingestion_config.curated_split_path, 'r') as f:
                exec(f.read(), split_vars)
            
            train_images = split_vars.get('train_images', [])
            train_labels = split_vars.get('train_labels', [])
            test_images = split_vars.get('test_images', [])
            test_labels = split_vars.get('test_labels', [])
            
            logging.info(f"Curated split loaded - Train: {len(train_images)}, Test: {len(test_images)}")
            
            return train_images, train_labels, test_images, test_labels
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def find_image_path(self, filename):
        # filename already includes the class folder (e.g., 'Leafsmut/BLAST1_001.jpg')
        dataset_path = Path(self.ingestion_config.raw_data_path)
        path = dataset_path / filename
        if path.exists():
            return str(path)
        return None

    def prepare_dataframes(self, train_images, train_labels, test_images, test_labels):
        """
        Convert split lists to DataFrames with full paths
        """
        try:
            logging.info("Preparing split DataFrames")
            
            # Create train DataFrame
            train_data = []
            for img, label in zip(train_images, train_labels):
                img_path = self.find_image_path(img)
                if img_path:
                    train_data.append({
                        'image_path': img_path,
                        'label': label,
                        'filename': img
                    })
            
            # Create test DataFrame
            test_data = []
            for img, label in zip(test_images, test_labels):
                img_path = self.find_image_path(img)
                if img_path:
                    test_data.append({
                        'image_path': img_path,
                        'label': label,
                        'filename': img
                    })
            
            train_df = pd.DataFrame(train_data)
            test_df = pd.DataFrame(test_data)
            
            logging.info(f"Curated DataFrames created - Train: {len(train_df)}, Test: {len(test_df)}")
            
            return train_df, test_df
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def save_splits(self, train_df, test_df):
        try:
            logging.info("Saving train/test splits to artifacts")
            
            os.makedirs('artifacts', exist_ok=True)
            
            save_object(file_path=self.ingestion_config.train_data_path,obj=train_df)
            save_object(file_path=self.ingestion_config.test_data_path,obj=test_df)
            
            logging.info(f"Train data saved to: {self.ingestion_config.train_data_path}")
            logging.info(f"Test data saved to: {self.ingestion_config.test_data_path}")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_ingestion(self):
        """
        Main method to initiate data ingestion process
        """
        logging.info("Entered the data ingestion method")
        
        try:
            # Use curated split from all.py
            train_images, train_labels, test_images, test_labels = self.load_curated_split()
            train_df, test_df = self.prepare_dataframes(train_images, train_labels, test_images, test_labels)
            
            train_data_path, test_data_path = self.save_splits(train_df, test_df)
            
            logging.info("Data ingestion completed successfully")
            
            return train_data_path, test_data_path
            
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    logging.info("Starting Rice Plant Disease Detection Pipeline")
    
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_loader, test_loader, class_to_idx = data_transformation.initiate_data_transformation(
        train_data, test_data, batch_size=32
    )
    
    trainer = ModelTrainer()
    results = trainer.initiate_model_training(
        train_df=pd.read_pickle(train_data),
        test_df=pd.read_pickle(test_data),
        class_to_idx=class_to_idx,
        epochs=10,
        learning_rate=1e-4
    )
    
    # Model Ensemble Evaluation
    logging.info("=== Model Ensemble Phase ===")
    ensemble = ModelEnsemble()
    ensemble_results = ensemble.initiate_ensemble_evaluation(
        test_loader=test_loader,
        weighting_method='accuracy'
    )
    
    # Final Pipeline Summary
    logging.info("=== Pipeline Summary ===")
    if 'best_model' in results:
        best = results['best_model']
        logging.info(f"Best Individual Model: {best['name'].upper()} - {best['accuracy']:.2f}% accuracy")
    
    if ensemble_results:
        logging.info(f"Best Ensemble Method: {ensemble_results['best_method']} - {ensemble_results['best_accuracy']:.4f} accuracy")
        
        # Calculate improvement
        best_individual = max(ensemble_results['individual_accuracies'].values())
        improvement = (ensemble_results['best_accuracy'] * 100) - best_individual
        logging.info(f"Ensemble improvement: +{improvement:.2f} percentage points")
    
    logging.info("Full pipeline completed successfully!")


