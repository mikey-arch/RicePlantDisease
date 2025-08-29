import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class RiceDiseaseDataset(Dataset):
    """Custom PyTorch Dataset for Rice Disease images"""
    
    def __init__(self, dataframe, transform=None, class_to_idx=None):
        self.dataframe = dataframe
        self.transform = transform
        
        if class_to_idx is None:
            self.classes = sorted(dataframe['label'].unique())
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        else:
            self.class_to_idx = class_to_idx
            self.classes = list(class_to_idx.keys())
        
        logging.info(f"Dataset created with {len(self.dataframe)} samples")
        logging.info(f"Classes: {self.classes}")
        logging.info(f"Class mapping: {self.class_to_idx}")
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = row['image_path']
        label = row['label']
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logging.error(f"Error loading image {image_path}: {e}")
            raise CustomException(e, sys)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Convert label to index e.g Bacterialblight:0, Brownspot :1, Leafsmut: 2
        label_idx = self.class_to_idx[label]
        
        return image, label_idx

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
        
    def get_transforms(self, image_size=224):
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return train_transform, val_transform
    
    def create_datasets(self, train_df, test_df, image_size=224, model_type='standard'):
        try:
            # Adjust image size for specific models
            if model_type == 'dinov2':
                image_size = 518
                logging.info(f"Using DINOv2 image size: {image_size}")
            else:
                logging.info(f"Using standard image size: {image_size}")
            
            # Get transforms
            train_transform, val_transform = self.get_transforms(image_size)
            
            # Create class mapping from training data
            classes = sorted(train_df['label'].unique())
            class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
            
            # Create datasets
            train_dataset = RiceDiseaseDataset(
                dataframe=train_df,
                transform=train_transform,
                class_to_idx=class_to_idx
            )
            
            test_dataset = RiceDiseaseDataset(
                dataframe=test_df,
                transform=val_transform,
                class_to_idx=class_to_idx
            )
            
            logging.info(f"Train dataset: {len(train_dataset)} samples")
            logging.info(f"Test dataset: {len(test_dataset)} samples")
            
            # Save preprocessor data
            preprocessor_data = {
                'class_to_idx': class_to_idx,
                'classes': classes,
                'image_size': image_size
            }
            
            save_object(
                file_path=self.transformation_config.preprocessor_path,
                obj=preprocessor_data
            )
            
            return train_dataset, test_dataset, class_to_idx
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def create_dataloaders(self, train_dataset, test_dataset, batch_size=32, num_workers=4):
        try:
            logging.info(f"Creating dataloaders with batch_size={batch_size}")
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True if torch.cuda.is_available() else False,
                drop_last=True
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True if torch.cuda.is_available() else False,
                drop_last=False
            )
            
            logging.info(f"Train loader: {len(train_loader)} batches")
            logging.info(f"Test loader: {len(test_loader)} batches")
            
            return train_loader, test_loader
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_data_path, test_data_path, image_size=224, batch_size=32):
        logging.info("Entered the data transformation method")
        
        try:
            train_df = pd.read_pickle(train_data_path)
            test_df = pd.read_pickle(test_data_path)
            
            logging.info(f"Loaded train data: {len(train_df)} samples")
            logging.info(f"Loaded test data: {len(test_df)} samples")
            
            train_dataset, test_dataset, class_to_idx = self.create_datasets(train_df, test_df, image_size)
            train_loader, test_loader = self.create_dataloaders(train_dataset, test_dataset, batch_size)
            
            logging.info("Data transformation completed successfully")
            
            return train_loader, test_loader, class_to_idx
            
        except Exception as e:
            raise CustomException(e, sys)
