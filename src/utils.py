import os
import sys
import pickle
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    """
    Save a Python object to a pickle file
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
        logging.info(f"Object saved to {file_path}")
        
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Load a Python object from a pickle file
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
            
        logging.info(f"Object loaded from {file_path}")
        return obj
        
    except Exception as e:
        raise CustomException(e, sys)

def save_model_with_metadata(model, model_type, num_classes, best_val_acc, history, model_info, save_path):
    """
    Save model with comprehensive metadata
    """
    try:
        model_data = {
            'model_state_dict': model.state_dict(),
            'model_type': model_type,
            'num_classes': num_classes,
            'best_val_acc': best_val_acc,
            'history': history,
            'model_info': model_info
        }
        
        torch.save(model_data, save_path)
        logging.info(f"Model {model_type} saved to {save_path}")
        
    except Exception as e:
        raise CustomException(e, sys)

def save_classification_report(model_type, y_true, y_pred, reports_path):
    """
    Save classification report and confusion matrix
    """
    try:
        class_names = ['Bacterialblight', 'Brownspot', 'Leafsmut']
        report = classification_report(y_true, y_pred, target_names=class_names)
        
        report_path = os.path.join(reports_path, f"{model_type}_classification_report.txt")
        
        with open(report_path, 'w') as f:
            f.write(f"Classification Report for {model_type}\n")
            f.write("=" * 50 + "\n")
            f.write(report)
            f.write("\n\nConfusion Matrix:\n")
            f.write(str(confusion_matrix(y_true, y_pred)))
        
        logging.info(f"Classification report saved for {model_type}")
        
    except Exception as e:
        raise CustomException(e, sys)

def plot_training_history(model_type, history, reports_path):
    """
    Plot and save training history curves
    """
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(history['train_loss'], label='Train Loss', marker='o')
        ax1.plot(history['val_loss'], label='Val Loss', marker='s')
        ax1.set_title(f'{model_type} - Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(history['train_acc'], label='Train Acc', marker='o')
        ax2.plot(history['val_acc'], label='Val Acc', marker='s')
        ax2.set_title(f'{model_type} - Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        plot_path = os.path.join(reports_path, f"{model_type}_training_history.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Training plots saved for {model_type}")
        
    except Exception as e:
        logging.error(f"Error saving plots: {e}")
        raise CustomException(e, sys)