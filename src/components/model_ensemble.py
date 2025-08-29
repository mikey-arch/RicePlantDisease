import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object, save_classification_report, plot_training_history

@dataclass
class EnsembleConfig:
    models_path: str = os.path.join('artifacts', 'models')
    ensemble_reports_path: str = os.path.join('artifacts', 'ensemble_reports')
    ensemble_model_path: str = os.path.join('artifacts', 'ensemble_model.pth')

class ModelEnsemble:
    def __init__(self):
        self.config = EnsembleConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.model_weights = {}
        
        # Create ensemble reports directory
        os.makedirs(self.config.ensemble_reports_path, exist_ok=True)
        
        logging.info(f"Ensemble system initialized on device: {self.device}")
    
    def load_trained_models(self, model_types=None):
        """
        Load all trained models from artifacts
        """
        try:
            if model_types is None:
                model_types = ['efficientnet', 'convnext', 'vit', 'swin', 'dinov2']
            
            logging.info(f"Loading trained models: {model_types}")
            
            for model_type in model_types:
                model_path = os.path.join(self.config.models_path, f"{model_type}_final.pth")
                
                if not os.path.exists(model_path):
                    logging.warning(f"Model file not found: {model_path}")
                    continue
                
                # Load model data
                model_data = torch.load(model_path, map_location=self.device)
                
                # Import the model class dynamically
                from src.components.model_trainer import ModelTrainer
                trainer = ModelTrainer()
                
                # Create model instance
                model = trainer.get_model(
                    model_type, 
                    num_classes=model_data['num_classes']
                )
                
                # Load trained weights
                model.load_state_dict(model_data['model_state_dict'])
                model.to(self.device)
                model.eval()
                
                self.models[model_type] = {
                    'model': model,
                    'accuracy': model_data.get('best_val_acc', 0),
                    'history': model_data.get('history', {}),
                    'model_info': model_data.get('model_info', {})
                }
                
                logging.info(f"Loaded {model_type}: {model_data.get('best_val_acc', 0):.2f}% accuracy")
            
            logging.info(f"Successfully loaded {len(self.models)} models")
            return self.models
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def calculate_ensemble_weights(self, method='accuracy'):
        """
        Calculate ensemble weights based on individual model performance
        """
        try:
            if method == 'accuracy':
                # Weight by accuracy
                accuracies = [info['accuracy'] for info in self.models.values()]
                total_accuracy = sum(accuracies)
                
                if total_accuracy == 0:
                    # Equal weights if no accuracy info
                    weights = [1.0 / len(self.models)] * len(self.models)
                else:
                    weights = [acc / total_accuracy for acc in accuracies]
                
            elif method == 'equal':
                # Equal weights
                weights = [1.0 / len(self.models)] * len(self.models)
                
            elif method == 'softmax_accuracy':
                # Softmax of accuracies for smoother weighting
                accuracies = np.array([info['accuracy'] for info in self.models.values()])
                weights = F.softmax(torch.tensor(accuracies / 10.0), dim=0).numpy()
                
            else:
                raise ValueError(f"Unknown weighting method: {method}")
            
            # Store weights
            model_names = list(self.models.keys())
            self.model_weights = dict(zip(model_names, weights))
            
            logging.info(f"Ensemble weights ({method}):")
            for name, weight in self.model_weights.items():
                logging.info(f"  {name}: {weight:.4f}")
                
            return self.model_weights
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def predict_ensemble(self, data_loader, method='weighted_average'):
        """
        Make ensemble predictions on a dataset
        """
        try:
            logging.info(f"Making ensemble predictions using {method}")
            
            all_targets = []
            
            # Collect predictions from all models
            model_predictions = {name: [] for name in self.models.keys()}
            model_probabilities = {name: [] for name in self.models.keys()}
            
            with torch.no_grad():
                for data, targets in tqdm(data_loader, desc="Ensemble Prediction"):
                    data = data.to(self.device)
                    targets = targets.to(self.device)
                    
                    # Get predictions from each model
                    for name, model_info in self.models.items():
                        model = model_info['model']
                        
                        # Handle different input sizes for different models
                        if name == 'dinov2' and data.shape[-1] != 518:
                            # Resize for DINOv2 if needed
                            model_data = F.interpolate(data, size=518, mode='bilinear', align_corners=False)
                        else:
                            model_data = data
                        
                        outputs = model(model_data)
                        probabilities = F.softmax(outputs, dim=1)
                        predictions = outputs.argmax(dim=1)
                        
                        model_predictions[name].extend(predictions.cpu().numpy())
                        model_probabilities[name].extend(probabilities.cpu().numpy())
                    
                    all_targets.extend(targets.cpu().numpy())
            
            # Convert to numpy arrays
            for name in self.models.keys():
                model_predictions[name] = np.array(model_predictions[name])
                model_probabilities[name] = np.array(model_probabilities[name])
            
            all_targets = np.array(all_targets)
            
            # Generate ensemble predictions
            if method == 'weighted_average':
                # Weighted average of probabilities
                ensemble_probs = np.zeros_like(model_probabilities[list(self.models.keys())[0]])
                
                for name, weight in self.model_weights.items():
                    if name in model_probabilities:
                        ensemble_probs += weight * model_probabilities[name]
                
                ensemble_predictions = np.argmax(ensemble_probs, axis=1)
                
            elif method == 'majority_vote':
                # Hard voting - majority wins
                prediction_matrix = np.array([model_predictions[name] for name in self.models.keys()])
                ensemble_predictions = []
                
                for i in range(len(all_targets)):
                    votes = prediction_matrix[:, i]
                    # Get majority vote
                    unique, counts = np.unique(votes, return_counts=True)
                    majority = unique[np.argmax(counts)]
                    ensemble_predictions.append(majority)
                
                ensemble_predictions = np.array(ensemble_predictions)
                
            elif method == 'weighted_vote':
                # Weighted hard voting
                ensemble_predictions = []
                
                for i in range(len(all_targets)):
                    weighted_votes = {}
                    for name, weight in self.model_weights.items():
                        if name in model_predictions:
                            pred = model_predictions[name][i]
                            weighted_votes[pred] = weighted_votes.get(pred, 0) + weight
                    
                    # Select prediction with highest weighted vote
                    best_pred = max(weighted_votes.keys(), key=lambda k: weighted_votes[k])
                    ensemble_predictions.append(best_pred)
                
                ensemble_predictions = np.array(ensemble_predictions)
            
            else:
                raise ValueError(f"Unknown ensemble method: {method}")
            
            # Calculate ensemble accuracy
            ensemble_accuracy = accuracy_score(all_targets, ensemble_predictions)
            logging.info(f"Ensemble accuracy ({method}): {ensemble_accuracy:.4f}")
            
            return {
                'ensemble_predictions': ensemble_predictions,
                'targets': all_targets,
                'individual_predictions': model_predictions,
                'individual_probabilities': model_probabilities,
                'ensemble_accuracy': ensemble_accuracy,
                'method': method
            }
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def evaluate_ensemble_methods(self, test_loader):
        """
        Evaluate different ensemble methods and find the best one
        """
        try:
            logging.info("Evaluating different ensemble methods")
            
            ensemble_methods = ['weighted_average', 'majority_vote', 'weighted_vote']
            results = {}
            
            for method in ensemble_methods:
                logging.info(f"Testing ensemble method: {method}")
                
                result = self.predict_ensemble(test_loader, method=method)
                accuracy = result['ensemble_accuracy']
                
                results[method] = {
                    'accuracy': accuracy,
                    'predictions': result['ensemble_predictions'],
                    'targets': result['targets']
                }
                
                logging.info(f"{method}: {accuracy:.4f}")
            
            # Find best method
            best_method = max(results.keys(), key=lambda k: results[k]['accuracy'])
            best_accuracy = results[best_method]['accuracy']
            
            logging.info(f"Best ensemble method: {best_method} with {best_accuracy:.4f} accuracy")
            
            return results, best_method
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def save_ensemble_results(self, results, method_name="ensemble"):
        """
        Save ensemble results and generate reports
        """
        try:
            predictions = results['predictions']
            targets = results['targets']
            
            # Save classification report
            save_classification_report(
                method_name, targets, predictions, 
                self.config.ensemble_reports_path
            )
            
            # Create ensemble summary
            summary_path = os.path.join(
                self.config.ensemble_reports_path, 
                f"{method_name}_summary.txt"
            )
            
            with open(summary_path, 'w') as f:
                f.write(f"Ensemble Results - {method_name}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Final Accuracy: {results['accuracy']:.4f}\n\n")
                
                f.write("Individual Model Performance:\n")
                for name, info in self.models.items():
                    f.write(f"  {name}: {info['accuracy']:.2f}%\n")
                
                f.write("\nEnsemble Weights:\n")
                for name, weight in self.model_weights.items():
                    f.write(f"  {name}: {weight:.4f}\n")
                
                f.write(f"\nImprovement over best individual: ")
                best_individual = max(info['accuracy'] for info in self.models.values())
                improvement = (results['accuracy'] * 100) - best_individual
                f.write(f"{improvement:.2f} percentage points\n")
            
            logging.info(f"Ensemble results saved to {self.config.ensemble_reports_path}")
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_ensemble_evaluation(self, test_loader, weighting_method='accuracy'):
        """
        Main method to run complete ensemble evaluation
        """
        logging.info("Starting ensemble evaluation")
        
        try:
            # Load all trained models
            self.load_trained_models()
            
            if len(self.models) == 0:
                logging.error("No trained models found!")
                return None
            
            # Calculate ensemble weights
            self.calculate_ensemble_weights(method=weighting_method)
            
            # Evaluate different ensemble methods
            all_results, best_method = self.evaluate_ensemble_methods(test_loader)
            
            # Save results for best method
            best_result = all_results[best_method]
            self.save_ensemble_results(best_result, f"best_ensemble_{best_method}")
            
            # Also save results for all methods
            for method, result in all_results.items():
                self.save_ensemble_results(result, f"ensemble_{method}")
            
            logging.info("Ensemble evaluation completed successfully")
            
            return {
                'best_method': best_method,
                'best_accuracy': best_result['accuracy'],
                'all_results': all_results,
                'individual_accuracies': {name: info['accuracy'] for name, info in self.models.items()}
            }
            
        except Exception as e:
            raise CustomException(e, sys)