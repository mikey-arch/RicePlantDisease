import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm
from dataclasses import dataclass
from tqdm import tqdm

from src.exception import CustomException
from src.logger import logging
from src.utils import save_model_with_metadata, save_classification_report, plot_training_history
from src.components.data_transformation import DataTransformation

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'models')
    model_reports_path: str = os.path.join('artifacts', 'model_reports')

class BaseModel(nn.Module):
    """Base class for all models"""
    def __init__(self, num_classes=3, model_name="base"):
        super(BaseModel, self).__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        
    def forward(self, x):
        raise NotImplementedError
        
    def get_model_info(self):
        return {
            'name': self.model_name,
            'num_classes': self.num_classes,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }

class EfficientNetModel(BaseModel):
    """EfficientNet with fine-tuning capability"""
    def __init__(self, num_classes=3, model_variant='efficientnet_b4', pretrained=True):
        super(EfficientNetModel, self).__init__(num_classes, f"EfficientNet_{model_variant}")
        
        self.backbone = timm.create_model(model_variant, pretrained=pretrained, num_classes=0)
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            self.feature_dim = features.shape[1]
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        self.freeze_backbone()
        
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

class ConvNeXtModel(BaseModel):
    """ConvNeXt with fine-tuning capability"""
    def __init__(self, num_classes=3, model_variant='convnext_base', pretrained=True):
        super(ConvNeXtModel, self).__init__(num_classes, f"ConvNeXt_{model_variant}")
        
        self.backbone = timm.create_model(model_variant, pretrained=pretrained, num_classes=0)
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            self.feature_dim = features.shape[1]
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        self.freeze_backbone()
        
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

class ViTModel(BaseModel):
    """Vision Transformer with fine-tuning capability"""
    def __init__(self, num_classes=3, model_variant='vit_base_patch16_224', pretrained=True):
        super(ViTModel, self).__init__(num_classes, f"ViT_{model_variant}")
        
        self.backbone = timm.create_model(model_variant, pretrained=pretrained, num_classes=0)
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            self.feature_dim = features.shape[1]
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 384),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(384, num_classes)
        )
        
        self.freeze_backbone()
        
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

class SwinModel(BaseModel):
    """Swin Transformer with fine-tuning capability"""
    def __init__(self, num_classes=3, model_variant='swin_base_patch4_window7_224', pretrained=True):
        super(SwinModel, self).__init__(num_classes, f"Swin_{model_variant}")
        
        self.backbone = timm.create_model(model_variant, pretrained=pretrained, num_classes=0)
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            self.feature_dim = features.shape[1]
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(0.4),
            nn.Linear(self.feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
        
        self.freeze_backbone()
        
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

class DINOv2Model(BaseModel):
    """DINOv2 with minimal fine-tuning"""
    def __init__(self, num_classes=3, model_variant='dinov2_base', pretrained=True):
        super(DINOv2Model, self).__init__(num_classes, f"DINOv2_{model_variant}")
        
        try:
            # Try timm first (preferred)
            self.backbone = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=pretrained, num_classes=0)
            self.feature_dim = self.backbone.num_features
        except Exception as e:
            logging.warning(f"Could not load DINOv2 from timm: {e}")
            try:
                # Fallback to torch.hub
                self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=pretrained)
                self.feature_dim = 768  # DINOv2 base feature dimension
            except Exception as e:
                logging.error(f"Could not load DINOv2 from torch.hub: {e}")
                # Final fallback - use ViT from timm as substitute
                logging.warning("Using ViT as DINOv2 substitute")
                self.backbone = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=0)
                self.feature_dim = self.backbone.num_features
        
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        self.freeze_backbone()
        
    def freeze_backbone(self):
        # Check if backbone has parameters (torch.hub models might not)
        if hasattr(self.backbone, 'parameters'):
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            logging.warning("Backbone doesn't have trainable parameters")
    
    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        return self.classifier(features)

class ModelTrainer:
    def __init__(self):
        self.trainer_config = ModelTrainerConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")
        
        # Create directories
        os.makedirs(self.trainer_config.trained_model_file_path, exist_ok=True)
        os.makedirs(self.trainer_config.model_reports_path, exist_ok=True)
        
    def get_model(self, model_type, num_classes=3):
        """Factory method to create models"""
        model_map = {
            'efficientnet': EfficientNetModel,
            'convnext': ConvNeXtModel,
            'vit': ViTModel,
            'swin': SwinModel,
            'dinov2': DINOv2Model
        }
        
        if model_type.lower() not in model_map:
            raise ValueError(f"Unknown model type: {model_type}")
            
        model_class = model_map[model_type.lower()]
        model = model_class(num_classes=num_classes)
        
        logging.info(f"Created {model_type} model")
        model_info = model.get_model_info()
        logging.info(f"Model info: {model_info}")
        
        return model
    
    def get_optimizer_and_scheduler(self, model, learning_rate=1e-4, weight_decay=1e-4):
        """Get optimizer and learning rate scheduler"""
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
        
        return optimizer, scheduler
    
    def train_epoch(self, model, train_loader, optimizer, criterion, device):
        """Train for one epoch"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, model, val_loader, criterion, device):
        """Validate for one epoch"""
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for data, target in pbar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                pbar.set_postfix({
                    'Val Loss': f'{val_loss/(pbar.n+1):.4f}',
                    'Val Acc': f'{100.*correct/total:.2f}%'
                })
        
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc, all_preds, all_targets
    
    def train_model(self, model_type, train_df, test_df, num_classes=3, 
                   epochs=10, learning_rate=1e-4):
        """Main training function"""
        try:
            logging.info(f"Starting training for {model_type}")
            
            # Create model-specific data loaders
            data_transformation = DataTransformation()
            train_dataset, test_dataset, class_to_idx = data_transformation.create_datasets(
                train_df, test_df, model_type=model_type
            )
            train_loader, val_loader = data_transformation.create_dataloaders(
                train_dataset, test_dataset, batch_size=32
            )
            
            # Create model
            model = self.get_model(model_type, num_classes)
            model = model.to(self.device)
            
            # Setup training
            criterion = nn.CrossEntropyLoss()
            optimizer, scheduler = self.get_optimizer_and_scheduler(model, learning_rate)
            
            # Training history
            history = {
                'train_loss': [], 'train_acc': [],
                'val_loss': [], 'val_acc': []
            }
            
            best_val_acc = 0.0
            best_model_state = None
            
            logging.info(f"Training {model_type} for {epochs} epochs")
            
            for epoch in range(epochs):
                logging.info(f"Epoch {epoch+1}/{epochs}")
                
                # Train
                train_loss, train_acc = self.train_epoch(
                    model, train_loader, optimizer, criterion, self.device
                )
                
                # Validate
                val_loss, val_acc, _, _ = self.validate_epoch(
                    model, val_loader, criterion, self.device
                )
                
                # Update scheduler
                scheduler.step()
                
                # Save history
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = model.state_dict().copy()
                
                logging.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                           f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Load best model for final evaluation
            if best_model_state:
                model.load_state_dict(best_model_state)
            
            # Final validation
            val_loss, val_acc, final_preds, final_targets = self.validate_epoch(
                model, val_loader, criterion, self.device
            )
            
            # Save model using utils
            model_save_path = os.path.join(
                self.trainer_config.trained_model_file_path, 
                f"{model_type}_final.pth"
            )
            
            save_model_with_metadata(
                model, model_type, num_classes, best_val_acc, 
                history, model.get_model_info(), model_save_path
            )
            
            # Save reports using utils
            save_classification_report(
                model_type, final_targets, final_preds, 
                self.trainer_config.model_reports_path
            )
            
            # Plot training history using utils
            plot_training_history(
                model_type, history, self.trainer_config.model_reports_path
            )
            
            logging.info(f"Training completed for {model_type}. Best Val Acc: {best_val_acc:.2f}%")
            
            return model_save_path, best_val_acc, history
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_model_training(self, train_df, test_df, class_to_idx, epochs=10, learning_rate=1e-4):
        """
        Main method to initiate training for all model types
        """
        logging.info("Entered the model training method")
        
        try:
            # Define models to train
            model_types = ['efficientnet', 'convnext', 'vit', 'swin', 'dinov2']
            results = {}
            num_classes = len(class_to_idx)
            
            logging.info(f"Training {len(model_types)} models with {num_classes} classes")
            logging.info(f"Class mapping: {class_to_idx}")
            
            for model_type in model_types:
                logging.info(f"Starting training for {model_type.upper()}")
                
                try:
                    model_path, best_acc, history = self.train_model(
                        model_type=model_type,
                        train_df=train_df,
                        test_df=test_df,
                        num_classes=num_classes,
                        epochs=epochs,
                        learning_rate=learning_rate
                    )
                    
                    results[model_type] = {
                        'path': model_path,
                        'accuracy': best_acc,
                        'history': history,
                        'status': 'success'
                    }
                    
                    logging.info(f"{model_type.upper()} completed successfully with accuracy: {best_acc:.2f}%")
                    
                except Exception as model_error:
                    logging.error(f"Error training {model_type}: {model_error}")
                    results[model_type] = {
                        'status': 'failed',
                        'error': str(model_error)
                    }
            
            # Generate summary
            successful_models = [(name, result['accuracy']) for name, result in results.items() 
                               if result['status'] == 'success']
            
            if successful_models:
                best_model = max(successful_models, key=lambda x: x[1])
                logging.info(f"Best performing model: {best_model[0].upper()} with {best_model[1]:.2f}% accuracy")
                
                results['best_model'] = {
                    'name': best_model[0],
                    'accuracy': best_model[1]
                }
            
            logging.info("Model training completed successfully")
            return results
            
        except Exception as e:
            raise CustomException(e, sys)