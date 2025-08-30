import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path
import torchvision.transforms as transforms

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

try:
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    LLAVA_AVAILABLE = True
except ImportError:
    LLAVA_AVAILABLE = False
    logging.warning("LLaVA not available. Install with: pip install transformers")

class PredictPipeline:
    def __init__(self, model_name="dinov2", load_llava=True):
        """
        Initialize prediction pipeline with DINOv2 (99.03% accuracy)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.model = None
        self.class_mapping = None
        self.transform = None
        self.llava_processor = None
        self.llava_model = None
        
        # Model paths
        self.model_path = os.path.join('artifacts', 'models', f'{model_name}_final.pth')
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        
        logging.info(f"Prediction pipeline initialized for {model_name} on {self.device}")
        
        # Load model and preprocessors
        self._load_model()
        self._load_preprocessors()
        
        # Load LLaVA for image analysis
        if load_llava and LLAVA_AVAILABLE:
            self._load_llava()
        
    def _load_model(self):
        """Load the trained DINOv2 model"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Load model data
            model_data = torch.load(self.model_path, map_location=self.device)
            
            # Import model class
            from src.components.model_trainer import ModelTrainer
            trainer = ModelTrainer()
            
            # Create model instance
            self.model = trainer.get_model(
                self.model_name, 
                num_classes=model_data['num_classes']
            )
            
            # Load trained weights
            self.model.load_state_dict(model_data['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Store model info for display
            self.model_accuracy = model_data.get('best_val_acc', 0)
            self.model_info = model_data.get('model_info', {})
            
            logging.info(f"Model loaded successfully: {self.model_accuracy:.2f}% accuracy")
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _load_preprocessors(self):
        """Load preprocessing information"""
        try:
            if os.path.exists(self.preprocessor_path):
                preprocessor_data = load_object(self.preprocessor_path)
                
                # Get class mapping
                self.class_mapping = preprocessor_data.get('class_to_idx', {})
                # Reverse mapping for predictions
                self.idx_to_class = {v: k for k, v in self.class_mapping.items()}
                
                logging.info(f"Loaded class mapping: {self.class_mapping}")
            else:
                # Fallback class mapping
                self.class_mapping = {'Bacterialblight': 0, 'Brownspot': 1, 'Leafsmut': 2}
                self.idx_to_class = {0: 'Bacterialblight', 1: 'Brownspot', 2: 'Leafsmut'}
                logging.warning("Using fallback class mapping")
            
            # Create transforms for DINOv2 (518x518 input size)
            self.transform = transforms.Compose([
                transforms.Resize((518, 518)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _load_llava(self):
        """Load LLaVA-NeXT model for image analysis"""
        try:
            logging.info("Loading LLaVA-NeXT model for image analysis...")
            
            # Use LLaVA-NeXT 7B model (more efficient than 13B)
            model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
            
            self.llava_processor = LlavaNextProcessor.from_pretrained(model_id)
            self.llava_model = LlavaNextForConditionalGeneration.from_pretrained(
                model_id, 
                torch_dtype=torch.float16, 
                low_cpu_mem_usage=True
            ).to(self.device)
            
            logging.info("LLaVA-NeXT model loaded successfully")
            
        except Exception as e:
            logging.error(f"Failed to load LLaVA-NeXT: {e}")
            self.llava_processor = None
            self.llava_model = None
    
    def preprocess_image(self, image_input):
        """
        Preprocess image for DINOv2 model
        """
        try:
            # Handle different input types
            if isinstance(image_input, str):
                # File path
                image = Image.open(image_input).convert('RGB')
            else:
                # PIL Image directly (for Streamlit uploads)
                image = image_input.convert('RGB')
            
            # Apply transforms
            image_tensor = self.transform(image)
            
            # Add batch dimension
            image_batch = image_tensor.unsqueeze(0).to(self.device)
            
            return image_batch, image
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def predict_single_image(self, image_input):
        """
        Make prediction on a single image
        """
        try:
            logging.info(f"Making prediction on image")
            
            # Preprocess image
            image_tensor, original_image = self.preprocess_image(image_input)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predicted_class_idx = outputs.argmax(dim=1).item()
                confidence = probabilities[0][predicted_class_idx].item()
            
            # Get class name
            predicted_class = self.idx_to_class[predicted_class_idx]
            
            # Get all class probabilities for display
            class_probs = {
                self.idx_to_class[i]: probabilities[0][i].item() 
                for i in range(len(self.idx_to_class))
            }
            
            result = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'all_probabilities': class_probs,
                'model_name': self.model_name.upper(),
                'model_accuracy': self.model_accuracy,
                'original_image': original_image
            }
            
            logging.info(f"Prediction: {predicted_class} ({confidence:.3f})")
            
            return result
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def generate_image_analysis(self, image_input):
        """
        Generate detailed analysis of the plant image using LLaVA-NeXT
        """
        if not self.llava_model or not self.llava_processor:
            return "LLaVA-NeXT model not available for detailed analysis."
        
        try:
            # Handle different input types
            if isinstance(image_input, str):
                image = Image.open(image_input).convert('RGB')
            else:
                image = image_input.convert('RGB')
            
            # Create conversation using proper chat template format
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text", 
                            "text": """Analyze this rice plant image in detail as an agricultural expert. Focus on:

1. Visible symptoms and abnormalities on the leaves
2. Patterns of discoloration, spots, or lesions  
3. Overall plant health indicators
4. Specific areas of concern
5. Severity of any visible disease symptoms

Provide a detailed botanical analysis suitable for agricultural diagnosis."""
                        }
                    ]
                }
            ]
            
            # Apply chat template
            prompt = self.llava_processor.apply_chat_template(conversation, add_generation_prompt=True)
            
            # Process inputs
            inputs = self.llava_processor(image, prompt, return_tensors="pt").to(self.device)
            
            # Generate analysis
            with torch.inference_mode():
                output = self.llava_model.generate(
                    **inputs,
                    max_new_tokens=500,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.llava_processor.tokenizer.eos_token_id
                )
            
            # Decode response and clean up
            full_response = self.llava_processor.decode(output[0], skip_special_tokens=True)
            
            # Extract just the assistant's response (after the prompt)
            if "[/INST]" in full_response:
                analysis = full_response.split("[/INST]")[-1].strip()
            else:
                # Fallback - try to find the actual response
                analysis = full_response.split("Provide a detailed botanical analysis suitable for agricultural diagnosis.")[-1].strip()
            
            logging.info("Generated image analysis successfully")
            return analysis if analysis else "Analysis completed but response was empty."
            
        except Exception as e:
            logging.error(f"Error generating image analysis: {e}")
            return f"Unable to generate detailed analysis: {str(e)}"
    
    def get_disease_info(self, disease_name):
        """
        Get detailed information about the predicted disease
        """
        disease_info = {
            'Bacterialblight': {
                'full_name': 'Bacterial Blight',
                'severity': 'High',
                'symptoms': [
                    'Water-soaked lesions on leaves',
                    'Yellow to brown leaf margins', 
                    'Wilting of infected leaves',
                    'Stunted plant growth'
                ],
                'treatment': [
                    'Apply copper-based fungicides',
                    'Remove and destroy infected plants',
                    'Improve field drainage',
                    'Use resistant rice varieties'
                ],
                'prevention': [
                    'Maintain proper plant spacing',
                    'Avoid overhead irrigation',
                    'Clean farming tools regularly',
                    'Crop rotation practices'
                ]
            },
            'Brownspot': {
                'full_name': 'Brown Spot',
                'severity': 'Medium',
                'symptoms': [
                    'Small brown spots on leaves',
                    'Circular lesions with dark borders',
                    'Spots may merge to form large patches',
                    'Premature leaf death'
                ],
                'treatment': [
                    'Apply recommended fungicides',
                    'Ensure balanced fertilization',
                    'Improve soil drainage',
                    'Remove crop residue'
                ],
                'prevention': [
                    'Use certified disease-free seeds',
                    'Maintain optimal plant nutrition',
                    'Avoid water stress',
                    'Practice field sanitation'
                ]
            },
            'Leafsmut': {
                'full_name': 'Leaf Smut',
                'severity': 'Medium',
                'symptoms': [
                    'Black powdery masses on leaves',
                    'Smut sori on leaf surface',
                    'Reduced photosynthesis',
                    'Overall plant weakness'
                ],
                'treatment': [
                    'Apply appropriate fungicides',
                    'Remove infected plant parts',
                    'Ensure good air circulation',
                    'Monitor field regularly'
                ],
                'prevention': [
                    'Use resistant varieties',
                    'Avoid dense planting',
                    'Maintain field hygiene',
                    'Proper seed treatment'
                ]
            }
        }
        
        return disease_info.get(disease_name, {
            'full_name': disease_name,
            'severity': 'Unknown',
            'symptoms': ['Symptoms not available'],
            'treatment': ['Consult agricultural expert'],
            'prevention': ['Follow general crop management practices']
        })
    
    def get_model_details(self):
        """
        Get detailed information about the model
        """
        return {
            'name': 'DINOv2 (Self-Supervised Vision Transformer)',
            'accuracy': f'{self.model_accuracy:.2f}%',
            'architecture': 'Vision Transformer with Self-Supervised Learning',
            'input_size': '518×518 pixels',
            'parameters': f"{self.model_info.get('num_parameters', 'N/A'):,}" if 'num_parameters' in self.model_info else 'N/A',
            'trainable_params': f"{self.model_info.get('trainable_parameters', 'N/A'):,}" if 'trainable_parameters' in self.model_info else 'N/A',
            'description': """
            DINOv2 is a state-of-the-art self-supervised vision transformer that learns 
            visual representations without labeled data. It achieves excellent performance 
            on agricultural disease detection through advanced feature learning.
            """.strip(),
            'advantages': [
                'Self-supervised learning approach',
                'Excellent generalization capability', 
                'High accuracy on agricultural data',
                'Robust to image variations'
            ]
        }

class PredictionHelper:
    """Helper class for batch predictions and utilities"""
    
    def __init__(self):
        self.pipeline = PredictPipeline()
    
    def get_sample_images(self, num_samples=6):
        """
        Get sample test images for demonstration
        """
        try:
            # Load from dedicated sample images folder
            sample_dir = os.path.join('artifacts', 'sample_images')
            
            if os.path.exists(sample_dir):
                sample_images = []
                
                # Get all sample images
                for filename in os.listdir(sample_dir):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        # Extract class name from filename prefix
                        class_name = filename.split('_')[0]
                        sample_path = os.path.join(sample_dir, filename)
                        
                        sample_images.append({
                            'image_path': sample_path,
                            'label': class_name,
                            'filename': filename
                        })
                
                return sample_images[:num_samples]
            else:
                logging.warning("No sample images folder found")
                return []
                
        except Exception as e:
            logging.error(f"Error loading sample images: {e}")
            return []
