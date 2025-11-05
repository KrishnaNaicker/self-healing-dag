"""
Inference Node - Runs classification using the fine-tuned model.
Enhanced with emoji detection and text preprocessing.
"""

import torch
import numpy as np
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
from pathlib import Path
import json
from rich.console import Console
from src.utils.emoji_processor import detect_emoji_emotion, preprocess_text

console = Console()


class InferenceNode:
    """
    Runs text classification inference using the fine-tuned LoRA model.
    Returns prediction, confidence score, and all class probabilities.
    
    ENHANCED FEATURES:
    - Emoji-only input detection
    - Text preprocessing (repeated chars normalization)
    - Improved edge case handling
    """
    
    def __init__(self, model_path: str = "models/fine_tuned/final_model"):
        """Initialize the inference node with the trained model."""
        self.model_path = Path(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load configuration
        with open(self.model_path / "training_config.json", 'r') as f:
            self.config = json.load(f)
        
        self.id2label = {int(k): v for k, v in self.config['id2label'].items()}
        self.label2id = self.config['label2id']
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        
        # Load base model
        base_model_name = self.config['model_name']
        self.model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=len(self.id2label),
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        # Load LoRA weights
        self.model = PeftModel.from_pretrained(self.model, str(self.model_path))
        self.model.to(self.device)
        self.model.eval()
        
        console.print(f"[green]✅ Model loaded successfully![/green]")
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute inference on input text.
        
        Args:
            state: Dictionary containing 'text' or 'input_text'
        
        Returns:
            Updated state with prediction results
        """
        # Handle both 'text' and 'input_text' keys
        input_text = state.get('text', state.get('input_text', ''))
        
        if not input_text:
            raise ValueError("No input text provided in state")
        
        # STEP 1: Check for emoji-only input
        is_emoji_dominant, emoji_emotion, emoji_confidence = detect_emoji_emotion(input_text)
        
        if is_emoji_dominant and emoji_confidence > 0.75:
            console.print(f"[magenta]😊 Emoji detected: {emoji_emotion} ({emoji_confidence:.1%})[/magenta]")
            
            # Create probability distribution for emoji-based prediction
            all_probs = {label: 0.01 for label in self.id2label.values()}
            all_probs[emoji_emotion] = emoji_confidence
            
            # Distribute remaining probability
            remaining_prob = 1.0 - emoji_confidence
            other_labels = [l for l in all_probs.keys() if l != emoji_emotion]
            for label in other_labels:
                all_probs[label] = remaining_prob / len(other_labels)
            
            # Update state with emoji-based prediction
            state.update({
                'predicted_label': emoji_emotion,
                'predicted_label_id': self.label2id[emoji_emotion],
                'confidence': emoji_confidence,
                'all_probabilities': all_probs,
                'text': input_text,  # Ensure 'text' key exists
                'emoji_detected': True
            })
            
            return state
        
        # STEP 2: Preprocess text (normalize repeated chars)
        processed_text = preprocess_text(input_text)
        
        # STEP 3: Tokenize
        inputs = self.tokenizer(
            processed_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(self.device)
        
        # STEP 4: Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)[0]
        
        # STEP 5: Get prediction
        predicted_label_id = torch.argmax(probabilities).item()
        predicted_label = self.id2label[predicted_label_id]
        confidence = probabilities[predicted_label_id].item()
        
        # STEP 6: Get all probabilities
        all_probs = {
            self.id2label[i]: prob.item() 
            for i, prob in enumerate(probabilities)
        }
        
        # STEP 7: Update state
        state.update({
            'predicted_label': predicted_label,
            'predicted_label_id': predicted_label_id,
            'confidence': confidence,
            'all_probabilities': all_probs,
            'text': input_text,  # Ensure original text is preserved
            'emoji_detected': False
        })
        
        return state


def create_inference_node(model_path: str = "models/fine_tuned/final_model"):
    """Factory function to create inference node."""
    return InferenceNode(model_path)