"""
Structured logging utility for the self-healing classifier.
Tracks predictions, fallbacks, and final decisions with timestamps.
"""

from loguru import logger
import sys
from pathlib import Path
from datetime import datetime
import json

class ClassificationLogger:
    def __init__(self, log_file: str = "./logs/predictions.log"):
        """Initialize the logger with file and console output."""
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Remove default logger
        logger.remove()
        
        # Add console logger with colors
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
            level="INFO",
            colorize=True
        )
        
        # Add file logger with structured format
        logger.add(
            self.log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            level="DEBUG",
            rotation="10 MB",
            retention="1 week"
        )
        
        self.session_start = datetime.now()
        self.stats = {
            "total_predictions": 0,
            "fallback_activations": 0,
            "user_corrections": 0,
            "correct_predictions": 0
        }
    
    def log_inference(self, input_text: str, predicted_label: str, 
                     confidence: float, all_probs: dict):
        """Log initial inference results."""
        self.stats["total_predictions"] += 1
        
        log_data = {
            "event": "INFERENCE",
            "input": input_text[:100] + "..." if len(input_text) > 100 else input_text,
            "predicted_label": predicted_label,
            "confidence": f"{confidence:.2%}",
            "all_probabilities": {k: f"{v:.2%}" for k, v in all_probs.items()}
        }
        
        logger.info(f"[InferenceNode] {json.dumps(log_data, indent=2)}")
    
    def log_confidence_check(self, confidence: float, threshold: float, 
                            trigger_fallback: bool):
        """Log confidence check results."""
        status = "FALLBACK_TRIGGERED" if trigger_fallback else "CONFIDENCE_OK"
        
        log_data = {
            "event": "CONFIDENCE_CHECK",
            "confidence": f"{confidence:.2%}",
            "threshold": f"{threshold:.2%}",
            "status": status
        }
        
        if trigger_fallback:
            self.stats["fallback_activations"] += 1
            logger.warning(f"[ConfidenceCheckNode] {json.dumps(log_data, indent=2)}")
        else:
            logger.info(f"[ConfidenceCheckNode] {json.dumps(log_data, indent=2)}")
    
    def log_fallback_request(self, question: str):
        """Log when fallback asks user for clarification."""
        log_data = {
            "event": "FALLBACK_REQUEST",
            "question": question
        }
        logger.warning(f"[FallbackNode] {json.dumps(log_data, indent=2)}")
    
    def log_user_response(self, user_input: str, corrected_label: str = None):
        """Log user's clarification response."""
        self.stats["user_corrections"] += 1
        
        log_data = {
            "event": "USER_CLARIFICATION",
            "user_response": user_input,
            "corrected_label": corrected_label
        }
        logger.info(f"[FallbackNode] {json.dumps(log_data, indent=2)}")
    
    def log_final_decision(self, final_label: str, source: str, 
                          original_label: str = None):
        """Log the final classification decision."""
        log_data = {
            "event": "FINAL_DECISION",
            "final_label": final_label,
            "source": source,  # "model", "user_correction", "backup_model"
            "original_prediction": original_label
        }
        
        if source == "model":
            self.stats["correct_predictions"] += 1
        
        logger.success(f"[Final] {json.dumps(log_data, indent=2)}")
        logger.info("-" * 80)
    
    def log_backup_model(self, backup_label: str, backup_confidence: float):
        """Log backup model prediction."""
        log_data = {
            "event": "BACKUP_MODEL",
            "backup_label": backup_label,
            "backup_confidence": f"{backup_confidence:.2%}"
        }
        logger.info(f"[BackupModel] {json.dumps(log_data, indent=2)}")
    
    def get_session_stats(self):
        """Get statistics for current session."""
        return {
            **self.stats,
            "session_duration": str(datetime.now() - self.session_start),
            "fallback_rate": f"{(self.stats['fallback_activations'] / max(self.stats['total_predictions'], 1)) * 100:.1f}%",
            "accuracy_rate": f"{(self.stats['correct_predictions'] / max(self.stats['total_predictions'], 1)) * 100:.1f}%"
        }
    
    def print_statistics(self):
        """Print session statistics in a formatted way."""
        stats = self.get_session_stats()
        
        print("\n" + "=" * 60)
        print("📊 SESSION STATISTICS")
        print("=" * 60)
        print(f"Total Predictions: {stats['total_predictions']}")
        print(f"Fallback Activations: {stats['fallback_activations']} ({stats['fallback_rate']})")
        print(f"User Corrections: {stats['user_corrections']}")
        print(f"Direct Correct Predictions: {stats['correct_predictions']} ({stats['accuracy_rate']})")
        print(f"Session Duration: {stats['session_duration']}")
        print("=" * 60 + "\n")

# Global logger instance
classification_logger = ClassificationLogger()