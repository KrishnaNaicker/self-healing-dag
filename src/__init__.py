"""
Self-Healing Text Classifier Package
A LangGraph-based classification pipeline with human-in-the-loop fallback.
"""

__version__ = "1.0.0"
__author__ = "Krishna Naicker"

from src.graph import create_classifier
from src.cli import SelfHealingCLI

__all__ = ['create_classifier', 'SelfHealingCLI']