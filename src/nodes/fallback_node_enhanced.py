"""
Enhanced Fallback Node with Backup Models (BONUS FEATURE)
Implements ensemble approach with sentiment analysis and zero-shot classification.
ENHANCED: Better sarcasm detection with pattern matching.
"""

from typing import Dict, Optional, List
from transformers import pipeline
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
import torch
import json
from pathlib import Path
from datetime import datetime
import re

console = Console()


class EnhancedFallbackNode:
    """
    Fallback node with backup models for improved accuracy.
    
    Backup Strategy:
    1. Primary model fails (low confidence) → Try sarcasm detection
    2. Try sentiment analysis
    3. Try zero-shot classification
    4. Ask user (interactive mode)
    """
    
    # Sarcasm detection patterns
    SARCASM_POSITIVE_WORDS = [
        'great', 'perfect', 'wonderful', 'brilliant', 'fantastic', 
        'lovely', 'best', 'amazing', 'excellent', 'superb', 'fabulous',
        'terrific', 'awesome', 'outstanding', 'marvelous'
    ]
    
    SARCASM_NEGATIVE_CONTEXT = [
        'monday', 'traffic', 'died', 'broke', 'stuck', 'late', 'wrong',
        'failed', 'problem', 'issue', 'error', 'crash', 'broken', 'down',
        'overtime', 'canceled', 'cancelled', 'lost', 'missing', 'fired',
        'sick', 'tired', 'exhausted', 'waiting', 'delayed'
    ]
    
    SARCASM_MARKERS = [
        'oh ', 'yeah right', 'sure', 'totally', 'obviously', 'clearly',
        'just what i', 'exactly what', 'so happy', 'so excited',
        'can barely contain', 'thrilled beyond'
    ]
    
    def __init__(self, interactive_mode: bool = True, enable_backup: bool = True):
        """Initialize with optional backup models."""
        self.interactive_mode = interactive_mode
        self.enable_backup = enable_backup
        self.fallback_log_path = Path("logs/fallback_log.jsonl")
        self.fallback_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize backup models
        if self.enable_backup:
            console.print("[yellow]🔧 Loading backup models...[/yellow]")
            
            # Backup Model 1: Sentiment Analysis (for sarcasm detection)
            try:
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=0 if torch.cuda.is_available() else -1
                )
                console.print("[green]✅ Sentiment analyzer loaded[/green]")
            except Exception as e:
                console.print(f"[yellow]⚠️ Sentiment analyzer unavailable: {e}[/yellow]")
                self.sentiment_analyzer = None
            
            # Backup Model 2: Zero-shot classifier (for edge cases)
            try:
                self.zero_shot = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",
                    device=0 if torch.cuda.is_available() else -1
                )
                console.print("[green]✅ Zero-shot classifier loaded[/green]")
            except Exception as e:
                console.print(f"[yellow]⚠️ Zero-shot classifier unavailable: {e}[/yellow]")
                self.zero_shot = None
        else:
            self.sentiment_analyzer = None
            self.zero_shot = None
        
        # Statistics tracking
        self.stats = {
            'total_fallbacks': 0,
            'sentiment_rescues': 0,
            'zero_shot_rescues': 0,
            'user_interventions': 0,
            'sarcasm_detections': 0,
            'confidence_scores': []
        }
    
    def __call__(self, state: Dict) -> Dict:
        """Process fallback with backup models."""
        text = state['text']
        predicted_label = state['predicted_label']
        confidence = state['confidence']
        all_probabilities = state['all_probabilities']
        
        console.print("\n[bold yellow]⚠️ FALLBACK ACTIVATED[/bold yellow]")
        console.print(f"[cyan]Primary prediction:[/cyan] {predicted_label} ({confidence:.1%})")
        
        self.stats['total_fallbacks'] += 1
        self.stats['confidence_scores'].append(confidence)
        
        # Try backup models
        final_label = None
        method = 'primary'
        
        # Strategy 0: Pattern-based Sarcasm Detection (FIRST!)
        if self.enable_backup and predicted_label in ['joy', 'love']:
            sarcasm_result = self._detect_sarcasm_patterns(text)
            if sarcasm_result:
                final_label = sarcasm_result
                method = 'sentiment_backup'
                self.stats['sarcasm_detections'] += 1
                self.stats['sentiment_rescues'] += 1
        
        # Strategy 1: Sentiment Analysis (if primary predicted positive emotion)
        if not final_label and self.enable_backup and self.sentiment_analyzer and predicted_label in ['joy', 'love']:
            sentiment_result = self._check_sentiment(text, predicted_label, confidence)
            if sentiment_result:
                final_label = sentiment_result
                method = 'sentiment_backup'
                self.stats['sentiment_rescues'] += 1
        
        # Strategy 2: Zero-shot classification (if still uncertain)
        if not final_label and self.enable_backup and self.zero_shot and confidence < 0.60:
            zero_shot_result = self._check_zero_shot(text)
            if zero_shot_result:
                final_label = zero_shot_result
                method = 'zero_shot_backup'
                self.stats['zero_shot_rescues'] += 1
        
        # Strategy 3: User intervention (interactive mode)
        if not final_label and self.interactive_mode:
            final_label = self._ask_user(text, predicted_label, all_probabilities)
            method = 'user_intervention'
            self.stats['user_interventions'] += 1
        
        # Fallback to primary if no backup worked
        if not final_label:
            final_label = predicted_label
            method = 'primary_fallback'
        
        # Log fallback
        self._log_fallback(text, predicted_label, final_label, confidence, method)
        
        # Update state
        state['final_label'] = final_label
        state['fallback_triggered'] = True
        state['fallback_method'] = method
        
        console.print(f"[green]✅ Final decision:[/green] {final_label} (via {method})")
        
        return state
    
    def _detect_sarcasm_patterns(self, text: str) -> Optional[str]:
        """
        Detect sarcasm using pattern matching.
        Checks for positive words + negative context combination.
        """
        text_lower = text.lower()
        
        # Check for explicit sarcasm markers
        for marker in self.SARCASM_MARKERS:
            if marker in text_lower:
                console.print(f"[magenta]🎭 Sarcasm marker detected: '{marker}'[/magenta]")
                return 'anger'
        
        # Check for positive words + negative context
        has_positive = any(word in text_lower for word in self.SARCASM_POSITIVE_WORDS)
        has_negative_context = any(word in text_lower for word in self.SARCASM_NEGATIVE_CONTEXT)
        
        if has_positive and has_negative_context:
            console.print("[magenta]🎭 Sarcasm pattern detected! (positive words + negative context)[/magenta]")
            return 'anger'
        
        # Check for exaggeration patterns (repeated punctuation)
        if re.search(r'(!\s*){2,}|(\.\s*){3,}', text):
            if has_positive:
                console.print("[magenta]🎭 Possible sarcasm (exaggeration detected)[/magenta]")
                return 'anger'
        
        return None
    
    def _check_sentiment(self, text: str, primary_label: str, confidence: float) -> Optional[str]:
        """
        Use sentiment analysis to detect sarcasm.
        If primary says 'joy' but sentiment is negative → probably sarcasm (anger).
        """
        console.print("\n[cyan]🔍 Checking sentiment with backup model...[/cyan]")
        
        try:
            result = self.sentiment_analyzer(text)[0]
            sentiment_label = result['label']  # 'positive', 'negative', or 'neutral'
            sentiment_score = result['score']
            
            console.print(f"[yellow]Sentiment: {sentiment_label} ({sentiment_score:.1%})[/yellow]")
            
            # Sarcasm detection: positive prediction but negative sentiment
            if primary_label in ['joy', 'love'] and sentiment_label == 'negative' and sentiment_score > 0.60:
                console.print("[magenta]🎭 Sarcasm detected! (sentiment contradiction)[/magenta]")
                return 'anger'
            
            # Neutral sentiment might indicate sarcasm too
            if primary_label in ['joy', 'love'] and sentiment_label == 'neutral' and sentiment_score > 0.70:
                console.print("[magenta]🎭 Possible sarcasm (neutral sentiment for positive words)[/magenta]")
                return 'anger'
            
            # Map sentiment to emotion if confident
            if sentiment_score > 0.80:
                mapping = {
                    'positive': 'joy',
                    'negative': 'sadness',
                    'neutral': 'neutral'
                }
                suggested = mapping.get(sentiment_label)
                if suggested and suggested != primary_label:
                    console.print(f"[cyan]💡 Sentiment suggests: {suggested}[/cyan]")
                    return suggested
        
        except Exception as e:
            console.print(f"[red]Sentiment analysis failed: {e}[/red]")
        
        return None
    
    def _check_zero_shot(self, text: str) -> Optional[str]:
        """Use zero-shot classification for edge cases."""
        console.print("\n[cyan]🔍 Trying zero-shot classification...[/cyan]")
        
        try:
            candidate_labels = [
                'joy', 'sadness', 'anger', 'fear', 
                'love', 'surprise', 'neutral', 'disgust'
            ]
            
            result = self.zero_shot(text, candidate_labels)
            top_label = result['labels'][0]
            top_score = result['scores'][0]
            
            console.print(f"[yellow]Zero-shot: {top_label} ({top_score:.1%})[/yellow]")
            
            # Accept if confident
            if top_score > 0.65:
                console.print(f"[cyan]💡 Zero-shot confident: {top_label}[/cyan]")
                return top_label
        
        except Exception as e:
            console.print(f"[red]Zero-shot failed: {e}[/red]")
        
        return None
    
    def _ask_user(self, text: str, predicted_label: str, probabilities: Dict) -> str:
        """Ask user for clarification (interactive mode)."""
        console.print("\n[bold cyan]🤔 Need your help![/bold cyan]")
        console.print(f'[yellow]Text:[/yellow] "{text}"')
        console.print(f"[yellow]AI thinks:[/yellow] {predicted_label}")
        
        # Show top 3 alternatives
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
        console.print("\n[cyan]Top alternatives:[/cyan]")
        for i, (label, prob) in enumerate(sorted_probs, 1):
            console.print(f"  {i}. {label} ({prob:.0%})")
        
        # Get user input
        console.print("\n[bold]What's the correct emotion?[/bold]")
        console.print("[dim]Enter number (1-3) or emotion name:[/dim]")
        
        user_input = Prompt.ask("[cyan]➤[/cyan]")
        
        # Parse input
        if user_input.isdigit() and 1 <= int(user_input) <= 3:
            return sorted_probs[int(user_input) - 1][0]
        elif user_input.lower() in probabilities:
            return user_input.lower()
        else:
            console.print("[yellow]Invalid input, using primary prediction[/yellow]")
            return predicted_label
    
    def _log_fallback(self, text: str, primary: str, final: str, confidence: float, method: str):
        """Log fallback event for statistics."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'text': text,
            'primary_prediction': primary,
            'final_label': final,
            'confidence': confidence,
            'method': method,
            'corrected': primary != final
        }
        
        with open(self.fallback_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def get_statistics(self) -> Dict:
        """Get fallback statistics."""
        return {
            'total_fallbacks': self.stats['total_fallbacks'],
            'sentiment_rescues': self.stats['sentiment_rescues'],
            'zero_shot_rescues': self.stats['zero_shot_rescues'],
            'user_interventions': self.stats['user_interventions'],
            'sarcasm_detections': self.stats['sarcasm_detections'],
            'avg_confidence': sum(self.stats['confidence_scores']) / len(self.stats['confidence_scores']) 
                if self.stats['confidence_scores'] else 0,
            'backup_success_rate': (self.stats['sentiment_rescues'] + self.stats['zero_shot_rescues']) 
                / max(self.stats['total_fallbacks'], 1)
        }
    
    def print_statistics(self):
        """Display fallback statistics."""
        stats = self.get_statistics()
        
        console.print("\n" + "="*80)
        console.print("[bold cyan]📊 FALLBACK STATISTICS (BONUS FEATURE)[/bold cyan]")
        console.print("="*80)
        console.print(f"Total Fallbacks: {stats['total_fallbacks']}")
        console.print(f"Sentiment Backup Rescues: {stats['sentiment_rescues']} ({stats['sentiment_rescues']/max(stats['total_fallbacks'],1)*100:.1f}%)")
        console.print(f"  └─ Sarcasm Detections: {stats['sarcasm_detections']}")
        console.print(f"Zero-Shot Rescues: {stats['zero_shot_rescues']} ({stats['zero_shot_rescues']/max(stats['total_fallbacks'],1)*100:.1f}%)")
        console.print(f"User Interventions: {stats['user_interventions']}")
        console.print(f"Average Confidence: {stats['avg_confidence']:.1%}")
        console.print(f"Backup Success Rate: {stats['backup_success_rate']:.1%}")
        console.print("="*80 + "\n")