"""
LangGraph DAG - Self-Healing Classification Pipeline with Backup Models (BONUS)
Orchestrates the flow between inference, confidence checking, and fallback.
Enhanced with ensemble backup models for improved accuracy.
"""

from typing import Dict, Any, TypedDict, Literal
from langgraph.graph import StateGraph, END
from src.nodes.inference_node import create_inference_node
from src.nodes.confidence_check_node import create_confidence_check_node
from src.nodes.fallback_node_enhanced import EnhancedFallbackNode
import yaml
from rich.console import Console
from pathlib import Path

console = Console()


class ClassificationState(TypedDict):
    """State schema for the classification pipeline."""
    # Input
    input_text: str
    text: str  # Alias for compatibility
    
    # Inference results
    predicted_label: str
    predicted_label_id: int
    confidence: float
    all_probabilities: Dict[str, float]
    
    # Confidence check
    should_fallback: bool
    confidence_status: str
    decision_reason: str
    
    # Fallback (Enhanced with backup models)
    fallback_triggered: bool
    fallback_method: str  # 'primary', 'sentiment_backup', 'zero_shot_backup', 'user_intervention'
    clarification_needed: bool
    user_corrected: bool
    backup_model_used: bool  # BONUS: Track if backup was used
    
    # Final output
    final_label: str
    source: str  # 'model', 'sentiment_backup', 'zero_shot_backup', 'user_correction'


class SelfHealingClassifier:
    """
    Main classification pipeline with self-healing mechanism.
    Uses LangGraph to orchestrate the DAG workflow.
    
    BONUS FEATURES:
    - Ensemble backup models (sentiment + zero-shot)
    - Advanced sarcasm detection
    - Comprehensive fallback statistics
    """
    
    def __init__(self, config_path: str = "config.yaml", enable_backup: bool = True):
        """
        Initialize the self-healing classifier pipeline.
        
        Args:
            config_path: Path to configuration file
            enable_backup: Enable backup models (BONUS feature)
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.confidence_threshold = self.config['dag']['confidence_threshold']
        self.enable_backup = enable_backup
        
        # Initialize nodes
        console.print("[cyan]🔧 Initializing pipeline nodes...[/cyan]")
        
        self.inference_node = create_inference_node()
        self.confidence_check_node = create_confidence_check_node(self.confidence_threshold)
        
        # BONUS: Enhanced fallback with backup models
        if self.enable_backup:
            console.print("[yellow]⚡ BONUS: Enabling backup models (ensemble approach)...[/yellow]")
            self.fallback_node = EnhancedFallbackNode(
                interactive_mode=True,
                enable_backup=True
            )
        else:
            # Fallback to basic fallback (compatibility)
            from src.nodes.fallback_node import create_fallback_node
            self.fallback_node = create_fallback_node(interactive_mode=True)
        
        # Build graph
        self.graph = self._build_graph()
        
        console.print("[green]✅ Self-healing classifier pipeline initialized[/green]")
        if self.enable_backup:
            console.print("[bold magenta]🎁 BONUS FEATURES ENABLED:[/bold magenta]")
            console.print("   • Sentiment-based sarcasm detection")
            console.print("   • Zero-shot classification fallback")
            console.print("   • Advanced confidence tracking")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph DAG."""
        # Create graph
        workflow = StateGraph(ClassificationState)
        
        # Add nodes
        workflow.add_node("inference", self._inference_wrapper)
        workflow.add_node("confidence_check", self.confidence_check_node)
        workflow.add_node("fallback", self._fallback_wrapper)
        workflow.add_node("accept", self._accept_prediction)
        
        # Set entry point
        workflow.set_entry_point("inference")
        
        # Add edges
        workflow.add_edge("inference", "confidence_check")
        
        # Conditional routing from confidence check
        workflow.add_conditional_edges(
            "confidence_check",
            self._route_decision,
            {
                "accept": "accept",
                "fallback": "fallback"
            }
        )
        
        # Terminal edges
        workflow.add_edge("accept", END)
        workflow.add_edge("fallback", END)
        
        # Compile
        return workflow.compile()
    
    def _inference_wrapper(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Wrapper to handle input_text → text conversion."""
        # Ensure 'text' field exists for inference node
        if 'input_text' in state and 'text' not in state:
            state['text'] = state['input_text']
        
        # Run inference
        result = self.inference_node(state)
        
        return result
    
    def _fallback_wrapper(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Wrapper for fallback node with source tracking."""
        # Ensure 'text' field exists
        if 'input_text' in state and 'text' not in state:
            state['text'] = state['input_text']
        
        # Run enhanced fallback
        result = self.fallback_node(state)
        
        # Set source based on fallback method
        method = result.get('fallback_method', 'primary_fallback')
        source_mapping = {
            'sentiment_backup': 'sentiment_backup',
            'zero_shot_backup': 'zero_shot_backup',
            'user_intervention': 'user_correction',
            'primary_fallback': 'model'
        }
        result['source'] = source_mapping.get(method, 'model')
        result['backup_model_used'] = method in ['sentiment_backup', 'zero_shot_backup']
        
        return result
    
    def _route_decision(self, state: Dict[str, Any]) -> Literal["accept", "fallback"]:
        """Route based on confidence check decision."""
        return "fallback" if state.get('should_fallback', False) else "accept"
    
    def _accept_prediction(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Accept the model's prediction (high confidence path)."""
        state['final_label'] = state['predicted_label']
        state['source'] = 'model'
        state['fallback_triggered'] = False
        state['backup_model_used'] = False
        
        console.print("\n[green]✅ High confidence - Prediction accepted![/green]")
        
        return state
    
    def classify(self, input_text: str) -> Dict[str, Any]:
        """
        Run classification on input text.
        
        Args:
            input_text: Text to classify
        
        Returns:
            Classification results with all intermediate states
        """
        # Initialize state
        initial_state = {
            'input_text': input_text,
            'text': input_text,  # Alias for compatibility
            'fallback_triggered': False,
            'clarification_needed': False,
            'user_corrected': False,
            'backup_model_used': False
        }
        
        # Run through graph
        result = self.graph.invoke(initial_state)
        
        return result
    
    def classify_with_user_input(
        self,
        input_text: str,
        user_clarification: str = None
    ) -> Dict[str, Any]:
        """
        Run classification with optional user clarification.
        Used by CLI for interactive mode.
        
        Args:
            input_text: Text to classify
            user_clarification: User's clarification if fallback triggered
        
        Returns:
            Final classification results
        """
        # Run initial classification
        result = self.classify(input_text)
        
        # If fallback was triggered and user provided input
        if result.get('clarification_needed') and user_clarification:
            # Process user input (if fallback node supports it)
            if hasattr(self.fallback_node, 'process_user_input'):
                result = self.fallback_node.process_user_input(result, user_clarification)
            result['source'] = 'user_correction'
            result['user_corrected'] = True
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get pipeline statistics (BONUS feature).
        
        Returns:
            Dictionary with fallback statistics
        """
        if hasattr(self.fallback_node, 'get_statistics'):
            return self.fallback_node.get_statistics()
        return {}
    
    def print_statistics(self):
        """Print fallback statistics (BONUS feature)."""
        if hasattr(self.fallback_node, 'print_statistics'):
            self.fallback_node.print_statistics()
        else:
            console.print("[yellow]Statistics not available (backup models disabled)[/yellow]")
    
    def visualize_statistics(self):
        """
        Generate visualization of fallback statistics (BONUS feature).
        Creates charts showing confidence curves and fallback frequency.
        """
        console.print("\n[cyan]📊 Generating fallback statistics visualizations...[/cyan]")
        
        try:
            from src.visualize_fallback_stats import main as visualize_main
            visualize_main()
        except ImportError:
            console.print("[yellow]⚠️ Visualization module not found. Run: pip install matplotlib seaborn[/yellow]")
        except Exception as e:
            console.print(f"[red]Error generating visualizations: {e}[/red]")


def create_classifier(
    config_path: str = "config.yaml",
    enable_backup: bool = True
) -> SelfHealingClassifier:
    """
    Factory function to create the classifier pipeline.
    
    Args:
        config_path: Path to configuration file
        enable_backup: Enable backup models (BONUS feature)
    
    Returns:
        Initialized SelfHealingClassifier instance
    """
    return SelfHealingClassifier(config_path, enable_backup=enable_backup)


# ============================================================================
# BONUS: Command-line utility for testing
# ============================================================================

def test_pipeline():
    """Quick test of the pipeline with sample inputs."""
    console.print("\n[bold cyan]🧪 Testing Self-Healing Pipeline[/bold cyan]\n")
    
    classifier = create_classifier(enable_backup=True)
    
    test_cases = [
        ("I'm so happy!", "Should be easy - clear joy"),
        ("Oh great, another Monday.", "Sarcasm test - backup should detect anger"),
        ("The movie was okay I guess", "Neutral/ambiguous test"),
        ("I love you so much!", "Love detection test"),
    ]
    
    for text, description in test_cases:
        console.print(f"\n[yellow]Test:[/yellow] {description}")
        console.print(f'[cyan]Input:[/cyan] "{text}"')
        
        result = classifier.classify(text)
        
        console.print(f"[green]Result:[/green] {result['final_label']}")
        console.print(f"[magenta]Source:[/magenta] {result['source']}")
        console.print(f"[blue]Confidence:[/blue] {result['confidence']:.1%}")
        if result.get('backup_model_used'):
            console.print("[bold yellow]⚡ BONUS: Backup model used![/bold yellow]")
        console.print("-" * 80)
    
    # Show statistics
    classifier.print_statistics()


if __name__ == "__main__":
    test_pipeline()