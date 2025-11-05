"""
Interactive CLI for Self-Healing Text Classifier
Provides a user-friendly interface for text classification with fallback handling.
"""

import sys
from pathlib import Path
from typing import Optional
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown
from rich.layout import Layout
from rich import box
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.graph import create_classifier
from src.utils.logger import ClassificationLogger

console = Console()


class SelfHealingCLI:
    """
    Interactive CLI for the self-healing text classifier.
    Handles user input, displays results, and manages the human-in-the-loop workflow.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the CLI application."""
        self.config_path = config_path
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.confidence_threshold = self.config['dag']['confidence_threshold']
        
        # Initialize logger
        self.logger = ClassificationLogger()
        
        # Initialize classifier (will load after training completes)
        self.classifier = None
        
        # Session stats
        self.session_start = datetime.now()
    
    def initialize_classifier(self):
        """Initialize the classifier (checks if model exists)."""
        model_path = Path("models/fine_tuned/final_model")
        
        if not model_path.exists():
            console.print("\n[red]❌ Error: Model not found![/red]")
            console.print(f"[yellow]Expected location: {model_path}[/yellow]")
            console.print("\n[cyan]💡 Please run training first:[/cyan]")
            console.print("[cyan]   python src/model_training.py[/cyan]\n")
            return False
        
        try:
            console.print("\n[yellow]🔄 Loading model...[/yellow]")
            self.classifier = create_classifier(self.config_path)
            console.print("[green]✅ Model loaded successfully![/green]\n")
            return True
        except Exception as e:
            console.print(f"\n[red]❌ Error loading model: {str(e)}[/red]\n")
            return False
    
    def print_welcome(self):
        """Display welcome screen."""
        welcome_text = f"""
# 🤖 Self-Healing Text Classifier

**A human-in-the-loop emotion classification system**

**Features:**
- 🎯 Multi-class emotion detection (6 emotions)
- 🛡️ Self-healing mechanism for uncertain predictions
- 💬 Interactive clarification when confidence is low
- 📊 Real-time confidence scoring
- 📝 Complete logging of all decisions

**Confidence Threshold:** {self.confidence_threshold:.0%}
**Session Started:** {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}

Type 'help' for commands or 'quit' to exit.
        """
        
        console.print(Panel(
            Markdown(welcome_text),
            border_style="green",
            box=box.DOUBLE
        ))
    
    def print_help(self):
        """Display help information."""
        help_table = Table(title="Available Commands", show_header=True)
        help_table.add_column("Command", style="cyan", width=15)
        help_table.add_column("Description", style="white")
        
        commands = [
            ("classify <text>", "Classify a text input"),
            ("<text>", "Shortcut: Just type text to classify"),
            ("stats", "Show session statistics"),
            ("help", "Show this help message"),
            ("clear", "Clear the screen"),
            ("quit / exit", "Exit the application")
        ]
        
        for cmd, desc in commands:
            help_table.add_row(cmd, desc)
        
        console.print(help_table)
        console.print()
    
    def display_inference_result(self, result: dict):
        """Display inference node results."""
        console.print("\n[bold cyan]═══ Inference Results ═══[/bold cyan]")
        
        # Main prediction
        console.print(f"\n[bold green]Predicted Label:[/bold green] {result['predicted_label']}")
        console.print(f"[bold yellow]Confidence:[/bold yellow] {result['confidence']:.2%}")
        
        # All probabilities table
        probs_table = Table(title="All Class Probabilities", show_header=True, box=box.SIMPLE)
        probs_table.add_column("Emotion", style="cyan")
        probs_table.add_column("Probability", style="yellow", justify="right")
        probs_table.add_column("Bar", style="green")
        
        sorted_probs = sorted(
            result['all_probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for emotion, prob in sorted_probs:
            bar = "█" * int(prob * 50)
            marker = " ⭐" if emotion == result['predicted_label'] else ""
            probs_table.add_row(
                f"{emotion}{marker}",
                f"{prob:.2%}",
                bar
            )
        
        console.print(probs_table)
        
        # Log inference
        self.logger.log_inference(
            result['input_text'],
            result['predicted_label'],
            result['confidence'],
            result['all_probabilities']
        )
    
    def display_confidence_check(self, result: dict):
        """Display confidence check results."""
        console.print("\n[bold cyan]═══ Confidence Check ═══[/bold cyan]")
        
        confidence = result['confidence']
        threshold = self.confidence_threshold
        status = result['confidence_status']
        
        if status == 'HIGH':
            console.print(f"\n[green]✅ Confidence OK:[/green] {confidence:.2%} >= {threshold:.0%}")
            console.print(f"[green]→ Accepting prediction: {result['predicted_label']}[/green]")
        else:
            console.print(f"\n[yellow]⚠️  Confidence Low:[/yellow] {confidence:.2%} < {threshold:.0%}")
            console.print(f"[yellow]→ Triggering fallback mechanism...[/yellow]")
        
        # Log confidence check
        self.logger.log_confidence_check(
            confidence,
            threshold,
            result['should_fallback']
        )
    
    def handle_fallback(self, result: dict) -> dict:
        """Handle fallback interaction with user."""
        console.print("\n[bold cyan]═══ Fallback Activated ═══[/bold cyan]")
        
        # Check if clarification_prompt exists
        if 'clarification_prompt' not in result:
            # Generate a simple prompt if missing
            console.print(f"\n[yellow]⚠️  Low confidence: {result['confidence']:.2%}[/yellow]")
            console.print(f"[yellow]Model predicted: {result['predicted_label']}[/yellow]")
            
            # Get top 2 from probabilities
            sorted_probs = sorted(
                result['all_probabilities'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:2]
            
            console.print("\n[cyan]Top candidates:[/cyan]")
            for i, (label, prob) in enumerate(sorted_probs, 1):
                console.print(f"  {i}. {label} ({prob:.0%})")
            
            result['suggested_labels'] = [label for label, _ in sorted_probs]
        else:
            # Show the generated prompt
            console.print(f"\n[yellow]{result['clarification_prompt']}[/yellow]")
        
        # Log fallback request
        self.logger.log_fallback_request(result.get('clarification_prompt', 'Clarification needed'))
        
        # Get user input
        console.print("\n[bold cyan]Your response:[/bold cyan]")
        console.print("[dim]Enter the number (1, 2) or the emotion name:[/dim]")
        
        user_input = Prompt.ask("[cyan]➤[/cyan]")
        
        # Process user input
        result = self.classifier.fallback_node.process_user_input(result, user_input)
        
        # Log user response
        self.logger.log_user_response(user_input, result.get('final_label'))
        
        return result
    
    def display_final_result(self, result: dict):
        """Display final classification result."""
        console.print("\n[bold green]═══ Final Decision ═══[/bold green]")
        
        final_label = result['final_label']
        source = result.get('source', 'unknown')
        
        # Create result panel
        if source == 'model':
            result_text = f"[bold green]✅ Label: {final_label}[/bold green]\n"
            result_text += f"[dim]Source: Model (high confidence)[/dim]\n"
            result_text += f"[dim]Confidence: {result['confidence']:.2%}[/dim]"
            border_color = "green"
        elif source == 'user_correction':
            result_text = f"[bold yellow]✅ Label: {final_label}[/bold yellow]\n"
            result_text += f"[dim]Source: User correction[/dim]\n"
            result_text += f"[dim]Original prediction: {result['predicted_label']} ({result['confidence']:.2%})[/dim]"
            border_color = "yellow"
        else:
            result_text = f"[bold cyan]✅ Label: {final_label}[/bold cyan]\n"
            result_text += f"[dim]Source: Backup model[/dim]"
            border_color = "cyan"
        
        console.print(Panel(
            result_text,
            title="Final Classification",
            border_style=border_color,
            box=box.ROUNDED
        ))
        
        # Log final decision
        self.logger.log_final_decision(
            final_label,
            source,
            result.get('predicted_label')
        )
    
    def classify_text(self, text: str):
        """
        Main classification workflow.
        
        Args:
            text: Input text to classify
        """
        # Display input
        console.print(f"\n[bold]Input:[/bold] [italic]\"{text}\"[/italic]")
        
        # Run inference through graph
        result = self.classifier.classify(text)
        
        # Display inference results
        self.display_inference_result(result)
        
        # Display confidence check
        self.display_confidence_check(result)
        
        # Handle fallback if needed
        if result.get('clarification_needed'):
            result = self.handle_fallback(result)
            result['source'] = 'user_correction'
        
        # Display final result
        self.display_final_result(result)
        
        console.print("\n" + "─" * 80 + "\n")
    
    def show_statistics(self):
        """Display session statistics."""
        console.print("\n")
        self.logger.print_statistics()
    
    def run(self):
        """Main CLI loop."""
        # Print welcome
        self.print_welcome()
        
        # Initialize classifier
        if not self.initialize_classifier():
            return
        
        console.print("[cyan]💡 Tip: Just type your text and press Enter to classify![/cyan]\n")
        
        # Main loop
        while True:
            try:
                # Get user input
                user_input = Prompt.ask("\n[bold cyan]➤[/bold cyan]").strip()
                
                if not user_input:
                    continue
                
                # Parse commands
                command = user_input.lower()
                
                if command in ['quit', 'exit', 'q']:
                    console.print("\n[yellow]📊 Final Statistics:[/yellow]")
                    self.show_statistics()
                    console.print("\n[green]👋 Thank you for using Self-Healing Classifier![/green]")
                    console.print("[green]Goodbye! 🎉[/green]\n")
                    break
                
                elif command == 'help':
                    self.print_help()
                
                elif command == 'stats':
                    self.show_statistics()
                
                elif command == 'clear':
                    console.clear()
                    self.print_welcome()
                
                elif command.startswith('classify '):
                    text = user_input[9:].strip()
                    if text:
                        self.classify_text(text)
                    else:
                        console.print("[yellow]⚠️  Please provide text to classify[/yellow]")
                
                else:
                    # Treat as direct text input
                    self.classify_text(user_input)
            
            except KeyboardInterrupt:
                console.print("\n\n[yellow]Interrupted by user[/yellow]")
                if Confirm.ask("\n[cyan]Do you want to exit?[/cyan]", default=False):
                    break
                else:
                    continue
            
            except Exception as e:
                console.print(f"\n[red]❌ Error: {str(e)}[/red]")
                console.print("[yellow]Type 'help' for usage information[/yellow]\n")


def main():
    """Entry point for CLI application."""
    cli = SelfHealingCLI()
    cli.run()


if __name__ == "__main__":
    main()