"""
Visualization utilities for the self-healing classifier.
Includes terminal-based charts and statistics display.
"""

import plotext as plt
from typing import List, Dict
from rich.console import Console
from rich.table import Table

console = Console()


def plot_confidence_curve(confidences: List[float], title: str = "Confidence Over Time"):
    """
    Plot confidence scores over time in terminal.
    
    Args:
        confidences: List of confidence scores
        title: Plot title
    """
    plt.clear_figure()
    plt.plot(confidences, marker="dot")
    plt.title(title)
    plt.xlabel("Prediction #")
    plt.ylabel("Confidence")
    plt.ylim(0, 1)
    plt.plotsize(80, 20)
    plt.show()


def plot_fallback_frequency(total: int, fallbacks: int):
    """
    Plot fallback frequency as a bar chart in terminal.
    
    Args:
        total: Total predictions
        fallbacks: Number of fallback activations
    """
    accepted = total - fallbacks
    fallback_rate = (fallbacks / total * 100) if total > 0 else 0
    
    plt.clear_figure()
    plt.bar(["Accepted", "Fallback"], [accepted, fallbacks])
    plt.title(f"Prediction Distribution (Fallback Rate: {fallback_rate:.1f}%)")
    plt.xlabel("Status")
    plt.ylabel("Count")
    plt.plotsize(60, 15)
    plt.show()


def display_class_distribution(predictions: Dict[str, int]):
    """
    Display class distribution as a table.
    
    Args:
        predictions: Dictionary of {class_name: count}
    """
    total = sum(predictions.values())
    
    table = Table(title="Prediction Distribution by Class")
    table.add_column("Class", style="cyan")
    table.add_column("Count", justify="right", style="green")
    table.add_column("Percentage", justify="right", style="yellow")
    table.add_column("Bar", style="blue")
    
    for class_name, count in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total * 100) if total > 0 else 0
        bar = "█" * int(percentage / 2)
        table.add_row(
            class_name,
            str(count),
            f"{percentage:.1f}%",
            bar
        )
    
    console.print(table)


def create_summary_report(stats: Dict):
    """
    Create a comprehensive summary report.
    
    Args:
        stats: Statistics dictionary
    """
    console.print("\n" + "=" * 80)
    console.print("[bold cyan]SESSION SUMMARY REPORT[/bold cyan]")
    console.print("=" * 80 + "\n")
    
    # Basic stats table
    stats_table = Table(show_header=False, box=None)
    stats_table.add_column("Metric", style="cyan", width=30)
    stats_table.add_column("Value", style="green", justify="right")
    
    stats_table.add_row("Total Predictions", str(stats.get('total_predictions', 0)))
    stats_table.add_row("Fallback Activations", str(stats.get('fallback_activations', 0)))
    stats_table.add_row("User Corrections", str(stats.get('user_corrections', 0)))
    stats_table.add_row("Direct Acceptances", str(stats.get('correct_predictions', 0)))
    stats_table.add_row("Fallback Rate", stats.get('fallback_rate', '0%'))
    stats_table.add_row("Session Duration", stats.get('session_duration', '0:00:00'))
    
    console.print(stats_table)
    console.print()