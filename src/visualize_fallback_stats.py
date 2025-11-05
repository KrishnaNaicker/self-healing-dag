"""
Fallback Statistics Visualization (BONUS FEATURE)
Creates confidence curves and fallback frequency charts.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from datetime import datetime

def load_fallback_logs():
    """Load fallback logs from JSONL file."""
    log_path = Path("logs/fallback_log.jsonl")
    if not log_path.exists():
        print("No fallback logs found!")
        return []
    
    logs = []
    with open(log_path, 'r') as f:
        for line in f:
            logs.append(json.loads(line))
    return logs

def plot_confidence_curves(logs):
    """Plot confidence scores over time."""
    confidences = [log['confidence'] for log in logs]
    
    plt.figure(figsize=(12, 6))
    plt.plot(confidences, marker='o', linestyle='-', alpha=0.6)
    plt.axhline(y=0.80, color='r', linestyle='--', label='Threshold (80%)')
    plt.axhline(y=0.65, color='orange', linestyle='--', label='Backup Trigger (65%)')
    plt.xlabel('Fallback Event #')
    plt.ylabel('Confidence Score')
    plt.title('Confidence Scores Over Multiple Inputs (Bonus Feature)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('logs/confidence_curves.png', dpi=150)
    print("✅ Confidence curves saved to logs/confidence_curves.png")

def plot_fallback_frequency(logs):
    """Plot fallback frequency by method."""
    methods = [log['method'] for log in logs]
    method_counts = Counter(methods)
    
    plt.figure(figsize=(10, 6))
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    plt.bar(method_counts.keys(), method_counts.values(), color=colors)
    plt.xlabel('Fallback Method')
    plt.ylabel('Frequency')
    plt.title('Fallback Frequency by Method (Bonus Feature)')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig('logs/fallback_frequency.png', dpi=150)
    print("✅ Fallback frequency saved to logs/fallback_frequency.png")

def plot_per_emotion_confidence(logs):
    """Plot confidence distribution per emotion."""
    emotion_confidences = {}
    for log in logs:
        emotion = log['primary_prediction']
        conf = log['confidence']
        if emotion not in emotion_confidences:
            emotion_confidences[emotion] = []
        emotion_confidences[emotion].append(conf)
    
    plt.figure(figsize=(12, 6))
    plt.boxplot(emotion_confidences.values(), labels=emotion_confidences.keys())
    plt.axhline(y=0.80, color='r', linestyle='--', alpha=0.5, label='Threshold')
    plt.xlabel('Emotion')
    plt.ylabel('Confidence Score')
    plt.title('Confidence Distribution by Emotion (Bonus Feature)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('logs/emotion_confidence_distribution.png', dpi=150)
    print("✅ Emotion confidence distribution saved to logs/emotion_confidence_distribution.png")

def main():
    """Generate all bonus visualizations."""
    print("\n🎨 Generating Fallback Statistics Visualizations (BONUS)...\n")
    
    logs = load_fallback_logs()
    if not logs:
        print("No data to visualize yet. Run some classifications first!")
        return
    
    print(f"Loaded {len(logs)} fallback events\n")
    
    plot_confidence_curves(logs)
    plot_fallback_frequency(logs)
    plot_per_emotion_confidence(logs)
    
    print("\n✅ All bonus visualizations generated in logs/ folder!")

if __name__ == "__main__":
    main()