"""
Stress test script for the self-healing classifier.
Tests edge cases, ambiguous inputs, and model limitations.
"""

from src.graph import create_classifier
from rich.console import Console
from rich.table import Table

console = Console()

# Test cases organized by difficulty
TEST_CASES = {
    "Neutral/Ambiguous": [
        "The movie was okay I guess",
        "It's fine, nothing special",
        "Meh, could be worse",
        "I don't really have an opinion",
        "Whatever, doesn't matter to me"
    ],
    "Sarcasm/Irony": [
        "Oh great, another Monday. Just perfect.",
        "Yeah, I totally love being stuck in traffic",
        "Wonderful, my phone just died. Best day ever!"
    ],
    "Mixed Emotions": [
        "I'm happy for you but also jealous",
        "The gift was thoughtful but I don't like it",
        "I love this but it scares me"
    ],
    "Subtle/Understated": [
        "I'm not thrilled about this",
        "That's... interesting",
        "How nice for you"
    ],
    "Clear Emotions (Should Work)": [
        "I'm absolutely furious and want to scream!",
        "This is the happiest day of my entire life!",
        "I'm terrified and shaking with fear",
        "My heart is broken and I can't stop crying"
    ]
}

def run_stress_test():
    """Run comprehensive stress test."""
    console.print("\n[bold cyan]🔥 STRESS TEST - Self-Healing Classifier[/bold cyan]\n")
    
    # Initialize classifier
    classifier = create_classifier()
    
    results = []
    
    for category, cases in TEST_CASES.items():
        console.print(f"\n[bold yellow]Testing: {category}[/bold yellow]")
        console.print("─" * 80)
        
        for text in cases:
            result = classifier.classify(text)
            
            predicted = result['predicted_label']
            confidence = result['confidence']
            fallback = result.get('fallback_triggered', False)
            
            # Determine if prediction seems correct
            status = "✅" if confidence >= 0.80 else "⚠️" if confidence >= 0.70 else "❌"
            
            console.print(f"\n{status} Input: \"{text}\"")
            console.print(f"   → {predicted} ({confidence:.1%}) | Fallback: {fallback}")
            
            results.append({
                'category': category,
                'text': text,
                'predicted': predicted,
                'confidence': confidence,
                'fallback': fallback
            })
    
    # Summary statistics
    console.print("\n\n[bold cyan]📊 STRESS TEST SUMMARY[/bold cyan]")
    console.print("═" * 80)
    
    table = Table(show_header=True)
    table.add_column("Category", style="cyan")
    table.add_column("Total", justify="right", style="white")
    table.add_column("High Conf (>80%)", justify="right", style="green")
    table.add_column("Med Conf (70-80%)", justify="right", style="yellow")
    table.add_column("Low Conf (<70%)", justify="right", style="red")
    table.add_column("Fallback Rate", justify="right", style="magenta")
    
    for category in TEST_CASES.keys():
        cat_results = [r for r in results if r['category'] == category]
        total = len(cat_results)
        high = len([r for r in cat_results if r['confidence'] >= 0.80])
        med = len([r for r in cat_results if 0.70 <= r['confidence'] < 0.80])
        low = len([r for r in cat_results if r['confidence'] < 0.70])
        fallbacks = len([r for r in cat_results if r['fallback']])
        fallback_rate = f"{fallbacks/total*100:.0f}%" if total > 0 else "0%"
        
        table.add_row(
            category,
            str(total),
            str(high),
            str(med),
            str(low),
            fallback_rate
        )
    
    console.print(table)
    
    # Overall stats
    total_tests = len(results)
    total_fallbacks = len([r for r in results if r['fallback']])
    avg_confidence = sum([r['confidence'] for r in results]) / total_tests
    
    console.print(f"\n[bold]Overall Statistics:[/bold]")
    console.print(f"  Total Tests: {total_tests}")
    console.print(f"  Fallback Activations: {total_fallbacks} ({total_fallbacks/total_tests*100:.1f}%)")
    console.print(f"  Average Confidence: {avg_confidence:.1%}")
    
    # Worst predictions
    console.print(f"\n[bold red]⚠️  Most Uncertain Predictions:[/bold red]")
    worst = sorted(results, key=lambda x: x['confidence'])[:5]
    for r in worst:
        console.print(f"  • \"{r['text']}\" → {r['predicted']} ({r['confidence']:.1%})")
    
    console.print("\n")


if __name__ == "__main__":
    run_stress_test()