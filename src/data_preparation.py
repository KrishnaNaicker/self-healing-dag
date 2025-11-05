"""
ULTIMATE Data Preparation - Handles ALL edge cases like a human.

Categories covered:
1. Edge Cases (emojis, keyboard smash, empty, punctuation, repetitions)
2. Subtle/Tricky (understated, indirect expressions)
3. Mixed Emotions (conflicting feelings)
4. Sarcasm/Irony (opposite of literal meaning)
5. Neutral/Ambiguous (unclear, neutral statements)

Uses: GoEmotions (real-world) + Custom synthetic examples
"""

import yaml
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from collections import Counter
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
import pandas as pd
import re

console = Console()


class UltimateDataPreparation:
    """
    Comprehensive data preparation that teaches the model to think like a human.
    Handles ALL edge cases, nuances, and complex emotional expressions.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.output_dir = Path("data/processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Comprehensive label set (7 core emotions + neutral)
        self.labels = [
            'joy',          # Happy, excited, delighted
            'sadness',      # Sad, disappointed, down
            'anger',        # Mad, frustrated, annoyed
            'fear',         # Scared, anxious, worried
            'love',         # Affection, care, adoration
            'surprise',     # Shocked, amazed, unexpected
            'neutral',      # Unclear, ambiguous, factual
            'disgust'       # Added for completeness (from GoEmotions)
        ]
    
    # ==================== CATEGORY 1: EDGE CASES ====================
    
    def create_edge_case_examples(self) -> Dataset:
        """
        Handle edge cases: emojis, keyboard smash, empty, punctuation, repetitions.
        Teaches model to extract emotion from non-standard text.
        """
        console.print("[yellow]Creating edge case examples...[/yellow]")
        
        examples = []
        
        # 1. Emoji-only expressions (map to emotions)
        emoji_mappings = [
            # Joy
            ("😊😊😊", "joy"),
            ("😁😁", "joy"),
            ("🎉🎉🎊", "joy"),
            ("❤️😍", "joy"),
            ("🥳🎈", "joy"),
            ("😄😃😀", "joy"),
            ("👍👍👍", "joy"),
            ("🙌🙌", "joy"),
            
            # Sadness
            ("😢😢😭", "sadness"),
            ("💔", "sadness"),
            ("😞😔", "sadness"),
            ("😿😭", "sadness"),
            ("🥺", "sadness"),
            
            # Anger
            ("😠😡", "anger"),
            ("🤬😤", "anger"),
            ("💢", "anger"),
            ("😾", "anger"),
            
            # Fear
            ("😨😱", "fear"),
            ("😰😓", "fear"),
            ("🙀", "fear"),
            
            # Love
            ("❤️💕💖", "love"),
            ("😍🥰", "love"),
            ("💑💏", "love"),
            ("💘💝", "love"),
            
            # Surprise
            ("😲😮", "surprise"),
            ("🤯", "surprise"),
            ("😳", "surprise"),
            
            # Neutral/Ambiguous
            ("🤔", "neutral"),
            ("😐", "neutral"),
            ("😶", "neutral"),
        ]
        
        # Add variations
        for emoji_text, emotion in emoji_mappings:
            examples.append({'text': emoji_text, 'label': emotion})
            # Add with spaces
            examples.append({'text': ' '.join(emoji_text), 'label': emotion})
        
        # 2. Keyboard smash (usually excitement or frustration)
        keyboard_smash_joy = [
            "asdfghjkl", "YESSSS", "AHHHHH", "omgomgomg", "skdjfhskdjfh",
            "jkdsfhjksdhf", "WOOOOO", "yayayayay", "woohooo"
        ]
        for text in keyboard_smash_joy:
            examples.append({'text': text, 'label': 'joy'})
            examples.append({'text': text.upper(), 'label': 'joy'})
        
        keyboard_smash_anger = [
            "ARGHHHHH", "grrrrr", "ughhhhhh", "SERIOUSLY",
            "jdkfhjsdhfkjshdf", "WHYYYY"
        ]
        for text in keyboard_smash_anger:
            examples.append({'text': text, 'label': 'anger'})
        
        # 3. Punctuation-heavy (emotion intensity markers)
        punctuation_examples = [
            ("!!!!!!", "joy"),
            ("?!?!?!", "surprise"),
            ("......", "sadness"),
            ("!!!!", "joy"),
            ("????", "neutral"),  # Confusion
            ("...", "neutral"),
            ("!?", "surprise"),
        ]
        for text, emotion in punctuation_examples:
            examples.append({'text': text, 'label': emotion})
        
        # 4. Word repetitions (emphasis)
        repetition_examples = [
            ("no no no no no", "anger"),
            ("yes yes yes yes", "joy"),
            ("ha ha ha ha", "joy"),
            ("please please please", "fear"),  # Desperation
            ("why why why", "sadness"),
            ("really really really", "surprise"),
            ("okay okay okay", "neutral"),
            ("the the the the", "neutral"),  # Nonsense
        ]
        for text, emotion in repetition_examples:
            examples.append({'text': text, 'label': emotion})
            examples.append({'text': text.upper(), 'label': emotion})
        
        # 5. Very short/minimal text
        minimal_examples = [
            ("k", "neutral"),
            ("ok", "neutral"),
            ("lol", "joy"),
            ("haha", "joy"),
            ("ugh", "anger"),
            ("meh", "neutral"),
            ("yay", "joy"),
            ("nah", "neutral"),
            ("sigh", "sadness"),
            ("wow", "surprise"),
            ("aww", "love"),
            ("eww", "disgust"),
        ]
        for text, emotion in minimal_examples:
            examples.append({'text': text, 'label': emotion})
        
        console.print(f"[green]✅ Created {len(examples)} edge case examples[/green]")
        return Dataset.from_pandas(pd.DataFrame(examples))
    
    # ==================== CATEGORY 2: SUBTLE/TRICKY ====================
    
    def create_subtle_examples(self) -> Dataset:
        """
        Subtle, understated, indirect expressions.
        Requires reading between the lines.
        """
        console.print("[yellow]Creating subtle/tricky examples...[/yellow]")
        
        examples = [
            # Understated sadness
            ("I'm fine", "neutral"),  # Could be sadness, but ambiguous
            ("It's whatever", "neutral"),
            ("I guess I'll manage", "sadness"),
            ("Not my best day", "sadness"),
            ("Could be better", "sadness"),
            ("I've had worse", "neutral"),
            ("I'll survive", "sadness"),
            
            # Understated joy
            ("Not too shabby", "joy"),
            ("I'm not complaining", "joy"),
            ("Could be worse", "neutral"),
            ("That's nice I suppose", "joy"),
            
            # Understated anger
            ("I'm not thrilled about this", "anger"),
            ("That's... interesting", "neutral"),  # Depends on context
            ("How nice for you", "anger"),  # Passive aggressive
            ("Good for you", "anger"),  # Often sarcastic
            ("Whatever you say", "anger"),
            ("If you say so", "neutral"),
            
            # Passive aggressive (anger masked)
            ("I'm so happy you could make it", "anger"),  # Late arrival
            ("Thanks for letting me know", "anger"),  # Should've told earlier
            ("That's one way to do it", "anger"),  # Wrong way
            ("Interesting choice", "neutral"),
            
            # Indirect expressions
            ("I see what you did there", "neutral"),
            ("That's something", "neutral"),
            ("Noted", "neutral"),
            ("Understood", "neutral"),
        ]
        
        console.print(f"[green]✅ Created {len(examples)} subtle examples[/green]")
        return Dataset.from_pandas(pd.DataFrame(examples, columns=['text', 'label']))
    
    # ==================== CATEGORY 3: MIXED EMOTIONS ====================
    
    def create_mixed_emotion_examples(self) -> Dataset:
        """
        Conflicting emotions in same sentence.
        Model learns to pick dominant emotion.
        """
        console.print("[yellow]Creating mixed emotion examples...[/yellow]")
        
        examples = [
            # Joy + Sadness
            ("I'm happy for you but I'll miss you", "sadness"),  # Dominant
            ("Congratulations, wish I could be there", "sadness"),
            ("Great news, though I'm a bit jealous", "joy"),  # Starts with joy
            
            # Love + Fear
            ("I love you but I'm scared", "fear"),
            ("I'm falling for you and it terrifies me", "love"),  # Vulnerable admission
            ("You make me so happy it scares me", "love"),
            
            # Joy + Anger
            ("I'm thrilled you got the job I applied for", "anger"),  # Sarcastic
            ("So happy to work overtime again", "anger"),  # Sarcasm
            
            # Joy + Worry
            ("I'm excited but nervous", "joy"),  # Excitement wins
            ("Can't wait but also terrified", "fear"),
            ("Looking forward to it with some anxiety", "joy"),
            
            # Gratitude + Disappointment
            ("Thanks for trying but it's not what I wanted", "sadness"),
            ("I appreciate it but I don't like it", "sadness"),
            ("The effort was nice but the result isn't", "sadness"),
            
            # Complex combinations
            ("I'm proud of you but wish it was me", "sadness"),
            ("Hilarious yet deeply concerning", "joy"),  # Dark humor
            ("Fascinating and terrifying at the same time", "fear"),
            ("I love this but hate that it's ending", "love"),
        ]
        
        console.print(f"[green]✅ Created {len(examples)} mixed emotion examples[/green]")
        return Dataset.from_pandas(pd.DataFrame(examples, columns=['text', 'label']))
    
    # ==================== CATEGORY 4: SARCASM/IRONY ====================
    
    def create_sarcasm_examples(self) -> Dataset:
        """
        Sarcastic/ironic statements - opposite of literal meaning.
        Hardest for models to learn.
        """
        console.print("[yellow]Creating sarcasm/irony examples...[/yellow]")
        
        examples = [
            # Sarcastic joy (actually anger/sadness)
            ("Oh great, another Monday. Just perfect.", "anger"),
            ("Yeah, I totally love being stuck in traffic", "anger"),
            ("Wonderful, my phone just died. Best day ever!", "anger"),
            ("Oh joy, more homework", "anger"),
            ("Fantastic, it's raining on my wedding day", "sadness"),
            ("Brilliant, I locked my keys in the car", "anger"),
            ("Perfect timing, as always", "anger"),
            ("Well this is fun", "anger"),  # Context: bad situation
            ("Living the dream", "sadness"),  # Context: struggling
            ("Just what I needed", "anger"),  # Context: problem
            
            # Ironic statements
            ("Sure, because that's exactly what I wanted", "anger"),
            ("Right, because that makes total sense", "anger"),
            ("Obviously the best idea ever", "anger"),
            ("Clearly someone thought this through", "anger"),
            
            # Dry humor / Deadpan sarcasm
            ("I'm thrilled beyond words", "anger"),  # Monotone context
            ("This is the highlight of my week", "sadness"),
            ("I can barely contain my excitement", "anger"),
            ("What a delightful surprise", "anger"),
            
            # Markers that help detect sarcasm
            ("Oh sure, that'll work", "anger"),
            ("Yeah right, good luck with that", "anger"),
            ("Totally, absolutely, definitely", "anger"),  # Over-emphasis
        ]
        
        console.print(f"[green]✅ Created {len(examples)} sarcasm examples[/green]")
        return Dataset.from_pandas(pd.DataFrame(examples, columns=['text', 'label']))
    
    # ==================== CATEGORY 5: NEUTRAL/AMBIGUOUS ====================
    
    def create_neutral_examples(self) -> Dataset:
        """
        Truly neutral, ambiguous, or factual statements.
        No clear emotion.
        """
        console.print("[yellow]Creating neutral/ambiguous examples...[/yellow]")
        
        examples = [
            # Lukewarm/ambiguous
            ("The movie was okay I guess", "neutral"),
            ("It's fine, nothing special", "neutral"),
            ("Meh, could be worse", "neutral"),
            ("It was alright", "neutral"),
            ("Not bad, not great", "neutral"),
            ("I have mixed feelings about this", "neutral"),
            ("I'm not sure how I feel", "neutral"),
            ("It's whatever", "neutral"),
            ("I don't really care", "neutral"),
            ("Doesn't matter to me", "neutral"),
            ("I'm indifferent", "neutral"),
            ("No strong feelings either way", "neutral"),
            ("I can take it or leave it", "neutral"),
            ("It's okay", "neutral"),
            ("Seems fine", "neutral"),
            
            # Factual statements (no emotion)
            ("I went to the store today", "neutral"),
            ("The meeting is at 3pm", "neutral"),
            ("It's raining outside", "neutral"),
            ("I ate lunch", "neutral"),
            ("The book has 300 pages", "neutral"),
            ("The color is blue", "neutral"),
            ("I'm working on a project", "neutral"),
            ("The temperature is 70 degrees", "neutral"),
            ("I read the article", "neutral"),
            ("The event starts tomorrow", "neutral"),
            ("Water is wet", "neutral"),
            ("The sky is blue", "neutral"),
            ("Today is Tuesday", "neutral"),
            ("I have a meeting", "neutral"),
            ("The report is due Friday", "neutral"),
            
            # Uncertain/Can't decide
            ("I don't know", "neutral"),
            ("Maybe", "neutral"),
            ("I guess so", "neutral"),
            ("Not sure", "neutral"),
            ("Could be", "neutral"),
            ("Possibly", "neutral"),
            ("We'll see", "neutral"),
            ("Hard to say", "neutral"),
            ("Depends", "neutral"),
            ("I'm on the fence", "neutral"),
        ]
        
        console.print(f"[green]✅ Created {len(examples)} neutral examples[/green]")
        return Dataset.from_pandas(pd.DataFrame(examples, columns=['text', 'label']))
    
    # ==================== NEGATION AUGMENTATION ====================
    
    def create_negation_examples(self, base_dataset: Dataset, num_examples: int = 1000) -> Dataset:
        """
        Teach model negation: "not happy" != "happy"
        Critical for understanding nuance.
        """
        console.print("[yellow]Creating negation-augmented examples...[/yellow]")
        
        negation_templates = [
            "I'm not {emotion}",
            "I don't feel {emotion}",
            "This is not {emotion}",
            "I'm definitely not {emotion}",
            "I wouldn't say I'm {emotion}",
            "Not feeling {emotion}",
            "Definitely not {emotion}",
            "Far from {emotion}",
        ]
        
        # Emotion opposites (for negation)
        opposites = {
            'joy': 'sadness',
            'sadness': 'joy',
            'anger': 'joy',
            'fear': 'joy',
            'love': 'neutral',
            'surprise': 'neutral',
            'neutral': 'neutral',
            'disgust': 'joy'
        }
        
        # Adjective forms
        emotion_adjectives = {
            'joy': ['happy', 'joyful', 'excited', 'thrilled', 'delighted'],
            'sadness': ['sad', 'unhappy', 'down', 'depressed', 'miserable'],
            'anger': ['angry', 'mad', 'furious', 'annoyed', 'irritated'],
            'fear': ['scared', 'afraid', 'terrified', 'anxious', 'worried'],
            'love': ['in love', 'loving', 'affectionate'],
            'surprise': ['surprised', 'shocked', 'amazed'],
            'neutral': ['neutral', 'indifferent'],
            'disgust': ['disgusted', 'repulsed']
        }
        
        examples = []
        sample_size = min(num_examples, len(base_dataset))
        
        for idx in range(sample_size):
            example = base_dataset[idx % len(base_dataset)]
            original_label = example['label']
            
            # Get opposite emotion
            opposite = opposites.get(original_label, 'neutral')
            
            # Get adjectives for original emotion
            adjectives = emotion_adjectives.get(original_label, [original_label])
            adjective = adjectives[idx % len(adjectives)]
            
            # Generate negated text
            template = negation_templates[idx % len(negation_templates)]
            negated_text = template.format(emotion=adjective)
            
            examples.append({
                'text': negated_text,
                'label': opposite
            })
        
        console.print(f"[green]✅ Created {len(examples)} negation examples[/green]")
        return Dataset.from_pandas(pd.DataFrame(examples))
    
    # ==================== GOEMOTIONS INTEGRATION ====================
    
    def load_goemotions(self) -> DatasetDict:
        """
        Load GoEmotions dataset (real-world Reddit comments).
        58K examples, 27 emotions, includes neutral.
        """
        console.print("[yellow]Loading GoEmotions (real-world data)...[/yellow]")
        
        try:
            # Load GoEmotions
            goemotions = load_dataset("go_emotions", "simplified", trust_remote_code=True)
            
            # GoEmotions label names (in order)
            goemotions_labels = [
                'admiration', 'amusement', 'anger', 'annoyance', 
                'approval', 'caring', 'confusion', 'curiosity', 
                'desire', 'disappointment', 'disapproval', 'disgust',
                'embarrassment', 'excitement', 'fear', 'gratitude',
                'grief', 'joy', 'love', 'nervousness', 'optimism',
                'pride', 'realization', 'relief', 'remorse', 'sadness',
                'surprise', 'neutral'
            ]
            
            # Map GoEmotions to our 8 classes
            label_mapping = {
                'admiration': 'joy',
                'amusement': 'joy',
                'anger': 'anger',
                'annoyance': 'anger',
                'approval': 'joy',
                'caring': 'love',
                'confusion': 'neutral',
                'curiosity': 'surprise',
                'desire': 'love',
                'disappointment': 'sadness',
                'disapproval': 'anger',
                'disgust': 'disgust',
                'embarrassment': 'sadness',
                'excitement': 'joy',
                'fear': 'fear',
                'gratitude': 'joy',
                'grief': 'sadness',
                'joy': 'joy',
                'love': 'love',
                'nervousness': 'fear',
                'optimism': 'joy',
                'pride': 'joy',
                'realization': 'surprise',
                'relief': 'joy',
                'remorse': 'sadness',
                'sadness': 'sadness',
                'surprise': 'surprise',
                'neutral': 'neutral'
            }
            
            def map_goemotions_label(example):
                """Map GoEmotions to our 8 classes."""
                # GoEmotions can have multiple labels, take the first one
                if isinstance(example['labels'], list) and len(example['labels']) > 0:
                    original_label_id = example['labels'][0]
                else:
                    original_label_id = 27  # default to neutral
                
                # Convert ID to label name
                if original_label_id < len(goemotions_labels):
                    original_label = goemotions_labels[original_label_id]
                    mapped_label = label_mapping[original_label]
                else:
                    mapped_label = 'neutral'
                
                return {
                    'text': example['text'],
                    'label': mapped_label
                }
            
            # Apply mapping and remove old columns
            goemotions = goemotions.map(
                map_goemotions_label,
                remove_columns=goemotions['train'].column_names
            )
            
            console.print(f"[green]✅ Loaded GoEmotions: {len(goemotions['train'])} train examples[/green]")
            return goemotions
            
        except Exception as e:
            console.print(f"[yellow]⚠️  Couldn't load GoEmotions: {e}[/yellow]")
            console.print("[yellow]Continuing without GoEmotions...[/yellow]")
            return None
    
    # ==================== MAIN PIPELINE ====================
    
    def merge_all_datasets(self):
        """Merge ALL datasets into comprehensive training set."""
        console.print("\n[bold cyan]🔥 CREATING ULTIMATE DATASET[/bold cyan]\n")
        
        all_datasets = []
        
        # 1. Base emotion dataset
        # 1. Base emotion dataset
        console.print("[cyan]1. Loading base emotion dataset...[/cyan]")
        base_emotion = load_dataset('emotion', trust_remote_code=True)
        label_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

        def map_base_labels(example):
            # Convert numeric label to string
            return {
                'text': example['text'],
                'label': label_names[example['label']]
            }

        # Apply mapping (this replaces the label column)
        base_emotion = base_emotion.map(
            map_base_labels,
            remove_columns=base_emotion['train'].column_names  # Remove ALL old columns
        )

        # Cast to plain features (no ClassLabel constraint)
        from datasets import Features, Value
        new_features = Features({
            'text': Value('string'),
            'label': Value('string')
        })

        base_emotion = base_emotion.cast(new_features)

        console.print(f"[green]✅ Base emotion: {len(base_emotion['train'])} examples[/green]")
        
        # 2. GoEmotions (real-world)
        console.print("\n[cyan]2. Loading GoEmotions (real-world data)...[/cyan]")
        goemotions = self.load_goemotions()
        if goemotions:
            goemotions = goemotions.cast(new_features)
        
        # 3. Edge cases
        console.print("\n[cyan]3. Creating edge case examples...[/cyan]")
        edge_cases = self.create_edge_case_examples()
        
        # 4. Subtle examples
        console.print("\n[cyan]4. Creating subtle/tricky examples...[/cyan]")
        subtle = self.create_subtle_examples()
        
        # 5. Mixed emotions
        console.print("\n[cyan]5. Creating mixed emotion examples...[/cyan]")
        mixed = self.create_mixed_emotion_examples()
        
        # 6. Sarcasm
        console.print("\n[cyan]6. Creating sarcasm/irony examples...[/cyan]")
        sarcasm = self.create_sarcasm_examples()
        
        # 7. Neutral
        console.print("\n[cyan]7. Creating neutral/ambiguous examples...[/cyan]")
        neutral = self.create_neutral_examples()
        
        # 8. Negation augmentation
        console.print("\n[cyan]8. Creating negation-augmented examples...[/cyan]")
        negation = self.create_negation_examples(base_emotion['train'], 1500)
        
        # Combine all
        console.print("\n[yellow]Merging all datasets...[/yellow]")
        
        # Split custom datasets into train/val/test (80/10/10)
        def split_dataset(ds, train_pct=0.8, val_pct=0.1):
            total = len(ds)
            train_size = int(total * train_pct)
            val_size = int(total * val_pct)
            
            # Get raw data
            data = ds.to_dict() if hasattr(ds, 'to_dict') else {'text': ds['text'], 'label': ds['label']}
            
            train_data = {k: v[:train_size] for k, v in data.items()}
            val_data = {k: v[train_size:train_size+val_size] for k, v in data.items()}
            test_data = {k: v[train_size+val_size:] for k, v in data.items()}
            
            return {
                'train': Dataset.from_dict(train_data, features=new_features),
                'val': Dataset.from_dict(val_data, features=new_features),
                'test': Dataset.from_dict(test_data, features=new_features)
            }
        
        edge_split = split_dataset(edge_cases)
        subtle_split = split_dataset(subtle)
        mixed_split = split_dataset(mixed)
        sarcasm_split = split_dataset(sarcasm)
        neutral_split = split_dataset(neutral)
        negation_split = split_dataset(negation)
        
        # Merge train sets
        train_datasets = [
            base_emotion['train'],
            edge_split['train'],
            subtle_split['train'],
            mixed_split['train'],
            sarcasm_split['train'],
            neutral_split['train'],
            negation_split['train']
        ]
        
        # Add GoEmotions if available (sample 10k to balance)
        if goemotions:
            sample_size = min(10000, len(goemotions['train']))
            goemotions_sample = goemotions['train'].shuffle(seed=42).select(range(sample_size))
            train_datasets.append(goemotions_sample)
        
        train_combined = concatenate_datasets(train_datasets).shuffle(seed=42)
        
        # Merge val sets
        val_combined = concatenate_datasets([
            base_emotion['validation'],
            edge_split['val'],
            subtle_split['val'],
            mixed_split['val'],
            sarcasm_split['val'],
            neutral_split['val'],
            negation_split['val']
        ]).shuffle(seed=42)
        
        # Merge test sets
        test_combined = concatenate_datasets([
            base_emotion['test'],
            edge_split['test'],
            subtle_split['test'],
            mixed_split['test'],
            sarcasm_split['test'],
            neutral_split['test'],
            negation_split['test']
        ]).shuffle(seed=42)
        
        final_dataset = DatasetDict({
            'train': train_combined,
            'validation': val_combined,
            'test': test_combined
        })
        
        console.print(f"\n[bold green]✅ ULTIMATE DATASET CREATED![/bold green]")
        console.print(f"   Train: {len(train_combined):,} examples")
        console.print(f"   Val: {len(val_combined):,} examples")
        console.print(f"   Test: {len(test_combined):,} examples")
        console.print(f"   TOTAL: {len(train_combined) + len(val_combined) + len(test_combined):,} examples\n")
        
        return final_dataset
    
    def analyze_and_save(self, dataset: DatasetDict):
        """Analyze dataset and save with label mappings."""
        console.print("\n[bold cyan]📊 Dataset Analysis[/bold cyan]\n")
        
        # First, validate and clean the labels
        console.print("[yellow]Validating labels...[/yellow]")
        
        def validate_and_fix_label(example):
            """Ensure label is one of our valid emotions."""
            label = example['label']
            
            # If it's a string number, it's invalid - map to neutral
            if isinstance(label, str) and label.isdigit():
                console.print(f"[red]Found invalid label '{label}', mapping to 'neutral'[/red]")
                return {'text': example['text'], 'label': 'neutral'}
            
            # If it's not in our valid labels, map to neutral
            if label not in self.labels:
                console.print(f"[red]Found unknown label '{label}', mapping to 'neutral'[/red]")
                return {'text': example['text'], 'label': 'neutral'}
            
            return example
        
        # Clean all splits
        dataset = dataset.map(validate_and_fix_label)
        
        console.print("[green]✅ Labels validated and cleaned[/green]\n")
        
        # Class distribution
        train_labels = [ex['label'] for ex in dataset['train']]
        label_counts = Counter(train_labels)
        
        table = Table(title="Class Distribution (Training Set)")
        table.add_column("Emotion", style="cyan", justify="left")
        table.add_column("Count", justify="right", style="green")
        table.add_column("Percentage", justify="right", style="yellow")
        table.add_column("Bar", style="blue")
        
        total = len(train_labels)
        max_count = max(label_counts.values())
        
        for label in self.labels:
            count = label_counts.get(label, 0)
            percentage = (count / total) * 100
            bar_length = int((count / max_count) * 30)
            bar = "█" * bar_length
            
            table.add_row(
                label.capitalize(),
                f"{count:,}",
                f"{percentage:.1f}%",
                bar
            )
        
        console.print(table)
        
        # Check for any labels not in our list
        unexpected_labels = set(train_labels) - set(self.labels)
        if unexpected_labels:
            console.print(f"\n[bold red]⚠️  WARNING: Found unexpected labels: {unexpected_labels}[/bold red]")
            console.print("[yellow]These will be filtered out...[/yellow]\n")
        
        # Create label mappings
        label_to_id = {label: idx for idx, label in enumerate(self.labels)}
        id_to_label = {idx: label for label, idx in label_to_id.items()}
        
        # Convert string labels to IDs
        def encode_labels(example):
            label = example['label']
            if label in label_to_id:
                example['label'] = label_to_id[label]
            else:
                # Fallback to neutral if somehow still invalid
                example['label'] = label_to_id['neutral']
            return example
        
        dataset = dataset.map(encode_labels)
        
        # Save label mapping
        mapping = {
            'label_to_id': label_to_id,
            'id_to_label': id_to_label,
            'num_labels': len(self.labels),
            'labels': self.labels
        }
        
        mapping_path = self.output_dir / "label_mapping.json"
        with open(mapping_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        console.print(f"\n[green]✅ Label mapping saved to {mapping_path}[/green]")
        
        # Save dataset
        dataset_path = self.output_dir / "emotion_dataset_ultimate"
        dataset.save_to_disk(str(dataset_path))
        
        console.print(f"[green]✅ Dataset saved to {dataset_path}[/green]")
        
        # Save summary
        summary = {
            'total_examples': len(dataset['train']) + len(dataset['validation']) + len(dataset['test']),
            'train_examples': len(dataset['train']),
            'val_examples': len(dataset['validation']),
            'test_examples': len(dataset['test']),
            'num_labels': len(self.labels),
            'labels': self.labels,
            'class_distribution': {label: label_counts.get(label, 0) for label in self.labels},
            'categories_covered': [
                'Edge Cases (emojis, keyboard smash, punctuation)',
                'Subtle/Tricky expressions',
                'Mixed emotions',
                'Sarcasm/Irony',
                'Neutral/Ambiguous',
                'Negation-augmented',
                'Real-world data (GoEmotions)'
            ]
        }
        
        summary_path = self.output_dir / "dataset_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        console.print(f"[green]✅ Summary saved to {summary_path}[/green]")
        
        return dataset
    
    def prepare(self):
        """Main preparation pipeline."""
        console.print("\n[bold green]🚀 ULTIMATE DATA PREPARATION STARTED![/bold green]")
        console.print("[bold cyan]Building dataset that handles ALL edge cases like a human...[/bold cyan]\n")
        
        # Merge all datasets
        dataset = self.merge_all_datasets()
        
        # Analyze and save
        dataset = self.analyze_and_save(dataset)
        
        console.print("\n[bold green]" + "="*80 + "[/bold green]")
        console.print("[bold green]✅ ULTIMATE DATASET READY FOR TRAINING![/bold green]")
        console.print("[bold green]" + "="*80 + "[/bold green]\n")
        
        console.print("[bold cyan]📋 What's Included:[/bold cyan]")
        console.print("  ✅ Edge cases: emojis, keyboard smash, punctuation, repetitions")
        console.print("  ✅ Subtle expressions: understated, passive-aggressive")
        console.print("  ✅ Mixed emotions: conflicting feelings handled")
        console.print("  ✅ Sarcasm/Irony: opposite meanings detected")
        console.print("  ✅ Neutral/Ambiguous: proper neutral class")
        console.print("  ✅ Negation: 'not happy' != 'happy'")
        console.print("  ✅ Real-world data: GoEmotions (Reddit comments)")
        console.print("\n[bold yellow]🎯 This model will think like a HUMAN![/bold yellow]\n")


def main():
    """Run ultimate data preparation."""
    prep = UltimateDataPreparation()
    prep.prepare()


if __name__ == "__main__":
    main()