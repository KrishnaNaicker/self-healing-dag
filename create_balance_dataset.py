"""
Create a properly balanced dataset - FINAL VERSION
Balances train, validation, AND test sets for fair evaluation.
"""

from datasets import load_from_disk, DatasetDict, Dataset
from collections import Counter
import random
import pandas as pd

print("🔧 Creating balanced emotion dataset...\n")

# Set seed for reproducibility
random.seed(42)

# Load original
dataset = load_from_disk("data/processed/emotion_dataset_ultimate")

print("="*80)
print("ORIGINAL DATASET DISTRIBUTION")
print("="*80)

for split_name in ['train', 'validation', 'test']:
    split_labels = [ex['label'] for ex in dataset[split_name]]
    label_counts = Counter(split_labels)
    print(f"\n{split_name.upper()} ({len(dataset[split_name])} samples):")
    for label, count in sorted(label_counts.items()):
        percentage = (count / len(split_labels)) * 100
        print(f"  Label {label}: {count:5d} ({percentage:5.1f}%)")

print("\n" + "="*80)
print("BALANCING PROCESS")
print("="*80)

def balance_by_label(dataset, target_per_class, split_name=""):
    """Balance dataset by downsampling majority and upsampling minority."""
    
    # Group indices by label
    label_indices = {}
    for idx, example in enumerate(dataset):
        label = example['label']
        if label not in label_indices:
            label_indices[label] = []
        label_indices[label].append(idx)
    
    # Balance each class
    balanced_indices = []
    
    print(f"\n{split_name}:")
    for label in sorted(label_indices.keys()):
        indices = label_indices[label]
        num_samples = len(indices)
        
        if num_samples > target_per_class:
            # Downsample (e.g., neutral)
            sampled = random.sample(indices, target_per_class)
            print(f"  Label {label}: Downsampled {num_samples:5d} → {target_per_class}")
        elif num_samples < target_per_class:
            # Upsample (repeat examples)
            repeats = (target_per_class // num_samples) + 1
            sampled = (indices * repeats)[:target_per_class]
            print(f"  Label {label}: Upsampled   {num_samples:5d} → {target_per_class}")
        else:
            # Already balanced
            sampled = indices
            print(f"  Label {label}: Kept        {num_samples:5d} → {target_per_class}")
        
        balanced_indices.extend(sampled)
    
    # Shuffle to mix labels
    random.shuffle(balanced_indices)
    
    return dataset.select(balanced_indices)

# Balance each split
print("\nBalancing TRAIN set (target: 3000 per class)...")
train_balanced = balance_by_label(dataset['train'], target_per_class=3000, split_name="TRAIN")

print("\nBalancing VALIDATION set (target: 300 per class)...")
val_balanced = balance_by_label(dataset['validation'], target_per_class=300, split_name="VALIDATION")

print("\nBalancing TEST set (target: 200 per class)...")
test_balanced = balance_by_label(dataset['test'], target_per_class=200, split_name="TEST")

print("\n" + "="*80)
print("BALANCED DATASET DISTRIBUTION")
print("="*80)

# Verify balancing
balanced_splits = {
    'train': train_balanced,
    'validation': val_balanced,
    'test': test_balanced
}

for split_name, split_data in balanced_splits.items():
    split_labels = [ex['label'] for ex in split_data]
    label_counts = Counter(split_labels)
    print(f"\n{split_name.upper()} ({len(split_data)} samples):")
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        percentage = (count / len(split_labels)) * 100
        print(f"  Label {label}: {count:5d} ({percentage:5.1f}%)")

# Create balanced dataset
balanced_dataset = DatasetDict({
    'train': train_balanced,
    'validation': val_balanced,
    'test': test_balanced
})

# Save
output_path = "data/processed/emotion_dataset_balanced"
print(f"\n{'='*80}")
print(f"SAVING BALANCED DATASET")
print(f"{'='*80}")
print(f"Output path: {output_path}")

balanced_dataset.save_to_disk(output_path)

print(f"\n✅ Balanced dataset saved successfully!")
print(f"\n{'='*80}")
print(f"FINAL SUMMARY")
print(f"{'='*80}")
print(f"Train set:      {len(train_balanced):6,} samples (24,000 = 8 classes × 3,000)")
print(f"Validation set: {len(val_balanced):6,} samples  (2,400 = 8 classes × 300)")
print(f"Test set:       {len(test_balanced):6,} samples  (1,600 = 8 classes × 200)")
print(f"Total:          {len(train_balanced) + len(val_balanced) + len(test_balanced):6,} samples")
print(f"\n🎯 All sets are now perfectly balanced!")
print(f"🚀 Ready for training with fair evaluation!\n")