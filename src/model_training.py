"""
Model Fine-Tuning with LoRA
Efficiently fine-tune DistilBERT on emotion classification.
"""

import os
import json
from pathlib import Path
from typing import Counter, Dict, Optional
import yaml
import torch
import numpy as np
from datasets import load_from_disk, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


class ModelTrainer:
    """Handles model fine-tuning with LoRA."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize trainer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.lora_config_dict = self.config['lora']
        self.training_config = self.config['training']
        self.dataset_config = self.config['dataset']
        
        # Load label mapping (FIXED: handle both naming conventions)
        with open('data/processed/label_mapping.json', 'r') as f:
            label_mapping = json.load(f)
        
        # Handle both 'id2label' and 'id_to_label' naming
        if 'id2label' in label_mapping:
            self.id2label = {int(k): v for k, v in label_mapping['id2label'].items()}
        elif 'id_to_label' in label_mapping:
            self.id2label = {int(k): v for k, v in label_mapping['id_to_label'].items()}
        else:
            raise KeyError("Could not find 'id2label' or 'id_to_label' in label mapping")
        
        if 'label2id' in label_mapping:
            self.label2id = label_mapping['label2id']
        elif 'label_to_id' in label_mapping:
            self.label2id = label_mapping['label_to_id']
        else:
            raise KeyError("Could not find 'label2id' or 'label_to_id' in label mapping")
        
        self.num_labels = len(self.id2label)
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        console.print(Panel.fit(
            f"[bold cyan]Model Trainer Initialized[/bold cyan]\n"
            f"Model: {self.model_config['name']}\n"
            f"Device: {self.device}\n"
            f"Classes: {self.num_labels}\n"
            f"LoRA Rank: {self.lora_config_dict['r']}",
            border_style="cyan"
        ))
    
    def load_dataset(self) -> DatasetDict:
        """Load preprocessed dataset."""
        console.print("\n[yellow]📂 Loading preprocessed dataset...[/yellow]")
        
        # FIXED: Use the correct dataset name from config
        dataset_name = self.dataset_config.get('name', 'emotion_dataset')
        dataset_path = Path("data/processed") / dataset_name
        
        if not dataset_path.exists():
            console.print(f"[red]❌ Dataset not found at {dataset_path}[/red]")
            console.print("[yellow]Looking for alternative paths...[/yellow]")
            
            # Try alternative paths
            alternatives = [
                Path("data/processed/emotion_dataset_ultimate"),
                Path("data/processed/emotion_dataset"),
            ]
            
            for alt_path in alternatives:
                if alt_path.exists():
                    dataset_path = alt_path
                    console.print(f"[green]✅ Found dataset at {dataset_path}[/green]")
                    break
            else:
                raise FileNotFoundError(f"Could not find dataset in any expected location")
        
        dataset = load_from_disk(str(dataset_path))
        
        console.print(f"[green]✅ Dataset loaded from {dataset_path}[/green]")
        console.print(f"[cyan]  Train: {len(dataset['train'])} samples[/cyan]")
        console.print(f"[cyan]  Validation: {len(dataset['validation'])} samples[/cyan]")
        console.print(f"[cyan]  Test: {len(dataset['test'])} samples[/cyan]")
        
        return dataset
    
    def load_tokenizer(self):
        """Load tokenizer."""
        console.print("\n[yellow]🔤 Loading tokenizer...[/yellow]")
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_config['name'])
        
        console.print(f"[green]✅ Tokenizer loaded: {self.model_config['name']}[/green]")
        
        return tokenizer
    
    def tokenize_dataset(self, dataset: DatasetDict, tokenizer):
        """Tokenize the dataset."""
        console.print("\n[yellow]⚙️ Tokenizing dataset...[/yellow]")
        
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=self.model_config['max_length']
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            desc="Tokenizing"
        )
        
        console.print("[green]✅ Tokenization complete![/green]")
        
        return tokenized_dataset
    
    def load_model(self):
        """Load pre-trained model."""
        console.print("\n[yellow]🤖 Loading pre-trained model...[/yellow]")
        
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_config['name'],
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        console.print(f"[green]✅ Model loaded: {self.model_config['name']}[/green]")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        console.print(f"[cyan]📊 Total parameters: {total_params:,}[/cyan]")
        
        return model
    
    def apply_lora(self, model):
        """Apply LoRA adapters to model."""
        console.print("\n[yellow]🔧 Applying LoRA adapters...[/yellow]")
        
        # LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=self.lora_config_dict['r'],
            lora_alpha=self.lora_config_dict['lora_alpha'],
            lora_dropout=self.lora_config_dict['lora_dropout'],
            target_modules=self.lora_config_dict['target_modules'],
            bias=self.lora_config_dict['bias']
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_percentage = 100 * trainable_params / total_params
        
        console.print(f"[green]✅ LoRA adapters applied![/green]")
        console.print(f"[cyan]📊 Trainable parameters: {trainable_params:,} ({trainable_percentage:.2f}%)[/cyan]")
        console.print(f"[cyan]📊 Frozen parameters: {total_params - trainable_params:,}[/cyan]")
        console.print(f"[magenta]🚀 Training {trainable_percentage:.2f}% of parameters![/magenta]")
        
        return model
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(self, model, tokenized_dataset, tokenizer):
        """Train the model."""
        console.print("\n[bold yellow]🏋️ Starting Training...[/bold yellow]")

        # Add class weights to combat remaining imbalance in test set
        from torch import nn
        import torch

        # Calculate class weights for loss function
        print("\n[yellow]⚖️ Calculating class weights...[/yellow]")
        train_labels = [ex['label'] for ex in tokenized_dataset['train']]
        label_counts = Counter(train_labels)

        # Inverse frequency weighting
        total = len(train_labels)
        class_weights = torch.tensor([
            total / (self.num_labels * label_counts.get(i, 1))
            for i in range(self.num_labels)
        ], dtype=torch.float32).to(self.device)

        console.print(f"[cyan]Class weights: {class_weights.tolist()}[/cyan]")

        # Apply to model (this goes BEFORE creating Trainer)
        # Note: We'll pass this to the model's forward pass
        model.config.class_weights = class_weights
        
        # Training arguments
        output_dir = Path(self.training_config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=int(self.training_config['num_epochs']),
            per_device_train_batch_size=int(self.training_config['batch_size']),
            per_device_eval_batch_size=int(self.training_config['batch_size']),
            warmup_steps=int(self.training_config['warmup_steps']),
            learning_rate=float(self.training_config.get('learning_rate', 2e-4)),  # FIXED: added
            logging_steps=int(self.training_config['logging_steps']),
            eval_steps=int(self.training_config['eval_steps']),
            save_steps=int(self.training_config['save_steps']),
            gradient_accumulation_steps=int(self.training_config['gradient_accumulation_steps']),
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            push_to_hub=False,
            report_to="none",  # Disable wandb/tensorboard for simplicity
            fp16=torch.cuda.is_available(),  # Mixed precision if GPU available
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset['validation'],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[]
        )
        
        # Train
        console.print(f"\n[cyan]🎯 Training for {self.training_config['num_epochs']} epochs...[/cyan]")
        console.print(f"[cyan]📦 Batch size: {self.training_config['batch_size']}[/cyan]")
        console.print(f"[cyan]📈 Learning rate: {training_args.learning_rate}[/cyan]\n")
        
        train_result = trainer.train()
        
        console.print("\n[green]✅ Training complete![/green]")
        
        # Save metrics
        metrics = train_result.metrics
        console.print("\n[cyan]📊 Training Metrics:[/cyan]")
        for key, value in metrics.items():
            console.print(f"  {key}: {value:.4f}")
        
        return trainer, train_result
    
    def evaluate(self, trainer, tokenized_dataset):
        """Evaluate on test set."""
        console.print("\n[yellow]📊 Evaluating on test set...[/yellow]")
        
        test_results = trainer.evaluate(tokenized_dataset['test'])
        
        console.print("\n[green]✅ Evaluation complete![/green]")
        
        # Display results
        table = Table(title="Test Set Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")
        
        for key, value in test_results.items():
            if isinstance(value, float):
                table.add_row(key, f"{value:.4f}")
            else:
                table.add_row(key, str(value))
        
        console.print(table)
        
        return test_results
    
    def save_model(self, model, tokenizer):
        """Save the fine-tuned model."""
        console.print("\n[yellow]💾 Saving model...[/yellow]")
        
        save_path = Path(self.training_config['output_dir']) / "final_model"
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA model
        model.save_pretrained(str(save_path))
        tokenizer.save_pretrained(str(save_path))
        
        # Save config
        config_save_path = save_path / "training_config.json"
        with open(config_save_path, 'w') as f:
            json.dump({
                'model_name': self.model_config['name'],
                'num_labels': self.num_labels,
                'id2label': self.id2label,
                'label2id': self.label2id,
                'lora_config': self.lora_config_dict,
                'training_config': self.training_config
            }, f, indent=2)
        
        console.print(f"[green]✅ Model saved to {save_path}[/green]")
        console.print(f"[cyan]📁 Files saved:[/cyan]")
        console.print(f"  • adapter_model.bin (LoRA weights)")
        console.print(f"  • adapter_config.json")
        console.print(f"  • tokenizer files")
        console.print(f"  • training_config.json")
        
        return save_path
    
    def generate_confusion_matrix(self, trainer, tokenized_dataset):
        """Generate and save confusion matrix."""
        console.print("\n[yellow]📈 Generating confusion matrix...[/yellow]")
        
        # Get predictions
        predictions = trainer.predict(tokenized_dataset['test'])
        pred_labels = np.argmax(predictions.predictions, axis=1)
        true_labels = predictions.label_ids
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=[self.id2label[i] for i in range(self.num_labels)],
            yticklabels=[self.id2label[i] for i in range(self.num_labels)]
        )
        plt.title('Confusion Matrix - Test Set')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save
        save_path = Path(self.training_config['output_dir']) / "confusion_matrix.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        console.print(f"[green]✅ Confusion matrix saved to {save_path}[/green]")


def main():
    """Main training pipeline."""
    console.print(Panel.fit(
        "[bold green]Self-Healing Classifier - Model Training[/bold green]\n"
        "[cyan]Fine-tuning DistilBERT with LoRA on Emotion Dataset[/cyan]",
        border_style="green"
    ))
    
    # Initialize trainer
    trainer_obj = ModelTrainer()
    
    # Load dataset
    dataset = trainer_obj.load_dataset()
    
    # Load tokenizer
    tokenizer = trainer_obj.load_tokenizer()
    
    # Tokenize dataset
    tokenized_dataset = trainer_obj.tokenize_dataset(dataset, tokenizer)
    
    # Load model
    model = trainer_obj.load_model()
    
    # Apply LoRA
    model = trainer_obj.apply_lora(model)
    
    # Train
    trainer, train_result = trainer_obj.train(model, tokenized_dataset, tokenizer)
    
    # Evaluate
    test_results = trainer_obj.evaluate(trainer, tokenized_dataset)
    
    # Save model
    save_path = trainer_obj.save_model(model, tokenizer)
    
    # Generate confusion matrix
    trainer_obj.generate_confusion_matrix(trainer, tokenized_dataset)
    
    # Final summary
    console.print("\n" + "=" * 80)
    console.print("[bold green]✅ MODEL TRAINING COMPLETE![/bold green]")
    console.print("=" * 80)
    console.print(f"\n[cyan]Model saved to: {save_path}[/cyan]")
    console.print(f"[cyan]Test Accuracy: {test_results['eval_accuracy']:.4f}[/cyan]")
    console.print(f"[cyan]Test F1 Score: {test_results['eval_f1']:.4f}[/cyan]")
    console.print(f"\n[yellow]➡️ Next Step: Test with stress_test.py![/yellow]\n")


if __name__ == "__main__":
    main()