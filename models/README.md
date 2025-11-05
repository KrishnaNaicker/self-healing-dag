# Models Directory

⚠️ **This directory is excluded from Git due to large file sizes.**

## Directory Structure (Not in Git)

```
models/
├── fine_tuned/
│   ├── final_model/
│   │   ├── adapter_model.bin          (~5MB)
│   │   ├── adapter_config.json
│   │   ├── config.json
│   │   ├── training_config.json
│   │   ├── tokenizer_config.json
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer.json
│   │   ├── vocab.txt
│   │   └── confusion_matrix.png
│   └── checkpoint-*/                   (training checkpoints)
└── backup/                             (backup models)
```

---

## 🎓 How to Get the Trained Model

### ⭐ **Recommended: Train from Scratch**

Train your own model (takes ~10 minutes on GPU):

```bash
# 1. Prepare dataset
python src/data_preparation.py

# 2. Train model with LoRA
python src/model_training.py
```

**This will create:**
- `models/fine_tuned/final_model/` with all necessary files
- Training takes ~10 min (GPU) or ~45 min (CPU)
- Achieves ~78% accuracy on balanced 8-emotion dataset

---

### 📦 **Training Output**

After training completes, you'll have:

```
models/fine_tuned/final_model/
├── adapter_model.bin          ✅ LoRA weights (~5MB)
├── adapter_config.json        ✅ LoRA configuration
├── config.json                ✅ Model configuration
├── training_config.json       ✅ Training parameters
├── tokenizer_config.json      ✅ Tokenizer settings
├── tokenizer.json             ✅ Tokenizer vocabulary
├── vocab.txt                  ✅ Word vocabulary
├── special_tokens_map.json    ✅ Special tokens
└── confusion_matrix.png       ✅ Performance visualization
```

---

## 📋 Training Requirements

### System Requirements

- **GPU (Recommended):** NVIDIA GPU with 4GB+ VRAM (e.g., GTX 1650, RTX 3060)
- **CPU (Alternative):** 8GB+ RAM, 4+ cores
- **Storage:** 10GB free space
- **Time:** ~10 minutes (GPU) / ~45 minutes (CPU)

### Software Requirements

```bash
# Install dependencies
pip install -r requirements.txt

# Key packages:
# - torch >= 2.0.0
# - transformers >= 4.35.0
# - peft >= 0.7.0
# - datasets >= 2.14.0
```

---

## 🚀 Quick Start

### Step 1: Prepare Dataset

```bash
python src/data_preparation.py
```

**What it does:**
- Downloads base emotion datasets
- Creates balanced 8-emotion dataset (24,000 training examples)
- Adds edge cases (emojis, sarcasm, mixed emotions)
- Saves to `data/processed/emotion_dataset_balanced/`

**Expected output:**
```
✅ Dataset created successfully!
Train: 24,000 examples (3,000 per emotion)
Val: 2,400 examples (300 per emotion)
Test: 1,600 examples (200 per emotion)
```

---

### Step 2: Train Model

```bash
python src/model_training.py
```

**What it does:**
- Loads DistilBERT base model
- Applies LoRA adapters (only 2% parameters trained)
- Fine-tunes on 8 emotions for 3 epochs
- Saves model to `models/fine_tuned/final_model/`

**Expected output:**
```
🏋️ Starting Training...
Epoch 1/3: 100%|████████| 1500/1500 [06:23<00:00, 4.76it/s]
Epoch 2/3: 100%|████████| 1500/1500 [06:21<00:00, 4.78it/s]
Epoch 3/3: 100%|████████| 1500/1500 [06:19<00:00, 4.80it/s]

✅ Training complete!

📊 Test Set Results:
┌─────────────────┬─────────┐
│ Metric          │ Value   │
├─────────────────┼─────────┤
│ eval_accuracy   │ 0.7850  │
│ eval_f1         │ 0.7829  │
└─────────────────┴─────────┘

✅ Model saved to models/fine_tuned/final_model
```

---

### Step 3: Test the Model

```bash
python src/cli.py
```

**Try these examples:**
```
➤: I'm so happy today!
→ joy (92.45%)

➤: Oh great, another Monday. Just perfect.
→ anger (sarcasm detected via backup model)

➤: 😀😊❤️
→ joy (emoji preprocessing)
```

---

## 🎯 Model Details

### Architecture

```
Base Model: DistilBERT (distilbert-base-uncased)
├── Total Parameters: 66M
├── LoRA Adapters:
│   ├── Rank (r): 8
│   ├── Alpha: 16
│   ├── Dropout: 0.1
│   ├── Target Modules: [q_lin, v_lin]
│   └── Trainable Parameters: ~1.3M (2%)
└── Output: 8 emotion classes
```

### Performance

| Metric | Score |
|--------|-------|
| Test Accuracy | 78.5% |
| Test Precision | 78.2% |
| Test Recall | 78.5% |
| Test F1 Score | 78.3% |
| Training Time | ~10 min (GPU) |
| Inference Speed | ~50 samples/sec |

### Emotions Classified

1. 😊 **Joy** - happiness, excitement, delight
2. 😢 **Sadness** - sorrow, disappointment, grief
3. 😠 **Anger** - frustration, rage, annoyance
4. 😱 **Fear** - anxiety, terror, worry
5. ❤️ **Love** - affection, adoration, care
6. 😮 **Surprise** - shock, amazement, astonishment
7. 😐 **Neutral** - indifference, ambiguity, calm
8. 🤢 **Disgust** - revulsion, contempt, distaste

---

## 💾 File Sizes

| File | Size | Description |
|------|------|-------------|
| `adapter_model.bin` | ~5 MB | LoRA weights (main trainable parameters) |
| `tokenizer.json` | ~2 MB | Tokenizer vocabulary and mappings |
| `vocab.txt` | ~230 KB | Word-level vocabulary |
| `config.json` | ~1 KB | Model configuration |
| Other configs | <10 KB | Various JSON settings |
| **Total** | **~7-8 MB** | Complete model package |

---

## 💻 Usage Example

```python
from src.graph import create_classifier

# Initialize classifier (loads from models/fine_tuned/final_model/)
classifier = create_classifier(enable_backup=True)

# Classify single text
result = classifier.classify("I'm so happy today!")

print(f"Emotion: {result['final_label']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Source: {result['source']}")

# Output:
# Emotion: joy
# Confidence: 92.45%
# Source: model
```

### Batch Processing

```python
texts = [
    "I'm so happy today!",
    "This is terrible.",
    "😀😊",
    "Oh great, just what I needed."
]

for text in texts:
    result = classifier.classify(text)
    print(f"{text} → {result['final_label']}")

# Output:
# I'm so happy today! → joy
# This is terrible. → sadness
# 😀😊 → joy
# Oh great, just what I needed. → anger
```

---

## 🐛 Troubleshooting

### Issue 1: CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solution:** Reduce batch size in `config.yaml`:

```yaml
training:
  batch_size: 8  # Reduce from 16
  gradient_accumulation_steps: 4  # Increase from 2
```

---

### Issue 2: Model Files Not Found

**Error:** `FileNotFoundError: models/fine_tuned/final_model not found`

**Solution:** Train the model first:

```bash
python src/model_training.py
```

---

### Issue 3: Dataset Not Found

**Error:** `FileNotFoundError: data/processed/emotion_dataset_balanced`

**Solution:** Prepare dataset first:

```bash
python src/data_preparation.py
```

---

### Issue 4: Slow Training on CPU

**Solution:** Use smaller batch size and enable mixed precision:

```yaml
training:
  batch_size: 4
  gradient_accumulation_steps: 8
```

---

### Issue 5: Import Errors

**Error:** `ModuleNotFoundError: No module named 'transformers'`

**Solution:** Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔧 Configuration

Edit `config.yaml` to customize training:

```yaml
model:
  name: "distilbert-base-uncased"
  num_labels: 8
  max_length: 128

lora:
  r: 8                    # Rank (complexity)
  lora_alpha: 16          # Scaling factor
  lora_dropout: 0.1
  target_modules: ["q_lin", "v_lin"]

training:
  num_epochs: 3           # Increase for better accuracy
  batch_size: 16          # Adjust based on GPU memory
  learning_rate: 1e-4
  warmup_steps: 100
```

---

## 📊 Training Logs

Training logs are saved to:
- **Console output:** Real-time progress
- **Model directory:** `models/fine_tuned/final_model/training_config.json`
- **Confusion matrix:** `models/fine_tuned/confusion_matrix.png`

---

## 🎓 Learning Resources

- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Parameter-Efficient Fine-Tuning
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108) - Efficient Transformer
- [PEFT Documentation](https://huggingface.co/docs/peft) - Hugging Face PEFT library
- [Transformers Guide](https://huggingface.co/docs/transformers) - Model training guide

---

## 📄 License

Model weights are released under **MIT License**.  
Base model (DistilBERT) is under **Apache 2.0 License**.

---

## 📞 Support

For issues or questions:
1. Check [Troubleshooting](#-troubleshooting) section above
2. Review [main README.md](../README.md) in project root
3. Open an issue on [GitHub](https://github.com/KrishnaNaicker/self-healing-dag/issues)

---

## 🙏 Acknowledgments

- **Hugging Face** for Transformers and PEFT libraries
- **Microsoft Research** for LoRA methodology
- **Google Research** for DistilBERT architecture

---

**Ready to train? Run:** `python src/model_training.py` 🚀