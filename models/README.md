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

## How to Get the Trained Model

### Option 1: Download Pre-trained Model (Recommended)

Download from the [Releases](https://github.com/KrishnaNaicker/self-healing-dag/releases) page:

```bash
# Create models directory
mkdir -p models/fine_tuned/final_model

# Download the model package
wget https://github.com/KrishnaNaicker/self-healing-dag/releases/download/v1.0/final_model.zip

# Extract
unzip final_model.zip -d models/fine_tuned/

# Verify
ls -lh models/fine_tuned/final_model/
```

**Expected files:**
- `adapter_model.bin` (~5MB) - LoRA weights
- `adapter_config.json` - LoRA configuration
- `config.json` - Model configuration
- `training_config.json` - Training parameters
- `tokenizer_config.json` - Tokenizer settings
- `tokenizer.json` - Tokenizer vocabulary
- `vocab.txt` - Word vocabulary
- `special_tokens_map.json` - Special tokens
- `confusion_matrix.png` - Performance visualization

---

### Option 2: Train from Scratch

Train your own model (takes ~10 minutes on GPU):

```bash
# 1. Prepare dataset
python src/data_preparation.py

# 2. Train model with LoRA
python src/model_training.py
```

**Training specifications:**
- Base Model: `distilbert-base-uncased`
- Method: LoRA (Low-Rank Adaptation)
- Trainable Parameters: ~1.3M (2% of total)
- Training Time: ~10 min (GPU) / ~45 min (CPU)
- Output: `models/fine_tuned/final_model/`

---

### Option 3: Use Hugging Face Hub (Future)

```bash
# Download from Hugging Face Hub
python -c "
from transformers import AutoModelForSequenceClassification
from peft import PeftModel

model = AutoModelForSequenceClassification.from_pretrained('KrishnaNaicker/emotion-classifier-lora')
model.save_pretrained('models/fine_tuned/final_model/')
"
```

---

## Model Details

### Architecture

```
DistilBERT (66M parameters)
├── Transformer Layers: 6
├── Hidden Size: 768
├── Attention Heads: 12
└── LoRA Adapters:
    ├── Rank (r): 8
    ├── Alpha: 16
    ├── Target Modules: [q_lin, v_lin]
    └── Trainable: 1.3M parameters (2%)
```

### Performance

| Metric | Score |
|--------|-------|
| Test Accuracy | 78.5% |
| Test F1 Score | 78.3% |
| Training Time | ~10 min (GPU) |
| Inference Speed | ~50 samples/sec |

### Emotions Classified

1. Joy
2. Sadness
3. Anger
4. Fear
5. Love
6. Surprise
7. Neutral
8. Disgust

---

## File Sizes

| File | Size | Description |
|------|------|-------------|
| `adapter_model.bin` | ~5 MB | LoRA weights (main model) |
| `config.json` | ~1 KB | Model configuration |
| `tokenizer.json` | ~2 MB | Tokenizer vocabulary |
| `vocab.txt` | ~230 KB | Word vocabulary |
| Other configs | <10 KB | Various settings |
| **Total** | **~7-8 MB** | Complete model package |

---

## Usage Example

```python
from src.graph import create_classifier

# Initialize classifier (loads from models/fine_tuned/final_model/)
classifier = create_classifier(enable_backup=True)

# Classify text
result = classifier.classify("I'm so happy today!")

print(f"Emotion: {result['final_label']}")
print(f"Confidence: {result['confidence']:.2%}")

# Output:
# Emotion: joy
# Confidence: 92.45%
```

---

## Troubleshooting

### Model Not Found Error

```
FileNotFoundError: models/fine_tuned/final_model not found
```

**Solution:** Download or train the model (see options above)

### Out of Memory During Training

**Solution:** Reduce batch size in `config.yaml`:

```yaml
training:
  batch_size: 8  # Reduce from 16
  gradient_accumulation_steps: 4  # Increase from 2
```

### Slow Inference

**Solution:** Use GPU or reduce max_length:

```yaml
model:
  max_length: 64  # Reduce from 128
```

---

## License

Model weights are released under MIT License.  
Base model (DistilBERT) is under Apache 2.0 License.

---

## Citation

```bibtex
@software{self_healing_emotion_classifier,
  author = {Naicker, Krishna},
  title = {Self-Healing Emotion Classifier with LangGraph},
  year = {2025},
  url = {https://github.com/KrishnaNaicker/self-healing-dag}
}
```

---

**For issues or questions, please open an issue on GitHub.**git add models/README.md