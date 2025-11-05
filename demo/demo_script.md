# 🎬 Demo Video Script

**Duration:** 3-4 minutes  
**Author:** Krishna Naicker  
**Date:** November 5, 2025

---

## Scene 1: Introduction (30 seconds)

**[Screen: Project title slide]**

"Hello! I'm Krishna Naicker, and today I'm presenting my Self-Healing Text Classifier - a production-ready emotion classification system with human-in-the-loop fallback mechanisms."

**[Screen: Architecture diagram]**

"Unlike traditional classifiers that blindly output predictions, my system detects when it's uncertain and requests human clarification, ensuring reliability in real-world applications."

---

## Scene 2: System Architecture (45 seconds)

**[Screen: LangGraph DAG visualization]**

"The system uses LangGraph to orchestrate a 3-node pipeline:

1. **Inference Node** - Runs our fine-tuned DistilBERT model with LoRA adapters
2. **Confidence Check Node** - Evaluates if confidence meets our 70% threshold
3. **Fallback Node** - Handles uncertain predictions through user clarification

This conditional workflow ensures we never blindly trust low-confidence predictions."

---

## Scene 3: Model Training (30 seconds)

**[Screen: Training output]**

"For training, I used LoRA - Low-Rank Adaptation - which fine-tunes only 1.3% of the model's parameters. This gives us 99% efficiency while maintaining 91% accuracy on the emotion dataset.

The model was trained for 3 epochs in about 35 minutes, achieving excellent results across all 6 emotion classes."

---

## Scene 4: Live Demo - High Confidence (45 seconds)

**[Screen: CLI running]**

"Let me show you the CLI interface. First, a high-confidence example:"

**[Type:]** `I absolutely love this amazing product!`

**[Explain as it runs:]**
"The model predicts 'love' with 94% confidence. Since this exceeds our 70% threshold, the system accepts it immediately. Notice the probability distribution showing all 6 classes - full transparency!"

---

## Scene 5: Live Demo - Fallback (60 seconds)

**[Screen: CLI]**

"Now, here's where the self-healing kicks in. Let's try an ambiguous input:"

**[Type:]** `The movie was painfully slow and boring.`

**[Explain as it runs:]**
"The model predicts 'sadness' but only with 54% confidence. The system detects this is below our threshold and triggers the fallback mechanism.

It asks me for clarification, showing the top 2 candidates: sadness at 54% and anger at 32%.

I know this was actually expressing anger, so I select option 2."

**[Select: 2]**

"The system corrects the prediction and logs this as a user correction. This is crucial for safety-critical applications where accuracy matters more than automation."

---

## Scene 6: Logging & Statistics (30 seconds)

**[Screen: Log file]**

"Every interaction is logged with timestamps - the initial prediction, confidence scores, fallback triggers, and final decisions. This creates a complete audit trail."

**[Screen: CLI stats command]**

**[Type:]** `stats`

"Here you can see session statistics showing fallback activation rate, user corrections, and overall accuracy improvement from the self-healing mechanism."

---

## Scene 7: Adaptability (20 seconds)

**[Screen: config.yaml]**

"The system is fully universal - it's not just for emotions. By simply changing the dataset in the config file, you can adapt it to sentiment analysis, topic classification, spam detection, or any text classification task."

---

## Scene 8: Conclusion (20 seconds)

**[Screen: Project summary]**

"In summary, this project demonstrates:
- Efficient fine-tuning with LoRA
- Intelligent fallback mechanisms
- Complete logging and transparency
- Universal adaptability

Thank you for watching! The complete code and documentation are available in the repository."

**[End screen with GitHub link]**

---

## Technical Details to Mention

- ✅ DistilBERT base model (66M parameters)
- ✅ LoRA fine-tuning (1.3% trainable)
- ✅ 91% test accuracy
- ✅ LangGraph for workflow orchestration
- ✅ Rich CLI interface
- ✅ Structured logging system
- ✅ 70% confidence threshold
- ✅ Human-in-the-loop design

---

## Demo Tips

1. **Prepare Examples Ahead**: Have 3-4 test sentences ready
2. **Show Both Paths**: High confidence AND fallback
3. **Emphasize Value**: Safety, reliability, audit trails
4. **Keep It Flowing**: Don't pause too long on any screen
5. **End Strong**: Summarize key achievements

---

## B-Roll Suggestions

- Code scrolling through key files
- Training metrics visualization
- Confusion matrix display
- Log file examples
- Architecture diagrams