"""
Emoji Preprocessing for Better Emotion Detection
Handles emoji-only inputs and converts them to emotions.
"""

import re
from collections import Counter
from typing import Tuple, Optional

# Comprehensive emoji to emotion mapping
EMOJI_TO_EMOTION = {
    # Joy/Happy
    '😀': 'joy', '😃': 'joy', '😄': 'joy', '😁': 'joy', '😆': 'joy',
    '😂': 'joy', '🤣': 'joy', '😊': 'joy', '😇': 'joy', '🙂': 'joy',
    '🙃': 'joy', '😉': 'joy', '😌': 'joy', '☺️': 'joy', '🤗': 'joy',
    '🥳': 'joy', '🎉': 'joy', '🎊': 'joy', '✨': 'joy', '🌟': 'joy',
    '⭐': 'joy', '💫': 'joy', '🎈': 'joy', '🎆': 'joy', '🎇': 'joy',
    
    # Love
    '🥰': 'love', '😍': 'love', '🤩': 'love', '😘': 'love', '😗': 'love',
    '😙': 'love', '😚': 'love', '❤️': 'love', '💕': 'love', '💖': 'love',
    '💗': 'love', '💘': 'love', '💝': 'love', '💞': 'love', '💓': 'love',
    '💑': 'love', '💏': 'love', '🌹': 'love', '💐': 'love', '🌺': 'love',
    
    # Sadness
    '😢': 'sadness', '😭': 'sadness', '😿': 'sadness', '💔': 'sadness',
    '😞': 'sadness', '😔': 'sadness', '😟': 'sadness', '😕': 'sadness',
    '🙁': 'sadness', '☹️': 'sadness', '😣': 'sadness', '😖': 'sadness',
    '😫': 'sadness', '😩': 'sadness', '🥺': 'sadness', '😪': 'sadness',
    
    # Anger
    '😠': 'anger', '😡': 'anger', '🤬': 'anger', '😤': 'anger',
    '💢': 'anger', '😾': 'anger', '👿': 'anger', '😈': 'anger',
    '💀': 'anger', '☠️': 'anger', '🔥': 'anger',
    
    # Fear
    '😨': 'fear', '😰': 'fear', '😱': 'fear', '😓': 'fear',
    '🙀': 'fear', '😧': 'fear', '😦': 'fear', '😵': 'fear',
    '🥶': 'fear', '😬': 'fear',
    
    # Surprise
    '😮': 'surprise', '😲': 'surprise', '😳': 'surprise', '🤯': 'surprise',
    '😯': 'surprise', '😦': 'surprise', '😧': 'surprise', '🎉': 'surprise',
    
    # Disgust
    '🤢': 'disgust', '🤮': 'disgust', '🤧': 'disgust', '🤒': 'disgust',
    '😷': 'disgust', '🤕': 'disgust', '🤑': 'disgust', '🤥': 'disgust',
    
    # Neutral
    '😐': 'neutral', '😑': 'neutral', '😶': 'neutral', '🤔': 'neutral',
    '🙄': 'neutral', '😏': 'neutral', '😒': 'neutral', '🤨': 'neutral',
}


def extract_emojis(text: str) -> list:
    """Extract all emojis from text."""
    # Unicode ranges for emojis
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA70-\U0001FAFF"  # extended emojis
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.findall(text)


def detect_emoji_emotion(text: str) -> Tuple[bool, Optional[str], float]:
    """
    Detect if text is primarily emojis and return emotion.
    
    Args:
        text: Input text
    
    Returns:
        (is_emoji_dominant, emotion, confidence)
        - is_emoji_dominant: True if text is mostly emojis
        - emotion: Detected emotion from emojis
        - confidence: Confidence score (0.0-1.0)
    """
    # Remove whitespace
    clean_text = text.strip()
    
    if not clean_text:
        return False, None, 0.0
    
    # Extract emojis
    emojis = extract_emojis(clean_text)
    
    if not emojis:
        return False, None, 0.0
    
    # Calculate emoji dominance (percentage of text that is emojis)
    emoji_char_count = sum(len(emoji) for emoji in emojis)
    text_length = len(clean_text)
    emoji_ratio = emoji_char_count / text_length
    
    # If less than 40% emojis, don't treat as emoji-only
    if emoji_ratio < 0.4:
        return False, None, 0.0
    
    # Map emojis to emotions
    emoji_emotions = []
    for emoji_str in emojis:
        for char in emoji_str:
            if char in EMOJI_TO_EMOTION:
                emoji_emotions.append(EMOJI_TO_EMOTION[char])
    
    # No recognized emojis
    if not emoji_emotions:
        # Unknown emoji, return neutral with low confidence
        return True, 'neutral', 0.55
    
    # Find dominant emotion
    emotion_counts = Counter(emoji_emotions)
    dominant_emotion, count = emotion_counts.most_common(1)[0]
    
    # Calculate confidence based on consensus
    consensus = count / len(emoji_emotions)
    
    # Base confidence on emoji ratio and consensus
    confidence = min(0.85 + (consensus * 0.10), 0.95)  # Cap at 95%
    
    # Boost confidence if very emoji-heavy
    if emoji_ratio > 0.8:
        confidence = min(confidence + 0.05, 0.98)
    
    return True, dominant_emotion, confidence


def normalize_repeated_chars(text: str) -> str:
    """
    Normalize repeated characters for better model understanding.
    Example: 'goooooo' → 'gooo', 'yesss' → 'yess'
    """
    # Keep max 3 consecutive repeated characters
    return re.sub(r'(.)\1{3,}', r'\1\1\1', text)


def preprocess_text(text: str) -> str:
    """
    Full text preprocessing pipeline.
    
    Args:
        text: Raw input text
    
    Returns:
        Preprocessed text
    """
    # Normalize repeated characters
    text = normalize_repeated_chars(text)
    
    # Strip extra whitespace
    text = ' '.join(text.split())
    
    return text