from textblob import TextBlob
import re

def analyze_sentiment(text):
    """
    High-accuracy sentiment analysis
    Returns: float from -1.0 (Negative) to 1.0 (Positive)
    """
    if not text:
        return 0.0
    
    try:
        text = str(text).strip()
        if len(text) < 3:
            return 0.0
        
        # Advanced text cleaning
        text = clean_text(text)
        
        # Get base sentiment
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Apply precision rules based on your dataset patterns
        polarity = apply_precision_rules(text, polarity, subjectivity)
        
        return float(polarity)
            
    except:
        return 0.0

def clean_text(text):
    """Advanced text cleaning"""
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[@#]', '', text)     # Remove @ and #
    text = re.sub(r'[^\w\s!?.,]', '', text)  # Remove special chars
    text = re.sub(r'\s+', ' ', text)     # Remove extra spaces
    return text.strip()

def apply_precision_rules(text, polarity, subjectivity):
    """Apply rules to improve precision based on your dataset"""
    text_lower = text.lower()
    
    # Strong positive indicators from your dataset
    strong_positive = [
        'love', 'amazing', 'awesome', 'great', 'best', 'wonderful', 'perfect',
        'excited', 'happy', 'enjoying', 'grateful', 'proud', 'inspired', 'success',
        'fantastic', 'excellent', 'brilliant', 'beautiful', 'outstanding'
    ]
    
    # Strong negative indicators from your dataset  
    strong_negative = [
        'hate', 'terrible', 'awful', 'worst', 'bad', 'angry', 'sad', 'disappointed',
        'fear', 'disgust', 'bitter', 'shame', 'injustice', 'horrible', 'hateful',
        'furious', 'annoying', 'miserable'
    ]
    
    # Neutral/uncertain indicators
    neutral_indicators = ['okay', 'maybe', 'perhaps', 'probably', 'might', 'could']
    
    # Count strong sentiment words
    pos_count = sum(1 for word in strong_positive if word in text_lower)
    neg_count = sum(1 for word in strong_negative if word in text_lower)
    neutral_count = sum(1 for word in neutral_indicators if word in text_lower)
    
    # Rule 1: Strong word dominance
    if pos_count > neg_count + 1:
        return max(polarity, 0.7)  # Boost to strongly positive
    elif neg_count > pos_count + 1:
        return min(polarity, -0.7)  # Boost to strongly negative
    
    # Rule 2: Question marks often indicate uncertainty
    if '?' in text and abs(polarity) < 0.3:
        return polarity * 0.5  # Reduce confidence for questions
    
    # Rule 3: High subjectivity with medium polarity -> boost
    if subjectivity > 0.7 and 0.2 < abs(polarity) < 0.6:
        polarity = polarity * 1.3
    
    # Rule 4: Very short texts are often neutral
    if len(text.split()) < 4 and abs(polarity) < 0.4:
        return 0.0
    
    # Rule 5: Neutral indicators reduce polarity
    if neutral_count > 0 and abs(polarity) < 0.5:
        polarity = polarity * 0.6
    
    # Ensure within bounds
    return max(min(polarity, 1.0), -1.0)