# ======================= IMPORTS ============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from helper import analyze_sentiment

warnings.filterwarnings('ignore')

# ======================= GLOBALS ============================
class_names = ['Negative', 'Neutral', 'Positive']
dataset_path_main = '3) Sentiment dataset.csv'
dataset_path_edited = '3) Sentiment dataset edited.csv'

# ======================= FUNCTIONS ==========================

def simplify_sentiment(sentiment_label):
    sentiment_label = sentiment_label.lower()
    
    # Positive emotions
    if any(word in sentiment_label for word in ['positive', 'joy', 'happy', 'excitement', 'content', 'gratitude', 
                                              'love', 'pride', 'hope', 'amusement', 'admiration', 'affection',
                                              'accomplishment', 'celebration', 'euphoria', 'elation', 'enjoyment',
                                              'enthusiasm', 'festivejoy', 'happiness', 'overjoyed', 'optimism',
                                              'playful', 'playfuljoy', 'positivity', 'satisfaction', 'triumph',
                                              'zest', 'blessed', 'compassion', 'compassionate', 'confident',
                                              'empowerment', 'fulfillment', 'grateful', 'harmony', 'heartwarming',
                                              'hopeful', 'inspired', 'kind', 'kindness', 'motivation', 'proud',
                                              'radiance', 'rejuvenation', 'relief', 'success', 'tenderness',
                                              'thrill', 'vibrancy', 'whimsy', 'wonderment', 'adoration',
                                              'arousal', 'energy', 'freedom', 'friendship', 'grandeur', 'marvel',
                                              'melodic', 'mesmerizing', 'romance', 'spark', 'adventure',
                                              'creative inspiration', 'creativity', 'culinary adventure',
                                              'culinaryodyssey', 'dazzle', 'elegance', 'enchantment', 'iconic',
                                              'imagination', 'runway creativity', 'whispers of the past',
                                              'artisticburst', 'celestial wonder', "nature's beauty", 
                                              "ocean's freedom", 'winter magic', 'dreamchaser', 'free-spirited',
                                              'innerjourney', 'envisioning history', 'joy in baking']):
        return 1
    
    # Negative emotions  
    elif any(word in sentiment_label for word in ['negative', 'sad', 'anger', 'fear', 'disgust', 'hate', 
                                                'despair', 'grief', 'loneliness', 'frustration', 'bitter',
                                                'devastated', 'disappointed', 'heartbreak', 'suffering',
                                                'anxiety', 'apprehensive', 'bad', 'betrayal', 'bitterness',
                                                'boredom', 'challenge', 'darkness', 'desolation', 'desperation',
                                                'dismissive', 'embarrassed', 'emotionalstorm', 'envious',
                                                'envy', 'exhaustion', 'fearful', 'frustrated', 'grief',
                                                'hate', 'heartache', 'helplessness', 'intimidation',
                                                'isolation', 'jealous', 'jealousy', 'loss', 'lostlove',
                                                'melancholy', 'miscalculation', 'mischievous', 'numbness',
                                                'obstacle', 'overwhelmed', 'pressure', 'regret', 'resentment',
                                                'ruins', 'sadness', 'shame', 'solitude', 'sorrow', 'suffering',
                                                'sympathy', 'yearning', 'ambivalence', 'bittersweet',
                                                'confusion', 'nostalgia', 'pensive']):
        return -1
    
    # Neutral emotions
    else:
        return 0
    

def load_dataset():
    """Load dataset and detect whether edited version exists."""
    try:
        df = pd.read_csv(dataset_path_edited)
        edited = True
         
    except:
        df = pd.read_csv(dataset_path_main)
        edited = False
         
    return df, edited


def preprocess_dataset(df: pd.DataFrame, edited_version: bool):
    """Preprocess dataset — handle columns, text sentiment, encoding."""
    if 'Country' in df.columns:
        df = df.drop('Country', axis=1)

    if not edited_version:
         
        for column_name in ['Text', 'Platform', 'Hashtags', 'Sentiment']:
            try:
                if column_name in ['Text', 'Comments', 'Hashtags']:
                    df[column_name] = df[column_name].apply(analyze_sentiment)
                elif column_name == 'Sentiment':
                    df[column_name] = df[column_name].apply(simplify_sentiment)
            except:
                pass
        df.to_csv(dataset_path_edited, index=False)
         
    return df


def prepare_features(df: pd.DataFrame):
    """Prepare features and target for model training."""
    feature_columns = ['Likes', 'Year', 'Month', 'Text', 'Hashtags']
    X = df[feature_columns]

    le = LabelEncoder()
    y = le.fit_transform(df['Sentiment'])
     

    # Scale numeric columns
    numeric_cols = ['Likes', 'Year', 'Month', 'Text', 'Hashtags']
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    return X, y, scaler


def train_model(X, y):
    """Train and tune Random Forest model with GridSearchCV."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

     
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(
            random_state=42, class_weight='balanced', n_jobs=-1
        ),
        param_grid=param_grid,
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

     
    return best_model, X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print metrics."""
    y_pred = model.predict(X_test)

    # Scores
    precision = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=None, zero_division=0)

     
    # for i, name in enumerate(class_names):
    #     print(f"{name}: Precision={precision[i]:.3f}, Recall={recall[i]:.3f}, F1={f1[i]:.3f}")

     
    # Confusion Matrix
    plt.figure(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


def analyze_feature_importance(model, X):
    """Plot feature importances."""
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

     
    plt.figure(figsize=(8, 5))
    sns.barplot(data=importance_df, x='Importance', y='Feature')
    plt.title('Random Forest Feature Importance')
    plt.tight_layout()
    plt.show()


def save_model(model, scaler):
    """Save the trained model and scaler."""
    joblib.dump(model, 'random_forest_sentiment_model.pkl')
    joblib.dump(scaler, 'feature_scaler.pkl')
     


# ======================= MAIN EXECUTION ======================
def main():
    df, edited = load_dataset()
    df = preprocess_dataset(df, edited)
    X, y, scaler = prepare_features(df)
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    evaluate_model(model, X_test, y_test)
    analyze_feature_importance(model, X)
    save_model(model, scaler)
     
if __name__ == "__main__":
    main()
