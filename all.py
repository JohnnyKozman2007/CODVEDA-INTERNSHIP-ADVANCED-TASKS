import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog, ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import instaloader
import re
import os,time
import threading
import sys
import os

warnings.filterwarnings('ignore')

# Add the current directory to Python path to import helper
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from helper import analyze_sentiment

# ======================= GLOBALS ============================
class_names = ['Negative', 'Neutral', 'Positive']
dataset_path_main = '3) Sentiment dataset.csv'
dataset_path_edited = '3) Sentiment dataset edited.csv'

# ======================= EXISTING FUNCTIONS FROM TASK 1 ====================



def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics."""
    y_pred = model.predict(X_test)

    # Scores
    precision = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=None, zero_division=0)

    results = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'classification_report': classification_report(y_test, y_pred, target_names=class_names),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'y_pred': y_pred,
        'y_test': y_test
    }
    
    return results

def analyze_feature_importance(model, X):
    """Return feature importance data."""
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return importance_df



# ======================= EXISTING FUNCTIONS FROM SCRAPER ====================

from scraper import extract_shortcode,scrape_post
from Task_1_advanced import simplify_sentiment,load_dataset,preprocess_dataset,prepare_features,train_model,save_model
# ======================= NEW INTEGRATION FUNCTIONS ====================

def prepare_scraped_data_for_analysis(df):
    """Prepare scraped data for sentiment analysis using existing functions"""
    analysis_df = df.copy()
    
    # Extract year and month from date
    analysis_df['Date'] = pd.to_datetime(analysis_df['Date'])
    analysis_df['Year'] = analysis_df['Date'].dt.year
    analysis_df['Month'] = analysis_df['Date'].dt.month
    
    # Use existing analyze_sentiment function for caption and hashtags
    analysis_df['Text'] = analysis_df['Caption'].apply(analyze_sentiment)
    analysis_df['Hashtags'] = analysis_df['Hashtags'].apply(
        lambda x: analyze_sentiment(x) if pd.notna(x) else 0
    )
    
    # Prepare features matching the training format
    features = ['Likes', 'Year', 'Month', 'Text', 'Hashtags']
    available_features = [f for f in features if f in analysis_df.columns]
    
    return analysis_df[available_features]

def predict_sentiment(scraped_data):
    """Predict sentiment using trained model"""
    try:
        model = joblib.load('random_forest_sentiment_model.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        
        # Scale the features
        scaled_features = scaler.transform(scraped_data)
        
        # Predict sentiment
        predictions = model.predict(scaled_features)
        probabilities = model.predict_proba(scaled_features)
        
        return predictions, probabilities
        
    except FileNotFoundError:
        raise Exception("Trained model not found. Please train the model first.")
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")



def scrape_and_analyze_post(link: str, callback=None):
    """Use existing scraper to get CSV, then analyze sentiment"""
    def scraping_thread():
        try:
            if callback: callback("Starting scraping process...")
            
            # First validate the link
            shortcode = extract_shortcode(link)
            if not shortcode:
                if callback: callback("error", "Invalid Instagram link.")
                return
            
            # Run the existing scraper - this will handle the GUI dialogs
            success = scrape_post(link)
            
             
            # Wait for file to be written
            time.sleep(2)
            
          
            if callback: callback("Scraping completed, analyzing sentiment...")
            
            # Check if file was created
            if not os.path.exists("instagram_data.csv"):
                if callback: callback("error", "Scraping failed - no data file created.")
                return
           
            # Load and analyze the data
            df = pd.read_csv("instagram_data.csv")
            
            if df.empty:
                if callback: callback("error", "No data found in scraped CSV file.")
                return
             
            # Prepare data for analysis
            analysis_data = prepare_scraped_data_for_analysis(df)
            
            # Predict sentiment
            predictions, probabilities = predict_sentiment(analysis_data)
            
            sentiment_labels = ['Negative', 'Neutral', 'Positive']
            sentiment_idx = predictions[0]
            sentiment_percentages = probabilities[0] * 100

            result_message = f"📊 Sentiment Analysis Results:\n\n"
            result_message += f"Overall Sentiment: {sentiment_labels[sentiment_idx]}\n\n"
            result_message += "Confidence Levels:\n"
            
            for i, label in enumerate(sentiment_labels):
                result_message += f"  {label}: {sentiment_percentages[i]:.1f}%\n"
            
            caption = df.iloc[0]['Caption']
            likes = df.iloc[0]['Likes']
            hashtags = df.iloc[0]['Hashtags']
            
            result_message += f"\n📝 Post Details:\n"
            result_message += f"Likes: {likes}\n"
            result_message += f"Hashtags: {hashtags if pd.notna(hashtags) else 'None'}\n"
            result_message += f"Caption Preview: {caption[:100]}..." if len(str(caption)) > 100 else f"Caption: {caption}"

            if callback: callback("complete", result_message)

        except:
            pass
    scraping_thread()
def train_model_from_file(file_path=None, callback=None):
    """Train model from CSV file using existing functions"""
    def training_thread():
        try:
            if callback: callback("Loading dataset...")
            
            if file_path:
                df = pd.read_csv(file_path)
                edited = False
            else:
                df, edited = load_dataset()
            
            if df is None:
                if callback: callback("error", "Failed to load dataset")
                return
                
            if callback: callback("Preprocessing data...")
            df_processed = preprocess_dataset(df, edited)
            
            if callback: callback("Preparing features...")
            X, y, scaler = prepare_features(df_processed)
            
            if callback: callback("Training model (this may take a while)...")
            model, X_train, X_test, y_train, y_test = train_model(X, y)
            
            if callback: callback("Saving model...")
            save_model(model, scaler)
            
            if callback: 
                callback("complete", "Model training completed successfully!")
                
        except Exception as e:
            if callback: callback("error", f"Training failed: {str(e)}")
    
    thread = threading.Thread(target=training_thread)
    thread.daemon = True
    thread.start()

def evaluate_model_gui(callback=None):
    """Evaluate the model and display results using existing functions"""
    def evaluation_thread():
        try:
            if callback: callback("Loading model...")
            model = joblib.load('random_forest_sentiment_model.pkl')
            
            if callback: callback("Loading dataset for evaluation...")
            df, edited = load_dataset()
            if df is None:
                if callback: callback("error", "Failed to load dataset for evaluation")
                return
            
            if callback: callback("Preprocessing evaluation data...")
            df_processed = preprocess_dataset(df, edited)
            
            if callback: callback("Preparing evaluation features...")
            X, y, _ = prepare_features(df_processed)
            
            if callback: callback("Splitting evaluation data...")
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            if callback: callback("Making predictions...")
            results = evaluate_model(model, X_test, y_test)
            
            if callback: callback("Analyzing feature importance...")
            feature_importance = analyze_feature_importance(model, pd.DataFrame(X, columns=['Likes', 'Year', 'Month', 'Text', 'Hashtags']))
            
            results['feature_importance'] = feature_importance
            results['model'] = model
            results['X_test'] = X_test
            results['y_test'] = y_test
            
            if callback: callback("complete", results)
                
        except FileNotFoundError:
            if callback: callback("error", "Trained model not found. Please train the model first.")
        except Exception as e:
            if callback: callback("error", f"Evaluation failed: {str(e)}")
    
    thread = threading.Thread(target=evaluation_thread)
    thread.daemon = True
    thread.start()

# ======================= MAIN GUI APPLICATION ====================

class SentimentAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Instagram Sentiment Analyzer")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        self.setup_gui()
    
    def setup_gui(self):
        # Main title
        title_frame = tk.Frame(self.root)
        title_frame.pack(pady=10)
        
        tk.Label(title_frame, text="📊 Instagram Sentiment Analyzer", 
                font=("Arial", 18, "bold")).pack()

        # Main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tab 1: Scraper & Analysis
        self.scraper_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.scraper_frame, text="Scrape & Analyze")
        
        # Tab 2: Model Training
        self.training_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.training_frame, text="Model Training")
        
        # Tab 3: Model Evaluation
        self.evaluation_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.evaluation_frame, text="Model Evaluation")
        
        self.setup_scraper_tab()
        self.setup_training_tab()
        self.setup_evaluation_tab()
        
        # Progress frame (common for all tabs)
        progress_frame = tk.Frame(self.root)
        progress_frame.pack(fill='x', padx=20, pady=10)
        
        self.progress_label = tk.Label(progress_frame, text="Ready", font=("Arial", 10))
        self.progress_label.pack()
        
        self.progress = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress.pack(fill='x', pady=5)

    def setup_scraper_tab(self):
        # Instagram Scraper Section
        scraper_title = tk.Label(self.scraper_frame, text="Instagram Post Scraper", 
                                font=("Arial", 14, "bold"))
        scraper_title.pack(pady=10)

        # Link entry
        link_frame = tk.Frame(self.scraper_frame)
        link_frame.pack(pady=10, fill='x', padx=20)
        
        tk.Label(link_frame, text="Instagram Post Link:", font=("Arial", 10)).pack(anchor='w')
        self.entry_link = tk.Entry(link_frame, width=70, font=("Arial", 10))
        self.entry_link.pack(fill='x', pady=5)

        # Scrape button
        self.btn_scrape = tk.Button(self.scraper_frame, text="Scrape & Analyze Post", 
                                  command=self.on_scrape, bg="#4CAF50", fg="white", 
                                  font=("Arial", 12, "bold"), width=20, height=2)
        self.btn_scrape.pack(pady=10)

        # Results display
        self.results_text = tk.Text(self.scraper_frame, height=15, width=80, font=("Arial", 10))
        self.results_text.pack(fill='both', expand=True, padx=20, pady=10)
        self.results_text.config(state=tk.DISABLED)

    def setup_training_tab(self):
        # Model Training Section
        training_title = tk.Label(self.training_frame, text="Model Training", 
                                 font=("Arial", 14, "bold"))
        training_title.pack(pady=10)

        # Training info
        info_text = """Train a new sentiment analysis model using your dataset.
The model will be saved as 'random_forest_sentiment_model.pkl'"""
        
        info_label = tk.Label(self.training_frame, text=info_text, font=("Arial", 10), 
                             justify=tk.LEFT, wraplength=600)
        info_label.pack(pady=10)

        # Training buttons
        train_buttons_frame = tk.Frame(self.training_frame)
        train_buttons_frame.pack(pady=20)

        self.btn_train_default = tk.Button(train_buttons_frame, text="Train with Default Dataset", 
                                         command=self.on_train_default, bg="#2196F3", fg="white", 
                                         font=("Arial", 11, "bold"), width=20, height=2)
        self.btn_train_default.pack(side=tk.LEFT, padx=10)

        self.btn_train_custom = tk.Button(train_buttons_frame, text="Train with Custom CSV", 
                                        command=self.on_train_custom, bg="#FF9800", fg="white", 
                                        font=("Arial", 11, "bold"), width=20, height=2)
        self.btn_train_custom.pack(side=tk.LEFT, padx=10)

        # Training log
        self.training_log = tk.Text(self.training_frame, height=15, width=80, font=("Arial", 9))
        self.training_log.pack(fill='both', expand=True, padx=20, pady=10)
        self.training_log.config(state=tk.DISABLED)

    def setup_evaluation_tab(self):
        # Model Evaluation Section
        eval_title = tk.Label(self.evaluation_frame, text="Model Evaluation", 
                             font=("Arial", 14, "bold"))
        eval_title.pack(pady=10)

        # Evaluation info
        info_text = """Evaluate the current model performance with detailed metrics and visualizations."""
        
        info_label = tk.Label(self.evaluation_frame, text=info_text, font=("Arial", 10), 
                             justify=tk.LEFT, wraplength=600)
        info_label.pack(pady=10)

        # Evaluation button
        self.btn_evaluate = tk.Button(self.evaluation_frame, text="Evaluate Model", 
                                    command=self.on_evaluate, bg="#9C27B0", fg="white", 
                                    font=("Arial", 12, "bold"), width=20, height=2)
        self.btn_evaluate.pack(pady=10)

        # Evaluation results frame
        self.eval_results_frame = tk.Frame(self.evaluation_frame)
        self.eval_results_frame.pack(fill='both', expand=True, padx=20, pady=10)

    def show_loading(self, show=True):
        if show:
            self.progress.start(10)
            self.progress_label.config(text="Processing...")
            # Disable all buttons
            for btn in [self.btn_scrape, self.btn_train_default, self.btn_train_custom, self.btn_evaluate]:
                btn.config(state=tk.DISABLED)
        else:
            self.progress.stop()
            self.progress_label.config(text="Ready")
            # Enable all buttons
            for btn in [self.btn_scrape, self.btn_train_default, self.btn_train_custom, self.btn_evaluate]:
                btn.config(state=tk.NORMAL)

    def update_progress(self, message, data=None):
        if message == "complete":
            self.show_loading(False)
            if isinstance(data, str):
                messagebox.showinfo("Success", data)
                self.display_scraping_result(data)
            elif isinstance(data, dict):
                self.display_evaluation_results(data)
        elif message == "error":
            self.show_loading(False)
            messagebox.showerror("Error", data)
        else:
            self.progress_label.config(text=message)

    def display_scraping_result(self, result):
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, result)
        self.results_text.config(state=tk.DISABLED)

    def log_training_message(self, message):
        self.training_log.config(state=tk.NORMAL)
        self.training_log.insert(tk.END, f"{message}\n")
        self.training_log.see(tk.END)
        self.training_log.config(state=tk.DISABLED)
        self.root.update()

    def on_scrape(self):
        link = self.entry_link.get().strip()
        if not link:
            messagebox.showwarning("Missing Input", "Please paste an Instagram post/reel link first.")
            return
        
        self.show_loading(True)
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Starting analysis...\n")
        self.results_text.config(state=tk.DISABLED)
        
        scrape_and_analyze_post(link, self.update_progress)

    def on_train_default(self):
        self.show_loading(True)
        self.training_log.config(state=tk.NORMAL)
        self.training_log.delete(1.0, tk.END)
        self.training_log.insert(tk.END, "Starting training with default dataset...\n")
        self.training_log.config(state=tk.DISABLED)
        
        train_model_from_file(None, self.update_training_progress)

    def on_train_custom(self):
        file_path = filedialog.askopenfilename(
            title="Select CSV file for training",
            filetypes=[("CSV files", "*.csv")]
        )
        if file_path:
            self.show_loading(True)
            self.training_log.config(state=tk.NORMAL)
            self.training_log.delete(1.0, tk.END)
            self.training_log.insert(tk.END, f"Starting training with custom dataset: {file_path}\n")
            self.training_log.config(state=tk.DISABLED)
            
            train_model_from_file(file_path, self.update_training_progress)

    def update_training_progress(self, message, data=None):
        if message == "complete":
            self.show_loading(False)
            self.log_training_message("Training completed successfully!")
            messagebox.showinfo("Success", "Model training completed successfully!")
        elif message == "error":
            self.show_loading(False)
            self.log_training_message(f"Error: {data}")
            messagebox.showerror("Error", data)
        else:
            self.log_training_message(message)

    def on_evaluate(self):
        self.show_loading(True)
        # Clear previous evaluation results
        for widget in self.eval_results_frame.winfo_children():
            widget.destroy()
        
        evaluate_model_gui(self.update_progress)

    def display_evaluation_results(self, results):
        """Display evaluation results in the evaluation tab"""
        # Create notebook for evaluation results
        eval_notebook = ttk.Notebook(self.eval_results_frame)
        eval_notebook.pack(fill='both', expand=True)
        
        # Tab 1: Metrics
        metrics_frame = ttk.Frame(eval_notebook)
        eval_notebook.add(metrics_frame, text="Classification Metrics")
        
        metrics_text = tk.Text(metrics_frame, wrap=tk.WORD, width=80, height=15)
        metrics_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        metrics_text.insert(tk.END, "Classification Report:\n\n")
        metrics_text.insert(tk.END, results['classification_report'])
        metrics_text.insert(tk.END, "\n\nPer-class Metrics:\n\n")
        for i, name in enumerate(class_names):
            metrics_text.insert(tk.END, f"{name}: Precision={results['precision'][i]:.3f}, Recall={results['recall'][i]:.3f}, F1={results['f1'][i]:.3f}\n")
        
        metrics_text.config(state=tk.DISABLED)
        
        # Tab 2: Confusion Matrix
        cm_frame = ttk.Frame(eval_notebook)
        eval_notebook.add(cm_frame, text="Confusion Matrix")
        
        fig_cm = plt.Figure(figsize=(6, 5))
        ax_cm = fig_cm.add_subplot(111)
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                   xticklabels=class_names, yticklabels=class_names)
        ax_cm.set_title('Confusion Matrix')
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('Actual')
        
        canvas_cm = FigureCanvasTkAgg(fig_cm, cm_frame)
        canvas_cm.draw()
        canvas_cm.get_tk_widget().pack(fill='both', expand=True)
        
        # Tab 3: Feature Importance
        fi_frame = ttk.Frame(eval_notebook)
        eval_notebook.add(fi_frame, text="Feature Importance")
        
        fig_fi = plt.Figure(figsize=(8, 5))
        ax_fi = fig_fi.add_subplot(111)
        importance_df = results['feature_importance']
        
        sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax_fi)
        ax_fi.set_title('Random Forest Feature Importance')
        fig_fi.tight_layout()
        
        canvas_fi = FigureCanvasTkAgg(fig_fi, fi_frame)
        canvas_fi.draw()
        canvas_fi.get_tk_widget().pack(fill='both', expand=True)

# ======================= MAIN EXECUTION ====================

if __name__ == "__main__":
    root = tk.Tk()
    app = SentimentAnalyzerApp(root)
    root.mainloop()