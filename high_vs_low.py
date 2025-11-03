import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             roc_auc_score, confusion_matrix, classification_report, 
                             roc_curve)
from sklearn.decomposition import PCA
import yfinance as yf
import joblib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap
import threading
import time
import os

class VolatilityPredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Volatility Predictor - Optimized")
        self.root.geometry("1000x800")
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.results = None
        self.X_test = None
        self.y_test = None
        self.progress = None
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create frames for each tab
        self.stats_frame = ttk.Frame(self.notebook)
        self.predict_frame = ttk.Frame(self.notebook)
        
        self.notebook.add(self.stats_frame, text='📊 Model Statistics')
        self.notebook.add(self.predict_frame, text='🔍 Predict Volatility')
        
        self.setup_stats_tab()
        self.setup_predict_tab()
        
        # Try to load cached model first, otherwise train asynchronously
        if not self.load_cached_model():
            self.train_model_async()
    
    def load_cached_model(self):
        """Load cached model if recent enough"""
        cache_file = 'volatility_model.pkl'
        if os.path.exists(cache_file):
            try:
                # Check if cache is less than 1 day old
                if os.path.getmtime(cache_file) > (time.time() - 86400):
                    model_data = joblib.load(cache_file)
                    self.model = model_data['model']
                    self.scaler = model_data['scaler']
                    self.feature_names = model_data['feature_names']
                    self.results = model_data['results']
                    self.X_test = model_data.get('X_test')
                    self.y_test = model_data.get('y_test')
                    
                    best_kernel = max(self.results.keys(), key=lambda k: self.results[k]['accuracy'])
                    self.status_label.config(
                        text=f"✅ Using cached model! Best: {best_kernel.upper()} (Accuracy: {self.results[best_kernel]['accuracy']:.2%})"
                    )
                    return True
            except Exception as e:
                print(f"Cache load failed: {e}")
        return False
    
    def setup_stats_tab(self):
        """Setup the statistics tab with model performance visualizations"""
        # Status label
        self.status_label = ttk.Label(self.stats_frame, text="🔄 Loading model...", 
                                     font=('Arial', 12, 'bold'))
        self.status_label.pack(pady=10)
        
        # Control frame
        control_frame = ttk.Frame(self.stats_frame)
        control_frame.pack(pady=10)
        
        self.retrain_btn = ttk.Button(control_frame, text="🔄 Retrain Model", 
                                     command=self.retrain_model, width=20)
        self.retrain_btn.pack(side=tk.LEFT, padx=5)
        
        self.performance_btn = ttk.Button(control_frame, text="📈 Show Performance", 
                                        command=self.show_performance, width=20)
        self.performance_btn.pack(side=tk.LEFT, padx=5)
        
        self.decision_btn = ttk.Button(control_frame, text="🎯 Decision Boundary", 
                                      command=self.show_decision_boundary, width=20)
        self.decision_btn.pack(side=tk.LEFT, padx=5)
        
        self.info_btn = ttk.Button(control_frame, text="ℹ️ Model Info", 
                                  command=self.show_model_info, width=20)
        self.info_btn.pack(side=tk.LEFT, padx=5)
        
        # Canvas for matplotlib figures
        self.stats_canvas_frame = ttk.Frame(self.stats_frame)
        self.stats_canvas_frame.pack(fill='both', expand=True, padx=10, pady=10)
    
    def setup_predict_tab(self):
        """Setup the prediction tab with feature input fields"""
        # Main frame
        main_frame = ttk.Frame(self.predict_frame, padding="20")
        main_frame.pack(fill='both', expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="🔍 Predict Stock Volatility", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Input frame
        input_frame = ttk.LabelFrame(main_frame, text="Enter Stock Data Features", padding="15")
        input_frame.pack(fill='x', pady=(0, 20))
        
        # Feature input fields
        self.feature_entries = {}
        
        features_config = [
            ('price_change', 'Price Change (%):', 'Daily price change as percentage (e.g., 0.02 for 2% gain, -0.015 for 1.5% loss)'),
            ('volume_change', 'Volume Change (%):', 'Daily volume change as percentage (e.g., 0.10 for 10% increase, -0.05 for 5% decrease)'),
            ('high_low_ratio', 'High-Low Ratio:', 'Ratio of high price to low price (e.g., 1.02 for 2% range, 1.10 for 10% range)'),
            ('volatility_5d', '5-Day Volatility:', 'Standard deviation of returns over 5 days (e.g., 0.015 for 1.5% daily volatility)'),
            ('price_ma_5', 'Price MA Ratio:', 'Current price vs 5-day moving average (e.g., 0.01 for 1% above MA, -0.005 for 0.5% below)'),
            ('volume_ma_5', 'Volume MA Ratio:', 'Current volume vs 5-day average volume (e.g., 1.2 for 20% above average, 0.8 for 20% below)'),
            ('price_range', 'Price Range (%):', 'Daily range as percentage of average price (e.g., 0.025 for 2.5% range)'),
            ('close_position', 'Close Position:', 'Where close price is in daily range (0-1, e.g., 0.8 for near high, 0.2 for near low)'),
            ('rsi_14', 'RSI (0-100):', 'Relative Strength Index (30-70 normal range, >70 overbought, <30 oversold)')
        ]
        
        for i, (feature_name, label_text, help_text) in enumerate(features_config):
            row_frame = ttk.Frame(input_frame)
            row_frame.pack(fill='x', pady=5)
            
            ttk.Label(row_frame, text=label_text, font=('Arial', 9, 'bold'), width=20).pack(side=tk.LEFT, padx=(0, 10))
            ttk.Label(row_frame, text=help_text, font=('Arial', 8), foreground='gray', wraplength=500).pack(side=tk.LEFT, fill='x', expand=True)
            
            self.feature_entries[feature_name] = ttk.Entry(row_frame, width=15)
            self.feature_entries[feature_name].pack(side=tk.RIGHT, padx=10)
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)
        
        self.predict_btn = ttk.Button(button_frame, text="🎯 Predict Volatility", 
                                     command=self.predict_volatility, width=20)
        self.predict_btn.pack(side=tk.LEFT, padx=10)
        
        self.high_vol_btn = ttk.Button(button_frame, text="📋 Load High Vol Example", 
                                      command=self.load_high_vol_example, width=20)
        self.high_vol_btn.pack(side=tk.LEFT, padx=10)
        
        self.low_vol_btn = ttk.Button(button_frame, text="📋 Load Low Vol Example", 
                                     command=self.load_low_vol_example, width=20)
        self.low_vol_btn.pack(side=tk.LEFT, padx=10)
        
        self.clear_btn = ttk.Button(button_frame, text="🧹 Clear All", 
                                   command=self.clear_all, width=20)
        self.clear_btn.pack(side=tk.LEFT, padx=10)
        
        # Results frame
        self.results_frame = ttk.LabelFrame(main_frame, text="Prediction Results", padding="15")
        self.results_frame.pack(fill='both', expand=True)
        
        self.results_text = tk.Text(self.results_frame, height=12, font=('Arial', 10))
        self.results_text.pack(fill='both', expand=True)
        
        scrollbar = ttk.Scrollbar(self.results_frame, orient="vertical", command=self.results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill='y')
        self.results_text.configure(yscrollcommand=scrollbar.set)
    
    def toggle_buttons_state(self, enabled):
        """Enable or disable buttons during training"""
        state = "normal" if enabled else "disabled"
        self.retrain_btn.config(state=state)
        self.performance_btn.config(state=state)
        self.decision_btn.config(state=state)
        self.info_btn.config(state=state)
        self.predict_btn.config(state=state)
        self.high_vol_btn.config(state=state)
        self.low_vol_btn.config(state=state)
        self.clear_btn.config(state=state)
    
    def train_model_async(self):
        """Train model in separate thread"""
        self.status_label.config(text="🔄 Training model...")
        
        # Add progress bar
        self.progress = ttk.Progressbar(self.stats_frame, mode='indeterminate')
        self.progress.pack(pady=5)
        self.progress.start()
        
        # Disable buttons during training
        self.toggle_buttons_state(False)
        
        # Run in separate thread
        thread = threading.Thread(target=self._train_model_thread)
        thread.daemon = True
        thread.start()
    
    def _train_model_thread(self):
        """Actual training in background thread"""
        try:
            # Get stock data
            symbols = ['AAPL', 'MSFT', 'GOOGL']  # Reduced for speed
            data = self.get_stock_data(symbols)
            processed_data = self.create_features_and_target(data)
            
            # Define features
            self.feature_names = [
                'price_change', 'volume_change', 'high_low_ratio', 'volatility_5d',
                'price_ma_5', 'volume_ma_5', 'price_range', 'close_position', 'rsi_14'
            ]
            
            X = processed_data[self.feature_names]
            y = processed_data['target']
            
            # Split and train
            X_train, self.X_test, y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(self.X_test)
            
            # Train multiple kernels for comparison
            kernels = ['linear', 'rbf']
            self.results = {}
            
            for kernel in kernels:
                model = SVC(kernel=kernel, random_state=42, probability=True, class_weight='balanced')
                model.fit(X_train_scaled, y_train)
                
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred, zero_division=0)
                recall = recall_score(self.y_test, y_pred, zero_division=0)
                auc = roc_auc_score(self.y_test, y_proba)
                
                self.results[kernel] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'auc': auc,
                    'y_pred': y_pred,
                    'y_proba': y_proba
                }
            
            # Select best model
            best_kernel = max(self.results.keys(), key=lambda k: self.results[k]['accuracy'])
            self.model = self.results[best_kernel]['model']
            
            # Save model
            self.save_model()
            
            # Schedule GUI update on main thread
            self.root.after(0, self._training_complete)
            
        except Exception as e:
            # Schedule error handling on main thread
            self.root.after(0, lambda: self._training_failed(str(e)))
    
    def _training_complete(self):
        """Called when training completes successfully"""
        self.progress.stop()
        self.progress.pack_forget()
        self.toggle_buttons_state(True)
        
        best_kernel = max(self.results.keys(), key=lambda k: self.results[k]['accuracy'])
        self.status_label.config(
            text=f"✅ Model trained! Best: {best_kernel.upper()} (Accuracy: {self.results[best_kernel]['accuracy']:.2%})"
        )
    
    def _training_failed(self, error_msg):
        """Called when training fails"""
        if self.progress:
            self.progress.stop()
            self.progress.pack_forget()
        self.toggle_buttons_state(True)
        self.status_label.config(text=f"❌ Training failed")
        messagebox.showerror("Error", f"Model training failed: {error_msg}")
    
    def get_stock_data(self, symbols, period="6mo"):
        """Fetch stock data with optimized parameters"""
        all_data = []
        
        for symbol in symbols:
            try:
                stock = yf.Ticker(symbol)
                df = stock.history(period=period, interval="1d")
                if len(df) > 10:  # Only use if we have enough data
                    df = df.reset_index()
                    df['symbol'] = symbol
                    all_data.append(df)
            except Exception as e:
                print(f"Failed to fetch {symbol}: {e}")
                continue
        
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    
    def create_features_and_target(self, data):
        """Create features and target"""
        df = data.copy()
        
        # Calculate daily volatility
        df['daily_volatility'] = (df['High'] - df['Low']) / df['Close']
        
        # Create features
        df['price_change'] = df['Close'].pct_change()
        df['volume_change'] = df['Volume'].pct_change()
        df['high_low_ratio'] = df['High'] / df['Low']
        df['volatility_5d'] = df['daily_volatility'].rolling(5).mean()
        df['price_ma_5'] = (df['Close'].rolling(5).mean() / df['Close'] - 1)
        df['volume_ma_5'] = df['Volume'] / df['Volume'].rolling(5).mean()
        df['price_range'] = (df['High'] - df['Low']) / df['Close']
        df['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        # Simple RSI calculation
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Target
        vol_threshold = df['daily_volatility'].median()
        df['target'] = (df['daily_volatility'] > vol_threshold).astype(int)
        
        return df.dropna()
    
    def save_model(self):
        """Save the trained model"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'results': self.results,
                'X_test': self.X_test,
                'y_test': self.y_test,
                'timestamp': pd.Timestamp.now()
            }
            joblib.dump(model_data, 'volatility_model.pkl')
        except Exception as e:
            print(f"Warning: Could not save model: {e}")
    
    def show_performance(self):
        """Show model performance visualizations asynchronously"""
        if self.results is None:
            messagebox.showerror("Error", "No model results available!")
            return
        
        self.status_label.config(text="🔄 Generating visualizations...")
        self.toggle_buttons_state(False)
        
        # Run in thread to avoid blocking
        thread = threading.Thread(target=self._generate_performance_plots)
        thread.daemon = True
        thread.start()
    
    def _generate_performance_plots(self):
        """Generate plots in background"""
        try:
            fig = self.create_performance_plots_fast()
            # Schedule GUI update on main thread
            self.root.after(0, lambda: self._display_figure(fig))
        except Exception as e:
            self.root.after(0, lambda: self._plot_generation_failed(str(e)))
    
    def _display_figure(self, fig):
        """Display the generated figure"""
        # Clear previous canvas
        for widget in self.stats_canvas_frame.winfo_children():
            widget.destroy()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.stats_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        self.status_label.config(text="✅ Visualizations ready!")
        self.toggle_buttons_state(True)
    
    def _plot_generation_failed(self, error_msg):
        """Handle plot generation failure"""
        self.status_label.config(text="❌ Visualization failed")
        self.toggle_buttons_state(True)
        messagebox.showerror("Error", f"Plot generation failed: {error_msg}")
    
    def create_performance_plots_fast(self):
        """Optimized performance plots"""
        # Create 2x2 grid for faster rendering
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('SVM Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. Metrics Comparison
        self._plot_simple_metrics(axes[0,0])
        
        # 2. Best model confusion matrix
        self._plot_best_confusion_matrix(axes[0,1])
        
        # 3. ROC Curves
        self._plot_roc_curves(axes[1,0])
        
        # 4. Feature importance (faster than decision boundary)
        self._plot_feature_importance(axes[1,1])
        
        plt.tight_layout()
        return fig
    
    def _plot_simple_metrics(self, ax):
        """Plot basic performance metrics"""
        metrics = ['accuracy', 'precision', 'recall', 'auc']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'AUC']
        
        x_pos = np.arange(len(metrics))
        width = 0.35
        
        for i, kernel in enumerate(self.results.keys()):
            scores = [self.results[kernel][metric] for metric in metrics]
            ax.bar(x_pos + i*width - width/2, scores, width, 
                  label=kernel.upper(), alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Performance Metrics by Kernel')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metric_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Add value labels
        for i, kernel in enumerate(self.results.keys()):
            scores = [self.results[kernel][metric] for metric in metrics]
            for j, score in enumerate(scores):
                ax.text(j + i*width - width/2, score + 0.02, f'{score:.3f}', 
                       ha='center', va='bottom', fontsize=8)
    
    def _plot_best_confusion_matrix(self, ax):
        """Plot confusion matrix for best model only"""
        best_kernel = max(self.results.keys(), key=lambda k: self.results[k]['accuracy'])
        cm = confusion_matrix(self.y_test, self.results[best_kernel]['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                   xticklabels=['Low Vol', 'High Vol'], 
                   yticklabels=['Low Vol', 'High Vol'],
                   cmap='Blues')
        ax.set_title(f'Best Model ({best_kernel.upper()}) - Confusion Matrix')
    
    def _plot_roc_curves(self, ax):
        """Plot ROC curves"""
        for kernel in self.results.keys():
            fpr, tpr, _ = roc_curve(self.y_test, self.results[kernel]['y_proba'])
            auc_score = self.results[kernel]['auc']
            ax.plot(fpr, tpr, label=f'{kernel} (AUC = {auc_score:.3f})', linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_feature_importance(self, ax):
        """Plot feature importance (faster alternative to decision boundary)"""
        try:
            if hasattr(self.model, 'coef_'):
                importances = np.abs(self.model.coef_[0])
            else:
                # For RBF kernel, use permutation importance approximation
                importances = np.ones(len(self.feature_names))
            
            indices = np.argsort(importances)
            ax.barh(range(len(indices)), importances[indices])
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([self.feature_names[i] for i in indices])
            ax.set_title('Feature Importance')
            ax.grid(True, alpha=0.3, axis='x')
        except Exception as e:
            ax.text(0.5, 0.5, f"Feature importance\nnot available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Feature Importance')
    
    def show_decision_boundary(self):
        """Show decision boundary visualization asynchronously"""
        if self.results is None:
            messagebox.showerror("Error", "No model results available!")
            return
        
        self.status_label.config(text="🔄 Generating decision boundary...")
        self.toggle_buttons_state(False)
        
        thread = threading.Thread(target=self._generate_decision_boundary)
        thread.daemon = True
        thread.start()
    
    def _generate_decision_boundary(self):
        """Generate decision boundary in background"""
        try:
            # Clear previous canvas
            self.root.after(0, lambda: [w.destroy() for w in self.stats_canvas_frame.winfo_children()])
            
            # Create decision boundary plot
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle('SVM Decision Boundaries', fontsize=16, fontweight='bold')
            
            # Plot for each kernel
            for i, kernel in enumerate(self.results.keys()):
                try:
                    self.plot_single_decision_boundary(axes[i], kernel)
                except Exception as e:
                    axes[i].text(0.5, 0.5, f"Decision Boundary\nNot Available\nfor {kernel} kernel",
                                ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(f'{kernel.upper()} Kernel - Decision Boundary')
            
            plt.tight_layout()
            
            # Schedule display on main thread
            self.root.after(0, lambda: self._display_decision_boundary(fig))
            
        except Exception as e:
            self.root.after(0, lambda: self._decision_boundary_failed(str(e)))
    
    def _display_decision_boundary(self, fig):
        """Display the decision boundary figure"""
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.stats_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        self.status_label.config(text="✅ Decision boundary ready!")
        self.toggle_buttons_state(True)
    
    def _decision_boundary_failed(self, error_msg):
        """Handle decision boundary generation failure"""
        self.status_label.config(text="❌ Decision boundary failed")
        self.toggle_buttons_state(True)
        messagebox.showerror("Error", f"Decision boundary generation failed: {error_msg}")
    
    def plot_single_decision_boundary(self, ax, kernel):
        """Plot decision boundary for a single kernel"""
        model = self.results[kernel]['model']
        
        # Use PCA to reduce to 2D
        pca = PCA(n_components=2)
        X_test_scaled = self.scaler.transform(self.X_test)
        X_pca = pca.fit_transform(X_test_scaled)
        
        # Create mesh with larger step for speed
        h = 0.05  # Increased from 0.02 for speed
        x_min, x_max = X_pca[:, 0].min() - 0.5, X_pca[:, 0].max() + 0.5
        y_min, y_max = X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Transform and predict
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        mesh_points_original = pca.inverse_transform(mesh_points)
        Z = model.predict(mesh_points_original)
        Z = Z.reshape(xx.shape)
        
        # Plot
        cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#0000FF'])
        
        ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=self.y_test, 
                           cmap=cmap_bold, alpha=0.8, edgecolor='black', s=30)
        
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        ax.set_title(f'{kernel.upper()} Kernel Decision Boundary')
        
        # Add accuracy info
        accuracy = self.results[kernel]['accuracy']
        ax.text(0.02, 0.98, f'Accuracy: {accuracy:.2%}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def predict_volatility(self):
        """Predict volatility based on user input"""
        if self.model is None:
            messagebox.showerror("Error", "Model not trained yet!")
            return
        
        try:
            # Get values from entries
            features = []
            for feature_name in self.feature_names:
                value = self.feature_entries[feature_name].get().strip()
                if not value:
                    messagebox.showerror("Error", f"Please enter value for {feature_name}")
                    return
                try:
                    features.append(float(value))
                except ValueError:
                    messagebox.showerror("Error", f"Invalid number for {feature_name}")
                    return
            
            # Convert to array and scale
            features_array = np.array(features).reshape(1, -1)
            features_scaled = self.scaler.transform(features_array)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0][1]
            
            # Display results
            self.display_results(features, prediction, probability)
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
    
    def display_results(self, features, prediction, probability):
        """Display prediction results"""
        self.results_text.delete(1.0, tk.END)
        
        result_text = "📊 VOLATILITY PREDICTION RESULTS\n"
        result_text += "=" * 50 + "\n\n"
        
        # Prediction with emoji
        volatility_type = "HIGH VOLATILITY" if prediction == 1 else "LOW VOLATILITY"
        color_indicator = "🔴" if prediction == 1 else "🟢"
        result_text += f"{color_indicator} PREDICTION: {volatility_type}\n"
        result_text += f"📈 Confidence: {probability:.2%}\n\n"
        
        # Feature values
        result_text += "🔍 INPUT FEATURE VALUES:\n"
        result_text += "-" * 25 + "\n"
        
        for i, feature_name in enumerate(self.feature_names):
            result_text += f"• {feature_name}: {features[i]:.4f}\n"
        
        result_text += "\n💡 TRADING IMPLICATIONS:\n"
        result_text += "-" * 25 + "\n"
        
        if prediction == 1:
            result_text += "• Expect larger price swings\n"
            result_text += "• Use wider stop-loss orders\n"
            result_text += "• Higher risk management required\n"
            result_text += "• Potential for larger gains/losses\n"
            result_text += "• Consider volatility-based strategies\n"
        else:
            result_text += "• Stable price action expected\n"
            result_text += "• Tighter stop-loss orders possible\n"
            result_text += "• Lower risk environment\n"
            result_text += "• More predictable price movements\n"
            result_text += "• Suitable for trend-following strategies\n"
        
        result_text += f"\n🎯 Model used: {self.model.kernel.upper()} kernel SVM"
        
        self.results_text.insert(1.0, result_text)
    
    def load_high_vol_example(self):
        """Load high volatility example data"""
        example_data = {
            'price_change': 0.045,      # 4.5% price increase
            'volume_change': 0.25,      # 25% volume increase
            'high_low_ratio': 1.08,     # 8% daily range
            'volatility_5d': 0.028,     # 2.8% daily volatility
            'price_ma_5': 0.015,        # 1.5% above MA
            'volume_ma_5': 1.35,        # 35% above average volume
            'price_range': 0.045,       # 4.5% daily range
            'close_position': 0.85,     # Closed near high of day
            'rsi_14': 72.5              # Overbought territory
        }
        
        for feature_name, value in example_data.items():
            self.feature_entries[feature_name].delete(0, tk.END)
            self.feature_entries[feature_name].insert(0, str(value))
        
        messagebox.showinfo("Example Loaded", "High volatility example data loaded!")
    
    def load_low_vol_example(self):
        """Load low volatility example data"""
        example_data = {
            'price_change': 0.008,      # 0.8% price increase
            'volume_change': -0.05,     # 5% volume decrease
            'high_low_ratio': 1.015,    # 1.5% daily range
            'volatility_5d': 0.012,     # 1.2% daily volatility
            'price_ma_5': -0.003,       # 0.3% below MA
            'volume_ma_5': 0.85,        # 15% below average volume
            'price_range': 0.018,       # 1.8% daily range
            'close_position': 0.45,     # Closed in middle of range
            'rsi_14': 52.0              # Neutral territory
        }
        
        for feature_name, value in example_data.items():
            self.feature_entries[feature_name].delete(0, tk.END)
            self.feature_entries[feature_name].insert(0, str(value))
        
        messagebox.showinfo("Example Loaded", "Low volatility example data loaded!")
    
    def clear_all(self):
        """Clear all input fields"""
        for entry in self.feature_entries.values():
            entry.delete(0, tk.END)
        self.results_text.delete(1.0, tk.END)
    
    def retrain_model(self):
        """Retrain the model with current data"""
        if messagebox.askyesno("Confirm", "Retrain model with latest market data?"):
            self.train_model_async()
    
    def show_model_info(self):
        """Show information about the trained model"""
        if self.model is None:
            messagebox.showinfo("Model Info", "No model trained yet!")
            return
        
        best_kernel = max(self.results.keys(), key=lambda k: self.results[k]['accuracy'])
        best_accuracy = self.results[best_kernel]['accuracy']
        
        info_text = f"🤖 MODEL INFORMATION\n"
        info_text += "=" * 30 + "\n"
        info_text += f"Best Model: SVM with {best_kernel.upper()} kernel\n"
        info_text += f"Best Accuracy: {best_accuracy:.2%}\n"
        info_text += f"Features Used: {len(self.feature_names)}\n"
        info_text += f"Class Weight: Balanced\n"
        info_text += f"\n📊 ALL KERNEL PERFORMANCE:\n"
        
        for kernel in self.results.keys():
            info_text += f"  {kernel.upper()}: Accuracy {self.results[kernel]['accuracy']:.2%}, "
            info_text += f"AUC {self.results[kernel]['auc']:.3f}\n"
        
        messagebox.showinfo("Model Info", info_text)

def main():
    root = tk.Tk()
    app = VolatilityPredictorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()