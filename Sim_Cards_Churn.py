import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.metrics import f1_score, precision_score, recall_score
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('TkAgg')

print("TensorFlow version:", tf.__version__)

# Global variables
model = None
scaler = None
label_encoders = {}
history = None
feature_means = {}

# Step 1: Load and Preprocess Both Datasets
print("Loading datasets...")
train_df = pd.read_csv('churn-bigml-80.csv')  # 80% training data
test_df = pd.read_csv('churn-bigml-20.csv')   # 20% testing data

print(f"Training set shape: {train_df.shape}")
print(f"Testing set shape: {test_df.shape}")

# Display dataset info
print("\nTraining set target distribution:")
print(train_df['Churn'].value_counts())
print("\nTesting set target distribution:")
print(test_df['Churn'].value_counts())

# Store original data statistics for reasonable defaults
original_train_df = train_df.copy()

# Preprocess both datasets
def preprocess_data(df):
    """Preprocess the dataframe for training"""
    df_processed = df.copy()
    
    # Convert categorical variables
    categorical_columns = ['State', 'International plan', 'Voice mail plan']
    
    for col in categorical_columns:
        label_encoder = LabelEncoder()
        df_processed[col] = label_encoder.fit_transform(df_processed[col])
        label_encoders[col] = label_encoder
    
    return df_processed

# Preprocess both datasets
train_processed = preprocess_data(train_df)
test_processed = preprocess_data(test_df)

# Store feature means for reasonable defaults
for col in train_processed.columns:
    if col != 'Churn':
        feature_means[col] = train_processed[col].mean()

# Separate features and target
X_train = train_processed.drop('Churn', axis=1)
y_train = train_processed['Churn'].astype(int)  # Convert False/True to 0/1

X_test = test_processed.drop('Churn', axis=1)
y_test = test_processed['Churn'].astype(int)

print(f"Processed training features shape: {X_train.shape}")
print(f"Processed testing features shape: {X_test.shape}")

# Step 2: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling completed!")
print(f"Scaled training data shape: {X_train_scaled.shape}")
print(f"Scaled testing data shape: {X_test_scaled.shape}")

# Step 3: Design Neural Network Architecture
input_dim = X_train_scaled.shape[1]
print(f"Input dimension: {input_dim}")

# Design the neural network architecture
model = keras.Sequential([
    # Input Layer
    keras.layers.Dense(64, activation='relu', input_shape=(input_dim,), name='input_layer'),
    keras.layers.BatchNormalization(),
    
    # Hidden Layer 1
    keras.layers.Dense(32, activation='relu', name='hidden_layer_1'),
    keras.layers.Dropout(0.3),
    
    # Hidden Layer 2  
    keras.layers.Dense(16, activation='relu', name='hidden_layer_2'),
    keras.layers.Dropout(0.2),
    
    # Output Layer (Binary classification for churn prediction)
    keras.layers.Dense(1, activation='sigmoid', name='output_layer')
])

print("=== NEURAL NETWORK ARCHITECTURE ===")
model.summary()

# Step 4: Compile the Model - FIXED METRIC NAMES
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',  # Binary classification loss
    metrics=['accuracy']  # Using only accuracy during training to avoid metric issues
)

print("Model compiled successfully!")

# Step 5: Train the Model with Backpropagation
print("=== TRAINING MODEL WITH BACKPROPAGATION ===")

# Train the model (backpropagation happens automatically in fit())
history = model.fit(
    X_train_scaled,
    y_train,
    batch_size=32,           # Mini-batch gradient descent
    epochs=50,               # Number of epochs
    validation_split=0.2,    # Use 20% of training data for validation
    verbose=1,              # Show training progress
    shuffle=True            # Shuffle training data
)

print("Training completed! Backpropagation was used to update weights.")

# Step 6: Evaluate Model Accuracy
print("=== MODEL EVALUATION ON TEST SET ===")

# Evaluate on the separate test set
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)

# Make predictions for additional metrics
y_pred_proba = model.predict(X_test_scaled, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Calculate additional metrics manually
test_precision = precision_score(y_test, y_pred)
test_recall = recall_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1-Score: {test_f1:.4f}")

# GUI Functions
def show_stats_and_graphs():
    """Display statistics and graphs in a new window"""
    stats_window = tk.Toplevel(root)
    stats_window.title("Model Statistics and Graphs")
    stats_window.geometry("1200x800")
    
    # Create notebook for tabs
    notebook = ttk.Notebook(stats_window)
    
    # Tab 1: Performance Metrics
    metrics_frame = ttk.Frame(notebook)
    notebook.add(metrics_frame, text="Performance Metrics")
    
    # Performance metrics
    metrics_text = f"""
    MODEL PERFORMANCE METRICS:
    =========================
    
    Test Accuracy: {test_accuracy:.4f}
    Test Loss: {test_loss:.4f}
    Test Precision: {test_precision:.4f}
    Test Recall: {test_recall:.4f}
    Test F1-Score: {test_f1:.4f}
    
    Training History:
    Final Training Accuracy: {history.history['accuracy'][-1]:.4f}
    Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}
    Final Training Loss: {history.history['loss'][-1]:.4f}
    Final Validation Loss: {history.history['val_loss'][-1]:.4f}
    
    Dataset Info:
    Training samples: {len(X_train)}
    Testing samples: {len(X_test)}
    Features: {X_train.shape[1]}
    """
    
    metrics_label = tk.Text(metrics_frame, wrap=tk.WORD, width=80, height=20, font=("Arial", 12))
    metrics_label.insert(tk.END, metrics_text)
    metrics_label.config(state=tk.DISABLED)
    metrics_label.pack(padx=10, pady=10)
    
    # Tab 2: Training Graphs
    graphs_frame = ttk.Frame(notebook)
    notebook.add(graphs_frame, text="Training Graphs")
    
    # Create matplotlib figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: Training and Validation Loss
    ax1.plot(history.history['loss'], label='Training Loss', linewidth=2, color='blue')
    ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='red')
    ax1.set_title('Training vs Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training and Validation Accuracy
    ax2.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2, color='blue')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='red')
    ax2.set_title('Training vs Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                xticklabels=['Not Churn', 'Churn'], 
                yticklabels=['Not Churn', 'Churn'])
    ax3.set_title('Confusion Matrix')
    ax3.set_xlabel('Predicted Label')
    ax3.set_ylabel('True Label')
    
    # Plot 4: ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    ax4.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax4.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax4.set_xlim([0.0, 1.0])
    ax4.set_ylim([0.0, 1.05])
    ax4.set_xlabel('False Positive Rate')
    ax4.set_ylabel('True Positive Rate')
    ax4.set_title('ROC Curve')
    ax4.legend(loc="lower right")
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Embed matplotlib figure in tkinter
    canvas = FigureCanvasTkAgg(fig, master=graphs_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(padx=10, pady=10)
    
    # Tab 3: Classification Report
    report_frame = ttk.Frame(notebook)
    notebook.add(report_frame, text="Classification Report")
    
    report = classification_report(y_test, y_pred, target_names=['Not Churn', 'Churn'])
    report_label = tk.Text(report_frame, wrap=tk.WORD, width=80, height=20, font=("Courier", 10))
    report_label.insert(tk.END, report)
    report_label.config(state=tk.DISABLED)
    report_label.pack(padx=10, pady=10)
    
    notebook.pack(expand=True, fill='both', padx=10, pady=10)

def get_reasonable_defaults():
    """Get reasonable default values for each feature"""
    defaults = {}
    
    # Use actual means from the training data for numerical features
    for col in X_train.columns:
        if col in ['International plan', 'Voice mail plan']:
            defaults[col] = 'No'  # Most common value
        elif col == 'State':
            defaults[col] = 'WV'  # Most common state in dataset
        else:
            # Use the mean value from training data (rounded for display)
            mean_val = feature_means[col]
            if 'minutes' in col.lower() or 'charge' in col.lower():
                defaults[col] = f"{mean_val:.1f}"
            elif 'calls' in col.lower():
                defaults[col] = f"{int(mean_val)}"
            else:
                defaults[col] = f"{mean_val:.0f}"
    
    return defaults

def predict_churn():
    """Predict churn based on user input"""
    try:
        # Get all input values
        input_data = []
        validation_errors = []
        
        for i, (col, var) in enumerate(feature_vars.items()):
            value_str = var.get().strip()
            
            if not value_str:
                validation_errors.append(f"{col}: Cannot be empty")
                continue
                
            if col in ['International plan', 'Voice mail plan']:
                # Convert Yes/No to 1/0
                if value_str.lower() in ['yes', 'y', 'true', '1']:
                    value = 1
                elif value_str.lower() in ['no', 'n', 'false', '0']:
                    value = 0
                else:
                    validation_errors.append(f"{col}: Must be 'Yes' or 'No'")
                    continue
                    
            elif col == 'State':
                # Convert state to encoded value
                state_name = value_str.upper()
                if state_name in label_encoders['State'].classes_:
                    value = label_encoders['State'].transform([state_name])[0]
                else:
                    # Use a default state if not found
                    value = label_encoders['State'].transform(['WV'])[0]
                    
            else:
                # Numerical values
                try:
                    value = float(value_str)
                    # Basic validation for numerical values
                    if value < 0:
                        validation_errors.append(f"{col}: Cannot be negative")
                        continue
                except ValueError:
                    validation_errors.append(f"{col}: Must be a number")
                    continue
            
            input_data.append(value)
        
        if validation_errors:
            error_msg = "Please fix the following errors:\n\n" + "\n".join(validation_errors)
            messagebox.showerror("Input Validation Error", error_msg)
            return
        
        # Convert to numpy array and scale
        input_array = np.array([input_data])
        input_scaled = scaler.transform(input_array)
        
        # Make prediction
        prediction_proba = model.predict(input_scaled, verbose=0)[0][0]
        prediction = "CHURN" if prediction_proba > 0.5 else "NO CHURN"
        confidence = prediction_proba if prediction_proba > 0.5 else 1 - prediction_proba
        
        # Show result with interpretation
        result_text = f"PREDICTION: {prediction}\n"
        result_text += f"Confidence: {confidence:.2%}\n"
        result_text += f"Probability: {prediction_proba:.4f}\n\n"
        
        # Add interpretation
        if prediction == "CHURN":
            if confidence > 0.8:
                result_text += "High risk of churn - immediate action recommended"
            elif confidence > 0.6:
                result_text += "Moderate risk of churn - proactive measures needed"
            else:
                result_text += "Low risk of churn - monitor customer"
        else:
            if confidence > 0.8:
                result_text += "Customer likely to stay - good retention"
            elif confidence > 0.6:
                result_text += "Customer probably staying - normal monitoring"
            else:
                result_text += "Uncertain prediction - gather more data"
        
        messagebox.showinfo("Churn Prediction Result", result_text)
        
    except Exception as e:
        messagebox.showerror("Prediction Error", f"An error occurred during prediction:\n{str(e)}")

def reset_to_defaults():
    """Reset all fields to reasonable defaults"""
    defaults = get_reasonable_defaults()
    for col, var in feature_vars.items():
        if col in defaults:
            var.set(defaults[col])

def create_gui():
    """Create the main GUI"""
    global root, feature_vars
    
    root = tk.Tk()
    root.title("Customer Churn Prediction System")
    root.geometry("900x700")
    
    # Main frame
    main_frame = ttk.Frame(root, padding="20")
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Title
    title_label = ttk.Label(main_frame, text="Customer Churn Prediction", 
                          font=("Arial", 16, "bold"))
    title_label.pack(pady=10)
    
    # Description
    desc_label = ttk.Label(main_frame, 
                          text="Enter customer details to predict if they will churn or not",
                          font=("Arial", 10))
    desc_label.pack(pady=5)
    
    # Create input frame with scrollbar
    input_container = ttk.Frame(main_frame)
    input_container.pack(fill=tk.BOTH, expand=True, pady=10)
    
    # Add scrollbar
    canvas = tk.Canvas(input_container)
    scrollbar = ttk.Scrollbar(input_container, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    input_frame = ttk.LabelFrame(scrollable_frame, text="Customer Features", padding="10")
    input_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Get feature names from training data
    feature_names = X_train.columns.tolist()
    feature_vars = {}
    
    # Get reasonable defaults
    defaults = get_reasonable_defaults()
    
    # Create input fields (2 columns for better layout)
    for i, feature in enumerate(feature_names):
        row = i // 2
        col = i % 2
        
        frame = ttk.Frame(input_frame)
        frame.grid(row=row, column=col, sticky="w", padx=15, pady=8)
        
        label = ttk.Label(frame, text=feature + ":", width=25, anchor="w")
        label.pack(side=tk.LEFT)
        
        var = tk.StringVar()
        # Set reasonable default value
        if feature in defaults:
            var.set(defaults[feature])
        entry = ttk.Entry(frame, textvariable=var, width=20)
        entry.pack(side=tk.LEFT, padx=5)
        
        feature_vars[feature] = var
    
    # Pack the canvas and scrollbar
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    # Button frame
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(pady=20)
    
    # Buttons
    stats_button = ttk.Button(button_frame, text="Show Stats & Graphs", 
                            command=show_stats_and_graphs)
    stats_button.pack(side=tk.LEFT, padx=10)
    
    predict_button = ttk.Button(button_frame, text="Predict Churn", 
                              command=predict_churn)
    predict_button.pack(side=tk.LEFT, padx=10)
    
    reset_button = ttk.Button(button_frame, text="Reset to Defaults", 
                            command=reset_to_defaults)
    reset_button.pack(side=tk.LEFT, padx=10)
    
    # Instructions
    instructions = """
    Instructions:
    1. Fill in all customer feature values (reasonable defaults are pre-filled)
    2. For 'International plan' and 'Voice mail plan': Enter 'Yes' or 'No'
    3. For 'State': Use state abbreviation (e.g., CA, NY, TX)
    4. Click 'Predict Churn' to get prediction with confidence level
    5. Click 'Reset to Defaults' to restore reasonable values
    6. Click 'Show Stats & Graphs' to see model performance
    
    Note: Using zeros for all values will give unrealistic predictions due to data scaling.
    """
    
    instructions_label = ttk.Label(main_frame, text=instructions, 
                                 font=("Arial", 9), justify=tk.LEFT)
    instructions_label.pack(pady=10)
    
    return root

# Run the GUI
if __name__ == "__main__":
    print("\n=== STARTING GUI ===")
    app = create_gui()
    app.mainloop()