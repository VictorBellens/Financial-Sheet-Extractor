import sys
print(sys.version)

import os
import cv2
import pytesseract
import re
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Set Tesseract path for macOS (Apple Silicon)
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

# --- Constants ---
ROOT_DIR = "/Users/triniroca/Desktop/stats_new/financial-data"
CATEGORIES = ["invoice", "email", "resume", "advertisement"]
MAX_SAMPLES_PER_CATEGORY = 1000
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')
MAX_NUM_WORDS = 10000
MAX_SEQUENCE_LENGTH = 200
MODEL_PATH = "/Users/triniroca/Desktop/stats_new/model11.h5"
TOKENIZER_PATH = "/Users/triniroca/Desktop/stats_new/tokenizer11.pkl"
ENCODER_PATH = "/Users/triniroca/Desktop/stats_new/encoder11.pkl"
PREPROCESSED_DIR = "/Users/triniroca/Desktop/stats_new/preprocessed/"

# Preprocessing and OCR
def preprocess_image_for_ocr(image_path):
    print(f"Preprocessing image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=30)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return thresh

def clean_text(text):
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^\w\s.,-/:]', '', text)
    return text.lower()

def extract_text_from_image(image_path):
    print(f"OCR processing: {image_path}")
    processed_img = preprocess_image_for_ocr(image_path)
    if processed_img is None:
        return ""
    try:
        text = pytesseract.image_to_string(processed_img, config='--psm 6')
        return clean_text(text)
    except Exception as e:
        print(f"OCR failed for {image_path}: {e}")
        return ""

# Information Extraction for Invoices
def extract_invoice_info(text):
    info = {}
    info['invoice_number'] = re.search(r'Invoice #\s*([\w-]+)', text, re.IGNORECASE).group(1) if re.search(r'Invoice #\s*([\w-]+)', text, re.IGNORECASE) else "N/A"
    info['invoice_date'] = re.search(r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b', text).group(1) if re.search(r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b', text) else "N/A"
    info['due_date'] = re.search(r'Due Date\s*[:\-]\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', text, re.IGNORECASE).group(1) if re.search(r'Due Date\s*[:\-]\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', text, re.IGNORECASE) else "N/A"
    info['issuer_name'] = re.search(r'From:\s*(.+?)\n', text, re.IGNORECASE).group(1) if re.search(r'From:\s*(.+?)\n', text, re.IGNORECASE) else "N/A"
    info['recipient_name'] = re.search(r'To:\s*(.+?)\n', text, re.IGNORECASE).group(1) if re.search(r'To:\s*(.+?)\n', text, re.IGNORECASE) else "N/A"
    info['total_amount'] = re.search(r'Total\s*[:$]\s*([\d,.]+)', text, re.IGNORECASE).group(1) if re.search(r'Total\s*[:$]\s*([\d,.]+)', text, re.IGNORECASE) else "N/A"
    return info

# Data Loading
def load_text_data():
    texts, labels = [], []
    for category in CATEGORIES:
        print(f"\nLoading category: {category}")
        folder = os.path.join(ROOT_DIR, category)
        if not os.path.exists(folder):
            print(f"Directory not found: {folder}")
            continue
        all_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTENSIONS)]
        sampled_files = all_files[:min(MAX_SAMPLES_PER_CATEGORY, len(all_files))]
        for i, file_path in enumerate(sampled_files, 1):
            text = extract_text_from_image(file_path)
            if text:
                texts.append(text)
                labels.append(category)
            print(f"Processed {i}/{len(sampled_files)} files in {category}")
    return texts, labels

# Tokenization
def tokenize_texts(texts):
    print("\nTokenizing text data...")
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    print("Tokenization complete.")
    return padded_sequences, tokenizer

# Label Encoding
def encode_labels(labels):
    print("Encoding labels...")
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels)
    print(f"Labels encoded as: {dict(zip(encoder.classes_, range(len(encoder.classes_))))}")
    return encoded, encoder

# Improved LSTM Model
def build_lstm_model(num_classes):
    print("\nBuilding LSTM model...")
    model = Sequential()
    model.add(Embedding(input_dim=MAX_NUM_WORDS, output_dim=128, input_length=MAX_SEQUENCE_LENGTH))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))  
    model.add(LSTM(64))
    model.add(Dropout(0.2))  
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

# Load Preprocessed Data
def load_preprocessed_data():
    if not all(os.path.exists(os.path.join(PREPROCESSED_DIR, f)) for f in 
               ["X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy"]):
        print("Preprocessed data not found.")
        return None, None, None, None
    X_train = np.load(os.path.join(PREPROCESSED_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(PREPROCESSED_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(PREPROCESSED_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(PREPROCESSED_DIR, "y_test.npy"))
    print("Loaded preprocessed data from", PREPROCESSED_DIR)
    return X_train, X_test, y_train, y_test

# Training and Saving
def train_model():
    # Try loading preprocessed data first
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    if X_train is None:  # If not found, preprocess from scratch
        texts, labels = load_text_data()
        if not texts:
            print("No valid text data extracted. Exiting.")
            return None, None, None
        X, tokenizer = tokenize_texts(texts)
        y, encoder = encode_labels(labels)
        y_categorical = to_categorical(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42, stratify=y)
        
        # Save preprocessed data
        os.makedirs(PREPROCESSED_DIR, exist_ok=True)
        np.save(os.path.join(PREPROCESSED_DIR, "X_train.npy"), X_train)
        np.save(os.path.join(PREPROCESSED_DIR, "X_test.npy"), X_test)
        np.save(os.path.join(PREPROCESSED_DIR, "y_train.npy"), y_train)
        np.save(os.path.join(PREPROCESSED_DIR, "y_test.npy"), y_test)
        print(f"Saved preprocessed data to {PREPROCESSED_DIR}")
    else:
        # Load tokenizer and encoder if preprocessed data exists
        if os.path.exists(TOKENIZER_PATH) and os.path.exists(ENCODER_PATH):
            with open(TOKENIZER_PATH, 'rb') as f:
                tokenizer = pickle.load(f)
            with open(ENCODER_PATH, 'rb') as f:
                encoder = pickle.load(f)
        else:
            print("Tokenizer or encoder missing despite preprocessed data. Reprocessing required.")
            return None, None, None

    model = build_lstm_model(num_classes=y_train.shape[1])

    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

    print("\nTraining model...")
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, 
                        callbacks=[early_stopping], verbose=1)

    print("\nEvaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    # Plot history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Save the model, tokenizer, and encoder
    model.save(MODEL_PATH)
    with open(TOKENIZER_PATH, 'wb') as f:
        pickle.dump(tokenizer, f)
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(encoder, f)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Tokenizer saved to {TOKENIZER_PATH}")
    print(f"Encoder saved to {ENCODER_PATH}")

    return model, tokenizer, encoder

# Load Saved Model
def load_trained_model():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH) and os.path.exists(ENCODER_PATH)):
        print("Saved model files not found. Please train the model first.")
        return None, None, None
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    with open(ENCODER_PATH, 'rb') as f:
        encoder = pickle.load(f)
    print("Loaded saved model, tokenizer, and encoder.")
    return model, tokenizer, encoder

# Process a Single Document
def process_document(image_path, model, tokenizer, encoder):
    text = extract_text_from_image(image_path)
    if not text:
        return {"category": "error", "info": {}}
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    prediction = model.predict(padded)
    category_id = np.argmax(prediction, axis=1)[0]
    category = encoder.inverse_transform([category_id])[0]
    info = extract_invoice_info(text) if category == "invoice" else {}
    return {"category": category, "info": info}

if __name__ == "__main__":
    print("Starting document classification using OCR and LSTM...\n")
    
    # Option to train or load
    TRAIN = True  # Set to True to train now
    if TRAIN:
        model, tokenizer, encoder = train_model()
    else:
        model, tokenizer, encoder = load_trained_model()
    
    if model and tokenizer and encoder:
        sample_doc = "/Users/triniroca/Desktop/stats_new/financial-data/invoice/2028724946.jpeg"
        result = process_document(sample_doc, model, tokenizer, encoder)
        print(f"Result: {result}")