import sys
import os
import json
import random
import cv2
import pytesseract
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Windows tesseract necessity
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"


ROOT_DIR = "./Dataset"
CATEGORIES = ["advertisement", "email", "invoice", "memo", "resume"]
MAX_SAMPLES_PER_CATEGORY = 1000
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')
MAX_NUM_WORDS = 10000
MAX_SEQUENCE_LENGTH = 200

DEBUG_SHOW_IMAGES = False
PROCESSED_DATA_FILE = "processed_data.json"



def preprocess_image_for_ocr(image_path):
    print(f"Preprocessing image: {image_path}")
    img = cv2.imread(image_path)

    if img is None:
        print("Could not load image!!!!!!!")
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #return gray # For more efficient processing

    # Apply denoising (Non-Local Means)
    denoised = cv2.fastNlMeansDenoising(gray, h=30, templateWindowSize=7, searchWindowSize=21)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # Adaptive Thresholding (local binarization)
    thresh = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    # Morphological closing (optional: bridge broken character lines)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    morph = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)

    # Optionally show image processing stages
    if DEBUG_SHOW_IMAGES:
        cv2.imshow("Original", img)
        cv2.imshow("Grayscale", gray)
        cv2.imshow("Denoised", denoised)
        cv2.imshow("CLAHE", enhanced)
        cv2.imshow("Morphological Closing", morph)
        cv2.imshow("Thresholded", thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Manually select best
    img = gray

    return img


def extract_text_from_image(image_path):
    print(f"OCR processing: {image_path}")
    processed_img = preprocess_image_for_ocr(image_path)
    if processed_img is None:
        return ""
    text = pytesseract.image_to_string(processed_img)
    if DEBUG_SHOW_IMAGES: print(text.strip())
    return text.strip()


def load_or_create_processed_data():
    texts, labels = load_processed_data()
    if not texts or not labels:
        texts = []
        labels = []

        for category in CATEGORIES:
            print(f"\nLoading category: {category}")
            folder = os.path.join(ROOT_DIR, category)
            all_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTENSIONS)]
            sampled_files = random.sample(all_files, min(MAX_SAMPLES_PER_CATEGORY, len(all_files)))

            for file_path in sampled_files:
                text = extract_text_from_image(file_path)
                if text:
                    texts.append(text)
                    labels.append(category)
                print()
        save_processed_data(texts, labels)

    return texts, labels


def save_processed_data(texts, labels):
    # Prepare data to be saved
    data = {
        "texts": texts,
        "labels": labels
    }

    try:
        with open(PROCESSED_DATA_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Processed data saved to {PROCESSED_DATA_FILE}")
    except Exception as e:
        print(f"Error saving processed data: {e}")


def load_processed_data():
    if not os.path.exists(PROCESSED_DATA_FILE):
        print("Processed data file not found. Please run OCR on the images first.")
        return None, None

    try:
        with open(PROCESSED_DATA_FILE, 'r') as f:
            data = json.load(f)
        
        texts = data.get("texts", [])
        labels = data.get("labels", [])
        
        if len(texts) == 0 or len(labels) == 0:
            print("Processed data is empty. Please run OCR on the images first.")
            return None, None
        
        print(f"Loaded processed data from {PROCESSED_DATA_FILE}")
        return texts, labels
    except Exception as e:
        print(f"Error loading processed data!!!! {e}")
        return None, None


def tokenize_texts(texts):
    print("\nTokenizing text data...")
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    print("Tokenization complete.")
    return padded_sequences, tokenizer


def encode_labels(labels):
    print("Encoding labels...")
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels)
    print(f"Labels encoded as: {dict(zip(encoder.classes_, range(len(encoder.classes_))))}")
    return encoded, encoder


def train_svm(X_train, y_train):
    print("\nTraining SVM model...")
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)
    return svm_model


def train_random_forest(X_train, y_train):
    print("\nTraining Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model


def build_lstm_model(num_classes):
    print("\nBuilding LSTM model...")
    model = Sequential()
    model.add(Embedding(input_dim=10000, output_dim=128, input_length=200))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model


def train_lstm(X_train, y_train, X_test, y_test):
    num_classes = len(CATEGORIES)
    model = build_lstm_model(num_classes)

    y_train_one_hot = to_categorical(y_train, num_classes=len(CATEGORIES))
    y_test_one_hot = to_categorical(y_test, num_classes=len(CATEGORIES))

    print("\nTraining LSTM model...")
    history = model.fit(X_train, y_train_one_hot, epochs=50, batch_size=32, validation_split=0.2)

    print("\nEvaluating LSTM model...")
    loss, accuracy = model.evaluate(X_test, y_test_one_hot)
    print(f"LSTM Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
    
    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('LSTM Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('LSTM Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


    return model, accuracy


def train_model():
    texts, labels = load_or_create_processed_data()
    X, tokenizer = tokenize_texts(texts)
    y, encoder = encode_labels(labels)
    
    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train SVM and Random Forest models
    #svm_model = train_svm(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    lstm_model, lstm_accuracy = train_lstm(X_train, y_train, X_test, y_test)

    # Evaluate models
    #svm_predictions = svm_model.predict(X_test)
    rf_predictions = rf_model.predict(X_test)

    #svm_accuracy = accuracy_score(y_test, svm_predictions)
    rf_accuracy = accuracy_score(y_test, rf_predictions)

    #print(f"SVM Accuracy: {svm_accuracy:.4f}")
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    print(f"LSTM Accuracy: {lstm_accuracy:.4f}")

    return rf_model, tokenizer, encoder


def predict_singular(image_path, rf_model, tokenizer, encoder):
    # OCR
    print(f"Processing image for prediction: {image_path}")
    text = extract_text_from_image(image_path)
    
    if not text:
        print("No text extracted from image.")
        return None
    
    # Tokenize
    print("Tokenizing text...")
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    
    # Predict
    print("Making prediction with Random Forest model...")
    prediction = rf_model.predict(padded_sequence)
    
    # Get label
    predicted_class_index = prediction[0]
    predicted_class_label = encoder.inverse_transform([predicted_class_index])[0]
    
    print(f"Predicted class: {predicted_class_label}")
    return predicted_class_label


if __name__ == "__main__":
    print("Starting document classification using OCR and Machine Learning...\n")
    rf_model, tokenizer, encoder = train_model()

    predict_singular("test_image.jpeg", rf_model, tokenizer, encoder)