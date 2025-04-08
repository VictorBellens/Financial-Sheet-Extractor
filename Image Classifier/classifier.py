import os
import cv2
import pytesseract
import re
import numpy as np
import pickle
import tempfile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Adjust this path based on your system
if os.name == 'nt':  # Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:  # macOS/Linux
    pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

# Constants
MAX_SEQUENCE_LENGTH = 200
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model11.h5")
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "tokenizer11.pkl")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "encoder11.pkl")

def preprocess_image_for_ocr(image_path):
    """Preprocess an image for better OCR results."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image: {image_path}")
            return None
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising (Non-Local Means) with enhanced parameters
        denoised = cv2.fastNlMeansDenoising(gray, h=30, templateWindowSize=7, searchWindowSize=21)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Apply adaptive thresholding for better text extraction
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological closing to bridge small gaps in text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        morph = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        
        # Return the thresholded image for best OCR results
        return thresh
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def clean_text(text):
    """Clean and normalize extracted text."""
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^\w\s.,-/:]', '', text)
    return text.lower()

def extract_text_from_image(image_path):
    """Extract text from an image using OCR."""
    try:
        processed_img = preprocess_image_for_ocr(image_path)
        if processed_img is None:
            return ""
            
        text = pytesseract.image_to_string(processed_img, config='--psm 6')
        return clean_text(text)
    except Exception as e:
        print(f"OCR failed for {image_path}: {e}")
        return ""

def extract_invoice_info(text):
    """Extract key information from invoice text."""
    info = {}
    
    # Extract invoice number
    invoice_number_match = re.search(r'invoice #\s*([\w-]+)', text, re.IGNORECASE)
    if not invoice_number_match:
        invoice_number_match = re.search(r'invoice\s*number\s*[:#]?\s*([\w-]+)', text, re.IGNORECASE)
    if not invoice_number_match:
        invoice_number_match = re.search(r'invoice\s*no\s*[:#]?\s*([\w-]+)', text, re.IGNORECASE)
    info['invoice_number'] = invoice_number_match.group(1) if invoice_number_match else "N/A"
    
    # Extract invoice date
    date_match = re.search(r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b', text)
    if not date_match:
        date_match = re.search(r'date\s*[:\-]\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', text, re.IGNORECASE)
    info['invoice_date'] = date_match.group(1) if date_match else "N/A"
    
    # Extract due date
    due_date_match = re.search(r'due date\s*[:\-]\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', text, re.IGNORECASE)
    if not due_date_match:
        due_date_match = re.search(r'due\s*[:\-]\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', text, re.IGNORECASE)
    info['due_date'] = due_date_match.group(1) if due_date_match else "N/A"
    
    # Extract issuer name (From:)
    issuer_match = re.search(r'from\s*:\s*(.+?)\n', text, re.IGNORECASE)
    if not issuer_match:
        # Look for company name at the top
        lines = text.split('\n')
        if len(lines) > 0:
            info['issuer_name'] = lines[0].strip()
        else:
            info['issuer_name'] = "N/A"
    else:
        info['issuer_name'] = issuer_match.group(1)
    
    # Extract recipient name (To:)
    recipient_match = re.search(r'to\s*:\s*(.+?)\n', text, re.IGNORECASE)
    if not recipient_match:
        recipient_match = re.search(r'bill to\s*:\s*(.+?)\n', text, re.IGNORECASE)
    info['recipient_name'] = recipient_match.group(1) if recipient_match else "N/A"
    
    # Extract total amount
    total_match = re.search(r'total\s*[:$]\s*([\d,.]+)', text, re.IGNORECASE)
    if not total_match:
        total_match = re.search(r'amount\s*due\s*[:$]\s*([\d,.]+)', text, re.IGNORECASE)
    if not total_match:
        total_match = re.search(r'balance\s*due\s*[:$]\s*([\d,.]+)', text, re.IGNORECASE)
    info['total_amount'] = total_match.group(1) if total_match else "N/A"
    
    return info

class DocumentClassifier:
    """Class for classifying documents using a pre-trained model."""
    
    def __init__(self):
        """Initialize the classifier, loading the model, tokenizer, and encoder."""
        self.model = None
        self.tokenizer = None
        self.encoder = None
        self.loaded = self.load_model()
        
    def load_model(self):
        """Load the pre-trained model."""
        try:
            self.model = load_model(MODEL_PATH)
            with open(TOKENIZER_PATH, 'rb') as f:
                self.tokenizer = pickle.load(f)
            with open(ENCODER_PATH, 'rb') as f:
                self.encoder = pickle.load(f)
            print("Model, tokenizer, and encoder loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
            
    def classify_document(self, image_path):
        """Classify a document image."""
        if not self.loaded:
            return {"category": "error", "info": {}, "confidence": 0.0}
            
        # Extract text from image
        text = extract_text_from_image(image_path)
        if not text:
            return {"category": "error", "info": {}, "confidence": 0.0}
            
        # Tokenize and pad the text
        sequence = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
        
        # Make prediction
        prediction = self.model.predict(padded)
        category_id = np.argmax(prediction, axis=1)[0]
        confidence = float(prediction[0][category_id])
        category = self.encoder.inverse_transform([category_id])[0]
        
        # Extract invoice info if the document is an invoice
        info = extract_invoice_info(text) if category == "invoice" else {}
        
        return {
            "category": category, 
            "info": info, 
            "confidence": confidence,
            "text": text
        }

def process_document(image_path):
    """
    Process a document image for classification and information extraction.
    This is a simplified wrapper for the DocumentClassifier class.
    """
    classifier = DocumentClassifier()
    return classifier.classify_document(image_path)

# For standalone testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        result = process_document(image_path)
        print(f"Document Type: {result['category']}")
        print(f"Confidence: {result['confidence']:.2%}")
        if result['category'] == 'invoice':
            print("\nInvoice Information:")
            for key, value in result['info'].items():
                print(f"{key.replace('_', ' ').title()}: {value}")
    else:
        print("Please provide an image path as an argument.") 