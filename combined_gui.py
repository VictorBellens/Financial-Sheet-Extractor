import os
import sys
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import json
import pickle
import tempfile
import io
import cv2
import pytesseract

# Use PyPDF2 for PDF text extraction (pure Python, no external dependencies)
import PyPDF2
from PIL import Image

# Global flag for lightweight mode (no TensorFlow)
LIGHTWEIGHT_MODE = False  # Set to True by default to avoid TensorFlow errors

# Only import TensorFlow-related modules if not in lightweight mode
if not LIGHTWEIGHT_MODE:
    try:
        import numpy as np
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        print("TensorFlow imported successfully")
    except ImportError as e:
        print(f"Error importing TensorFlow: {e}")
        print("Running in lightweight mode (classification disabled)")
        LIGHTWEIGHT_MODE = True

# Import components from other modules
sys.path.append('./Invoice_Extractor')
from invoice_extractor import InvoiceExtractor, process_directory

# Import image processing capabilities
sys.path.append('./Data_Processor')
from main import preprocess_image_for_ocr, extract_text_from_image

# Only import classifier components if not in lightweight mode
if not LIGHTWEIGHT_MODE:
    sys.path.append('./Image Classifier')
    from stats import extract_text_from_image as classify_extract_text
    from stats import process_document, MAX_SEQUENCE_LENGTH

# Create an enhanced invoice extractor that can handle both PDFs and images
class EnhancedInvoiceExtractor(InvoiceExtractor):
    def __init__(self, file_path=None, is_image=False):
        # For image files, we don't call the parent constructor directly
        if is_image:
            self.pdf_path = file_path  # Store the path but it's actually an image
            self.is_image = True
            
            # Process the image to extract text
            self.text_content = self._extract_image_text()
            self.lines = self.text_content.split('\n')
            
            # Form extractor will be None for images
            self.form_extractor = None
            
            # Extract filename parts for improved extraction
            self.filename_info = self._extract_from_filename()
            
            # Initialize validator and other components
            from extraction_validator import ExtractionValidator
            self.validator = ExtractionValidator()
            self.stopwords = self._load_stopwords()
            self.vendor_patterns = self._load_vendor_patterns()
            
            # Process document content
            if self.text_content:
                self.document_sections = self._extract_document_sections()
                self.document_type = self._classify_document()
                self.entities = self._extract_named_entities()
                self.field_relationships = self._build_field_relationships()
            else:
                self.document_sections = {}
                self.document_type = "unknown"
                self.entities = []
                self.field_relationships = {}
        else:
            # For PDF files, use the parent constructor
            super().__init__(file_path)
            self.is_image = False
    
    def _extract_image_text(self):
        """Extract text from image file using OCR."""
        try:
            # Use the Data_Processor's image preprocessing and text extraction
            return extract_text_from_image(self.pdf_path)
        except Exception as e:
            print(f"Error extracting text from image {self.pdf_path}: {e}")
            return ""
    
    def _extract_text(self):
        """Override to handle both PDFs and images."""
        if self.is_image:
            return self._extract_image_text()
        else:
            return super()._extract_text()

# Function to extract text directly from PDF using PyPDF2
def extract_text_from_pdf(pdf_path):
    """Extract text directly from a PDF file"""
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            
            # Extract text from each page
            for page_num in range(min(num_pages, 5)):  # Limit to first 5 pages for efficiency
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
                
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

# Function to process a directory with both PDFs and images
def enhanced_process_directory(directory_path, output_file):
    """Process all PDFs and images in a directory and extract invoice data."""
    results = []
    
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        # Skip non-file items
        if not os.path.isfile(file_path):
            continue
        
        # Check if it's an image file
        is_image = filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.bmp'))
        
        # Check if it's a PDF file
        is_pdf = filename.lower().endswith('.pdf')
        
        # Skip files that are neither images nor PDFs
        if not (is_image or is_pdf):
            continue
        
        try:
            print(f"Processing: {filename}")
            
            # Use the enhanced extractor
            extractor = EnhancedInvoiceExtractor(file_path, is_image=is_image)
            data = extractor.extract_all()
            
            # Add filename to the result
            result_dict = data.to_dict()
            result_dict['filename'] = filename
            
            results.append(result_dict)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Save results to output file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Processed {len(results)} files. Results saved to {output_file}")
    return results

class CombinedFinancialExtractorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Financial Document Processor" + (" (Lightweight)" if LIGHTWEIGHT_MODE else ""))
        self.root.geometry("900x700")
        self.root.minsize(900, 700)
        
        # Store temporary files for cleanup
        self.temp_files = []
        
        # Load ML models only if not in lightweight mode
        self.classifier_model = None
        self.tokenizer = None
        self.encoder = None
        
        if not LIGHTWEIGHT_MODE:
            self.load_models()
        
        # Setup UI
        self.setup_ui()
        
    def load_models(self):
        if LIGHTWEIGHT_MODE:
            return
            
        # Load the image classifier model
        try:
            model_path = os.path.join("Image Classifier", "model.h5")
            tokenizer_path = os.path.join("Image Classifier", "tokenizer.pkl")
            encoder_path = os.path.join("Image Classifier", "encoder.pkl")
            
            self.classifier_model = load_model(model_path)
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            with open(encoder_path, 'rb') as f:
                self.encoder = pickle.load(f)
            print("Loaded image classifier model successfully")
        except Exception as e:
            print(f"Error loading classifier model: {e}")
            self.classifier_model = None
            self.tokenizer = None
            self.encoder = None
        
    def setup_ui(self):
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create frames for each tab
        self.main_frame = ttk.Frame(self.notebook)
        self.results_frame = ttk.Frame(self.notebook)
        
        # Add tabs to notebook
        self.notebook.add(self.main_frame, text="Document Processing")
        self.notebook.add(self.results_frame, text="Results")
        
        # Setup main processing frame
        self.setup_main_frame()
        
        # Setup results frame
        self.setup_results_frame()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w")
        status_bar.pack(side="bottom", fill="x")
        
        # Store current data
        self.current_data = None
        self.classification_result = None
        
        # Set up cleanup on window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def on_closing(self):
        # Clean up temporary files
        self.cleanup_temp_files()
        # Destroy the window
        self.root.destroy()
        
    def cleanup_temp_files(self):
        """Clean up any temporary files that were created"""
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Removed temporary file: {file_path}")
            except Exception as e:
                print(f"Error removing temporary file {file_path}: {e}")
                
        # Also clean up any temporary directories that might have been created
        for temp_dir in [os.path.dirname(f) for f in self.temp_files if os.path.dirname(f)]:
            try:
                if os.path.exists(temp_dir) and os.path.isdir(temp_dir):
                    os.rmdir(temp_dir)
                    print(f"Removed temporary directory: {temp_dir}")
            except Exception as e:
                print(f"Error removing temporary directory {temp_dir}: {e}")
                
        self.temp_files = []
        
    def setup_main_frame(self):
        # Frame for file selection
        file_frame = ttk.LabelFrame(self.main_frame, text="Document Selection")
        file_frame.pack(fill="x", padx=10, pady=10)
        
        # Single file processing
        ttk.Label(file_frame, text="Select Document:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.file_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path_var, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=5, pady=5)
        
        # Batch processing
        ttk.Label(file_frame, text="Or Process Directory:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.dir_path_var = tk.StringVar(value="Data")  # Default to Data directory
        ttk.Entry(file_frame, textvariable=self.dir_path_var, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_directory).grid(row=1, column=2, padx=5, pady=5)
        
        # Processing options frame
        options_frame = ttk.LabelFrame(self.main_frame, text="Processing Options")
        options_frame.pack(fill="x", padx=10, pady=10)
        
        # Create classification option (disabled in lightweight mode)
        self.classify_var = tk.BooleanVar(value=False if LIGHTWEIGHT_MODE else True)
        classify_checkbox = ttk.Checkbutton(options_frame, text="Classify Document Type", variable=self.classify_var)
        classify_checkbox.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        if LIGHTWEIGHT_MODE:
            classify_checkbox.configure(state="disabled")
            ttk.Label(options_frame, text="(Disabled in lightweight mode)", foreground="gray").grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Create invoice extraction option
        self.extract_invoice_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Extract Invoice Data", variable=self.extract_invoice_var).grid(row=1, column=0, padx=5, pady=5, sticky="w")
        
        # Create data processing option
        self.process_data_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Process Data", variable=self.process_data_var).grid(row=2, column=0, padx=5, pady=5, sticky="w")
        
        # Output filename
        ttk.Label(options_frame, text="Output Filename:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.output_filename_var = tk.StringVar(value="combined_results.json")
        ttk.Entry(options_frame, textvariable=self.output_filename_var, width=30).grid(row=3, column=1, padx=5, pady=5, sticky="w")
        
        # Process buttons frame
        button_frame = ttk.Frame(self.main_frame)
        button_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Button(button_frame, text="Process Single File", command=self.process_single_file).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Process Directory", command=self.process_directory).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Clear", command=self.clear_data).pack(side="left", padx=5)
        
    def setup_results_frame(self):
        # Create a frame for classification results
        classification_frame = ttk.LabelFrame(self.results_frame, text="Document Classification")
        classification_frame.pack(fill="x", padx=10, pady=10)
        
        if LIGHTWEIGHT_MODE:
            ttk.Label(classification_frame, text="Classification is disabled in lightweight mode").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        else:
            ttk.Label(classification_frame, text="Document Type:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
            self.doc_type_var = tk.StringVar(value="")
            ttk.Label(classification_frame, textvariable=self.doc_type_var, font=("Arial", 10, "bold")).grid(row=0, column=1, padx=5, pady=5, sticky="w")
            
            ttk.Label(classification_frame, text="Confidence:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
            self.confidence_var = tk.StringVar(value="")
            ttk.Label(classification_frame, textvariable=self.confidence_var).grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        # Create a frame for invoice extraction results
        invoice_frame = ttk.LabelFrame(self.results_frame, text="Extracted Invoice Data")
        invoice_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create treeview for results
        self.tree = ttk.Treeview(invoice_frame)
        self.tree["columns"] = ("Value")
        self.tree.column("#0", width=200, minwidth=200)
        self.tree.column("Value", width=680, minwidth=200)
        self.tree.heading("#0", text="Field")
        self.tree.heading("Value", text="Value")
        
        # Add fields to treeview
        self.tree.insert("", "end", text="Invoice Number", iid="invoice_number", values=(""))
        self.tree.insert("", "end", text="Invoice Date", iid="invoice_date", values=(""))
        self.tree.insert("", "end", text="Due Date", iid="due_date", values=(""))
        self.tree.insert("", "end", text="Issuer Name", iid="issuer_name", values=(""))
        self.tree.insert("", "end", text="Recipient Name", iid="recipient_name", values=(""))
        self.tree.insert("", "end", text="Total Amount", iid="total_amount", values=(""))
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(invoice_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack treeview and scrollbar
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bottom buttons
        button_frame = ttk.Frame(self.results_frame)
        button_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Button(button_frame, text="Save Results", command=self.save_data).pack(side="left", padx=5)
        
    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Document",
            filetypes=[("PDF Files", "*.pdf"), ("Image Files", "*.png *.jpg *.jpeg"), ("All Files", "*.*")]
        )
        if file_path:
            self.file_path_var.set(file_path)
            
    def browse_directory(self):
        dir_path = filedialog.askdirectory(title="Select Directory with Documents")
        if dir_path:
            self.dir_path_var.set(dir_path)
    
    def classify_document(self, file_path):
        if LIGHTWEIGHT_MODE or not self.classifier_model or not self.tokenizer or not self.encoder:
            return {"category": "unknown", "confidence": 0}
        
        try:
            # Different processing for PDF vs image files
            if file_path.lower().endswith('.pdf'):
                self.status_var.set("Extracting text from PDF for classification...")
                self.root.update_idletasks()
                
                # Extract text directly from PDF
                text = extract_text_from_pdf(file_path)
                if not text:
                    return {"category": "unknown", "confidence": 0}
            else:
                # Use image processing for non-PDF files
                self.status_var.set("Processing image for classification...")
                self.root.update_idletasks()
                
                # Use the Image Classifier's extraction function
                text = classify_extract_text(file_path)
                if not text:
                    return {"category": "unknown", "confidence": 0}
                    
            # Process the extracted text for classification
            sequence = self.tokenizer.texts_to_sequences([text])
            padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
            prediction = self.classifier_model.predict(padded)
            
            category_id = np.argmax(prediction, axis=1)[0]
            confidence = float(prediction[0][category_id])
            category = self.encoder.inverse_transform([category_id])[0]
            
            return {"category": category, "confidence": confidence}
        except Exception as e:
            print(f"Classification error: {e}")
            return {"category": "error", "confidence": 0}
    
    def process_single_file(self):
        file_path = self.file_path_var.get()
        if not file_path:
            messagebox.showerror("Error", "Please select a file.")
            return
            
        if not os.path.exists(file_path):
            messagebox.showerror("Error", f"File not found: {file_path}")
            return
            
        try:
            self.status_var.set(f"Processing {os.path.basename(file_path)}...")
            self.root.update_idletasks()
            
            results = {}
            
            # Determine if the file is an image
            is_image = file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.bmp'))
            
            # Classify document if selected and not in lightweight mode
            if not LIGHTWEIGHT_MODE and self.classify_var.get():
                classification = self.classify_document(file_path)
                self.classification_result = classification
                results["document_type"] = classification["category"]
                results["classification_confidence"] = classification["confidence"]
                
                # Update classification results
                self.doc_type_var.set(classification["category"])
                self.confidence_var.set(f"{classification['confidence']:.2%}")
            
            # Extract invoice data if selected 
            # In lightweight mode, always extract invoice data if selected
            # In normal mode, check classification if classify_var is True
            should_extract = self.extract_invoice_var.get() and (
                LIGHTWEIGHT_MODE or 
                not self.classify_var.get() or 
                self.classification_result["category"] == "invoice"
            )
            
            if should_extract:
                # Use the enhanced extractor that can handle both PDFs and images
                extractor = EnhancedInvoiceExtractor(file_path, is_image=is_image)
                invoice_data = extractor.extract_all()
                self.current_data = invoice_data
                
                # Update treeview
                self.tree.item("invoice_number", values=(invoice_data.invoice_number or ""))
                self.tree.item("invoice_date", values=(invoice_data.invoice_date or ""))
                self.tree.item("due_date", values=(invoice_data.due_date or ""))
                self.tree.item("issuer_name", values=(invoice_data.issuer_name or ""))
                self.tree.item("recipient_name", values=(invoice_data.recipient_name or ""))
                self.tree.item("total_amount", values=(invoice_data.total_amount or ""))
                
                # Add to results
                results["invoice_data"] = invoice_data.to_dict() if invoice_data else {}
            
            # Switch to results tab
            self.notebook.select(self.results_frame)
            
            self.status_var.set(f"Successfully processed {os.path.basename(file_path)}")
            
            # Store results
            self.current_results = results
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process file: {str(e)}")
            self.status_var.set("Error processing file")
    
    def process_directory(self):
        dir_path = self.dir_path_var.get()
        if not dir_path:
            messagebox.showerror("Error", "Please select a directory.")
            return
            
        if not os.path.exists(dir_path):
            messagebox.showerror("Error", f"Directory not found: {dir_path}")
            return
            
        try:
            self.status_var.set(f"Processing documents in {dir_path}...")
            self.root.update_idletasks()
            
            output_file = self.output_filename_var.get()
            
            # If invoice extraction is selected, process directory with enhanced directory processor
            if self.extract_invoice_var.get():
                # Use the enhanced process_directory function that handles both PDFs and images
                enhanced_process_directory(dir_path, output_file)
            
            # If classification is selected and not in lightweight mode
            if not LIGHTWEIGHT_MODE and self.classify_var.get():
                all_results = []
                
                for filename in os.listdir(dir_path):
                    # Include both image and PDF extensions
                    if filename.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                        file_path = os.path.join(dir_path, filename)
                        
                        # Determine if the file is an image
                        is_image = filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.bmp'))
                        
                        # Classify the document
                        classification = self.classify_document(file_path)
                        
                        # Extract invoice data if it's an invoice
                        invoice_data = None
                        if self.extract_invoice_var.get() and classification["category"] == "invoice":
                            try:
                                extractor = EnhancedInvoiceExtractor(file_path, is_image=is_image)
                                invoice_data = extractor.extract_all()
                            except Exception as e:
                                print(f"Error extracting invoice data from {filename}: {e}")
                        
                        # Add to results
                        result = {
                            "filename": filename,
                            "document_type": classification["category"],
                            "classification_confidence": classification["confidence"]
                        }
                        
                        if invoice_data:
                            result["invoice_data"] = invoice_data.to_dict()
                            
                        all_results.append(result)
                
                # Save combined results
                with open(output_file, 'w') as f:
                    json.dump(all_results, f, indent=2)
            
            messagebox.showinfo("Success", f"Processing complete!\nResults saved to {output_file}")
            self.status_var.set(f"Successfully processed directory. Results saved to {output_file}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process directory: {str(e)}")
            self.status_var.set("Error processing directory")
    
    def save_data(self):
        if not hasattr(self, 'current_results') or not self.current_results:
            messagebox.showerror("Error", "No data to save.")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Extracted Data",
            defaultextension=".json",
            initialfile=self.output_filename_var.get(),
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(self.current_results, f, indent=2)
                self.status_var.set(f"Data saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save data: {str(e)}")
    
    def clear_data(self):
        # Clear file path
        self.file_path_var.set("")
        
        # Clear results
        self.current_data = None
        self.classification_result = None
        self.current_results = None
        
        # Clear classification results if not in lightweight mode
        if not LIGHTWEIGHT_MODE:
            self.doc_type_var.set("")
            self.confidence_var.set("")
        
        # Clear treeview
        self.tree.item("invoice_number", values=(""))
        self.tree.item("invoice_date", values=(""))
        self.tree.item("due_date", values=(""))
        self.tree.item("issuer_name", values=(""))
        self.tree.item("recipient_name", values=(""))
        self.tree.item("total_amount", values=(""))
        
        self.status_var.set("Ready")
        
        # Clean up any temporary files
        self.cleanup_temp_files()

if __name__ == "__main__":
    root = tk.Tk()
    app = CombinedFinancialExtractorGUI(root)
    root.mainloop() 