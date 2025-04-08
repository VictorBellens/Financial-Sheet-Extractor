import os
import sys
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import json
import tempfile
import threading
import time
from PIL import Image, ImageTk

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add paths using absolute paths
sys.path.append(os.path.join(current_dir, 'utility'))
sys.path.append(os.path.join(current_dir, 'Invoice_Extractor'))
sys.path.append(os.path.join(current_dir, 'Image Classifier'))

# Now import the modules
from image_utils import preprocess_image_pipeline
from invoice_extractor import InvoiceExtractor, process_directory
from Invoice_Extractor.enhanced_image_extractor import ImageInvoiceExtractor
from classifier import DocumentClassifier, process_document

# Global flag for lightweight mode (no TensorFlow)
LIGHTWEIGHT_MODE = False

class FinancialDocumentProcessor:
    """Core document processing functionality separate from the GUI."""
    
    def __init__(self):
        self.temp_files = []
        self.classifier = None if LIGHTWEIGHT_MODE else DocumentClassifier()
        self.image_extractor = ImageInvoiceExtractor()
        
    def cleanup_temp_files(self):
        """Clean up any temporary files that were created."""
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Removed temporary file: {file_path}")
            except Exception as e:
                print(f"Error removing temporary file {file_path}: {e}")
        self.temp_files = []
    
    def process_image(self, image_path):
        """Process a single image file."""
        try:
            # Create a temporary directory for processed images
            temp_dir = tempfile.mkdtemp()
            
            # Phase 1: Document Classification (if enabled)
            # -------------------------------------------------
            doc_category = "invoice"  # Default category
            doc_confidence = 1.0
            text_content = ""
            
            if not LIGHTWEIGHT_MODE and self.classifier:
                # Preprocess the image for classification
                jpeg_path, enhanced_path = preprocess_image_pipeline(image_path, temp_dir)
                
                # Add to temp files list for cleanup
                if jpeg_path:
                    self.temp_files.append(jpeg_path)
                if enhanced_path:
                    self.temp_files.append(enhanced_path)
                
                # If preprocessing failed, return error
                if not enhanced_path:
                    return {"error": "Failed to preprocess image for classification"}
                
                # Classify the document
                classification_result = self.classifier.classify_document(enhanced_path)
                
                # Extract the category and confidence
                doc_category = classification_result.get("category", "unknown")
                doc_confidence = classification_result.get("confidence", 0.0)
                text_content = classification_result.get("text", "")
            
            # Phase 2: Invoice Information Extraction (if applicable)
            # -------------------------------------------------
            invoice_data = {}
            
            if doc_category == "invoice":
                # Use the specialized image invoice extractor
                invoice_result = self.image_extractor.extract(image_path, use_enhanced=True)
                invoice_data = invoice_result.to_dict()
            
            # Combine and return the results
            result = {
                "category": doc_category,
                "confidence": doc_confidence,
                "info": invoice_data,
                "text": text_content
            }
            
            return result
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return {"error": str(e)}
    
    def process_pdf(self, pdf_path):
        """Process a PDF document."""
        try:
            # Extract invoice data using InvoiceExtractor
            extractor = InvoiceExtractor(pdf_path)
            invoice_data = extractor.extract_all()
            
            # Convert to dict for easier handling
            result = {
                "category": "invoice",
                "confidence": 1.0,
                "info": invoice_data.to_dict(),
                "text": extractor.text_content
            }
            
            return result
            
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return {"error": str(e)}
    
    def process_file(self, file_path):
        """Process a single file (PDF or image)."""
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}
        
        try:
            # Determine file type
            file_ext = os.path.splitext(file_path.lower())[1]
            
            # Process based on file type
            if file_ext in ['.pdf']:
                return self.process_pdf(file_path)
            elif file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                return self.process_image(file_path)
            else:
                return {"error": f"Unsupported file type: {file_ext}"}
                
        except Exception as e:
            print(f"Error processing file: {e}")
            return {"error": str(e)}
    
    def process_directory(self, dir_path, output_file):
        """Process all files in a directory."""
        if not os.path.exists(dir_path):
            return {"error": f"Directory not found: {dir_path}"}
            
        results = []
        
        try:
            for filename in os.listdir(dir_path):
                file_path = os.path.join(dir_path, filename)
                
                # Skip directories and non-supported files
                if os.path.isdir(file_path):
                    continue
                    
                file_ext = os.path.splitext(filename.lower())[1]
                if file_ext not in ['.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                    continue
                
                # Process file
                result = self.process_file(file_path)
                
                # Add filename to result
                result["filename"] = filename
                
                # Add to results
                results.append(result)
                
            # Save results to output file
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
                
            return {"success": True, "count": len(results), "output_file": output_file}
            
        except Exception as e:
            print(f"Error processing directory: {e}")
            return {"error": str(e)}


class FinancialDocumentProcessorGUI:
    """GUI for the Financial Document Processor."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Financial Document Processor" + (" (Lightweight)" if LIGHTWEIGHT_MODE else ""))
        self.root.geometry("800x600")
        self.root.minsize(800, 600)
        
        # Create processor
        self.processor = FinancialDocumentProcessor()
        
        # Setup UI
        self.setup_ui()
        
        # Set up cleanup on window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Store current data
        self.current_results = None
        self.processing_thread = None
        
    def on_closing(self):
        """Handle window closing."""
        # Cleanup temp files
        self.processor.cleanup_temp_files()
        # Destroy window
        self.root.destroy()
        
    def setup_ui(self):
        """Set up the user interface."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
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
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(self.root, variable=self.progress_var, maximum=100)
        self.progress.pack(side="bottom", fill="x", padx=5, pady=2)
        
    def setup_main_frame(self):
        """Set up the main processing frame."""
        # File selection frame
        file_frame = ttk.LabelFrame(self.main_frame, text="Document Selection")
        file_frame.pack(fill="x", padx=5, pady=5)
        
        # Single file processing
        ttk.Label(file_frame, text="Select Document:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.file_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path_var, width=40).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=5, pady=5)
        
        # Batch processing
        ttk.Label(file_frame, text="Or Process Directory:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.dir_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.dir_path_var, width=40).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_directory).grid(row=1, column=2, padx=5, pady=5)
        
        # Output filename
        ttk.Label(file_frame, text="Output Filename:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.output_filename_var = tk.StringVar(value="results.json")
        ttk.Entry(file_frame, textvariable=self.output_filename_var, width=40).grid(row=2, column=1, padx=5, pady=5)
        
        # Process buttons frame
        button_frame = ttk.Frame(self.main_frame)
        button_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(button_frame, text="Process File", command=self.process_single_file).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Process Directory", command=self.process_directory).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Clear", command=self.clear_data).pack(side="left", padx=5)
        
        # Options frame
        options_frame = ttk.LabelFrame(self.main_frame, text="Options")
        options_frame.pack(fill="x", padx=5, pady=5)
        
        # Classification option
        self.classify_var = tk.BooleanVar(value=True)
        classify_chk = ttk.Checkbutton(options_frame, text="Classify Document Type", variable=self.classify_var)
        classify_chk.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        if LIGHTWEIGHT_MODE:
            classify_chk.configure(state="disabled")
            ttk.Label(options_frame, text="(Disabled in lightweight mode)", foreground="gray").grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Preprocessing option
        self.preprocess_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Preprocess Images", variable=self.preprocess_var).grid(row=1, column=0, padx=5, pady=5, sticky="w")
        
        # Image preview frame
        preview_frame = ttk.LabelFrame(self.main_frame, text="Document Preview")
        preview_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create canvas for image preview
        self.preview_canvas = tk.Canvas(preview_frame, bg="white")
        self.preview_canvas.pack(fill="both", expand=True, padx=5, pady=5)
        
    def setup_results_frame(self):
        """Set up the results frame."""
        # Classification results frame
        classification_frame = ttk.LabelFrame(self.results_frame, text="Document Classification")
        classification_frame.pack(fill="x", padx=5, pady=5)
        
        # Classification results
        ttk.Label(classification_frame, text="Document Type:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.doc_type_var = tk.StringVar()
        ttk.Label(classification_frame, textvariable=self.doc_type_var, font=("Arial", 10, "bold")).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        ttk.Label(classification_frame, text="Confidence:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.confidence_var = tk.StringVar()
        ttk.Label(classification_frame, textvariable=self.confidence_var).grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        # Invoice data frame
        invoice_frame = ttk.LabelFrame(self.results_frame, text="Extracted Data")
        invoice_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create treeview for invoice data
        self.tree = ttk.Treeview(invoice_frame)
        self.tree["columns"] = ("Value")
        self.tree.column("#0", width=150, minwidth=150)
        self.tree.column("Value", width=400, minwidth=200)
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
        button_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(button_frame, text="Save Results", command=self.save_results).pack(side="left", padx=5)
        
    def browse_file(self):
        """Open a file dialog to select a document."""
        file_path = filedialog.askopenfilename(
            title="Select Document",
            filetypes=[
                ("All Supported Files", "*.pdf *.jpg *.jpeg *.png *.tiff *.bmp"),
                ("PDF Files", "*.pdf"),
                ("Image Files", "*.jpg *.jpeg *.png *.tiff *.bmp"),
                ("All Files", "*.*")
            ]
        )
        if file_path:
            self.file_path_var.set(file_path)
            self.display_preview(file_path)
            
    def browse_directory(self):
        """Open a directory dialog to select a folder."""
        dir_path = filedialog.askdirectory(title="Select Directory with Documents")
        if dir_path:
            self.dir_path_var.set(dir_path)
            
    def display_preview(self, file_path):
        """Display a preview of the selected document."""
        try:
            # Clear canvas
            self.preview_canvas.delete("all")
            
            # Check file type
            file_ext = os.path.splitext(file_path.lower())[1]
            
            if file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                # For image files, show both original and processed versions
                # Create a temporary directory for processed images
                temp_dir = tempfile.mkdtemp()
                self.temp_preview_dir = temp_dir
                
                # Process the image to show the enhanced version
                if self.preprocess_var.get():
                    self.status_var.set(f"Preprocessing image for preview...")
                    self.root.update_idletasks()
                    
                    # Preprocess the image to show enhancement
                    jpeg_path, enhanced_path = preprocess_image_pipeline(file_path, temp_dir)
                    
                    # Add to temp files for cleanup
                    if jpeg_path:
                        self.processor.temp_files.append(jpeg_path)
                    if enhanced_path:
                        self.processor.temp_files.append(enhanced_path)
                        
                    # If preprocessing succeeded, use the enhanced image for preview
                    if enhanced_path:
                        preview_path = enhanced_path
                        is_processed = True
                    else:
                        preview_path = file_path
                        is_processed = False
                else:
                    preview_path = file_path
                    is_processed = False
                
                # Display image preview
                img = Image.open(preview_path)
                
                # Resize if needed
                canvas_width = self.preview_canvas.winfo_width()
                canvas_height = self.preview_canvas.winfo_height()
                
                if canvas_width > 1 and canvas_height > 1:
                    # Calculate aspect ratio
                    img_width, img_height = img.size
                    aspect_ratio = img_width / img_height
                    
                    if img_width > canvas_width:
                        img_width = canvas_width
                        img_height = int(img_width / aspect_ratio)
                        
                    if img_height > canvas_height:
                        img_height = canvas_height
                        img_width = int(img_height * aspect_ratio)
                        
                    img = img.resize((img_width, img_height), Image.LANCZOS)
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(img)
                
                # Keep a reference to avoid garbage collection
                self.photo = photo
                
                # Display on canvas
                self.preview_canvas.create_image(
                    canvas_width//2, canvas_height//2, 
                    image=photo, anchor="center"
                )
                
                # Display a label indicating this is the processed version
                if is_processed:
                    self.preview_canvas.create_text(
                        canvas_width//2, 20,
                        text="Processed Image Preview",
                        font=("Arial", 10, "bold"),
                        fill="green"
                    )
                
                self.status_var.set("Ready")
                
            elif file_ext == '.pdf':
                # Display PDF icon or "PDF Document" text
                canvas_width = self.preview_canvas.winfo_width()
                canvas_height = self.preview_canvas.winfo_height()
                
                self.preview_canvas.create_text(
                    canvas_width//2, canvas_height//2,
                    text="PDF Document",
                    font=("Arial", 14, "bold"),
                    fill="gray"
                )
                
        except Exception as e:
            print(f"Error displaying preview: {e}")
            self.status_var.set(f"Error in preview: {str(e)}")
            
            # Display error message on canvas
            canvas_width = self.preview_canvas.winfo_width()
            canvas_height = self.preview_canvas.winfo_height()
            
            self.preview_canvas.create_text(
                canvas_width//2, canvas_height//2,
                text=f"Error displaying preview:\n{str(e)}",
                font=("Arial", 10),
                fill="red",
                justify="center"
            )
            
    def process_single_file(self):
        """Process a single document file."""
        file_path = self.file_path_var.get()
        if not file_path:
            messagebox.showerror("Error", "Please select a file to process.")
            return
            
        # Set status
        self.status_var.set(f"Processing {os.path.basename(file_path)}...")
        self.progress_var.set(10)
        self.root.update_idletasks()
        
        # Process in a separate thread to avoid freezing UI
        self.processing_thread = threading.Thread(
            target=self._process_file_thread,
            args=(file_path,)
        )
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Start monitoring thread progress
        self.root.after(100, self._check_processing_thread)
        
    def _process_file_thread(self, file_path):
        """Thread function to process a file."""
        try:
            # Process file
            self.current_results = self.processor.process_file(file_path)
            
        except Exception as e:
            self.current_results = {"error": str(e)}
            
    def _check_processing_thread(self):
        """Monitor the processing thread."""
        if self.processing_thread and self.processing_thread.is_alive():
            # Update progress bar (simulate progress)
            current = self.progress_var.get()
            if current < 90:
                self.progress_var.set(current + 5)
            
            # Check again later
            self.root.after(100, self._check_processing_thread)
        else:
            # Processing complete
            self.progress_var.set(100)
            
            # Update UI with results
            self._update_results_ui()
            
            # Reset
            self.processing_thread = None
            
    def _update_results_ui(self):
        """Update the UI with processing results."""
        if not self.current_results:
            return
            
        # Check for errors
        if "error" in self.current_results:
            messagebox.showerror("Error", f"Processing failed: {self.current_results['error']}")
            self.status_var.set("Error: Processing failed")
            return
            
        # Update classification results
        self.doc_type_var.set(self.current_results.get("category", "Unknown"))
        self.confidence_var.set(f"{self.current_results.get('confidence', 0):.2%}")
        
        # Update invoice data if available
        info = self.current_results.get("info", {})
        if not info and self.current_results.get("category") == "invoice":
            # Try to get info from text
            from classifier import extract_invoice_info
            info = extract_invoice_info(self.current_results.get("text", ""))
            
        # Update treeview
        self.tree.item("invoice_number", values=(info.get("invoice_number", "")))
        self.tree.item("invoice_date", values=(info.get("invoice_date", "")))
        self.tree.item("due_date", values=(info.get("due_date", "")))
        self.tree.item("issuer_name", values=(info.get("issuer_name", "")))
        self.tree.item("recipient_name", values=(info.get("recipient_name", "")))
        self.tree.item("total_amount", values=(info.get("total_amount", "")))
        
        # Switch to results tab
        self.notebook.select(self.results_frame)
        
        self.status_var.set(f"Successfully processed document")
        
    def process_directory(self):
        """Process all documents in a directory."""
        dir_path = self.dir_path_var.get()
        if not dir_path:
            messagebox.showerror("Error", "Please select a directory to process.")
            return
            
        output_file = self.output_filename_var.get()
        
        # Set status
        self.status_var.set(f"Processing documents in {dir_path}...")
        self.progress_var.set(10)
        self.root.update_idletasks()
        
        # Process in a separate thread to avoid freezing UI
        self.processing_thread = threading.Thread(
            target=self._process_directory_thread,
            args=(dir_path, output_file)
        )
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Start monitoring thread progress
        self.root.after(100, self._check_directory_thread)
        
    def _process_directory_thread(self, dir_path, output_file):
        """Thread function to process a directory."""
        try:
            # Process directory
            self.current_results = self.processor.process_directory(dir_path, output_file)
            
        except Exception as e:
            self.current_results = {"error": str(e)}
            
    def _check_directory_thread(self):
        """Monitor the directory processing thread."""
        if self.processing_thread and self.processing_thread.is_alive():
            # Update progress bar (simulate progress)
            current = self.progress_var.get()
            if current < 90:
                self.progress_var.set(current + 2)
            
            # Check again later
            self.root.after(200, self._check_directory_thread)
        else:
            # Processing complete
            self.progress_var.set(100)
            
            # Show results
            if self.current_results:
                if "error" in self.current_results:
                    messagebox.showerror("Error", f"Processing failed: {self.current_results['error']}")
                    self.status_var.set("Error: Processing failed")
                else:
                    count = self.current_results.get("count", 0)
                    output_file = self.current_results.get("output_file", "")
                    messagebox.showinfo("Success", f"Processed {count} documents.\nResults saved to {output_file}")
                    self.status_var.set(f"Successfully processed {count} documents")
            
            # Reset
            self.processing_thread = None
            
    def save_results(self):
        """Save the current results to a file."""
        if not self.current_results:
            messagebox.showerror("Error", "No results to save.")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(self.current_results, f, indent=2)
                    
                self.status_var.set(f"Results saved to {file_path}")
                messagebox.showinfo("Success", f"Results saved to {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results: {e}")
                
    def clear_data(self):
        """Clear all current data and reset the UI."""
        # Clear file paths
        self.file_path_var.set("")
        self.dir_path_var.set("")
        
        # Clear results
        self.current_results = None
        
        # Clear classification results
        self.doc_type_var.set("")
        self.confidence_var.set("")
        
        # Clear invoice data
        self.tree.item("invoice_number", values=(""))
        self.tree.item("invoice_date", values=(""))
        self.tree.item("due_date", values=(""))
        self.tree.item("issuer_name", values=(""))
        self.tree.item("recipient_name", values=(""))
        self.tree.item("total_amount", values=(""))
        
        # Clear preview
        self.preview_canvas.delete("all")
        
        # Reset progress
        self.progress_var.set(0)
        
        # Reset status
        self.status_var.set("Ready")
        
        # Clean up temp files
        self.processor.cleanup_temp_files()


if __name__ == "__main__":
    root = tk.Tk()
    app = FinancialDocumentProcessorGUI(root)
    root.mainloop() 