import os
import sys
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import json
from invoice_extractor import InvoiceExtractor, process_directory

class InvoiceExtractorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Invoice Data Extractor")
        self.root.geometry("800x600")
        self.root.minsize(800, 600)
        
        self.setup_ui()
        
    def setup_ui(self):
        # Frame for file selection
        file_frame = ttk.LabelFrame(self.root, text="Invoice File Selection")
        file_frame.pack(fill="x", padx=10, pady=10)
        
        # Single file processing
        ttk.Label(file_frame, text="Select Invoice PDF:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.file_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path_var, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(file_frame, text="Process File", command=self.process_single_file).grid(row=0, column=3, padx=5, pady=5)
        
        # Batch processing
        ttk.Label(file_frame, text="Or Process Directory:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.dir_path_var = tk.StringVar(value="Data")  # Default to Data directory
        ttk.Entry(file_frame, textvariable=self.dir_path_var, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_directory).grid(row=1, column=2, padx=5, pady=5)
        ttk.Button(file_frame, text="Process Directory", command=self.process_directory).grid(row=1, column=3, padx=5, pady=5)
        
        # Frame for results
        results_frame = ttk.LabelFrame(self.root, text="Extracted Data")
        results_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create treeview for results
        self.tree = ttk.Treeview(results_frame)
        self.tree["columns"] = ("Value")
        self.tree.column("#0", width=200, minwidth=200)
        self.tree.column("Value", width=580, minwidth=200)
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
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack treeview and scrollbar
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bottom buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Button(button_frame, text="Save Data", command=self.save_data).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Clear", command=self.clear_data).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Exit", command=self.root.quit).pack(side="right", padx=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w")
        status_bar.pack(side="bottom", fill="x")
        
        # Store current data
        self.current_data = None
        
    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Invoice PDF",
            filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")]
        )
        if file_path:
            self.file_path_var.set(file_path)
            
    def browse_directory(self):
        dir_path = filedialog.askdirectory(title="Select Directory with Invoice PDFs")
        if dir_path:
            self.dir_path_var.set(dir_path)
    
    def process_single_file(self):
        file_path = self.file_path_var.get()
        if not file_path:
            messagebox.showerror("Error", "Please select a PDF file.")
            return
            
        if not os.path.exists(file_path):
            messagebox.showerror("Error", f"File not found: {file_path}")
            return
            
        try:
            self.status_var.set(f"Processing {os.path.basename(file_path)}...")
            self.root.update_idletasks()
            
            extractor = InvoiceExtractor(file_path)
            invoice_data = extractor.extract_all()
            self.current_data = invoice_data
            
            # Update treeview
            self.tree.item("invoice_number", values=(invoice_data.invoice_number or ""))
            self.tree.item("invoice_date", values=(invoice_data.invoice_date or ""))
            self.tree.item("due_date", values=(invoice_data.due_date or ""))
            self.tree.item("issuer_name", values=(invoice_data.issuer_name or ""))
            self.tree.item("recipient_name", values=(invoice_data.recipient_name or ""))
            self.tree.item("total_amount", values=(invoice_data.total_amount or ""))
            
            self.status_var.set(f"Successfully processed {os.path.basename(file_path)}")
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
            self.status_var.set(f"Processing invoices in {dir_path}...")
            self.root.update_idletasks()
            
            output_file = "extracted_invoices.json"
            process_directory(dir_path, output_file)
            
            messagebox.showinfo("Success", f"Processing complete!\nResults saved to {output_file}")
            self.status_var.set(f"Successfully processed directory. Results saved to {output_file}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process directory: {str(e)}")
            self.status_var.set("Error processing directory")
    
    def save_data(self):
        if not self.current_data:
            messagebox.showerror("Error", "No data to save.")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Extracted Data",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(self.current_data.to_dict(), f, indent=2)
                self.status_var.set(f"Data saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save data: {str(e)}")
    
    def clear_data(self):
        self.file_path_var.set("")
        self.current_data = None
        
        # Clear treeview
        self.tree.item("invoice_number", values=(""))
        self.tree.item("invoice_date", values=(""))
        self.tree.item("due_date", values=(""))
        self.tree.item("issuer_name", values=(""))
        self.tree.item("recipient_name", values=(""))
        self.tree.item("total_amount", values=(""))
        
        self.status_var.set("Ready")

if __name__ == "__main__":
    root = tk.Tk()
    app = InvoiceExtractorGUI(root)
    root.mainloop() 