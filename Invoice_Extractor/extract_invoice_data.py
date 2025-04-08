import sys
import os
import json
from invoice_extractor import InvoiceExtractor

def main():
    """
    Extract data from a single invoice or process all invoices in the Data directory.
    
    Usage:
        python extract_invoice_data.py [invoice_pdf_path]
        
    If no argument is provided, processes all invoices in the Data directory.
    """
    if len(sys.argv) > 1:
        # Process a single invoice
        invoice_path = sys.argv[1]
        if not os.path.exists(invoice_path):
            print(f"Error: File {invoice_path} not found.")
            return
            
        extractor = InvoiceExtractor(invoice_path)
        invoice_data = extractor.extract_all()
        
        print("\n=== Extracted Invoice Data ===")
        print(f"Invoice Number: {invoice_data.invoice_number}")
        print(f"Invoice Date: {invoice_data.invoice_date}")
        print(f"Due Date: {invoice_data.due_date}")
        print(f"Issuer Name: {invoice_data.issuer_name}")
        print(f"Recipient Name: {invoice_data.recipient_name}")
        print(f"Total Amount: {invoice_data.total_amount}")
        
        # Save to JSON file
        output_file = os.path.basename(invoice_path).replace('.pdf', '.json')
        with open(output_file, 'w') as f:
            json.dump(invoice_data.to_dict(), f, indent=2)
        print(f"\nData saved to {output_file}")
        
    else:
        # Process all invoices in the Data directory
        from invoice_extractor import process_directory
        process_directory("Data")
        
if __name__ == "__main__":
    main() 