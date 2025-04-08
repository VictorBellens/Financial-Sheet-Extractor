import PyPDF2
from typing import Dict, Optional, Any

class PDFFormExtractor:
    """Extracts form field data from fillable PDF forms."""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.form_fields = self._extract_form_fields()
        
    def _extract_form_fields(self) -> Dict[str, Any]:
        """Extract all form fields from the PDF."""
        try:
            with open(self.pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                if reader.is_encrypted:
                    try:
                        reader.decrypt('')  # Try empty password
                    except:
                        print(f"Warning: Could not decrypt PDF {self.pdf_path}")
                        return {}
                
                # Extract form fields if available
                if reader.is_form:
                    return reader.get_form_text_fields() or {}
                return {}
        except Exception as e:
            print(f"Error extracting form fields from {self.pdf_path}: {e}")
            return {}
    
    def has_form_fields(self) -> bool:
        """Check if the PDF has any form fields."""
        return len(self.form_fields) > 0
    
    def get_invoice_number(self) -> Optional[str]:
        """Try to find invoice number in form fields."""
        field_patterns = [
            "invoice number", "invoice no", "invoice #", "invoice_number", 
            "invoicenumber", "invoice_no", "invoiceno"
        ]
        return self._find_field_by_patterns(field_patterns)
    
    def get_invoice_date(self) -> Optional[str]:
        """Try to find invoice date in form fields."""
        field_patterns = [
            "invoice date", "date", "invoice_date", "invoicedate"
        ]
        return self._find_field_by_patterns(field_patterns)
    
    def get_due_date(self) -> Optional[str]:
        """Try to find due date in form fields."""
        field_patterns = [
            "due date", "payment due", "due_date", "duedate", "payment_due"
        ]
        return self._find_field_by_patterns(field_patterns)
    
    def get_issuer_name(self) -> Optional[str]:
        """Try to find issuer name in form fields."""
        field_patterns = [
            "issuer", "from", "seller", "company", "vendor", "issuer_name", "company_name"
        ]
        return self._find_field_by_patterns(field_patterns)
    
    def get_recipient_name(self) -> Optional[str]:
        """Try to find recipient name in form fields."""
        field_patterns = [
            "bill to", "recipient", "customer", "client", "buyer", "bill_to", "ship_to",
            "recipient_name", "customer_name"
        ]
        return self._find_field_by_patterns(field_patterns)
    
    def get_total_amount(self) -> Optional[str]:
        """Try to find total amount in form fields."""
        field_patterns = [
            "total", "amount", "total amount", "balance due", "amount due", "grand total",
            "total_amount", "amount_due"
        ]
        return self._find_field_by_patterns(field_patterns)
    
    def _find_field_by_patterns(self, patterns: list) -> Optional[str]:
        """Find a field value by checking multiple possible field names."""
        # First check for exact matches
        for field_name, value in self.form_fields.items():
            if field_name.lower() in patterns and value:
                return str(value).strip()
        
        # Then check for partial matches
        for field_name, value in self.form_fields.items():
            for pattern in patterns:
                if pattern in field_name.lower() and value:
                    return str(value).strip()
        
        return None
    
    def get_all_fields(self) -> Dict[str, Any]:
        """Return all form fields for inspection."""
        return self.form_fields


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        extractor = PDFFormExtractor(pdf_path)
        print(f"Form fields: {extractor.has_form_fields()}")
        if extractor.has_form_fields():
            print(f"All fields: {extractor.get_all_fields()}")
            print(f"Invoice Number: {extractor.get_invoice_number()}")
            print(f"Invoice Date: {extractor.get_invoice_date()}")
            print(f"Due Date: {extractor.get_due_date()}")
            print(f"Issuer: {extractor.get_issuer_name()}")
            print(f"Recipient: {extractor.get_recipient_name()}")
            print(f"Total Amount: {extractor.get_total_amount()}")
    else:
        print("Usage: python pdf_form_extractor.py path/to/invoice.pdf") 