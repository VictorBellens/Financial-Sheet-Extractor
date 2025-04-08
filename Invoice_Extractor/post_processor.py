import re
import os
from typing import Dict, Any, List, Tuple, Optional, Set
from dateutil import parser as date_parser
import datetime
from decimal import Decimal, InvalidOperation


class PostProcessor:
    """
    Post-processes extracted invoice data to improve accuracy through:
    1. Cross-validation between fields
    2. Auto-correction of common extraction errors
    3. Consistency checks between fields
    4. Prioritization of high-confidence data sources
    """
    
    def __init__(self):
        # Initialize with common errors to correct
        self.common_replacements = {
            # Common OCR errors
            "O": "0",
            "I": "1",
            "l": "1",
            "S": "5",
            "G": "6",
            "B": "8",
            # Common text patterns to clean
            "Ship Mode:": "",
            "Ship Mode": "",
            "Standard Class": "",
            "First Class": "",
            "Second Class": "",
            "Item Quantity Rate Amount": "",
        }
        
        # Common expressions for dates
        self.date_formats = [
            r"\d{1,2}/\d{1,2}/\d{2,4}",          # MM/DD/YYYY or DD/MM/YYYY
            r"\d{1,2}-\d{1,2}-\d{2,4}",          # MM-DD-YYYY or DD-MM-YYYY
            r"\d{1,2}\.\d{1,2}\.\d{2,4}",        # MM.DD.YYYY or DD.MM.YYYY
            r"[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}" # Month DD, YYYY
        ]
        
        # Load stopwords
        self.stopwords = self._load_stopwords()
    
    def _load_stopwords(self) -> Set[str]:
        """Load common stopwords from file."""
        stopwords_file = os.path.join(os.path.dirname(__file__), 'common_stopwords.txt')
        
        # Create default empty set
        stopwords = set()
        
        # Try to load stopwords file if it exists
        if os.path.exists(stopwords_file):
            try:
                with open(stopwords_file, 'r', encoding='utf-8') as f:
                    # Read each line, strip whitespace, and convert to lowercase
                    stopwords = set(line.strip().lower() for line in f if line.strip())
                print(f"Loaded {len(stopwords)} stopwords for post-processing")
            except Exception as e:
                print(f"Error loading stopwords: {e}")
        else:
            print(f"Stopwords file not found at {stopwords_file}")
        
        return stopwords
    
    def process(self, extracted_data: Dict[str, Any], filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Process extracted invoice data to improve accuracy.
        
        Args:
            extracted_data: Dictionary of extracted invoice data
            filename: Optional invoice filename for additional context
            
        Returns:
            Improved invoice data
        """
        # Make a copy to avoid modifying the original
        data = extracted_data.copy()
        
        # Extract filename info if available
        filename_info = self._extract_from_filename(filename) if filename else {}
        
        # Clean and correct fields
        data = self._clean_text_fields(data)
        
        # Prioritize filename data for certain fields
        data = self._prioritize_filename_data(data, filename_info)
        
        # Fix date fields
        data = self._fix_date_fields(data)
        
        # Fix amount fields
        data = self._fix_amount_fields(data)
        
        # Check and fix consistency between fields
        data = self._ensure_field_consistency(data)
        
        # Update confidence based on field corrections
        data = self._update_confidence(data, filename_info)
        
        return data
    
    def _extract_from_filename(self, filename: str) -> Dict[str, str]:
        """Extract information from the invoice filename."""
        result = {}
        if not filename:
            return result
            
        # Pattern for "invoice_Name_12345.pdf"
        match = re.match(r'invoice_([^_]+)_(\d+)(?:\.pdf)?', filename, re.IGNORECASE)
        if match:
            recipient_name = match.group(1).replace('_', ' ')
            invoice_number = match.group(2)
            
            result['recipient_name'] = recipient_name
            result['invoice_number'] = invoice_number
            
            # Set high confidence for these fields from filename
            result['confidence'] = {
                'recipient_name': 0.95,
                'invoice_number': 0.95
            }
        
        return result
    
    def _clean_text_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean text fields to remove common errors and irrelevant text."""
        text_fields = ['issuer_name', 'recipient_name']
        
        for field in text_fields:
            if field in data and data[field]:
                value = data[field]
                
                # Apply common replacements
                for pattern, replacement in self.common_replacements.items():
                    value = value.replace(pattern, replacement)
                
                # Remove consecutive spaces
                value = re.sub(r'\s+', ' ', value).strip()
                
                # Check if the entire text is a stopword
                if value.lower() in self.stopwords:
                    data[field] = ""
                    continue
                
                # Filter out words that are stopwords
                words = value.split()
                filtered_words = [word for word in words if word.lower() not in self.stopwords]
                value = " ".join(filtered_words)
                
                # Skip if value became empty or irrelevant
                if not value or value.lower() in ['ship to', 'bill to', 'customer', 'client']:
                    data[field] = ""
                    continue
                    
                data[field] = value
        
        return data
    
    def _prioritize_filename_data(self, data: Dict[str, Any], filename_info: Dict[str, Any]) -> Dict[str, Any]:
        """Prioritize data extracted from filename for certain fields."""
        # Fields that are more reliable from filename
        if 'recipient_name' in filename_info:
            # Update only if filename info has higher confidence
            filename_confidence = filename_info.get('confidence', {}).get('recipient_name', 0.95)
            current_confidence = data.get('confidence', {}).get('recipient_name', 0)
            
            if filename_confidence > current_confidence:
                data['recipient_name'] = filename_info['recipient_name']
                if 'confidence' in data:
                    data['confidence']['recipient_name'] = filename_confidence
        
        if 'invoice_number' in filename_info:
            # Update only if filename info has higher confidence or current is missing
            filename_confidence = filename_info.get('confidence', {}).get('invoice_number', 0.95)
            current_confidence = data.get('confidence', {}).get('invoice_number', 0)
            
            if filename_confidence > current_confidence or not data.get('invoice_number'):
                data['invoice_number'] = filename_info['invoice_number']
                if 'confidence' in data:
                    data['confidence']['invoice_number'] = filename_confidence
        
        return data
    
    def _fix_date_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Fix and standardize date fields."""
        date_fields = ['invoice_date', 'due_date']
        
        for field in date_fields:
            if field in data and data[field]:
                value = data[field]
                
                # Try to parse the date
                try:
                    # Check if the value matches a date pattern
                    if not any(re.search(pattern, value) for pattern in self.date_formats):
                        # Try to find a date pattern in the value
                        for pattern in self.date_formats:
                            match = re.search(pattern, value)
                            if match:
                                value = match.group(0)
                                break
                    
                    # Parse the date and standardize format
                    parsed_date = date_parser.parse(value)
                    data[field] = parsed_date.strftime("%b %d, %Y")
                    
                    # Update confidence
                    if 'confidence' in data:
                        data['confidence'][field] = max(data.get('confidence', {}).get(field, 0), 0.7)
                        
                except (ValueError, date_parser.ParserError):
                    # If we can't parse it, leave it as is
                    pass
        
        return data
    
    def _fix_amount_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Fix and standardize amount fields."""
        if 'total_amount' in data and data['total_amount']:
            value = data['total_amount']
            
            # Clean amount format
            value = re.sub(r'[^\d.,]', '', value)
            
            # Ensure proper formatting
            try:
                # Handle European format (1.234,56)
                if ',' in value and '.' in value:
                    if value.rindex(',') > value.rindex('.'):
                        value = value.replace('.', '')
                        value = value.replace(',', '.')
                    else:
                        value = value.replace(',', '')
                elif ',' in value and '.' not in value:
                    if len(value.split(',')[1]) <= 2:  # Likely decimal separator
                        value = value.replace(',', '.')
                    else:  # Likely thousands separator
                        value = value.replace(',', '')
                
                # Convert to Decimal and back to string
                amount = Decimal(value)
                
                # Update structured amount value if present
                if 'amount_value' in data:
                    data['amount_value'] = amount
                
                # Format with thousands separator
                data['total_amount'] = f"{amount:,.2f}".replace(',', ' ')
                
                # Update confidence
                if 'confidence' in data:
                    data['confidence']['total_amount'] = max(data.get('confidence', {}).get('total_amount', 0), 0.8)
                    
            except (ValueError, InvalidOperation):
                # If we can't parse it, leave it as is
                pass
        
        return data
    
    def _ensure_field_consistency(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure consistency between fields."""
        # Check invoice date and due date consistency
        if 'invoice_date' in data and data['invoice_date'] and 'due_date' in data and data['due_date']:
            try:
                invoice_date = date_parser.parse(data['invoice_date']).date()
                due_date = date_parser.parse(data['due_date']).date()
                
                # Due date should be on or after invoice date
                if due_date < invoice_date:
                    # Fix: Set due date to invoice date + 30 days
                    due_date = invoice_date + datetime.timedelta(days=30)
                    data['due_date'] = due_date.strftime("%b %d, %Y")
                    
                    # Lower confidence in fixed date
                    if 'confidence' in data:
                        data['confidence']['due_date'] = 0.5
            except (ValueError, date_parser.ParserError):
                pass
        
        # Check recipient name consistency with context
        if 'recipient_name' in data and data['recipient_name']:
            name = data['recipient_name']
            
            # Check if the name is in the stopwords
            if name.lower() in self.stopwords:
                data['recipient_name'] = ""
                if 'confidence' in data:
                    data['confidence']['recipient_name'] = 0.0
                return data
            
            # Check for common false positives
            false_positives = ['ship to', 'bill to', 'ship mode', 'standard class', 'first class']
            
            if any(fp in name.lower() for fp in false_positives):
                # Clear the field
                data['recipient_name'] = ""
                
                # Set confidence to zero
                if 'confidence' in data:
                    data['confidence']['recipient_name'] = 0.0
        
        # Also check issuer name
        if 'issuer_name' in data and data['issuer_name']:
            name = data['issuer_name']
            
            # Check if the name is in the stopwords
            if name.lower() in self.stopwords:
                data['issuer_name'] = ""
                if 'confidence' in data:
                    data['confidence']['issuer_name'] = 0.0
        
        return data
    
    def _update_confidence(self, data: Dict[str, Any], filename_info: Dict[str, Any]) -> Dict[str, Any]:
        """Update overall confidence based on field corrections."""
        if 'confidence' not in data:
            return data
            
        # Define field weights
        weights = {
            'invoice_number': 1.5,
            'invoice_date': 1.2,
            'due_date': 0.8,
            'issuer_name': 1.0,
            'recipient_name': 1.0,
            'total_amount': 1.3
        }
        
        # Boost confidence for fields that match filename info
        if 'recipient_name' in filename_info and 'recipient_name' in data:
            if data['recipient_name'] == filename_info['recipient_name']:
                data['confidence']['recipient_name'] = 0.95
                
        if 'invoice_number' in filename_info and 'invoice_number' in data:
            if data['invoice_number'] == filename_info['invoice_number']:
                data['confidence']['invoice_number'] = 0.95
        
        # Lower confidence for empty fields after cleaning
        for field in ['issuer_name', 'recipient_name']:
            if field in data and not data[field] and field in data.get('confidence', {}):
                data['confidence'][field] = 0.0
        
        # Recalculate overall confidence
        total_weight = sum(weights.get(field, 1.0) for field in data['confidence'].keys())
        overall_confidence = sum(
            data['confidence'].get(field, 0) * weights.get(field, 1.0)
            for field in data['confidence'].keys()
        ) / total_weight if total_weight > 0 else 0.0
        
        data['overall_confidence'] = overall_confidence
        
        return data


def batch_process(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process a batch of invoice extraction results.
    
    Args:
        results: List of dictionaries with 'filename' and 'data' keys
        
    Returns:
        List of processed results
    """
    processor = PostProcessor()
    processed_results = []
    
    for item in results:
        filename = item.get('filename', '')
        data = item.get('data', {})
        
        # Process the data
        processed_data = processor.process(data, filename)
        
        # Add to results
        processed_results.append({
            'filename': filename,
            'data': processed_data
        })
    
    return processed_results


if __name__ == "__main__":
    # Simple test if run directly
    import sys
    import json
    
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
        
        try:
            with open(json_path, 'r') as f:
                results = json.load(f)
                
            processed_results = batch_process(results)
            
            # Save processed results
            output_path = os.path.splitext(json_path)[0] + "_processed.json"
            with open(output_path, 'w') as f:
                json.dump(processed_results, f, indent=2)
                
            print(f"Processed {len(results)} invoices. Results saved to {output_path}")
            
            # Print a sample of before/after
            if processed_results:
                sample = processed_results[0]
                print("\nSample Before/After:")
                print(f"Filename: {sample['filename']}")
                print("Before:", json.dumps(results[0]['data'], indent=2))
                print("After:", json.dumps(sample['data'], indent=2))
        
        except Exception as e:
            print(f"Error processing results: {e}")
    else:
        print("Usage: python post_processor.py path/to/results.json") 