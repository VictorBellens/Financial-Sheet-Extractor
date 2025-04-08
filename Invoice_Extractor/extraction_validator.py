import re
import datetime
from typing import Dict, Optional, Tuple, Any
from decimal import Decimal, InvalidOperation
from dateutil import parser as date_parser

class ExtractionValidator:
    """
    Validates extracted invoice data and assigns confidence scores.
    Provides methods to check if data is reasonable and estimate confidence.
    """
    
    def __init__(self):
        self.validation_results = {}
        self.confidence_scores = {}
    
    def validate_invoice_number(self, invoice_number: Optional[str]) -> Tuple[bool, float]:
        """
        Validate invoice number and assign confidence score.
        
        Returns:
            Tuple of (is_valid, confidence_score)
        """
        if not invoice_number:
            return False, 0.0
        
        # Common invoice number patterns
        patterns = [
            r'^\d{4,10}$',  # 4-10 digits
            r'^\d{2,8}-\d{2,8}$',  # Format like 12345-67890
            r'^[A-Z]{1,5}-\d{4,10}$',  # Format like INV-12345
            r'^[A-Z]{1,5}\d{4,10}$'  # Format like INV12345
        ]
        
        # Check for pattern match
        pattern_match = any(re.match(pattern, invoice_number) for pattern in patterns)
        
        # Basic validation: non-empty and reasonable length
        is_valid = len(invoice_number) >= 3 and len(invoice_number) <= 20
        
        # Calculate confidence
        confidence = 0.0
        if is_valid:
            # Base confidence for valid format
            confidence = 0.6
            
            # Boost for pattern match
            if pattern_match:
                confidence += 0.3
            
            # Boost for all-digit invoice numbers (very common)
            if invoice_number.isdigit():
                confidence += 0.1
                
            # Cap confidence at 1.0
            confidence = min(confidence, 1.0)
        
        self.validation_results['invoice_number'] = is_valid
        self.confidence_scores['invoice_number'] = confidence
        
        return is_valid, confidence
    
    def validate_date(self, date_str: Optional[str], field_name: str) -> Tuple[bool, float, Optional[datetime.date]]:
        """
        Validate a date string and assign confidence score.
        
        Args:
            date_str: Date string to validate
            field_name: Name of the field ('invoice_date' or 'due_date')
            
        Returns:
            Tuple of (is_valid, confidence_score, parsed_date or None)
        """
        if not date_str:
            self.validation_results[field_name] = False
            self.confidence_scores[field_name] = 0.0
            return False, 0.0, None
        
        try:
            # Try to parse the date
            parsed_date = date_parser.parse(date_str).date()
            
            # Check if date is reasonable - accept dates from 30 years ago to 5 years in future
            today = datetime.date.today()
            thirty_years_ago = today.replace(year=today.year - 30)
            five_years_future = today.replace(year=today.year + 5)
            
            is_valid = thirty_years_ago <= parsed_date <= five_years_future
            
            # Calculate confidence
            confidence = 0.0
            if is_valid:
                # Base confidence for valid date
                confidence = 0.7
                
                # Higher confidence for invoice dates in past 5 years
                if field_name == 'invoice_date':
                    five_years_ago = today.replace(year=today.year - 5)
                    if five_years_ago <= parsed_date <= today:
                        confidence += 0.3
                    # Still reasonable confidence for older dates
                    elif thirty_years_ago <= parsed_date < five_years_ago:
                        confidence += 0.2
                
                # Higher confidence for due dates in reasonable range from today
                if field_name == 'due_date':
                    if parsed_date >= today:
                        # Future due date
                        days_diff = (parsed_date - today).days
                        if days_diff <= 90:  # Within 90 days
                            confidence += 0.3
                        elif days_diff <= 180:  # Within 180 days
                            confidence += 0.2
                        else:
                            confidence += 0.1
                    else:
                        # Past due date
                        days_overdue = (today - parsed_date).days
                        if days_overdue <= 90:  # Overdue within 90 days
                            confidence += 0.2
                        elif days_overdue <= 365:  # Overdue within 1 year
                            confidence += 0.1
                
                # Cap confidence at 1.0
                confidence = min(confidence, 1.0)
            
            self.validation_results[field_name] = is_valid
            self.confidence_scores[field_name] = confidence
            
            return is_valid, confidence, parsed_date
            
        except (ValueError, TypeError, date_parser.ParserError):
            self.validation_results[field_name] = False
            self.confidence_scores[field_name] = 0.0
            return False, 0.0, None
    
    def validate_name(self, name: Optional[str], field_name: str) -> Tuple[bool, float]:
        """
        Validate a name (issuer or recipient) and assign confidence score.
        
        Args:
            name: Name string to validate
            field_name: Name of the field ('issuer_name' or 'recipient_name')
            
        Returns:
            Tuple of (is_valid, confidence_score)
        """
        if not name:
            self.validation_results[field_name] = False
            self.confidence_scores[field_name] = 0.0
            return False, 0.0
        
        # Basic validation: non-empty and reasonable length
        is_valid = 2 <= len(name) <= 100
        
        # Check for common false positives and shipping information
        false_positives = [
            'bill to', 'ship to', 'invoice', 'receipt', 'statement', 'customer',
            'client', 'balance due', 'date', 'amount', 'total', 'item', 'quantity',
            'ship mode', 'standard class', 'first class', 'second class',
            'rate', 'product', 'service', 'description'
        ]
        
        # Check for shipping information or table headers
        if re.search(r'^ship\s+mode:', name.lower()):
            is_valid = False
        
        # Check if the name contains any false positive keywords
        for fp in false_positives:
            if fp in name.lower():
                # If it's just a part of a longer name, it might still be valid
                if len(name.split()) <= 3 or name.lower().startswith(fp):
                    is_valid = False
                    break
        
        # Calculate confidence
        confidence = 0.0
        if is_valid:
            # Base confidence for valid name
            confidence = 0.5
            
            # Higher confidence for longer names (more likely to be real company names)
            if len(name) > 3:
                confidence += 0.1
            
            # Higher confidence for names with common business identifiers
            business_terms = ['inc', 'llc', 'ltd', 'corp', 'co', 'company', 'gmbh', 'corporation']
            if any(term in name.lower() for term in business_terms):
                confidence += 0.3
            
            # Higher confidence for capitalized words (proper names)
            if any(word[0].isupper() for word in name.split() if word):
                confidence += 0.1
            
            # Lower confidence for names with numbers (less likely for company names)
            if re.search(r'\d', name):
                confidence -= 0.1
            
            # Cap confidence at 1.0
            confidence = min(confidence, 1.0)
            confidence = max(confidence, 0.0)  # Ensure it's not negative
        
        self.validation_results[field_name] = is_valid
        self.confidence_scores[field_name] = confidence
        
        return is_valid, confidence
    
    def validate_amount(self, amount_str: Optional[str]) -> Tuple[bool, float, Optional[Decimal]]:
        """
        Validate total amount and assign confidence score.
        
        Returns:
            Tuple of (is_valid, confidence_score, decimal_amount or None)
        """
        if not amount_str:
            self.validation_results['total_amount'] = False
            self.confidence_scores['total_amount'] = 0.0
            return False, 0.0, None
        
        # Remove any non-numeric characters except decimal point and comma
        amount_str = re.sub(r'[^\d.,]', '', amount_str)
        
        try:
            # Handle different number formats
            if ',' in amount_str and '.' in amount_str:
                # If both comma and period exist, determine which is the decimal separator
                if amount_str.rindex(',') > amount_str.rindex('.'):
                    # European format: 1.234,56
                    amount_str = amount_str.replace('.', '')
                    amount_str = amount_str.replace(',', '.')
                else:
                    # US/UK format: 1,234.56
                    amount_str = amount_str.replace(',', '')
            elif ',' in amount_str and '.' not in amount_str:
                # If only comma exists, it might be a decimal separator or thousands separator
                if amount_str.count(',') == 1 and len(amount_str.split(',')[1]) <= 2:
                    # Likely a decimal separator: 1234,56
                    amount_str = amount_str.replace(',', '.')
                else:
                    # Likely thousands separators: 1,234,567
                    amount_str = amount_str.replace(',', '')
            
            # Convert to Decimal
            amount = Decimal(amount_str)
            
            # Check if amount is reasonable
            is_valid = amount >= 0 and amount < 1000000  # Reasonable range for most invoices
            
            # Zero amounts can be valid (paid invoices, credit notes)
            if amount == 0:
                is_valid = True
            
            # Calculate confidence
            confidence = 0.0
            if is_valid:
                # Base confidence for valid amount
                confidence = 0.7
                
                # Higher confidence for amounts with cents (typical for invoices)
                if amount % 1 != 0:
                    confidence += 0.1
                
                # Higher confidence for common invoice amounts
                if 10 <= amount <= 10000:
                    confidence += 0.2
                elif amount <= 100000:
                    confidence += 0.1
                
                # Lower confidence for zero amounts (often just placeholders)
                if amount == 0:
                    confidence -= 0.3
                
                # Cap confidence at 1.0
                confidence = min(confidence, 1.0)
            
            self.validation_results['total_amount'] = is_valid
            self.confidence_scores['total_amount'] = confidence
            
            return is_valid, confidence, amount
            
        except (ValueError, TypeError, InvalidOperation):
            self.validation_results['total_amount'] = False
            self.confidence_scores['total_amount'] = 0.0
            return False, 0.0, None
    
    def validate_all(self, invoice_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate all fields in the invoice data and assign confidence scores.
        
        Args:
            invoice_data: Dictionary containing extracted invoice data
            
        Returns:
            Dictionary with validation results and confidence scores added
        """
        # Clear previous results
        self.validation_results = {}
        self.confidence_scores = {}
        
        # Validate each field
        self.validate_invoice_number(invoice_data.get('invoice_number'))
        self.validate_date(invoice_data.get('invoice_date'), 'invoice_date')
        self.validate_date(invoice_data.get('due_date'), 'due_date')
        self.validate_name(invoice_data.get('issuer_name'), 'issuer_name')
        self.validate_name(invoice_data.get('recipient_name'), 'recipient_name')
        self.validate_amount(invoice_data.get('total_amount'))
        
        # Calculate overall confidence score
        fields = ['invoice_number', 'invoice_date', 'due_date', 'issuer_name', 'recipient_name', 'total_amount']
        present_fields = [f for f in fields if f in self.confidence_scores]
        
        if present_fields:
            # Weight invoice number, invoice date, and total amount more heavily
            weights = {
                'invoice_number': 1.5,
                'invoice_date': 1.2,
                'total_amount': 1.3,
                'issuer_name': 1.0,
                'recipient_name': 1.0,
                'due_date': 0.8
            }
            
            total_weight = sum(weights[f] for f in present_fields)
            overall_confidence = sum(self.confidence_scores[f] * weights[f] for f in present_fields) / total_weight
        else:
            overall_confidence = 0.0
        
        # Add validation results and confidence scores to the invoice data
        result = invoice_data.copy()
        result['validation'] = self.validation_results.copy()
        result['confidence'] = self.confidence_scores.copy()
        result['overall_confidence'] = overall_confidence
        
        return result
    
    def is_data_consistent(self, invoice_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if the data is internally consistent.
        
        Args:
            invoice_data: Dictionary containing extracted invoice data
            
        Returns:
            Tuple of (is_consistent, inconsistency_reason)
        """
        # Check invoice date and due date relationship
        invoice_date_str = invoice_data.get('invoice_date')
        due_date_str = invoice_data.get('due_date')
        
        if invoice_date_str and due_date_str:
            try:
                invoice_date = date_parser.parse(invoice_date_str).date()
                due_date = date_parser.parse(due_date_str).date()
                
                # Due date should be on or after invoice date
                if due_date < invoice_date:
                    return False, "Due date is before invoice date"
                
                # Due date shouldn't be too far in the future from invoice date
                days_diff = (due_date - invoice_date).days
                if days_diff > 365:  # More than 1 year payment terms is suspicious
                    return False, f"Due date is {days_diff} days after invoice date"
                    
            except (ValueError, TypeError, date_parser.ParserError):
                pass
        
        # Add other consistency checks as needed
        return True, ""


if __name__ == "__main__":
    # Test the validator
    validator = ExtractionValidator()
    
    # Test invoice number validation
    test_numbers = ["123456", "INV-12345", "ABC123", "X", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    for num in test_numbers:
        valid, conf = validator.validate_invoice_number(num)
        print(f"Invoice #{num}: Valid={valid}, Confidence={conf:.2f}")
    
    # Test date validation
    test_dates = ["2023-06-01", "01/01/2020", "Jun 5, 2023", "2030-01-01", "1990-01-01", "garbage"]
    for date in test_dates:
        valid, conf, parsed = validator.validate_date(date, 'invoice_date')
        print(f"Date '{date}': Valid={valid}, Confidence={conf:.2f}, Parsed={parsed}")
    
    # Test amount validation
    test_amounts = ["100.00", "$1,234.56", "1.234,56", "0", "1000000000", "garbage"]
    for amount in test_amounts:
        valid, conf, decimal = validator.validate_amount(amount)
        print(f"Amount '{amount}': Valid={valid}, Confidence={conf:.2f}, Decimal={decimal}")
        
    # Test name validation
    test_names = ["ABC Company Inc.", "Ship Mode: First Class", "John Smith", "Item Quantity Rate Amount", "Standard Corporation"]
    for name in test_names:
        valid, conf = validator.validate_name(name, 'issuer_name')
        print(f"Name '{name}': Valid={valid}, Confidence={conf:.2f}") 