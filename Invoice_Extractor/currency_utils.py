import re
from typing import Tuple, Optional
from decimal import Decimal, InvalidOperation

class CurrencyProcessor:
    """Utility class for detecting and standardizing currency amounts."""
    
    # Currency symbols and their corresponding codes
    CURRENCY_SYMBOLS = {
        '$': 'USD',
        '€': 'EUR',
        '£': 'GBP',
        '¥': 'JPY',
        '₹': 'INR',
        'C$': 'CAD',
        'A$': 'AUD',
        'Fr': 'CHF',
        'kr': 'SEK',
        'R': 'ZAR',
        '₽': 'RUB',
        '₴': 'UAH',
        '₺': 'TRY'
    }
    
    # Currency codes that might appear in text
    CURRENCY_CODES = {
        'USD', 'EUR', 'GBP', 'JPY', 'INR', 'CAD', 'AUD', 
        'CHF', 'SEK', 'NOK', 'DKK', 'ZAR', 'RUB', 'UAH', 'TRY'
    }
    
    @classmethod
    def detect_and_standardize(cls, amount_text: str) -> Tuple[Optional[Decimal], Optional[str]]:
        """
        Detect currency type and standardize the amount.
        
        Args:
            amount_text: Text containing a currency amount
            
        Returns:
            Tuple of (standardized amount as Decimal, currency code)
        """
        if not amount_text:
            return None, None
            
        # Remove all whitespace
        amount_text = re.sub(r'\s', '', amount_text)
        
        # Try to find currency symbol or code
        currency_code = None
        
        # Check for currency symbols
        for symbol, code in cls.CURRENCY_SYMBOLS.items():
            if symbol in amount_text:
                currency_code = code
                amount_text = amount_text.replace(symbol, '')
                break
        
        # If no symbol found, check for currency codes
        if not currency_code:
            for code in cls.CURRENCY_CODES:
                if code in amount_text:
                    currency_code = code
                    amount_text = amount_text.replace(code, '')
                    break
        
        # Default to USD if no currency identified
        if not currency_code:
            currency_code = 'USD'
        
        # Extract the numeric amount
        # This handles formats like 1,234.56 or 1.234,56 (European format)
        amount_match = re.search(r'([\d,.]+)', amount_text)
        if not amount_match:
            return None, currency_code
            
        amount_str = amount_match.group(1)
        
        # Determine if the amount uses comma as decimal separator
        if ',' in amount_str and '.' in amount_str:
            # If both comma and period exist, the last one is the decimal separator
            if amount_str.rindex(',') > amount_str.rindex('.'):
                # European format: 1.234,56
                amount_str = amount_str.replace('.', '')
                amount_str = amount_str.replace(',', '.')
            else:
                # US/UK format: 1,234.56
                amount_str = amount_str.replace(',', '')
        elif ',' in amount_str and '.' not in amount_str:
            # If only comma exists, it could be a decimal separator or a thousands separator
            if amount_str.count(',') == 1 and len(amount_str.split(',')[1]) <= 2:
                # Likely a decimal separator: 1234,56
                amount_str = amount_str.replace(',', '.')
            else:
                # Likely thousands separators: 1,234,567
                amount_str = amount_str.replace(',', '')
        
        # Convert to Decimal
        try:
            amount = Decimal(amount_str)
            return amount, currency_code
        except InvalidOperation:
            return None, currency_code
    
    @classmethod
    def format_for_display(cls, amount: Decimal, currency_code: str = 'USD') -> str:
        """
        Format a decimal amount for display with appropriate currency symbol.
        
        Args:
            amount: Decimal amount to format
            currency_code: Currency code (default: USD)
            
        Returns:
            Formatted currency string
        """
        if amount is None:
            return ""
            
        # Format with 2 decimal places
        amount_str = f"{amount:.2f}"
        
        # Get symbol for the currency code
        symbol = '$'  # Default to USD
        for sym, code in cls.CURRENCY_SYMBOLS.items():
            if code == currency_code:
                symbol = sym
                break
        
        # Format based on currency
        if currency_code in ['USD', 'CAD', 'AUD']:
            return f"{symbol}{amount_str}"
        elif currency_code == 'EUR':
            return f"€{amount_str}"
        elif currency_code == 'GBP':
            return f"£{amount_str}"
        elif currency_code == 'JPY':
            # JPY typically doesn't use decimal places
            return f"¥{int(amount)}"
        else:
            # For other currencies, show code and amount
            return f"{currency_code} {amount_str}"


def test_currency_processor():
    """Test the currency processor with various formats."""
    test_cases = [
        ("$1,234.56", (Decimal('1234.56'), 'USD')),
        ("1,234.56 USD", (Decimal('1234.56'), 'USD')),
        ("€1.234,56", (Decimal('1234.56'), 'EUR')),
        ("1.234,56 EUR", (Decimal('1234.56'), 'EUR')),
        ("£99.99", (Decimal('99.99'), 'GBP')),
        ("JPY 10000", (Decimal('10000'), 'JPY')),
        ("¥10,000", (Decimal('10000'), 'JPY')),
        ("CAD 50.00", (Decimal('50.00'), 'CAD')),
        ("50.00", (Decimal('50.00'), 'USD')),  # Default to USD
        ("1,234,567.89", (Decimal('1234567.89'), 'USD')),  # Large number, default currency
    ]
    
    for input_text, expected_output in test_cases:
        result = CurrencyProcessor.detect_and_standardize(input_text)
        print(f"Input: {input_text}")
        print(f"Result: {result}")
        print(f"Expected: {expected_output}")
        print(f"Pass: {result == expected_output}")
        print("---")


if __name__ == "__main__":
    test_currency_processor() 