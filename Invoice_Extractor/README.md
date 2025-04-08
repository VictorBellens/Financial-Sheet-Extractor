# Invoice Information Extractor

A tool for extracting key information from PDF invoices, such as:
- Invoice Number
- Invoice Date
- Due Date
- Issuer Name
- Recipient Name
- Total Amount

## Requirements

- Python 3.6+
- Required packages: pdfplumber, PyPDF2

## Installation

1. Clone this repository or download the source code
2. Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage

### Command-line Interface

To process a single invoice:
```
python extract_invoice_data.py path/to/invoice.pdf
```

To process all invoices in the Data directory:
```
python extract_invoice_data.py
```

### GUI Application

For a more user-friendly experience, you can use the GUI application:
```
python invoice_gui.py
```

The GUI allows you to:
- Browse and select individual invoice PDFs
- Process an entire directory of invoices
- View extracted data in a clean interface
- Save results as JSON

## File Structure

- `invoice_extractor.py`: Core extraction logic and utility functions
- `extract_invoice_data.py`: Command-line script for invoice processing
- `invoice_gui.py`: GUI application for invoice processing
- `requirements.txt`: Required Python packages
- `Data/`: Directory containing sample invoice PDFs

## How It Works

The extractor uses text extraction and pattern matching to identify key invoice information. When standard extraction patterns don't match, it falls back to alternative strategies, such as extracting data from the filename.

The extraction process follows these steps:
1. Extract all text from the PDF using pdfplumber
2. Apply regular expression patterns to locate the relevant information
3. Use fallback methods if standard patterns don't yield results
4. Return structured data ready for use

## Extending the Extractor

To add support for additional fields or invoice formats:

1. Add new field definitions to the `InvoiceData` class
2. Create new extraction methods in the `InvoiceExtractor` class
3. Add the new fields to the extraction workflow in the `extract_all` method 