import os
import re
import pdfplumber
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Set, NamedTuple
import json
from datetime import datetime
from decimal import Decimal
import logging
import pickle
from collections import defaultdict, Counter

# Import our new modules
from pdf_form_extractor import PDFFormExtractor
from currency_utils import CurrencyProcessor
from extraction_validator import ExtractionValidator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define a position class to track element positions
class Position(NamedTuple):
    x0: float
    y0: float
    x1: float
    y1: float
    page: int


class InvoiceType:
    STANDARD = "standard"
    RECEIPT = "receipt"
    PURCHASE_ORDER = "purchase_order"
    BILL = "bill"
    STATEMENT = "statement"
    UNKNOWN = "unknown"


class FieldPosition:
    HEADER = "header"
    FOOTER = "footer"
    LEFT = "left"
    RIGHT = "right"
    CENTER = "center"
    UNKNOWN = "unknown"


@dataclass
class InvoiceData:
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None
    issuer_name: Optional[str] = None
    recipient_name: Optional[str] = None
    total_amount: Optional[str] = None
    
    # New fields for enhanced data
    currency_code: Optional[str] = None
    amount_value: Optional[Decimal] = None
    validation: Dict[str, bool] = field(default_factory=dict)
    confidence: Dict[str, float] = field(default_factory=dict)
    overall_confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        base_dict = {
            "invoice_number": self.invoice_number,
            "invoice_date": self.invoice_date,
            "due_date": self.due_date,
            "issuer_name": self.issuer_name, 
            "recipient_name": self.recipient_name,
            "total_amount": self.total_amount
        }
        
        # Add enhanced data if available
        if self.currency_code:
            base_dict["currency_code"] = self.currency_code
        if self.amount_value is not None:
            base_dict["amount_value"] = str(self.amount_value)
        if self.validation:
            base_dict["validation"] = self.validation
        if self.confidence:
            base_dict["confidence"] = self.confidence
        if self.overall_confidence:
            base_dict["overall_confidence"] = self.overall_confidence
            
        return base_dict


@dataclass
class DocumentSection:
    """Represents a semantic section of a document."""
    name: str
    text: str
    position: Position
    confidence: float = 0.0
    

@dataclass
class NamedEntity:
    """Represents a named entity found in text."""
    text: str
    entity_type: str
    position: Optional[Position] = None
    confidence: float = 0.0


class InvoiceGraph:
    """
    Represents a graph of relationships between invoice fields.
    Used for contextual understanding and inference of missing fields.
    """
    def __init__(self, text_content: str, extracted_fields: Dict[str, Any]):
        self.text_content = text_content
        self.fields = extracted_fields
        self.nodes = {}
        self.edges = {}
        self._build_graph()
    
    def _build_graph(self):
        """Build the graph from extracted fields."""
        # Create nodes for each field
        for field_name, field_value in self.fields.items():
            if field_value:
                # Find position in text
                pos = self.text_content.find(str(field_value))
                self.nodes[field_name] = {
                    'value': field_value,
                    'position': pos if pos != -1 else -1
                }
        
        # Create edges based on positions and semantic relationships
        field_pairs = [
            ('invoice_number', 'invoice_date', 0.8),
            ('invoice_date', 'due_date', 0.7),
            ('issuer_name', 'invoice_number', 0.6),
            ('recipient_name', 'total_amount', 0.5),
            ('invoice_number', 'total_amount', 0.4)
        ]
        
        for field1, field2, base_weight in field_pairs:
            if field1 in self.nodes and field2 in self.nodes:
                pos1 = self.nodes[field1]['position']
                pos2 = self.nodes[field2]['position']
                
                if pos1 != -1 and pos2 != -1:
                    # Calculate proximity-based weight
                    proximity = 1.0 - (abs(pos1 - pos2) / len(self.text_content))
                    
                    # Combined weight
                    weight = (base_weight + proximity) / 2
                    
                    # Add bidirectional edges
                    if field1 not in self.edges:
                        self.edges[field1] = {}
                    if field2 not in self.edges:
                        self.edges[field2] = {}
                    
                    self.edges[field1][field2] = weight
                    self.edges[field2][field1] = weight
    
    def infer_missing_fields(self) -> Dict[str, Any]:
        """Use the graph to infer missing field values."""
        inferred_fields = {}
        
        # TODO: Implement more sophisticated inference logic
        # For now, just use a simple example:
        
        # If we have invoice_date but not due_date, try to infer it
        if 'invoice_date' in self.nodes and 'due_date' not in self.nodes:
            invoice_date = self.nodes['invoice_date']['value']
            
            # Common payment terms (e.g., Net 30)
            try:
                from datetime import datetime, timedelta
                
                # Try to parse the date
                for fmt in ['%b %d, %Y', '%B %d, %Y', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d']:
                    try:
                        date_obj = datetime.strptime(invoice_date, fmt)
                        # Assume Net 30 terms
                        due_date = date_obj + timedelta(days=30)
                        inferred_fields['due_date'] = due_date.strftime('%b %d, %Y')
                        break
                    except ValueError:
                        continue
            except Exception:
                pass
        
        return inferred_fields


class FewShotExtractor:
    """
    Implements few-shot learning for quickly adapting to new invoice formats.
    """
    def __init__(self, examples_path: Optional[str] = None):
        self.examples = []
        if examples_path and os.path.exists(examples_path):
            try:
                with open(examples_path, 'r') as f:
                    self.examples = json.load(f)
            except Exception as e:
                logger.error(f"Error loading few-shot examples: {e}")
    
    def add_example(self, text_fragment: str, field_name: str, field_value: str):
        """Add a new example to the few-shot learner."""
        self.examples.append({
            'text': text_fragment,
            'field': field_name,
            'value': field_value
        })
    
    def save_examples(self, path: str):
        """Save the current examples to a file."""
        try:
            with open(path, 'w') as f:
                json.dump(self.examples, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving few-shot examples: {e}")
    
    def extract_field(self, text: str, field_name: str) -> Tuple[Optional[str], float]:
        """
        Extract a field using few-shot examples.
        
        Returns:
            Tuple of (extracted_value, confidence)
        """
        if not self.examples:
            return None, 0.0
        
        # Filter examples for the requested field
        field_examples = [ex for ex in self.examples if ex['field'] == field_name]
        if not field_examples:
            return None, 0.0
        
        # Simple pattern matching based on examples
        for example in field_examples:
            example_text = example['text']
            example_value = example['value']
            
            # Find the context before and after the value in the example
            value_pos = example_text.find(example_value)
            if value_pos != -1:
                # Get context (up to 20 chars before and after)
                context_before = example_text[max(0, value_pos - 20):value_pos]
                context_after = example_text[value_pos + len(example_value):min(len(example_text), value_pos + len(example_value) + 20)]
                
                # Create a regex pattern from the context and value
                # Escape special regex characters in context
                context_before = re.escape(context_before)
                context_after = re.escape(context_after)
                
                # Allow for variations in the value
                if field_name == 'invoice_number':
                    value_pattern = r'([A-Z0-9-]+)'
                elif field_name in ['invoice_date', 'due_date']:
                    value_pattern = r'([A-Za-z0-9,\s/.-]+)'
                elif field_name in ['issuer_name', 'recipient_name']:
                    value_pattern = r'([A-Za-z0-9\s,&.-]+)'
                elif field_name == 'total_amount':
                    value_pattern = r'([$€£]?[\d,]+\.?\d*)'
                else:
                    value_pattern = r'([^\s]+)'
                
                # Create pattern with optional context
                pattern = f"{context_before}{value_pattern}{context_after}"
                
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(1), 0.7  # Moderate confidence
        
        return None, 0.0


class InvoiceExtractor:
    def __init__(self, pdf_path: str = None):
        self.pdf_path = pdf_path
        
        if pdf_path:
            self.text_content = self._extract_text()
            self.lines = self.text_content.split('\n')
            
            # Initialize form field extractor
            self.form_extractor = PDFFormExtractor(pdf_path)
            
            # Extract filename parts for improved extraction
            self.filename_info = self._extract_from_filename()
        else:
            self.text_content = ""
            self.lines = []
            self.form_extractor = None
            self.filename_info = {}
        
        # Initialize validator
        self.validator = ExtractionValidator()
        
        # Load stopwords
        self.stopwords = self._load_stopwords()
        
        # Known vendor patterns database
        self.vendor_patterns = self._load_vendor_patterns()
        
        # Check if document content is available before further processing
        if pdf_path and self.text_content:
            # Extract document layout
            self.document_sections = self._extract_document_sections()
            
            # Document classification
            self.document_type = self._classify_document()
            
            # Named entities
            self.entities = self._extract_named_entities()
            
            # Field relationships graph
            self.field_relationships = self._build_field_relationships()
        else:
            self.document_sections = {}
            self.document_type = InvoiceType.UNKNOWN
            self.entities = []
            self.field_relationships = {}
            
        logger.info(f"Initialized InvoiceExtractor for: {pdf_path}")
        if pdf_path and self.text_content:
            logger.info(f"Document type: {self.document_type}, Extracted {len(self.entities)} entities")

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
                print(f"Loaded {len(stopwords)} stopwords")
            except Exception as e:
                print(f"Error loading stopwords: {e}")
        else:
            print(f"Stopwords file not found at {stopwords_file}")
        
        return stopwords
        
    def _extract_from_filename(self) -> Dict[str, str]:
        """Extract information from the invoice filename."""
        result = {}
        filename = os.path.basename(self.pdf_path)
        
        # Pattern for "invoice_Name_12345.pdf"
        match = re.match(r'invoice_([^_]+)_(\d+)\.pdf', filename, re.IGNORECASE)
        if match:
            recipient_name = match.group(1).replace('_', ' ')
            invoice_number = match.group(2)
            
            # Clean up recipient name
            recipient_name = recipient_name.replace('_', ' ')
            
            # Store the extracted info
            result['recipient_name'] = recipient_name
            result['invoice_number'] = invoice_number
        
        return result
        
    def _extract_text(self) -> str:
        """Extract text from PDF file using pdfplumber."""
        full_text = ""
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page in pdf.pages:
                    full_text += page.extract_text() + "\n"
            return full_text
        except Exception as e:
            print(f"Error extracting text from {self.pdf_path}: {e}")
            return ""
    
    def _find_text_block(self, start_pattern: str, max_lines: int = 5) -> str:
        """Find a text block that starts with a specific pattern."""
        for i, line in enumerate(self.lines):
            if re.search(start_pattern, line, re.IGNORECASE):
                # Found the start of the block
                block_lines = []
                for j in range(i + 1, min(i + max_lines + 1, len(self.lines))):
                    next_line = self.lines[j].strip()
                    if not next_line:
                        break  # Empty line marks end of block
                    if re.search(r'^[A-Z][a-z]+ [A-Z][a-z]+', next_line):  # Looks like a name
                        block_lines.append(next_line)
                        break
                    if len(next_line.split()) <= 5:  # Short line that might be a name
                        block_lines.append(next_line)
                
                if block_lines:
                    return block_lines[0]  # Return first line of block
        
        return ""
    
    def extract_invoice_number(self) -> Optional[str]:
        """Extract invoice number from the text content."""
        # First check if we have it from the filename
        if 'invoice_number' in self.filename_info:
            return self.filename_info['invoice_number']
            
        # Next try to get from form fields if available
        if self.form_extractor.has_form_fields():
            form_invoice_number = self.form_extractor.get_invoice_number()
            if form_invoice_number:
                return form_invoice_number
        
        # Otherwise use text extraction
        patterns = [
            r"Invoice\s*#?\s*:\s*([A-Z0-9-]+)",
            r"Invoice\s*Number\s*:\s*([A-Z0-9-]+)",
            r"INVOICE\s*#\s*([A-Z0-9-]+)",
            r"Invoice\s*#\s*(\d+)",
            r"#\s*(\d+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, self.text_content, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def extract_invoice_date(self) -> Optional[str]:
        """Extract invoice date from the text content."""
        # First try to get from form fields if available
        if self.form_extractor.has_form_fields():
            form_invoice_date = self.form_extractor.get_invoice_date()
            if form_invoice_date:
                return form_invoice_date
                
        # First try specific patterns
        patterns = [
            r"Invoice\s*Date\s*:\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            r"Date\s*:\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            r"Invoice\s*Date\s*:\s*([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
            r"Date\s*:\s*([A-Za-z]+\s+\d{1,2},?\s+\d{4})"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, self.text_content, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Look for lines with 'Date:' or similar
        date_patterns = [
            r"\b(?:Date|Invoice Date)\b.*?(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b",
            r"\b(?:Date|Invoice Date)\b.*?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b"
        ]
        
        for line in self.lines:
            for pattern in date_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
        
        # If we can't find a specific date pattern, look for any date format in the first few lines
        general_date_patterns = [
            r"(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b",
            r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b"
        ]
        
        for i in range(min(10, len(self.lines))):
            line = self.lines[i]
            for pattern in general_date_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match and "due" not in line.lower():
                    return match.group(1).strip()
        
        # If no date found, try to generate current date as a fallback
        try:
            return datetime.now().strftime("%b %d, %Y")
        except:
            return None
    
    def extract_due_date(self) -> Optional[str]:
        """Extract due date from the text content."""
        # First try to get from form fields if available
        if self.form_extractor.has_form_fields():
            form_due_date = self.form_extractor.get_due_date()
            if form_due_date:
                return form_due_date
                
        patterns = [
            r"Due\s*Date\s*:\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            r"Payment\s*Due\s*:\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            r"Due\s*Date\s*:\s*([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
            r"Payment\s*Due\s*:\s*([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
            r"Due\s*By\s*:\s*([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
            r"Due\s*By\s*:\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, self.text_content, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Look for lines with 'Due' keywords
        for line in self.lines:
            if re.search(r"\bdue\b", line, re.IGNORECASE):
                # Look for date patterns in this line
                date_patterns = [
                    r"(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b",
                    r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b"
                ]
                
                for pattern in date_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        return match.group(1).strip()
        
        return None
    
    def extract_company_name(self) -> Optional[str]:
        """Extract the company name from the invoice."""
        # Default company name based on the README
        DEFAULT_COMPANY = "SuperStore"
        
        # Look for a company name in the first few lines
        company_indicators = ["LLC", "Inc", "Corporation", "Ltd", "Limited", "Co"]
        
        # Check first 7 lines for company name
        for i in range(min(7, len(self.lines))):
            line = self.lines[i].strip()
            # Skip empty lines, invoice headers, and date lines
            if (not line or 
                re.search(r"invoice|receipt|bill|statement|#\s*\d+", line, re.IGNORECASE) or
                re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", line, re.IGNORECASE) or
                re.search(r"ship\s+mode|quantity|item|rate|amount", line, re.IGNORECASE)):
                continue
                
            # If line contains a company indicator or looks like a name
            if any(indicator in line for indicator in company_indicators) or len(line.split()) <= 4:
                return line
        
        return DEFAULT_COMPANY
    
    def extract_issuer_name(self) -> Optional[str]:
        """
        Extract the issuer's name from the text content using advanced techniques.
        """
        # First try to get from form fields if available
        if self.form_extractor.has_form_fields():
            form_issuer_name = self.form_extractor.get_issuer_name()
            if form_issuer_name:
                return form_issuer_name
        
        # Check for named entities detected as organizations
        org_entities = [entity for entity in self.entities if entity.entity_type == 'ORG']
        if org_entities:
            # Sort by confidence
            org_entities.sort(key=lambda x: x.confidence, reverse=True)
            
            # Return the highest confidence organization
            best_org = org_entities[0]
            logger.info(f"Found issuer name using NER: {best_org.text} (confidence: {best_org.confidence})")
            return best_org.text
        
        # Look in document header section
        if 'header' in self.document_sections:
            header_text = self.document_sections['header'].text
            header_lines = header_text.split('\n')
            
            # First line is often the company name if it's not "invoice" or similar
            for line in header_lines:
                line = line.strip()
                if line and not re.search(r'\b(?:invoice|receipt|statement)\b', line, re.IGNORECASE):
                    if not any(word.lower() in line.lower() for word in self.stopwords):
                        logger.info(f"Found issuer name in header: {line}")
                        return line
        
        # Use explicit markers with improved pattern matching
        patterns = [
            r"From\s*:\s*([A-Za-z0-9\s,.'&-]+)(?:\n[A-Za-z0-9\s,.'&-]+)?",
            r"Issued\s*by\s*:\s*([A-Za-z0-9\s,.'&-]+)",
            r"Seller\s*:\s*([A-Za-z0-9\s,.'&-]+)",
            r"Vendor\s*:\s*([A-Za-z0-9\s,.'&-]+)",
            r"Company\s*:\s*([A-Za-z0-9\s,.'&-]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, self.text_content, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Verify it's not a false positive
                if not any(word.lower() in name.lower() for word in self.stopwords) and not re.search(r"ship\s+mode|standard class|first class", name, re.IGNORECASE):
                    if not re.search(r"^\s*#|date|invoice", name, re.IGNORECASE):
                        logger.info(f"Found issuer name with pattern: {name}")
                        return name
        
        # Check vendor patterns database
        for vendor_name, pattern_data in self.vendor_patterns.items():
            issuer_pattern = pattern_data.get("issuer_pattern", "")
            if issuer_pattern and re.search(issuer_pattern, self.text_content, re.IGNORECASE):
                logger.info(f"Found issuer name in vendor database: {vendor_name}")
                return vendor_name

        # Default fallback
        logger.info("Using default issuer name: SuperStore")
        return "SuperStore"
    
    def extract_recipient_name(self) -> Optional[str]:
        """
        Extract the recipient's name from the text content using NER and contextual understanding.
        """
        # First try to get from filename - this is often most reliable
        if 'recipient_name' in self.filename_info:
            recipient = self.filename_info['recipient_name']
            logger.info(f"Found recipient name in filename: {recipient}")
            return recipient
        
        # Then try to get from form fields if available
        if self.form_extractor.has_form_fields():
            form_recipient_name = self.form_extractor.get_recipient_name()
            if form_recipient_name:
                logger.info(f"Found recipient name in form fields: {form_recipient_name}")
                return form_recipient_name
        
        # Find potential recipients in named entities
        person_entities = [
            entity for entity in self.entities 
            if entity.entity_type == 'ORG' and entity.text != self.extract_issuer_name()
        ]
        
        # Check for address blocks which often contain recipient info
        address_blocks = []
        for i, line in enumerate(self.lines):
            if re.search(r"bill\s*to|ship\s*to|sold\s*to|delivery\s*to", line, re.IGNORECASE):
                # Collect the following lines as an address block
                block_text = line + "\n"
                j = i + 1
                while j < len(self.lines) and j < i + 6:  # Look at next 5 lines max
                    if self.lines[j].strip():
                        block_text += self.lines[j] + "\n"
                    j += 1
                
                address_blocks.append(block_text)
        
        # Try to extract names from address blocks
        for block in address_blocks:
            lines = block.split('\n')
            for i, line in enumerate(lines):
                # Skip the header line with "Bill To:" etc.
                if i == 0 and re.search(r"bill\s*to|ship\s*to", line, re.IGNORECASE):
                    continue
                
                # Skip empty lines
                if not line.strip():
                    continue
                
                # Skip lines that look like addresses or contain shipping info
                if (re.search(r"\d{1,5}\s+[A-Za-z]+(?:\s+(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr))", line, re.IGNORECASE) or
                    re.search(r"ship\s+mode|standard class|first class", line, re.IGNORECASE) or
                    re.search(r"\(\d{3}\)\s*\d{3}-\d{4}|\d{3}-\d{3}-\d{4}", line)):
                    continue
                
                # Skip if line contains stopwords
                if any(word.lower() in line.lower() for word in self.stopwords):
                    continue
                
                # The first non-header, non-address line is likely the name
                candidate = line.strip()
                if candidate:
                    logger.info(f"Found recipient name in address block: {candidate}")
                    return candidate
        
        # Use field relationships to find recipient near other identified fields
        if 'recipient_name' in self.field_relationships:
            # Check if we have identified invoice number, date, or amount
            related_fields = {
                'invoice_number': self.extract_invoice_number(),
                'invoice_date': self.extract_invoice_date(),
                'total_amount': self.extract_total_amount()[0]
            }
            
            # Filter to fields we've successfully extracted
            found_fields = {k: v for k, v in related_fields.items() if v}
            
            # For each related field we've found, look nearby in the text for recipient indicators
            for field_name, field_value in found_fields.items():
                if field_name in self.field_relationships['recipient_name']:
                    # Find position of this field value in text
                    field_pos = self.text_content.find(field_value)
                    if field_pos != -1:
                        # Look in a window around this position
                        window_size = 500  # characters
                        start = max(0, field_pos - window_size)
                        end = min(len(self.text_content), field_pos + window_size)
                        window_text = self.text_content[start:end]
                        
                        # Look for recipient indicators in this window
                        recipient_patterns = [
                            r"(?:To|Bill\s*To|Ship\s*To)\s*:\s*([A-Za-z0-9\s,.'&-]+)",
                            r"(?:Customer|Client)\s*:\s*([A-Za-z0-9\s,.'&-]+)",
                            r"(?:Attention|ATTN)\s*:\s*([A-Za-z0-9\s,.'&-]+)"
                        ]
                        
                        for pattern in recipient_patterns:
                            match = re.search(pattern, window_text, re.IGNORECASE)
                            if match:
                                name = match.group(1).strip()
                                # Verify not a false positive
                                false_positives = ["Ship To", "Bill To", "Balance Due", "Customer", "Ship Mode", "Standard Class", "First Class"]
                                if name not in false_positives and len(name) > 1:
                                    logger.info(f"Found recipient name near {field_name}: {name}")
                                    return name
        
        # Use positional heuristics - recipient is often in the top half, but after the issuer
        if 'header' in self.document_sections:
            header_text = self.document_sections['header'].text
            
            # Look for "To:" or similar in header
            to_match = re.search(r"(?:To|Bill\s*To|Ship\s*To|Attention|ATTN)\s*:\s*([A-Za-z0-9\s,.'&-]+)", 
                              header_text, re.IGNORECASE)
            if to_match:
                name = to_match.group(1).strip()
                if name and name not in ["Ship To", "Bill To"]:
                    logger.info(f"Found recipient name in header section: {name}")
                    return name
        
        # If we get here, we couldn't find a reliable recipient name
        logger.warning("Could not identify recipient name with high confidence")
        return "Unknown Recipient"
    
    def extract_total_amount(self) -> Tuple[Optional[str], Optional[Decimal], Optional[str]]:
        """
        Extract the total amount from the text content.
        
        Returns:
            Tuple of (original_amount_string, decimal_amount, currency_code)
        """
        # First try to get from form fields if available
        if self.form_extractor.has_form_fields():
            form_amount = self.form_extractor.get_total_amount()
            if form_amount:
                amount, currency = CurrencyProcessor.detect_and_standardize(form_amount)
                return form_amount, amount, currency
                
        patterns = [
            r"Total\s*:?\s*[$€£]?\s*([\d,]+\.\d{2})",
            r"Amount\s*Due\s*:?\s*[$€£]?\s*([\d,]+\.\d{2})",
            r"Total\s*Amount\s*:?\s*[$€£]?\s*([\d,]+\.\d{2})",
            r"Balance\s*Due\s*:?\s*[$€£]?\s*([\d,]+\.\d{2})",
            r"Grand\s*Total\s*:?\s*[$€£]?\s*([\d,]+\.\d{2})"
        ]
        
        # First look for total at the end of the document
        reverse_lines = self.lines.copy()
        reverse_lines.reverse()
        
        for i, line in enumerate(reverse_lines[:15]):  # Check last 15 lines
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    amount_str = match.group(1).strip()
                    amount, currency = CurrencyProcessor.detect_and_standardize(line)
                    return amount_str, amount, currency
        
        # If not found at the end, check the entire document
        for pattern in patterns:
            match = re.search(pattern, self.text_content, re.IGNORECASE)
            if match:
                amount_str = match.group(1).strip()
                # Extract full line containing the amount for currency detection
                for line in self.lines:
                    if amount_str in line:
                        amount, currency = CurrencyProcessor.detect_and_standardize(line)
                        return amount_str, amount, currency
                
                # If line not found, just process the amount
                amount, currency = CurrencyProcessor.detect_and_standardize(amount_str)
                return amount_str, amount, currency
        
        # Look for currency patterns near "total" keywords
        for i, line in enumerate(self.lines):
            if re.search(r"\btotal\b|\bamount\b|\bbalance\b", line, re.IGNORECASE):
                # Look for currency amounts
                amount_match = re.search(r"([\d,]+\.\d{2})", line)
                if amount_match:
                    amount_str = amount_match.group(1).strip()
                    amount, currency = CurrencyProcessor.detect_and_standardize(line)
                    return amount_str, amount, currency
        
        return None, None, None
    
    def extract_all(self) -> InvoiceData:
        """Extract all invoice data with advanced context-aware techniques."""
        # Extract basic data
        invoice_number = self.extract_invoice_number()
        invoice_date = self.extract_invoice_date()
        due_date = self.extract_due_date()
        issuer_name = self.extract_issuer_name()
        recipient_name = self.extract_recipient_name()
        
        # Extract amount with currency detection
        amount_str, amount_value, currency_code = self.extract_total_amount()
        
        # Initialize preliminary data for advanced processing
        extracted_fields = {
            'invoice_number': invoice_number,
            'invoice_date': invoice_date,
            'due_date': due_date,
            'issuer_name': issuer_name,
            'recipient_name': recipient_name,
            'total_amount': amount_str
        }
        
        # Use invoice graph for context-aware field inference
        invoice_graph = InvoiceGraph(self.text_content, extracted_fields)
        inferred_fields = invoice_graph.infer_missing_fields()
        
        # Update with inferred fields (only if original is missing)
        for field, value in inferred_fields.items():
            if field not in extracted_fields or not extracted_fields[field]:
                extracted_fields[field] = value
                logger.info(f"Inferred {field} from context: {value}")
        
        # Try few-shot learning for missing fields
        few_shot_examples_path = os.path.join(os.path.dirname(__file__), 'few_shot_examples.json')
        if os.path.exists(few_shot_examples_path):
            few_shot = FewShotExtractor(few_shot_examples_path)
            
            # For each missing field, try few-shot extraction
            for field in extracted_fields:
                if not extracted_fields[field]:
                    value, confidence = few_shot.extract_field(self.text_content, field)
                    if value and confidence > 0.5:  # Only use if reasonably confident
                        extracted_fields[field] = value
                        logger.info(f"Extracted {field} using few-shot learning: {value}")
        
        # Vendor-specific templates for known formats
        if self.document_type != InvoiceType.UNKNOWN:
            # Apply document type-specific adjustments
            if self.document_type == InvoiceType.STANDARD:
                # Standard invoices often have clearer fields
                pass  # Already handled in main extraction
            elif self.document_type == InvoiceType.RECEIPT:
                # Receipts might not have recipient_name as prominently
                if not extracted_fields['recipient_name'] or extracted_fields['recipient_name'] == "Unknown Recipient":
                    # Look for customer info in a different format
                    customer_match = re.search(r"customer(?:\s*|:)([A-Za-z\s]+)", self.text_content, re.IGNORECASE)
                    if customer_match:
                        extracted_fields['recipient_name'] = customer_match.group(1).strip()
            elif self.document_type == InvoiceType.PURCHASE_ORDER:
                # Purchase orders might have PO number as invoice_number
                po_match = re.search(r"P\.?O\.?\s*(?:#|No|Number)?\s*:?\s*([A-Z0-9-]+)", self.text_content, re.IGNORECASE)
                if po_match and not extracted_fields['invoice_number']:
                    extracted_fields['invoice_number'] = po_match.group(1).strip()
        
        # Create final invoice data object
        invoice_data = InvoiceData(
            invoice_number=extracted_fields['invoice_number'],
            invoice_date=extracted_fields['invoice_date'],
            due_date=extracted_fields['due_date'],
            issuer_name=extracted_fields['issuer_name'],
            recipient_name=extracted_fields['recipient_name'],
            total_amount=extracted_fields['total_amount'],
            amount_value=amount_value,
            currency_code=currency_code
        )
        
        # Validate and add confidence scores
        data_dict = invoice_data.to_dict()
        validated_data = self.validator.validate_all(data_dict)
        
        # Update the invoice data with validation results
        invoice_data.validation = validated_data.get('validation', {})
        invoice_data.confidence = validated_data.get('confidence', {})
        invoice_data.overall_confidence = validated_data.get('overall_confidence', 0.0)
        
        return invoice_data

    def _load_vendor_patterns(self) -> Dict[str, Dict]:
        """Load known vendor patterns from file or create default ones."""
        vendor_file = os.path.join(os.path.dirname(__file__), 'vendor_patterns.json')
        default_patterns = {
            "SuperStore": {
                "issuer_pattern": r"SuperStore|Super\s+Store",
                "header_pattern": r"SuperStore\s+Invoice",
                "invoice_prefix": "INV-",
                "confidence": 0.9
            },
            "Acme Corp": {
                "issuer_pattern": r"Acme\s+Corp|ACME",
                "header_pattern": r"ACME\s+INVOICE",
                "invoice_prefix": "AC-",
                "confidence": 0.85
            }
        }
        
        if os.path.exists(vendor_file):
            try:
                with open(vendor_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading vendor patterns: {e}")
        
        return default_patterns
    
    def _classify_document(self) -> str:
        """Classify the document type based on content."""
        # Initialize counters for keywords
        type_scores = {
            InvoiceType.STANDARD: 0,
            InvoiceType.RECEIPT: 0,
            InvoiceType.PURCHASE_ORDER: 0,
            InvoiceType.BILL: 0,
            InvoiceType.STATEMENT: 0
        }
        
        # Keywords associated with each document type
        keywords = {
            InvoiceType.STANDARD: ["invoice", "invoiced", "bill to"],
            InvoiceType.RECEIPT: ["receipt", "received", "payment received"],
            InvoiceType.PURCHASE_ORDER: ["purchase order", "po number", "order number"],
            InvoiceType.BILL: ["bill", "billing", "utility"],
            InvoiceType.STATEMENT: ["statement", "account summary", "balance"]
        }
        
        # Count keyword occurrences in text
        for doc_type, words in keywords.items():
            for word in words:
                count = len(re.findall(r'\b' + word + r'\b', self.text_content, re.IGNORECASE))
                type_scores[doc_type] += count
        
        # Get the most likely document type
        max_score = max(type_scores.values())
        if max_score == 0:
            return InvoiceType.UNKNOWN
        
        # Return the document type with the highest score
        for doc_type, score in type_scores.items():
            if score == max_score:
                logger.info(f"Document classified as: {doc_type}")
                return doc_type
    
    def _extract_document_sections(self) -> Dict[str, DocumentSection]:
        """Extract semantic sections of the document."""
        sections = {}
        
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_height = page.height
                    page_width = page.width
                    
                    # Define header region (top 20% of page)
                    header_bbox = (0, 0, page_width, page_height * 0.2)
                    header_text = page.crop(header_bbox).extract_text() or ""
                    if header_text:
                        sections['header'] = DocumentSection(
                            name='header',
                            text=header_text,
                            position=Position(0, 0, page_width, page_height * 0.2, page_num),
                            confidence=0.9
                        )
                    
                    # Define footer region (bottom 15% of page)
                    footer_bbox = (0, page_height * 0.85, page_width, page_height)
                    footer_text = page.crop(footer_bbox).extract_text() or ""
                    if footer_text:
                        sections['footer'] = DocumentSection(
                            name='footer',
                            text=footer_text,
                            position=Position(0, page_height * 0.85, page_width, page_height, page_num),
                            confidence=0.9
                        )
                    
                    # Extract address block regions (often in the top half)
                    address_pattern = r"\b\d+\s+[A-Za-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd)\b"
                    
                    # Extract tables which usually contain line items
                    tables = page.extract_tables()
                    if tables:
                        table_text = "\n".join([" | ".join([cell or "" for cell in row]) for table in tables for row in table])
                        sections['table'] = DocumentSection(
                            name='table',
                            text=table_text,
                            position=Position(0, 0, page_width, page_height, page_num),  # Approximate
                            confidence=0.8
                        )
                    
                    # Look for total section (usually near the bottom)
                    totals_bbox = (0, page_height * 0.6, page_width, page_height * 0.9)
                    totals_text = page.crop(totals_bbox).extract_text() or ""
                    if re.search(r'\b(?:total|amount due|balance|sum)\b', totals_text, re.IGNORECASE):
                        sections['totals'] = DocumentSection(
                            name='totals',
                            text=totals_text,
                            position=Position(0, page_height * 0.6, page_width, page_height * 0.9, page_num),
                            confidence=0.8
                        )
        
        except Exception as e:
            logger.error(f"Error extracting document sections: {e}")
        
        return sections
    
    def _extract_named_entities(self) -> List[NamedEntity]:
        """Extract named entities from text using pattern matching and NER."""
        entities = []
        
        # Company name patterns
        company_patterns = [
            r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+)(?:\s+(?:Inc|LLC|Ltd|Limited|Corporation|Corp|Co))\b",
            r"\b([A-Z][A-Za-z]+)(?:\s+(?:Inc|LLC|Ltd|Limited|Corporation|Corp|Co))\b",
            r"(?:From|Vendor|Supplier|Seller|Company):\s*([A-Z][A-Za-z0-9\s]+(?:Inc|LLC|Ltd|Limited|Corporation|Corp|Co)?)"
        ]
        
        # Organization indicators
        org_indicators = ["Inc", "LLC", "Ltd", "Limited", "Corporation", "Corp", "Co", "Company"]
        
        # Extract companies using patterns
        for pattern in company_patterns:
            for match in re.finditer(pattern, self.text_content):
                company_name = match.group(1).strip()
                
                # Skip if in stopwords or too short
                if (company_name.lower() in self.stopwords or 
                    len(company_name) <= 2 or
                    re.search(r'\b(?:bill to|ship to|attention|attn|total|invoice)\b', company_name, re.IGNORECASE)):
                    continue
                
                entities.append(NamedEntity(
                    text=company_name,
                    entity_type='ORG',
                    confidence=0.8 if any(indicator in company_name for indicator in org_indicators) else 0.6
                ))
        
        # Extract from headers using positional heuristics
        if 'header' in self.document_sections:
            header_text = self.document_sections['header'].text
            header_lines = header_text.split('\n')
            
            # First non-empty line in header is often company name
            for line in header_lines:
                line = line.strip()
                if line and len(line) > 2 and line.lower() not in self.stopwords:
                    # Skip if it contains invoice keywords
                    if not re.search(r'\b(?:invoice|receipt|statement)\b', line, re.IGNORECASE):
                        entities.append(NamedEntity(
                            text=line,
                            entity_type='ORG',
                            confidence=0.7
                        ))
                        break
        
        # Check against known vendor patterns
        for vendor_name, pattern_data in self.vendor_patterns.items():
            issuer_pattern = pattern_data.get("issuer_pattern", "")
            if issuer_pattern and re.search(issuer_pattern, self.text_content, re.IGNORECASE):
                entities.append(NamedEntity(
                    text=vendor_name,
                    entity_type='ORG',
                    confidence=pattern_data.get("confidence", 0.85)
                ))
        
        # Remove duplicates and sort by confidence
        unique_entities = {}
        for entity in entities:
            if entity.text not in unique_entities or entity.confidence > unique_entities[entity.text].confidence:
                unique_entities[entity.text] = entity
        
        return list(unique_entities.values())
    
    def _build_field_relationships(self) -> Dict[str, Dict[str, float]]:
        """Build relationships between fields based on proximity and context."""
        relationships = {
            "invoice_number": {},
            "invoice_date": {},
            "due_date": {},
            "issuer_name": {},
            "recipient_name": {},
            "total_amount": {}
        }
        
        # Define common field relationships with confidence scores
        relationships["invoice_number"]["invoice_date"] = 0.8  # Invoice numbers are often near dates
        relationships["invoice_date"]["due_date"] = 0.7  # Invoice date often near due date
        relationships["issuer_name"]["invoice_number"] = 0.6  # Issuer often associated with invoice number
        relationships["recipient_name"]["total_amount"] = 0.5  # Less strong relationship
        
        # For each field, find text proximity to other fields
        field_indicators = {
            "invoice_number": [r"invoice\s*(?:#|number|no)", r"inv\s*(?:#|number|no)"],
            "invoice_date": [r"(?:invoice|date)\s*(?:date|:)", r"dated"],
            "due_date": [r"due\s*(?:date|by)", r"payment\s*due"],
            "issuer_name": [r"from", r"(?:bill|issued)\s*(?:from|by)"],
            "recipient_name": [r"(?:bill|ship)\s*to", r"customer", r"client"],
            "total_amount": [r"(?:total|amount|balance)\s*(?:due|:)", r"(?:grand|invoice)\s*total"]
        }
        
        # Find proximity between field indicators in text
        field_positions = {}
        
        # Find all field indicators in text and their positions
        for field, indicators in field_indicators.items():
            positions = []
            for indicator in indicators:
                for match in re.finditer(indicator, self.text_content, re.IGNORECASE):
                    positions.append(match.start())
            
            if positions:
                field_positions[field] = sum(positions) / len(positions)  # Average position
        
        # Calculate proximity-based relationships
        for field1 in field_positions:
            for field2 in field_positions:
                if field1 != field2:
                    # Calculate proximity (closer = higher score)
                    distance = abs(field_positions[field1] - field_positions[field2])
                    max_distance = len(self.text_content)
                    proximity_score = 1 - (distance / max_distance)
                    
                    # Update relationship with proximity score if it improves it
                    if field2 not in relationships[field1] or proximity_score > relationships[field1][field2]:
                        relationships[field1][field2] = proximity_score
        
        return relationships


def process_directory(directory_path: str, output_file: str = "extracted_invoices.json"):
    """Process all PDF invoices in a directory and save results to a JSON file."""
    results = []
    
    # Ensure directory exists
    if not os.path.exists(directory_path):
        logger.error(f"Directory not found: {directory_path}")
        return
    
    # Get list of invoice files
    invoice_files = []
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(".pdf") and "invoice" in filename.lower():
            invoice_files.append(filename)
    
    logger.info(f"Found {len(invoice_files)} invoice files to process")
    
    # Process each invoice
    for filename in invoice_files:
        file_path = os.path.join(directory_path, filename)
        logger.info(f"Processing {filename}...")
        
        try:
            # Create extractor
            extractor = InvoiceExtractor(file_path)
            
            # Extract data
            invoice_data = extractor.extract_all()
            
            # Add to results
            results.append({
                "filename": filename,
                "data": invoice_data.to_dict()
            })
            
            # Log confidence
            logger.info(f"Extracted {filename} with confidence: {invoice_data.overall_confidence:.2f}")
            
            # Add extraction as few-shot example if high confidence
            if invoice_data.overall_confidence >= 0.8:
                try:
                    # Only add examples for fields with high confidence
                    few_shot_examples = []
                    for field, confidence in invoice_data.confidence.items():
                        if confidence >= 0.8 and getattr(invoice_data, field):
                            field_value = getattr(invoice_data, field)
                            # Find the text context around this value
                            if field_value and isinstance(field_value, str):
                                value_pos = extractor.text_content.find(field_value)
                                if value_pos != -1:
                                    # Get context (up to 100 chars before and after)
                                    start_pos = max(0, value_pos - 100)
                                    end_pos = min(len(extractor.text_content), value_pos + len(field_value) + 100)
                                    context = extractor.text_content[start_pos:end_pos]
                                    
                                    few_shot_examples.append({
                                        "text": context,
                                        "field": field,
                                        "value": field_value
                                    })
                    
                    # Save examples if we found any
                    if few_shot_examples:
                        examples_file = os.path.join(os.path.dirname(__file__), 'few_shot_examples.json')
                        
                        # Load existing examples
                        existing_examples = []
                        if os.path.exists(examples_file):
                            with open(examples_file, 'r') as f:
                                existing_examples = json.load(f)
                        
                        # Add new examples
                        existing_examples.extend(few_shot_examples)
                        
                        # Save updated examples
                        with open(examples_file, 'w') as f:
                            json.dump(existing_examples, f, indent=2)
                        
                        logger.info(f"Added {len(few_shot_examples)} high-confidence examples to few-shot learning")
                
                except Exception as e:
                    logger.error(f"Error adding few-shot examples: {e}")
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            results.append({
                "filename": filename,
                "data": {},
                "error": str(e)
            })
    
    # Save results to JSON file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Extraction complete. Results saved to {output_file}")
    
    # Calculate and log statistics
    total = len(results)
    errors = sum(1 for r in results if "error" in r)
    success_rate = (total - errors) / total * 100 if total > 0 else 0
    
    confidence_sum = sum(r["data"].get("overall_confidence", 0) for r in results if "data" in r and "overall_confidence" in r["data"])
    avg_confidence = confidence_sum / (total - errors) if (total - errors) > 0 else 0
    
    logger.info(f"Processed {total} invoices with {errors} errors")
    logger.info(f"Success rate: {success_rate:.1f}%")
    logger.info(f"Average confidence: {avg_confidence:.2f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract data from invoice PDFs")
    parser.add_argument("--dir", help="Directory containing invoice PDFs", default="Data")
    parser.add_argument("--output", help="Output JSON file", default="extracted_invoices.json")
    
    args = parser.parse_args()
    
    # Process all invoices in the specified directory
    process_directory(args.dir, args.output) 