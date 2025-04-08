import re
import pdfplumber
from typing import Dict, List, Tuple, Any, Optional
import json

class LayoutAnalyzer:
    """
    Analyzes the layout structure of invoices to better locate fields.
    
    This helps identify different regions of the invoice (header, address blocks,
    line items table, totals section) for more accurate field extraction.
    """
    
    def __init__(self, pdf_path: str):
        """Initialize with path to PDF file."""
        self.pdf_path = pdf_path
        self.layout_data = self._analyze_layout()
    
    def _analyze_layout(self) -> Dict[str, Any]:
        """Analyze the layout of the invoice and identify key regions."""
        result = {
            "regions": {},
            "tables": [],
            "address_blocks": [],
            "line_item_region": None,
            "total_region": None,
            "header_region": None,
        }
        
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                # Analyze first page (most invoices are single page)
                if len(pdf.pages) > 0:
                    page = pdf.pages[0]
                    
                    # Extract text with positions
                    text_with_positions = self._extract_text_with_positions(page)
                    
                    # Identify header region (top 20% of page)
                    page_height = page.height
                    header_y_threshold = page_height * 0.2
                    header_elements = [t for t in text_with_positions if t['top'] < header_y_threshold]
                    if header_elements:
                        result["header_region"] = {
                            "elements": header_elements,
                            "bbox": self._get_region_bbox(header_elements),
                            "text": "\n".join([t['text'] for t in header_elements])
                        }
                    
                    # Detect tables
                    tables = page.extract_tables()
                    if tables:
                        for i, table in enumerate(tables):
                            if table and len(table) > 1:  # Must have at least header + one row
                                # Convert table to text representation
                                table_text = self._table_to_text(table)
                                result["tables"].append({
                                    "index": i,
                                    "rows": len(table),
                                    "cols": len(table[0]) if table[0] else 0,
                                    "content": table,
                                    "text": table_text
                                })
                                
                                # Identify if this is the line items table
                                if self._is_line_items_table(table_text):
                                    result["line_item_region"] = i
                    
                    # Find address blocks (common formatting patterns)
                    address_blocks = self._find_address_blocks(text_with_positions)
                    result["address_blocks"] = address_blocks
                    
                    # Find total section (usually bottom 25% of page)
                    total_y_threshold = page_height * 0.75
                    total_elements = [t for t in text_with_positions if t['top'] > total_y_threshold]
                    if total_elements:
                        result["total_region"] = {
                            "elements": total_elements,
                            "bbox": self._get_region_bbox(total_elements),
                            "text": "\n".join([t['text'] for t in total_elements])
                        }
        except Exception as e:
            print(f"Error analyzing layout of {self.pdf_path}: {e}")
        
        return result
    
    def _extract_text_with_positions(self, page) -> List[Dict[str, Any]]:
        """Extract text with position information."""
        words = page.extract_words()
        lines_by_position = {}
        
        # Group words by vertical position
        for word in words:
            # Round y-position to group words in the same line
            y_pos = round(word['top'])
            if y_pos not in lines_by_position:
                lines_by_position[y_pos] = []
            lines_by_position[y_pos].append(word)
        
        # Sort lines by y-position and words by x-position within each line
        result = []
        for y_pos in sorted(lines_by_position.keys()):
            line_words = sorted(lines_by_position[y_pos], key=lambda w: w['x0'])
            text = " ".join([w['text'] for w in line_words])
            
            # Get bounding box of the whole line
            x0 = min([w['x0'] for w in line_words])
            x1 = max([w['x1'] for w in line_words])
            y0 = min([w['top'] for w in line_words])
            y1 = max([w['bottom'] for w in line_words])
            
            result.append({
                'text': text,
                'x0': x0,
                'x1': x1,
                'top': y0,
                'bottom': y1,
                'width': x1 - x0,
                'height': y1 - y0
            })
        
        return result
    
    def _get_region_bbox(self, elements: List[Dict[str, Any]]) -> Tuple[float, float, float, float]:
        """Get bounding box for a list of elements."""
        if not elements:
            return (0, 0, 0, 0)
        
        x0 = min([e['x0'] for e in elements])
        x1 = max([e['x1'] for e in elements])
        y0 = min([e['top'] for e in elements])
        y1 = max([e['bottom'] for e in elements])
        
        return (x0, y0, x1, y1)
    
    def _table_to_text(self, table: List[List[str]]) -> str:
        """Convert a table to a text representation."""
        if not table:
            return ""
            
        # Clean up table cells
        cleaned_table = []
        for row in table:
            cleaned_row = []
            for cell in row:
                # Replace None with empty string
                cell_text = str(cell).strip() if cell is not None else ""
                cleaned_row.append(cell_text)
            cleaned_table.append(cleaned_row)
            
        # Join cells into text
        return "\n".join([" | ".join(row) for row in cleaned_table])
    
    def _is_line_items_table(self, table_text: str) -> bool:
        """Determine if a table is the line items table."""
        # Common headers for line item tables
        line_item_headers = [
            "item", "description", "quantity", "price", "amount", 
            "product", "service", "rate", "total", "unit", "qty"
        ]
        
        # Check if table has line item headers
        first_line = table_text.split('\n')[0].lower() if table_text else ""
        header_matches = sum(1 for header in line_item_headers if header in first_line)
        
        # Needs at least 2 of the common headers to be considered a line items table
        return header_matches >= 2
    
    def _find_address_blocks(self, text_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify address blocks in the text elements."""
        address_blocks = []
        address_patterns = [
            r"\d+\s+[A-Za-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr)",  # Street address
            r"[A-Za-z]+,\s*[A-Z]{2}\s+\d{5}",  # City, State ZIP
            r"\(\d{3}\)\s*\d{3}-\d{4}|\d{3}-\d{3}-\d{4}"  # Phone number
        ]
        
        # Group consecutive lines that may form an address block
        current_block = []
        current_block_y = None
        
        sorted_elements = sorted(text_elements, key=lambda e: e['top'])
        
        for elem in sorted_elements:
            text = elem['text']
            
            # Check if this line matches any address pattern
            is_address_like = any(re.search(pattern, text) for pattern in address_patterns)
            
            # Or check if it follows common address formats (e.g. "City, ST 12345")
            if not is_address_like:
                # City, State ZIP pattern
                if re.search(r"[A-Za-z ]+,?\s+[A-Z]{2}\s+\d{5}", text):
                    is_address_like = True
            
            # If this is part of the current block
            if current_block and abs(elem['top'] - current_block_y) < 20:  # Close enough vertically
                if is_address_like or len(current_block) < 4:  # Allow a few non-address lines
                    current_block.append(elem)
                    current_block_y = elem['top']
                else:
                    # End current block and save it if it contains address-like elements
                    if any(any(re.search(pattern, e['text']) for pattern in address_patterns) for e in current_block):
                        address_blocks.append({
                            "elements": current_block.copy(),
                            "bbox": self._get_region_bbox(current_block),
                            "text": "\n".join([e['text'] for e in current_block])
                        })
                    current_block = []
                    current_block_y = None
            
            # Start a new block
            elif is_address_like:
                if current_block:
                    # End and save previous block
                    if any(any(re.search(pattern, e['text']) for pattern in address_patterns) for e in current_block):
                        address_blocks.append({
                            "elements": current_block.copy(),
                            "bbox": self._get_region_bbox(current_block),
                            "text": "\n".join([e['text'] for e in current_block])
                        })
                current_block = [elem]
                current_block_y = elem['top']
        
        # Add the last block if any
        if current_block:
            if any(any(re.search(pattern, e['text']) for pattern in address_patterns) for e in current_block):
                address_blocks.append({
                    "elements": current_block,
                    "bbox": self._get_region_bbox(current_block),
                    "text": "\n".join([e['text'] for e in current_block])
                })
        
        return address_blocks
    
    def find_field_region(self, field_name: str) -> Optional[str]:
        """
        Find the most likely region containing a specific field.
        
        Args:
            field_name: Name of the field to locate ('invoice_number', 'invoice_date', etc.)
            
        Returns:
            Text content of the region or None if not found
        """
        # Define region patterns for different fields
        field_patterns = {
            'invoice_number': [r'invoice\s*#', r'invoice\s*no', r'invoice\s*number'],
            'invoice_date': [r'invoice\s*date', r'date', r'issued'],
            'due_date': [r'due\s*date', r'payment\s*due', r'due\s*by'],
            'issuer_name': [r'from', r'seller', r'vendor', r'issued\s*by'],
            'recipient_name': [r'to', r'bill\s*to', r'ship\s*to', r'customer', r'client'],
            'total_amount': [r'total', r'balance\s*due', r'amount\s*due', r'grand\s*total']
        }
        
        # Select patterns for the requested field
        patterns = field_patterns.get(field_name, [])
        if not patterns:
            return None
        
        # Check header region for invoice number, dates
        if field_name in ['invoice_number', 'invoice_date'] and 'header_region' in self.layout_data:
            header_text = self.layout_data['header_region'].get('text', '')
            for pattern in patterns:
                if re.search(pattern, header_text, re.IGNORECASE):
                    return header_text
        
        # Check address blocks for issuer/recipient
        if field_name in ['issuer_name', 'recipient_name']:
            for block in self.layout_data.get('address_blocks', []):
                block_text = block.get('text', '')
                for pattern in patterns:
                    if re.search(pattern, block_text, re.IGNORECASE):
                        return block_text
        
        # Check total region for amounts
        if field_name == 'total_amount' and 'total_region' in self.layout_data:
            total_text = self.layout_data['total_region'].get('text', '')
            for pattern in patterns:
                if re.search(pattern, total_text, re.IGNORECASE):
                    return total_text
        
        # Search in all regions as fallback
        for region_name in ['header_region', 'total_region']:
            if region_name in self.layout_data:
                region_text = self.layout_data[region_name].get('text', '')
                for pattern in patterns:
                    if re.search(pattern, region_text, re.IGNORECASE):
                        return region_text
        
        return None
    
    def get_address_blocks(self) -> List[str]:
        """Get text content of all identified address blocks."""
        return [block.get('text', '') for block in self.layout_data.get('address_blocks', [])]
    
    def get_line_items_table(self) -> Optional[str]:
        """Get the line items table if found."""
        line_item_idx = self.layout_data.get('line_item_region')
        if line_item_idx is not None:
            tables = self.layout_data.get('tables', [])
            if 0 <= line_item_idx < len(tables):
                return tables[line_item_idx].get('text', '')
        return None
    
    def get_total_section(self) -> Optional[str]:
        """Get the total section text if found."""
        if 'total_region' in self.layout_data:
            return self.layout_data['total_region'].get('text', '')
        return None
    
    def get_header_section(self) -> Optional[str]:
        """Get the header section text if found."""
        if 'header_region' in self.layout_data:
            return self.layout_data['header_region'].get('text', '')
        return None


if __name__ == "__main__":
    # Simple test if run directly
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        analyzer = LayoutAnalyzer(pdf_path)
        
        print("=== Layout Analysis Results ===")
        print("\nHeader Section:")
        print(analyzer.get_header_section() or "Not found")
        
        print("\nAddress Blocks:")
        for i, block in enumerate(analyzer.get_address_blocks()):
            print(f"Block {i+1}:")
            print(block)
            print("-" * 40)
        
        print("\nLine Items Table:")
        print(analyzer.get_line_items_table() or "Not found")
        
        print("\nTotal Section:")
        print(analyzer.get_total_section() or "Not found")
        
        print("\nInvoice Number Location:")
        print(analyzer.find_field_region('invoice_number') or "Not found")
        
        print("\nInvoice Date Location:")
        print(analyzer.find_field_region('invoice_date') or "Not found")
        
        print("\nRecipient Location:")
        print(analyzer.find_field_region('recipient_name') or "Not found")
    else:
        print("Usage: python layout_analyzer.py path/to/invoice.pdf") 