import os
import sys
import json
import tempfile
import cv2
import numpy as np
from typing import Dict, Any, Optional, List
import logging
import re

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from Invoice_Extractor.invoice_extractor import InvoiceData
from Invoice_Extractor.ocr_processor import OCRProcessor
from utility.image_utils import preprocess_image_pipeline

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageInvoiceExtractor:
    """
    Specialized extractor for processing image-based invoices.
    Uses the enhanced image preprocessing pipeline for better OCR results.
    """
    
    def __init__(self):
        """Initialize the image invoice extractor."""
        self.ocr_processor = OCRProcessor()
        self.temp_files = []
        logger.info("Image Invoice Extractor initialized")
    
    def extract(self, image_path: str, use_enhanced: bool = True) -> InvoiceData:
        """
        Extract invoice data from an image.
        
        Args:
            image_path: Path to the image file
            use_enhanced: Whether to use enhanced preprocessing
            
        Returns:
            InvoiceData object with extracted information
        """
        try:
            logger.info(f"Extracting data from image: {os.path.basename(image_path)}")
            
            # Create a temporary directory
            temp_dir = tempfile.mkdtemp()
            
            # Process image for better OCR
            if use_enhanced:
                # Use our enhanced preprocessing pipeline
                jpeg_path, enhanced_path = preprocess_image_pipeline(image_path, temp_dir)
                
                # Add to temp files for later cleanup
                if jpeg_path:
                    self.temp_files.append(jpeg_path)
                if enhanced_path:
                    self.temp_files.append(enhanced_path)
                    
                # Use the enhanced image if available
                ocr_path = enhanced_path if enhanced_path else image_path
            else:
                ocr_path = image_path
            
            # Process the image using OCR processor
            ocr_result = self.ocr_processor.process_file(ocr_path, preprocess=True)
            
            # Check for errors
            if "error" in ocr_result:
                logger.error(f"OCR processing failed: {ocr_result['error']}")
                return InvoiceData()
            
            # Extract text from OCR result
            text_content = ocr_result.get("text", "")
            
            # Extract invoice information from the text
            invoice_data = self._extract_invoice_data(text_content)
            
            # Calculate confidence scores
            confidence_scores = self._calculate_confidence(invoice_data, text_content)
            
            # Update invoice data with confidence scores
            for field, confidence in confidence_scores.items():
                invoice_data.confidence[field] = confidence
            
            # Calculate overall confidence
            if confidence_scores:
                invoice_data.overall_confidence = sum(confidence_scores.values()) / len(confidence_scores)
            
            logger.info(f"Extraction complete with confidence: {invoice_data.overall_confidence:.2f}")
            return invoice_data
            
        except Exception as e:
            logger.error(f"Error extracting from image: {str(e)}")
            return InvoiceData()
        finally:
            # Clean up temporary files
            self._cleanup_temp_files()
    
    def _extract_invoice_data(self, text: str) -> InvoiceData:
        """
        Extract invoice data from OCR text.
        
        Args:
            text: OCR-extracted text
            
        Returns:
            InvoiceData object with extracted fields
        """
        invoice_data = InvoiceData()
        
        # Extract invoice number
        invoice_number_match = re.search(r'invoice\s*#?\s*[:\-]?\s*([a-zA-Z0-9\-]+)', text, re.IGNORECASE)
        if not invoice_number_match:
            invoice_number_match = re.search(r'invoice\s*number\s*[:\-]?\s*([a-zA-Z0-9\-]+)', text, re.IGNORECASE)
        if invoice_number_match:
            invoice_data.invoice_number = invoice_number_match.group(1).strip()
        
        # Extract invoice date
        date_match = re.search(r'(invoice)?\s*date\s*[:\-]?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})', text, re.IGNORECASE)
        if date_match:
            invoice_data.invoice_date = date_match.group(2).strip()
        
        # Extract due date
        due_date_match = re.search(r'due\s*date\s*[:\-]?\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})', text, re.IGNORECASE)
        if due_date_match:
            invoice_data.due_date = due_date_match.group(1).strip()
        
        # Extract issuer name (look for From: or company at top)
        issuer_match = re.search(r'from\s*[:\-]?\s*([A-Za-z0-9\s&,\.]+)', text, re.IGNORECASE)
        if not issuer_match:
            # Try to get the company name from the first line
            lines = text.strip().split('\n')
            if lines and lines[0]:
                issuer_candidate = lines[0].strip()
                # Check if it's a reasonable company name (not too short, not just numbers)
                if len(issuer_candidate) > 3 and not issuer_candidate.isdigit():
                    invoice_data.issuer_name = issuer_candidate
        else:
            invoice_data.issuer_name = issuer_match.group(1).strip()
        
        # Extract recipient name (look for To: or Bill To:)
        recipient_match = re.search(r'(?:to|bill\s*to)\s*[:\-]?\s*([A-Za-z0-9\s&,\.]+)', text, re.IGNORECASE)
        if recipient_match:
            invoice_data.recipient_name = recipient_match.group(1).strip()
        
        # Extract total amount
        total_match = re.search(r'(?:total|amount due|balance due|grand total)[:\s]*[$£€¥]?\s*([\d,\.]+)', text, re.IGNORECASE)
        if total_match:
            invoice_data.total_amount = total_match.group(1).strip()
        
        return invoice_data
    
    def _calculate_confidence(self, invoice_data: InvoiceData, text: str) -> Dict[str, float]:
        """
        Calculate confidence scores for extracted fields.
        
        Args:
            invoice_data: Extracted invoice data
            text: OCR-extracted text
            
        Returns:
            Dictionary of field confidence scores
        """
        confidence = {}
        
        # Check if field is present and estimate confidence
        if invoice_data.invoice_number:
            confidence["invoice_number"] = 0.8 if re.search(r'invoice\s*#', text, re.IGNORECASE) else 0.6
        
        if invoice_data.invoice_date:
            confidence["invoice_date"] = 0.8 if re.search(r'date', text, re.IGNORECASE) else 0.6
        
        if invoice_data.due_date:
            confidence["due_date"] = 0.9 if re.search(r'due\s*date', text, re.IGNORECASE) else 0.7
        
        if invoice_data.issuer_name:
            confidence["issuer_name"] = 0.7 if re.search(r'from', text, re.IGNORECASE) else 0.5
        
        if invoice_data.recipient_name:
            confidence["recipient_name"] = 0.8 if re.search(r'to|bill', text, re.IGNORECASE) else 0.6
        
        if invoice_data.total_amount:
            confidence["total_amount"] = 0.9 if re.search(r'total', text, re.IGNORECASE) else 0.7
        
        return confidence
    
    def _cleanup_temp_files(self):
        """Clean up temporary files created during processing."""
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug(f"Removed temporary file: {file_path}")
            except Exception as e:
                logger.warning(f"Error removing temporary file {file_path}: {e}")
        
        self.temp_files = []


def process_image(image_path: str) -> Dict[str, Any]:
    """
    Process an image file to extract invoice data.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with extracted invoice data
    """
    extractor = ImageInvoiceExtractor()
    result = extractor.extract(image_path)
    return result.to_dict()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"Processing: {image_path}")
        result = process_image(image_path)
        print(json.dumps(result, indent=2))
    else:
        print("Please provide an image path.") 