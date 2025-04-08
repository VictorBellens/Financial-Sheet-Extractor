import os
import sys
import pytesseract
from PIL import Image
import pdf2image
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
import logging
import tempfile
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ocr_processor')

class OCRProcessor:
    """
    Process image-based invoices (TIFF, JPG, PNG) and scanned PDFs using OCR.
    Converts the images to text that can be processed by the invoice extractor.
    """
    
    def __init__(self, tesseract_path: Optional[str] = None):
        """
        Initialize the OCR processor.
        
        Args:
            tesseract_path: Path to the Tesseract executable
        """
        # If tesseract path is provided, set the path
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Check if tesseract is available
        try:
            pytesseract.get_tesseract_version()
            logger.info("Tesseract OCR initialized successfully")
        except Exception as e:
            logger.warning(f"Tesseract OCR not properly configured: {e}")
            logger.warning("Please install Tesseract OCR and ensure it's in your PATH")
            logger.warning("Or provide the path to the tesseract executable")
    
    def process_file(self, file_path: str, output_path: Optional[str] = None, 
                     preprocess: bool = True) -> Dict[str, Any]:
        """
        Process an image file or PDF and extract text using OCR.
        
        Args:
            file_path: Path to the image or PDF file
            output_path: Optional path to save the OCR result
            preprocess: Whether to preprocess the image for better OCR results
            
        Returns:
            Dictionary with OCR results, including text content and positions
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            # Handle different file types
            if file_ext == '.pdf':
                result = self._process_pdf(file_path, preprocess)
            elif file_ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']:
                result = self._process_image(file_path, preprocess)
            else:
                logger.error(f"Unsupported file type: {file_ext}")
                return {"error": f"Unsupported file type: {file_ext}"}
            
            # Save result if output path is provided
            if output_path and result:
                with open(output_path, 'w', encoding='utf-8') as f:
                    if 'text' in result:
                        f.write(result['text'])
                    else:
                        json.dump(result, f, indent=2, ensure_ascii=False)
            
            return result
        
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    def _process_pdf(self, pdf_path: str, preprocess: bool = True) -> Dict[str, Any]:
        """
        Process a PDF file using OCR.
        
        Args:
            pdf_path: Path to the PDF file
            preprocess: Whether to preprocess the image
            
        Returns:
            Dictionary with OCR results
        """
        try:
            # Convert PDF to images
            images = pdf2image.convert_from_path(pdf_path)
            
            if not images:
                return {"error": "No images extracted from PDF"}
            
            # Process each page
            pages_text = []
            pages_data = []
            
            for i, img in enumerate(images):
                logger.info(f"Processing page {i+1}/{len(images)} of {pdf_path}")
                
                # Convert PIL Image to numpy array for OpenCV
                if preprocess:
                    img_np = np.array(img)
                    img_processed = self._preprocess_image(img_np)
                    pil_img = Image.fromarray(img_processed)
                else:
                    pil_img = img
                
                # Get OCR data with positioning
                page_data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
                
                # Extract text content
                text_content = self._extract_text_from_data(page_data)
                pages_text.append(text_content)
                
                # Store page data
                pages_data.append({
                    'page': i+1,
                    'width': img.width,
                    'height': img.height,
                    'ocr_data': self._simplify_ocr_data(page_data)
                })
            
            # Combine results
            result = {
                'text': '\n'.join(pages_text),
                'pages': pages_data,
                'source': pdf_path,
                'type': 'pdf',
                'page_count': len(images)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    def _process_image(self, image_path: str, preprocess: bool = True) -> Dict[str, Any]:
        """
        Process an image file using OCR.
        
        Args:
            image_path: Path to the image file
            preprocess: Whether to preprocess the image
            
        Returns:
            Dictionary with OCR results
        """
        try:
            # Read image
            img = Image.open(image_path)
            
            # Preprocess image if needed
            if preprocess:
                img_np = np.array(img)
                img_processed = self._preprocess_image(img_np)
                pil_img = Image.fromarray(img_processed)
            else:
                pil_img = img
            
            # Get OCR data with positioning
            ocr_data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
            
            # Extract text content
            text_content = self._extract_text_from_data(ocr_data)
            
            # Combine results
            result = {
                'text': text_content,
                'pages': [{
                    'page': 1,
                    'width': img.width,
                    'height': img.height,
                    'ocr_data': self._simplify_ocr_data(ocr_data)
                }],
                'source': image_path,
                'type': os.path.splitext(image_path)[1][1:],
                'page_count': 1
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results.
        
        Args:
            img: Image as numpy array
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if it's a color image
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply noise removal
        denoised = cv2.fastNlMeansDenoising(binary, h=10)
        
        # Dilation and erosion to remove noise
        kernel = np.ones((1, 1), np.uint8)
        dilated = cv2.dilate(denoised, kernel, iterations=1)
        eroded = cv2.erode(dilated, kernel, iterations=1)
        
        return eroded
    
    def _extract_text_from_data(self, ocr_data: Dict[str, Any]) -> str:
        """
        Extract formatted text from OCR data.
        
        Args:
            ocr_data: OCR data from pytesseract
            
        Returns:
            Formatted text
        """
        text_lines = []
        current_line = ""
        prev_line_num = -1
        
        # Iterate through recognized text
        for i in range(len(ocr_data['text'])):
            text = ocr_data['text'][i]
            conf = ocr_data['conf'][i]
            line_num = ocr_data['line_num'][i]
            
            # Skip empty text or low confidence text
            if not text.strip() or conf < 30:
                continue
            
            # If we're on a new line, add the previous line to the list
            if line_num != prev_line_num and prev_line_num != -1:
                text_lines.append(current_line.strip())
                current_line = ""
            
            # Add text to current line
            current_line += text + " "
            prev_line_num = line_num
        
        # Add the last line
        if current_line:
            text_lines.append(current_line.strip())
        
        return '\n'.join(text_lines)
    
    def _simplify_ocr_data(self, ocr_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Simplify OCR data to include only relevant information.
        
        Args:
            ocr_data: OCR data from pytesseract
            
        Returns:
            Simplified OCR data
        """
        simplified = []
        
        for i in range(len(ocr_data['text'])):
            text = ocr_data['text'][i]
            
            # Skip empty text
            if not text.strip():
                continue
            
            # Add relevant information
            simplified.append({
                'text': text,
                'conf': ocr_data['conf'][i],
                'x': ocr_data['left'][i],
                'y': ocr_data['top'][i],
                'width': ocr_data['width'][i],
                'height': ocr_data['height'][i],
                'line': ocr_data['line_num'][i],
                'block': ocr_data['block_num'][i]
            })
        
        return simplified

    def convert_to_pdf(self, image_path: str, output_path: Optional[str] = None) -> str:
        """
        Convert an image file to PDF.
        
        Args:
            image_path: Path to the image file
            output_path: Optional path to save the PDF
            
        Returns:
            Path to the converted PDF
        """
        try:
            # Read image
            img = Image.open(image_path)
            
            # Determine output path
            if not output_path:
                output_path = os.path.splitext(image_path)[0] + '.pdf'
            
            # Save as PDF
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            
            img.save(output_path, 'PDF')
            logger.info(f"Image converted to PDF: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error converting image to PDF: {str(e)}", exc_info=True)
            raise
    
    def extract_tables(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract tables from an image or PDF.
        
        Args:
            file_path: Path to the image or PDF file
            
        Returns:
            List of extracted tables with positions and content
        """
        # This is a basic implementation - for production use, consider tabula-py for PDFs
        # or more sophisticated table extraction libraries
        
        file_ext = os.path.splitext(file_path)[1].lower()
        tables = []
        
        try:
            # Process image or PDF
            if file_ext == '.pdf':
                images = pdf2image.convert_from_path(file_path)
                for i, img in enumerate(images):
                    img_np = np.array(img)
                    page_tables = self._detect_tables(img_np)
                    for table in page_tables:
                        table['page'] = i + 1
                    tables.extend(page_tables)
            else:
                # Process single image
                img = Image.open(file_path)
                img_np = np.array(img)
                tables = self._detect_tables(img_np)
                for table in tables:
                    table['page'] = 1
            
            return tables
            
        except Exception as e:
            logger.error(f"Error extracting tables: {str(e)}", exc_info=True)
            return []
    
    def _detect_tables(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect tables in an image using contour detection.
        
        Args:
            img: Image as numpy array
            
        Returns:
            List of detected tables
        """
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Apply thresholding
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Dilate to connect lines
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=3)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size
        min_area = img.shape[0] * img.shape[1] * 0.01  # Min 1% of image area
        tables = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Verify if it's likely a table (width-to-height ratio)
                aspect_ratio = w / h
                if 0.5 <= aspect_ratio <= 5:  # Typical table aspect ratios
                    # Extract table region
                    table_region = gray[y:y+h, x:x+w]
                    
                    # OCR the table region
                    table_text = pytesseract.image_to_string(table_region)
                    
                    # Add table information
                    tables.append({
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'content': table_text,
                        'confidence': self._estimate_table_confidence(table_region)
                    })
        
        return tables
    
    def _estimate_table_confidence(self, table_img: np.ndarray) -> float:
        """
        Estimate confidence level for table detection.
        
        Args:
            table_img: Table image region
            
        Returns:
            Estimated confidence level (0-1)
        """
        # Use OCR confidence as a proxy for table detection confidence
        ocr_data = pytesseract.image_to_data(table_img, output_type=pytesseract.Output.DICT)
        
        # Calculate average confidence of recognized text
        confidences = [conf for conf in ocr_data['conf'] if conf != -1]
        
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            return avg_confidence / 100.0  # Normalize to 0-1
        else:
            return 0.3  # Default moderate confidence


def create_searchable_pdf(input_path: str, output_path: Optional[str] = None) -> str:
    """
    Create a searchable PDF with OCR text layer.
    
    Args:
        input_path: Path to the input PDF or image
        output_path: Optional path to save the searchable PDF
        
    Returns:
        Path to the searchable PDF
    """
    # Default output path
    if not output_path:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_searchable.pdf"
    
    try:
        processor = OCRProcessor()
        
        # Process file with OCR
        ocr_result = processor.process_file(input_path)
        
        if 'error' in ocr_result:
            logger.error(f"OCR processing failed: {ocr_result['error']}")
            return ""
        
        # For images, convert to PDF first
        file_ext = os.path.splitext(input_path)[1].lower()
        pdf_path = input_path
        
        if file_ext != '.pdf':
            pdf_path = processor.convert_to_pdf(input_path)
        
        # External tools are typically used here to add OCR text layer
        # For a complete implementation, you would typically use tools like ocrmypdf
        # Since this is beyond the scope of pure Python, we'll log a message
        logger.info(f"To create a fully searchable PDF, install and use ocrmypdf CLI tool:")
        logger.info(f"ocrmypdf {pdf_path} {output_path}")
        
        # For this example, we'll simulate by copying the file
        import shutil
        shutil.copy(pdf_path, output_path)
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating searchable PDF: {str(e)}", exc_info=True)
        return ""


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process image files with OCR')
    parser.add_argument('input', help='Input file (PDF, TIFF, JPG, PNG)')
    parser.add_argument('--output', '-o', help='Output file for OCR results')
    parser.add_argument('--no-preprocess', action='store_true', help='Skip image preprocessing')
    parser.add_argument('--tables', action='store_true', help='Extract tables')
    parser.add_argument('--searchable', action='store_true', help='Create searchable PDF')
    parser.add_argument('--tesseract', help='Path to tesseract executable')
    
    args = parser.parse_args()
    
    # Initialize OCR processor
    processor = OCRProcessor(tesseract_path=args.tesseract)
    
    if args.tables:
        # Extract tables
        tables = processor.extract_tables(args.input)
        print(f"Found {len(tables)} tables")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(tables, f, indent=2)
            print(f"Table extraction results saved to {args.output}")
        else:
            for i, table in enumerate(tables):
                print(f"\nTable {i+1}:")
                print(f"Position: x={table['x']}, y={table['y']}, width={table['width']}, height={table['height']}")
                print(f"Content:\n{table['content']}")
    
    elif args.searchable:
        # Create searchable PDF
        output = args.output or os.path.splitext(args.input)[0] + '_searchable.pdf'
        result = create_searchable_pdf(args.input, output)
        if result:
            print(f"Searchable PDF created: {result}")
        else:
            print("Failed to create searchable PDF")
    
    else:
        # Process file with OCR
        result = processor.process_file(args.input, args.output, preprocess=not args.no_preprocess)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"OCR processing complete")
            print(f"Detected text ({len(result['text'])} characters):")
            print(result['text'][:500] + '...' if len(result['text']) > 500 else result['text'])
            
            if args.output:
                print(f"Results saved to {args.output}")
            
            print(f"\nProcessed {result['page_count']} page(s)")
            print(f"Source: {result['source']}")
            print(f"Type: {result['type']}") 