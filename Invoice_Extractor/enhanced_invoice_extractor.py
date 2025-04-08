import os
import sys
import json
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from invoice_extractor import InvoiceExtractor
from pdf_form_extractor import PDFFormExtractor
from ml_extractor import MLExtractor
from extraction_validator import validate_extraction
from currency_utils import detect_and_standardize_currency
from layout_analyzer import LayoutAnalyzer
from post_processor import PostProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('invoice_extraction.log')
    ]
)
logger = logging.getLogger('enhanced_invoice_extractor')

class EnhancedInvoiceExtractor:
    """
    Enhanced invoice extractor that combines multiple extraction methods:
    1. PDF form field extraction
    2. Rule-based extraction
    3. Machine learning based extraction
    4. Layout-aware extraction
    
    It selects the best results based on confidence scores and
    applies post-processing for consistency and accuracy.
    """
    
    def __init__(self, ml_model_path: Optional[str] = None):
        """
        Initialize the enhanced invoice extractor.
        
        Args:
            ml_model_path: Path to the ML model file. If None, a default path is used.
        """
        self.invoice_extractor = InvoiceExtractor()
        self.pdf_form_extractor = PDFFormExtractor()
        
        # Initialize ML extractor with model path
        if not ml_model_path and os.path.exists('ml_model.pkl'):
            ml_model_path = 'ml_model.pkl'
        
        # Check if we have a BERT model
        bert_model_exists = False
        if ml_model_path:
            if os.path.exists(f"{ml_model_path}_manifest.json"):
                bert_model_exists = True
                logger.info(f"Using BERT-based ML extractor with model: {ml_model_path}")
            elif os.path.exists(ml_model_path):
                logger.info(f"Using legacy ML extractor with model: {ml_model_path}")
        
        self.ml_extractor = MLExtractor(model_path=ml_model_path)
        self.using_bert = bert_model_exists
        
        # Initialize post-processor
        self.post_processor = PostProcessor()
        
        # Thresholds for confidence
        self.ml_confidence_threshold = 0.7
        self.rule_based_confidence_threshold = 0.8
        self.form_field_confidence_threshold = 0.9
        
        # For BERT models, use a slightly different threshold
        if bert_model_exists:
            self.ml_confidence_threshold = 0.6
        
        # Store the current invoice path for context
        self.current_invoice_path = None
        
        logger.info("Enhanced invoice extractor initialized")
    
    def extract_all(self, pdf_path: str, validate: bool = True) -> Dict[str, Any]:
        """
        Extract all invoice information using multiple methods and combine the results.
        
        Args:
            pdf_path: Path to the PDF invoice
            validate: Whether to validate the extraction results
            
        Returns:
            Dictionary with extracted invoice information
        """
        self.current_invoice_path = pdf_path
        filename = os.path.basename(pdf_path)
        logger.info(f"Extracting data from: {filename}")
        
        try:
            # Step 1: Extract from form fields (if available)
            form_field_results = self.pdf_form_extractor.extract(pdf_path)
            
            # Step 2: Rule-based extraction
            rule_based_results = self.invoice_extractor.extract_all(pdf_path)
            
            # Step 3: Layout analysis for improved context
            try:
                layout_analyzer = LayoutAnalyzer(pdf_path)
                # Extract with layout context
                rule_based_results = self._enhance_with_layout(rule_based_results, layout_analyzer)
            except Exception as e:
                logger.warning(f"Layout analysis failed: {e}")
            
            # Step 4: ML-based extraction
            try:
                ml_results = self.ml_extractor.extract(pdf_path)
                
                # Log confidence scores for ML extraction
                if self.using_bert and 'confidence' in ml_results:
                    logger.info(f"BERT model confidence scores: {ml_results['confidence']}")
                
            except Exception as e:
                logger.warning(f"ML extraction failed: {e}")
                ml_results = {}
            
            # Step 5: Combine all results
            combined_results = self._combine_extraction_results(
                form_field_results, rule_based_results, ml_results
            )
            
            # Step 6: Standardize currency
            if 'total_amount' in combined_results:
                combined_results['total_amount'] = detect_and_standardize_currency(
                    combined_results['total_amount']
                )
            
            # Step 7: Validate results
            if validate:
                validation_results = validate_extraction(combined_results)
                combined_results.update(validation_results)
            
            # Step 8: Post-process for accuracy and consistency
            combined_results = self.post_processor.process(combined_results, filename)
            
            logger.info(f"Extraction complete for {filename} with confidence: {combined_results.get('overall_confidence', 'N/A')}")
            return combined_results
            
        except Exception as e:
            logger.error(f"Error extracting from {filename}: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    def _enhance_with_layout(self, results: Dict[str, Any], layout_analyzer: LayoutAnalyzer) -> Dict[str, Any]:
        """
        Enhance extraction results with layout analysis information.
        
        Args:
            results: Current extraction results
            layout_analyzer: LayoutAnalyzer instance with page analysis
            
        Returns:
            Enhanced extraction results
        """
        enhanced_results = results.copy()
        
        # Use layout analysis to improve field extraction
        for field in ['invoice_number', 'invoice_date', 'due_date', 'total_amount']:
            field_region = layout_analyzer.find_field_region(field)
            if field_region and field_region.get('text'):
                # Only update if field is empty or has low confidence
                current_confidence = results.get('confidence', {}).get(field, 0)
                if field not in results or not results[field] or current_confidence < 0.7:
                    enhanced_results[field] = field_region['text']
                    if 'confidence' in enhanced_results:
                        enhanced_results['confidence'][field] = field_region.get('confidence', 0.65)
        
        # Improve company name extraction using header region
        header = layout_analyzer.get_header_section()
        if header and 'issuer_name' not in results:
            # Extract potential company name from header
            lines = header.split('\n')
            if lines and len(lines) > 0:
                # First line is often the company name
                enhanced_results['issuer_name'] = lines[0].strip()
                if 'confidence' in enhanced_results:
                    enhanced_results['confidence']['issuer_name'] = 0.6
        
        # Use address blocks for better name extraction
        address_blocks = layout_analyzer.get_address_blocks()
        if address_blocks:
            for block in address_blocks:
                if 'type' in block:
                    if block['type'] == 'from' and ('issuer_name' not in enhanced_results or not enhanced_results['issuer_name']):
                        lines = block['text'].split('\n')
                        if lines and len(lines) > 0:
                            enhanced_results['issuer_name'] = lines[0].strip()
                            if 'confidence' in enhanced_results:
                                enhanced_results['confidence']['issuer_name'] = 0.7
                    
                    elif block['type'] == 'to' and ('recipient_name' not in enhanced_results or not enhanced_results['recipient_name']):
                        lines = block['text'].split('\n')
                        if lines and len(lines) > 0:
                            enhanced_results['recipient_name'] = lines[0].strip()
                            if 'confidence' in enhanced_results:
                                enhanced_results['confidence']['recipient_name'] = 0.7
        
        return enhanced_results
    
    def _combine_extraction_results(self, 
                                   form_field_results: Dict[str, Any], 
                                   rule_based_results: Dict[str, Any], 
                                   ml_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine extraction results from different methods based on confidence.
        
        Args:
            form_field_results: Results from PDF form fields
            rule_based_results: Results from rule-based extraction
            ml_results: Results from ML-based extraction
            
        Returns:
            Combined results with highest confidence values
        """
        combined_results = {}
        combined_confidence = {}
        
        # Fields to extract
        fields = ['invoice_number', 'invoice_date', 'due_date', 'issuer_name', 
                  'recipient_name', 'total_amount']
        
        # Copy confidence scores if available
        rule_based_confidence = rule_based_results.get('confidence', {})
        ml_confidence = ml_results.get('confidence', {})
        
        # BERT models typically have higher quality, so we can give more weight to its confidence
        ml_confidence_boost = 1.2 if self.using_bert else 1.0
        
        # Form fields have highest confidence if available
        for field in fields:
            if field in form_field_results and form_field_results[field]:
                combined_results[field] = form_field_results[field]
                combined_confidence[field] = self.form_field_confidence_threshold
            else:
                # Compare rule-based and ML confidence
                rule_conf = rule_based_confidence.get(field, 0)
                # Apply ML confidence boost for BERT models
                ml_conf = ml_confidence.get(field, 0) * ml_confidence_boost
                
                # Log confidence comparison for debugging
                logger.debug(f"Field {field}: Rule confidence = {rule_conf}, ML confidence = {ml_conf}")
                
                # Choose result with highest confidence
                if ml_conf >= self.ml_confidence_threshold and ml_conf > rule_conf:
                    if field in ml_results and ml_results[field]:
                        combined_results[field] = ml_results[field]
                        combined_confidence[field] = ml_conf
                elif rule_conf >= self.rule_based_confidence_threshold:
                    if field in rule_based_results and rule_based_results[field]:
                        combined_results[field] = rule_based_results[field]
                        combined_confidence[field] = rule_conf
                # Fallback to any available result
                elif field in rule_based_results and rule_based_results[field]:
                    combined_results[field] = rule_based_results[field]
                    combined_confidence[field] = rule_conf
                elif field in ml_results and ml_results[field]:
                    combined_results[field] = ml_results[field]
                    combined_confidence[field] = ml_conf
        
        # Add confidence scores to combined results
        combined_results['confidence'] = combined_confidence
        
        # Calculate overall confidence
        if combined_confidence:
            overall_confidence = sum(combined_confidence.values()) / len(combined_confidence)
            combined_results['overall_confidence'] = overall_confidence
        
        return combined_results
    
    def train_ml_model(self, training_data_path: str, output_model_path: str = 'ml_model.pkl') -> None:
        """
        Train the ML model with labeled data.
        
        Args:
            training_data_path: Path to the JSON file with labeled training data
            output_model_path: Path to save the trained model
        """
        try:
            with open(training_data_path, 'r') as f:
                training_data = json.load(f)
            
            # Train the model
            self.ml_extractor.train(training_data, output_model_path)
            logger.info(f"ML model trained successfully with {len(training_data)} samples")
            
            # Update model path
            self.ml_extractor.load_model(output_model_path)
            self.using_bert = True
            
        except Exception as e:
            logger.error(f"Error training ML model: {str(e)}", exc_info=True)
            raise
    
    def get_extraction_method(self, field: str, combined_results: Dict[str, Any]) -> str:
        """
        Determine which extraction method was used for a specific field.
        
        Args:
            field: The field name
            combined_results: The combined extraction results
            
        Returns:
            The name of the extraction method used
        """
        confidence = combined_results.get('confidence', {}).get(field, 0)
        
        if confidence >= self.form_field_confidence_threshold:
            return "form_field"
        elif confidence >= self.ml_confidence_threshold and self.using_bert:
            return "bert_ml"
        elif confidence >= self.ml_confidence_threshold:
            return "machine_learning"
        else:
            return "rule_based"


def process_directory(input_dir: str, output_file: str = 'extracted_invoices.json', 
                     ml_model_path: Optional[str] = None) -> None:
    """
    Process all invoice files in a directory and save results to a JSON file.
    
    Args:
        input_dir: Directory containing invoice files
        output_file: Path to save the JSON output
        ml_model_path: Path to the ML model file
    """
    extractor = EnhancedInvoiceExtractor(ml_model_path=ml_model_path)
    results = []
    
    # Process each PDF file
    start_time = datetime.now()
    logger.info(f"Starting batch processing of directory: {input_dir}")
    
    # Count files to process
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    total_files = len(pdf_files)
    logger.info(f"Found {total_files} PDF files to process")
    
    processed_count = 0
    error_count = 0
    
    for filename in pdf_files:
        file_path = os.path.join(input_dir, filename)
        logger.info(f"Processing file {processed_count+1}/{total_files}: {filename}")
        
        try:
            # Extract data from the invoice
            extraction_result = extractor.extract_all(file_path)
            
            # Add file info
            result = {
                'filename': filename,
                'data': extraction_result
            }
            results.append(result)
            processed_count += 1
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}", exc_info=True)
            results.append({
                'filename': filename,
                'error': str(e)
            })
            error_count += 1
    
    # Calculate statistics
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    avg_time = duration / total_files if total_files > 0 else 0
    
    stats = {
        'total_files': total_files,
        'processed_successfully': processed_count,
        'errors': error_count,
        'processing_time_seconds': duration,
        'average_time_per_file_seconds': avg_time,
        'timestamp': end_time.isoformat()
    }
    
    # Add statistics to results
    output_data = {
        'results': results,
        'stats': stats
    }
    
    # Save results to JSON file
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Processing complete. Results saved to {output_file}")
    logger.info(f"Processed {processed_count} files with {error_count} errors in {duration:.2f} seconds")
    
    # Apply post-processing to results
    try:
        from post_processor import batch_process
        processed_results = batch_process(results)
        
        # Save processed results
        processed_output_file = os.path.splitext(output_file)[0] + "_processed.json"
        with open(processed_output_file, 'w') as f:
            json.dump({
                'results': processed_results,
                'stats': stats
            }, f, indent=2)
        
        logger.info(f"Post-processing complete. Processed results saved to {processed_output_file}")
    except Exception as e:
        logger.error(f"Error during post-processing: {str(e)}", exc_info=True)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process invoices using the Enhanced Invoice Extractor')
    parser.add_argument('input', help='Input PDF file or directory containing PDF files')
    parser.add_argument('--output', '-o', help='Output JSON file path', default='extracted_invoices.json')
    parser.add_argument('--model', '-m', help='ML model file path', default=None)
    parser.add_argument('--train', '-t', help='Training data JSON file for ML model', default=None)
    parser.add_argument('--validate', '-v', action='store_true', help='Validate extraction results')
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = EnhancedInvoiceExtractor(ml_model_path=args.model)
    
    # Train model if training data provided
    if args.train:
        print(f"Training ML model with data from {args.train}")
        extractor.train_ml_model(args.train, output_model_path=args.model or 'ml_model.pkl')
    
    # Process input
    if os.path.isdir(args.input):
        # Process directory
        process_directory(args.input, args.output, ml_model_path=args.model)
    elif os.path.isfile(args.input) and args.input.lower().endswith('.pdf'):
        # Process single file
        try:
            results = extractor.extract_all(args.input, validate=args.validate)
            
            # Save results
            with open(args.output, 'w') as f:
                json.dump({
                    'results': [{
                        'filename': os.path.basename(args.input),
                        'data': results
                    }]
                }, f, indent=2)
                
            print(f"Extraction complete. Results saved to {args.output}")
            
            # Print summary
            print("\nExtraction Summary:")
            for field, value in results.items():
                if field != 'confidence':
                    confidence = results.get('confidence', {}).get(field, 'N/A')
                    method = extractor.get_extraction_method(field, results) if field in results.get('confidence', {}) else 'N/A'
                    print(f"{field}: {value} (Confidence: {confidence}, Method: {method})")
                    
        except Exception as e:
            print(f"Error processing {args.input}: {str(e)}")
    else:
        print("Invalid input. Please provide a PDF file or directory containing PDF files.") 