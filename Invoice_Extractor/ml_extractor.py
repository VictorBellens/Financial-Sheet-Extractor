import os
import re
import json
import pickle
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import torch
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from dataclasses import dataclass
import pandas as pd
import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ml_extractor')

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

@dataclass
class FieldCandidate:
    """
    A candidate for a field value with its context and confidence.
    """
    field_name: str  # The field name (e.g., 'invoice_number')
    value: str  # The candidate value
    context: str  # Surrounding text context
    confidence: float  # Confidence score between 0 and 1
    position: Tuple[int, int] = None  # Optional position in document (start, end)
    page: int = 0  # Page number (for multi-page documents)


class InvoiceDataset(Dataset):
    """
    PyTorch dataset for BERT fine-tuning on invoice fields.
    """
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Convert dict of tensors to tensors and remove the batch dimension
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class BERTFieldExtractor:
    """
    Field extractor using BERT for field classification and extraction.
    """
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
        self.model.to(device)
        self.field_tokens = {}  # Field-specific tokens
        self.confidence_threshold = 0.7
        
    def train(self, train_texts, train_labels, val_texts=None, val_labels=None, 
              batch_size=8, epochs=3, learning_rate=2e-5, max_length=512):
        """
        Fine-tune BERT on invoice field extraction.
        """
        # Create datasets
        train_dataset = InvoiceDataset(train_texts, train_labels, self.tokenizer, max_length)
        
        # Create data loaders
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
        
        # Validation set
        if val_texts is not None and val_labels is not None:
            val_dataset = InvoiceDataset(val_texts, val_labels, self.tokenizer, max_length)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_dataloader = None
        
        # Prepare optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Training loop
        logger.info(f"Starting BERT fine-tuning for {epochs} epochs")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            total_loss = 0
            
            for batch in tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
            avg_train_loss = total_loss / len(train_dataloader)
            logger.info(f"Average training loss: {avg_train_loss:.4f}")
            
            # Validation
            if val_dataloader:
                self.model.eval()
                val_loss = 0
                predictions = []
                true_labels = []
                
                with torch.no_grad():
                    for batch in val_dataloader:
                        batch = {k: v.to(device) for k, v in batch.items()}
                        outputs = self.model(**batch)
                        val_loss += outputs.loss.item()
                        
                        logits = outputs.logits
                        preds = torch.argmax(logits, dim=1).cpu().numpy()
                        predictions.extend(preds)
                        true_labels.extend(batch['labels'].cpu().numpy())
                
                avg_val_loss = val_loss / len(val_dataloader)
                accuracy = accuracy_score(true_labels, predictions)
                f1 = f1_score(true_labels, predictions, average='weighted')
                
                logger.info(f"Validation loss: {avg_val_loss:.4f}")
                logger.info(f"Validation accuracy: {accuracy:.4f}")
                logger.info(f"Validation F1: {f1:.4f}")
        
        logger.info("BERT fine-tuning complete")
        
    def save(self, path):
        """Save the model and tokenizer"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_path = f"{path}_model"
        tokenizer_path = f"{path}_tokenizer"
        config_path = f"{path}_config.json"
        
        # Save model and tokenizer
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(tokenizer_path)
        
        # Save additional config
        config = {
            'field_tokens': self.field_tokens,
            'confidence_threshold': self.confidence_threshold
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f)
            
        logger.info(f"Saved model and config to {path}")
        
    @classmethod
    def load(cls, path):
        """Load a saved model"""
        model_path = f"{path}_model"
        tokenizer_path = f"{path}_tokenizer"
        config_path = f"{path}_config.json"
        
        instance = cls.__new__(cls)
        
        # Load tokenizer and model
        instance.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        instance.model = BertForSequenceClassification.from_pretrained(model_path)
        instance.model.to(device)
        
        # Load config
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            instance.field_tokens = config.get('field_tokens', {})
            instance.confidence_threshold = config.get('confidence_threshold', 0.7)
        else:
            instance.field_tokens = {}
            instance.confidence_threshold = 0.7
            
        return instance
    
    def predict(self, texts, field_name):
        """
        Predict field extraction for multiple text segments.
        """
        # Create dataset
        labels = [0] * len(texts)  # Dummy labels
        dataset = InvoiceDataset(texts, labels, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
        
        # Prediction
        self.model.eval()
        predictions = []
        confidences = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                outputs = self.model(**batch)
                
                # Get predictions
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1).cpu().numpy()
                confidence = probs[:, 1].cpu().numpy()  # Confidence of positive class
                
                predictions.extend(preds)
                confidences.extend(confidence)
        
        # Create field candidates
        candidates = []
        for i, (text, pred, conf) in enumerate(zip(texts, predictions, confidences)):
            if pred == 1:  # Positive class
                # Extract the actual value from text
                value = self._extract_value_from_text(text, field_name)
                
                candidates.append(FieldCandidate(
                    field_name=field_name,
                    value=value,
                    context=text,
                    confidence=float(conf)
                ))
        
        # Sort by confidence
        candidates = sorted(candidates, key=lambda x: x.confidence, reverse=True)
        
        return candidates
    
    def _extract_value_from_text(self, text, field_name):
        """
        Extract actual field value from identified text segment.
        Uses field-specific patterns to extract values.
        """
        # Different extraction patterns for different field types
        patterns = {
            'invoice_number': [
                r"invoice\s*(?:#|number|no)?\s*[:.]?\s*([A-Za-z0-9-]+)",
                r"invoice\s*[:.]?\s*([A-Za-z0-9-]+)",
                r"#\s*(\d+)"
            ],
            'invoice_date': [
                r"(?:invoice|date)\s*(?:date|of)?\s*[:.]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
                r"(?:invoice|date)\s*(?:date|of)?\s*[:.]?\s*([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})"
            ],
            'due_date': [
                r"(?:due|payment)\s*(?:date|by)?\s*[:.]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
                r"(?:due|payment)\s*(?:date|by)?\s*[:.]?\s*([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})"
            ],
            'total_amount': [
                r"(?:total|amount|balance)\s*(?:due|:)?\s*[\$€£]?\s*([\d,]+\.\d{2})",
                r"[\$€£]\s*([\d,]+\.\d{2})"
            ]
        }
        
        # Default to capture everything
        default_pattern = r"(?:^|\s)([A-Za-z0-9-]+)(?:\s|$)"
        
        # Try field-specific patterns first
        field_patterns = patterns.get(field_name, [default_pattern])
        
        for pattern in field_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # If no pattern matched, return cleaned text
        # Remove common headers and keep only the potential value
        cleaned_text = re.sub(r'invoice|number|date|due|total|amount|:|#', '', text, flags=re.IGNORECASE)
        return cleaned_text.strip()


class MLExtractor:
    """
    ML-based invoice extractor using BERT for field identification and extraction.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the ML extractor with an optional model path.
        """
        self.field_extractors = {}
        self.confidence_threshold = 0.7
        self.context_size = 100  # Characters before and after for context window
        
        if model_path and os.path.exists(f"{model_path}_model"):
            self.load_model(model_path)
        else:
            logger.info("No model loaded. Use train() to train models for field extraction.")
    
    def train(self, training_data: List[Dict[str, Any]], output_path: str) -> None:
        """
        Train ML models for field extraction.
        
        Args:
            training_data: List of training examples with text content and labeled fields
            output_path: Path to save the trained model
        """
        logger.info(f"Training ML models with {len(training_data)} examples")
        
        # Prepare training data for each field
        fields = ['invoice_number', 'invoice_date', 'due_date', 'issuer_name', 
                  'recipient_name', 'total_amount']
        
        # Map of field name to its training data
        field_data = {field: {'texts': [], 'labels': []} for field in fields}
        
        for example in training_data:
            text = example.get('text', '')
            if not text:
                continue
                
            # Get labeled fields
            labeled_fields = example.get('data', {})
            
            # Generate positive and negative examples for each field
            for field in fields:
                field_value = labeled_fields.get(field)
                if not field_value:
                    continue
                    
                # Find the field value in the text
                value_pattern = re.escape(str(field_value))
                matches = re.finditer(value_pattern, text, re.IGNORECASE)
                
                for match in matches:
                    start, end = match.span()
                    
                    # Extract context around the match
                    context_start = max(0, start - self.context_size)
                    context_end = min(len(text), end + self.context_size)
                    context = text[context_start:context_end]
                    
                    # Add positive example
                    field_data[field]['texts'].append(context)
                    field_data[field]['labels'].append(1)
                    
                    # Generate negative examples from other parts of the text
                    neg_indices = []
                    for i in range(3):  # Generate 3 negative examples
                        neg_start = (start + (i+1) * len(text) // 5) % len(text)
                        neg_end = min(neg_start + (end - start), len(text))
                        
                        if neg_start >= context_start and neg_start <= context_end:
                            continue  # Skip if overlaps with positive example
                            
                        neg_context_start = max(0, neg_start - self.context_size // 2)
                        neg_context_end = min(len(text), neg_end + self.context_size // 2)
                        
                        field_data[field]['texts'].append(text[neg_context_start:neg_context_end])
                        field_data[field]['labels'].append(0)
        
        # Train models for each field
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        for field in fields:
            texts = field_data[field]['texts']
            labels = field_data[field]['labels']
            
            if len(texts) < 10:  # Skip if not enough examples
                logger.warning(f"Not enough examples for {field}. Skipping.")
                continue
                
            logger.info(f"Training model for {field} with {len(texts)} examples")
            
            # Split data
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            # Create and train extractor
            extractor = BERTFieldExtractor()
            extractor.train(
                train_texts, 
                train_labels,
                val_texts, 
                val_labels,
                batch_size=8,
                epochs=3
            )
            
            # Save field-specific model
            field_path = f"{output_path}_{field}"
            extractor.save(field_path)
            
            # Add to field extractors
            self.field_extractors[field] = extractor
            
        # Save a manifest
        manifest = {
            'fields': list(self.field_extractors.keys()),
            'confidence_threshold': self.confidence_threshold,
            'context_size': self.context_size
        }
        
        with open(f"{output_path}_manifest.json", 'w') as f:
            json.dump(manifest, f)
            
        logger.info(f"ML models trained and saved to {output_path}")
    
    def load_model(self, model_path: str) -> None:
        """
        Load trained ML models for field extraction.
        
        Args:
            model_path: Path to the trained model files
        """
        # Load manifest
        manifest_path = f"{model_path}_manifest.json"
        
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
                
            fields = manifest.get('fields', [])
            self.confidence_threshold = manifest.get('confidence_threshold', 0.7)
            self.context_size = manifest.get('context_size', 100)
        else:
            # Try to infer fields from available model files
            fields = []
            for field in ['invoice_number', 'invoice_date', 'due_date', 'issuer_name', 
                         'recipient_name', 'total_amount']:
                if os.path.exists(f"{model_path}_{field}_model"):
                    fields.append(field)
        
        # Load each field extractor
        for field in fields:
            field_path = f"{model_path}_{field}"
            try:
                self.field_extractors[field] = BERTFieldExtractor.load(field_path)
                logger.info(f"Loaded model for {field}")
            except Exception as e:
                logger.error(f"Error loading model for {field}: {e}")
        
        logger.info(f"Loaded ML models for {len(self.field_extractors)} fields")
    
    def extract(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract invoice information from a PDF using ML models.
        
        Args:
            pdf_path: Path to the PDF invoice
            
        Returns:
            Dictionary with extracted invoice information and confidence scores
        """
        # First extract text from PDF
        text = self._extract_text(pdf_path)
        
        if not text:
            logger.warning(f"No text extracted from {pdf_path}")
            return {}
            
        # Extract fields
        extracted_fields = {}
        confidence_scores = {}
        
        for field, extractor in self.field_extractors.items():
            # Generate candidate texts
            candidate_texts = self._generate_candidate_texts(text, field)
            
            # Get field candidates
            candidates = extractor.predict(candidate_texts, field)
            
            # Select best candidate
            if candidates:
                best_candidate = candidates[0]  # Already sorted by confidence
                
                # Only use if confidence is above threshold
                if best_candidate.confidence >= self.confidence_threshold:
                    extracted_fields[field] = best_candidate.value
                    confidence_scores[field] = best_candidate.confidence
        
        # Calculate overall confidence
        if confidence_scores:
            overall_confidence = sum(confidence_scores.values()) / len(confidence_scores)
        else:
            overall_confidence = 0.0
            
        # Combine results
        result = {
            **extracted_fields,
            'confidence': confidence_scores,
            'overall_confidence': overall_confidence
        }
        
        return result
    
    def _extract_text(self, pdf_path: str) -> str:
        """
        Extract text from PDF file.
        """
        try:
            import pdfplumber
            
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                        
                return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ""
    
    def _generate_candidate_texts(self, text: str, field: str) -> List[str]:
        """
        Generate candidate text segments for field extraction.
        
        Args:
            text: Full document text
            field: Field name to extract
            
        Returns:
            List of text segments to evaluate
        """
        candidates = []
        
        # Split into lines
        lines = text.split('\n')
        
        # Field-specific patterns to help identify relevant sections
        field_patterns = {
            'invoice_number': ['invoice', 'number', '#', 'no'],
            'invoice_date': ['invoice', 'date', 'issued'],
            'due_date': ['due', 'payment', 'pay by'],
            'issuer_name': ['from', 'seller', 'company'],
            'recipient_name': ['to', 'bill to', 'ship to', 'customer'],
            'total_amount': ['total', 'amount', 'balance', 'due', 'pay']
        }
        
        patterns = field_patterns.get(field, [])
        
        # Add individual lines
        for i, line in enumerate(lines):
            # Skip empty lines
            if not line.strip():
                continue
                
            # Check if line contains field-related keywords
            if any(pattern in line.lower() for pattern in patterns):
                # Add the line with context
                start_idx = max(0, i - 2)
                end_idx = min(len(lines), i + 3)
                context = '\n'.join(lines[start_idx:end_idx])
                candidates.append(context)
                
                # Also add just the line itself
                candidates.append(line)
        
        # Generate sliding windows over the text
        window_size = 200
        step_size = 100
        
        for i in range(0, len(text) - window_size + 1, step_size):
            window = text[i:i+window_size]
            candidates.append(window)
        
        # Remove duplicates while preserving order
        unique_candidates = []
        seen = set()
        
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                unique_candidates.append(candidate)
        
        return unique_candidates


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train or use ML invoice extractor')
    parser.add_argument('--train', action='store_true', help='Train ML models')
    parser.add_argument('--extract', action='store_true', help='Extract data from invoice')
    parser.add_argument('--data', help='Path to training data JSON file')
    parser.add_argument('--model', help='Path to save/load ML model')
    parser.add_argument('--invoice', help='Path to invoice PDF for extraction')
    
    args = parser.parse_args()
    
    if args.train and args.data and args.model:
        # Load training data
        with open(args.data, 'r') as f:
            training_data = json.load(f)
            
        # Train models
        extractor = MLExtractor()
        extractor.train(training_data, args.model)
        
    elif args.extract and args.invoice and args.model:
        # Load model and extract data
        extractor = MLExtractor(args.model)
        result = extractor.extract(args.invoice)
        
        print(json.dumps(result, indent=2))
    else:
        parser.print_help() 