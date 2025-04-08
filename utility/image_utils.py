import os
import cv2
import numpy as np
from PIL import Image
import tempfile
import uuid

def convert_to_jpeg(input_path, output_dir=None):
    """
    Converts an image to JPEG format.
    
    Args:
        input_path: Path to the input image
        output_dir: Directory to save the output. If None, a temp file is created.
        
    Returns:
        Path to the converted JPEG image
    """
    try:
        # Check if input file exists
        if not os.path.exists(input_path):
            print(f"Error: Input file {input_path} does not exist")
            return None
            
        # Determine output path
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            filename = f"{uuid.uuid4().hex}.jpg"
            output_path = os.path.join(output_dir, filename)
        else:
            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            output_path = temp_file.name
            temp_file.close()
        
        # Open and convert image
        image = Image.open(input_path)
        
        # If image is PNG with transparency, convert to white background
        if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
            # Create a white background
            bg = Image.new('RGB', image.size, (255, 255, 255))
            # Paste the image on the background
            if image.mode == 'RGBA':
                bg.paste(image, mask=image.split()[3])
            else:
                bg.paste(image, mask=image.convert('RGBA').split()[3])
            # Save as JPEG
            bg.save(output_path, 'JPEG', quality=95)
        else:
            # Convert to RGB if not already (e.g., CMYK)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            # Save as JPEG
            image.save(output_path, 'JPEG', quality=95)
            
        print(f"Converted {input_path} to JPEG: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error converting image to JPEG: {e}")
        return None

def enhance_image_for_ocr(image_path, output_dir=None):
    """
    Enhances an image for better OCR results.
    
    Args:
        image_path: Path to the input image (should be a JPEG or compatible format)
        output_dir: Directory to save the enhanced image. If None, a temp file is created.
        
    Returns:
        Path to the enhanced image
    """
    try:
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image {image_path}")
            return None
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising (Non-Local Means) with enhanced parameters
        denoised = cv2.fastNlMeansDenoising(gray, h=30, templateWindowSize=7, searchWindowSize=21)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Apply adaptive thresholding for better text extraction
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological closing to bridge small gaps in text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        morph = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        
        # Create processed image variants
        processed_images = {
            "gray": gray,
            "denoised": denoised,
            "enhanced": enhanced,
            "thresh": thresh,
            "morph": morph
        }
        
        # Determine output path
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            filename = f"enhanced_{uuid.uuid4().hex}.jpg"
            output_path = os.path.join(output_dir, filename)
        else:
            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            output_path = temp_file.name
            temp_file.close()
        
        # Save the enhanced image (using the thresholded version for best OCR results)
        cv2.imwrite(output_path, thresh)
        
        print(f"Enhanced image saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error enhancing image: {e}")
        return None

def preprocess_image_pipeline(input_path, temp_dir=None):
    """
    Complete preprocessing pipeline: Convert to JPEG and enhance for OCR.
    
    Args:
        input_path: Path to the input image
        temp_dir: Directory to save temporary files. If None, system temp dir is used.
        
    Returns:
        Tuple of (original_jpeg_path, enhanced_image_path)
    """
    # Convert to JPEG first
    jpeg_path = convert_to_jpeg(input_path, temp_dir)
    if not jpeg_path:
        return None, None
        
    # Enhance the JPEG image
    enhanced_path = enhance_image_for_ocr(jpeg_path, temp_dir)
    
    return jpeg_path, enhanced_path 