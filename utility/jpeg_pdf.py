import os
import zipfile
import tempfile
from pdf2image import convert_from_path
from PIL import Image

def convert_pdfs_to_jpegs(zip_path, output_folder):
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Create a temporary directory for extracted files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Process each file in the extracted directory and its subdirectories
        for root, dirs, files in os.walk(temp_dir):
            for filename in files:
                if filename.lower().endswith('.pdf'):
                    # Get the relative path from temp_dir to maintain structure
                    rel_path = os.path.relpath(root, temp_dir)
                    # Create corresponding output directory
                    output_subdir = os.path.join(output_folder, rel_path)
                    os.makedirs(output_subdir, exist_ok=True)
                    
                    pdf_path = os.path.join(root, filename)
                    try:
                        # Convert PDF to images (returns a list of images, one per page)
                        images = convert_from_path(pdf_path)
                        
                        # For each page in the PDF
                        for i, image in enumerate(images):
                            # Create JPEG filename (add page number if PDF has multiple pages)
                            if len(images) > 1:
                                jpeg_filename = f"{os.path.splitext(filename)[0]}_page{i+1}.jpg"
                            else:
                                jpeg_filename = f"{os.path.splitext(filename)[0]}.jpg"
                            
                            jpeg_path = os.path.join(output_subdir, jpeg_filename)
                            # Save as JPEG
                            image.save(jpeg_path, 'JPEG')
                            print(f"Converted {os.path.join(rel_path, filename)} page {i+1} to JPEG")
                    except Exception as e:
                        print(f"Error processing {os.path.join(rel_path, filename)}: {str(e)}")

if __name__ == "__main__":
    # Get the user's home directory
    home_dir = os.path.expanduser("~")
    
    # Set the paths
    zip_path = os.path.join(home_dir, "Downloads", "company documents dataset.zip")
    output_folder = "archive_jpegs"  # Changed output folder name
    
    # Check if the zip file exists
    if not os.path.exists(zip_path):
        print(f"Error: Could not find {zip_path}")
        print("Please make sure the file 'company documents dataset.zip' exists in your Downloads folder")
    else:
        print(f"Processing zip file: {zip_path}")
        print(f"JPEGs will be saved to: {os.path.abspath(output_folder)}")
        convert_pdfs_to_jpegs(zip_path, output_folder)
