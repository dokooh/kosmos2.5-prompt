import os
import sys
import io  # Missing import that was needed
from pathlib import Path
import argparse
from PIL import Image

# Fix the PyMuPDF import issue
try:
    import fitz  # Try importing fitz first
    # Check if this is actually PyMuPDF by testing for a PyMuPDF-specific attribute
    if not hasattr(fitz, 'open'):
        raise ImportError("Wrong fitz package")
except (ImportError, AttributeError):
    # If fitz import fails or it's the wrong package, try PyMuPDF directly
    try:
        import pymupdf as fitz
    except ImportError:
        print("Error: PyMuPDF is not installed.")
        print("Please install it with: pip install PyMuPDF")
        sys.exit(1)

def create_output_folder(pdf_path, output_dir=None):
    """Create output folder based on PDF name or use specified directory."""
    pdf_name = Path(pdf_path).stem
    
    if output_dir:
        folder_path = Path(output_dir) / f"{pdf_name}_pages"
    else:
        folder_path = Path(pdf_path).parent / f"{pdf_name}_pages"
    
    # Create folder if it doesn't exist
    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path

def pdf_to_images(pdf_path, output_dir=None, dpi=150, image_format='PNG'):
    """
    Convert PDF pages to images.
    
    Args:
        pdf_path (str): Path to the PDF file
        output_dir (str): Output directory (optional)
        dpi (int): Resolution for the images (default: 150)
        image_format (str): Image format (default: PNG)
    """
    try:
        # Check if PDF file exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Create output folder
        output_folder = create_output_folder(pdf_path, output_dir)
        print(f"Output folder: {output_folder}")
        
        # Open the PDF
        pdf_document = fitz.open(pdf_path)
        total_pages = len(pdf_document)
        
        print(f"Processing {total_pages} pages from: {Path(pdf_path).name}")
        
        # Convert each page to image
        for page_num in range(total_pages):
            # Get the page
            page = pdf_document[page_num]
            
            # Create a matrix for the desired DPI
            mat = fitz.Matrix(dpi/72, dpi/72)  # 72 is the default DPI
            
            # Render page to an image
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("ppm")
            img = Image.open(io.BytesIO(img_data))
            
            # Save the image
            image_filename = f"page_{page_num + 1:03d}.png"
            image_path = output_folder / image_filename
            
            img.save(image_path, image_format)
            print(f"Saved: {image_filename}")
        
        # Close the PDF
        pdf_document.close()
        
        print(f"\nCompleted! All {total_pages} pages saved to: {output_folder}")
        return output_folder
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return None

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Convert PDF pages to PNG images')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('-o', '--output', help='Output directory (optional)')
    parser.add_argument('-d', '--dpi', type=int, default=150, 
                       help='DPI for images (default: 150)')
    parser.add_argument('-f', '--format', default='PNG', 
                       choices=['PNG', 'JPEG', 'TIFF'],
                       help='Image format (default: PNG)')
    
    args = parser.parse_args()
    
    # Convert PDF to images
    result = pdf_to_images(
        pdf_path=args.pdf_path,
        output_dir=args.output,
        dpi=args.dpi,
        image_format=args.format
    )
    
    if result:
        print(f"\nSuccess! Images saved in: {result}")
    else:
        print("Failed to process PDF")
        sys.exit(1)

if __name__ == "__main__":
    main()