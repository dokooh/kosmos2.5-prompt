import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from typing import List, Tuple
import shutil

def get_args():
    parser = argparse.ArgumentParser(description='Batch process images with Kosmos-2.5 for markdown and OCR tasks')
    parser.add_argument('--input_folder', '-i', type=str, required=True,
                       help='Path to input folder containing images')
    parser.add_argument('--output_folder', '-o', type=str, required=True,
                       help='Path to output folder for results')
    parser.add_argument('--device', '-d', type=str, default='cuda:0',
                       help='Device to use (default: cuda:0, use cpu for CPU)')
    parser.add_argument('--max_tokens', '-m', type=int, default=1024,
                       help='Maximum number of tokens to generate (default: 1024)')
    parser.add_argument('--skip_md', action='store_true',
                       help='Skip markdown generation')
    parser.add_argument('--skip_ocr', action='store_true',
                       help='Skip OCR processing')
    parser.add_argument('--image_extensions', type=str, nargs='+', 
                       default=['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'],
                       help='Image file extensions to process (default: jpg, jpeg, png, bmp, tiff, webp)')
    return parser.parse_args()

def find_images(input_folder: str, extensions: List[str]) -> List[str]:
    """Find all image files in the input folder"""
    image_files = []
    input_path = Path(input_folder)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    
    for ext in extensions:
        # Case insensitive search
        pattern = f"*{ext.lower()}"
        image_files.extend(input_path.glob(pattern))
        pattern = f"*{ext.upper()}"
        image_files.extend(input_path.glob(pattern))
    
    # Remove duplicates and convert to strings
    image_files = list(set([str(f) for f in image_files]))
    image_files.sort()
    
    return image_files

def create_output_structure(output_folder: str) -> Tuple[str, str, str, str]:
    """Create output folder structure"""
    output_path = Path(output_folder)
    
    # Create main output folder
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subfolders
    markdown_folder = output_path / "markdown"
    ocr_folder = output_path / "ocr"
    ocr_text_folder = output_path / "ocr_text"
    logs_folder = output_path / "logs"
    
    markdown_folder.mkdir(exist_ok=True)
    ocr_folder.mkdir(exist_ok=True)
    ocr_text_folder.mkdir(exist_ok=True)
    logs_folder.mkdir(exist_ok=True)
    
    return str(markdown_folder), str(ocr_folder), str(ocr_text_folder), str(logs_folder)

def run_markdown_task(image_path: str, output_folder: str, device: str, max_tokens: int, logs_folder: str) -> bool:
    """Run markdown generation task"""
    image_name = Path(image_path).stem
    output_file = Path(output_folder) / f"{image_name}_markdown.txt"
    log_file = Path(logs_folder) / f"{image_name}_md.log"
    
    cmd = [
        sys.executable, "md.py",
        "--image", image_path,
        "--device", device,
        "--max_tokens", str(max_tokens)
    ]
    
    try:
        print(f"Processing markdown for: {Path(image_path).name}")
        
        # Run the command and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        
        # Log the output
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Return code: {result.returncode}\n")
            f.write(f"STDOUT:\n{result.stdout}\n")
            f.write(f"STDERR:\n{result.stderr}\n")
        
        if result.returncode == 0:
            # Extract markdown from stdout
            lines = result.stdout.split('\n')
            in_markdown = False
            markdown_content = []
            
            for line in lines:
                if "GENERATED MARKDOWN:" in line:
                    in_markdown = True
                    continue
                elif "=" * 50 in line and in_markdown:
                    break
                elif in_markdown:
                    markdown_content.append(line)
            
            # Save markdown to file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(markdown_content).strip())
            
            print(f"✓ Markdown saved: {output_file}")
            return True
        else:
            print(f"✗ Markdown failed for {Path(image_path).name}")
            return False
            
    except Exception as e:
        print(f"✗ Error processing markdown for {Path(image_path).name}: {e}")
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"Error: {str(e)}\n")
        return False

def run_ocr_task(image_path: str, ocr_folder: str, ocr_text_folder: str, device: str, max_tokens: int, logs_folder: str) -> bool:
    """Run OCR task"""
    image_name = Path(image_path).stem
    output_image = Path(ocr_folder) / f"{image_name}_ocr.png"
    output_text = Path(ocr_text_folder) / f"{image_name}_ocr.txt"
    log_file = Path(logs_folder) / f"{image_name}_ocr.log"
    
    cmd = [
        sys.executable, "ocr.py",
        "--image", image_path,
        "--output", str(output_image),
        "--text_output", str(output_text),
        "--device", device,
        "--max_tokens", str(max_tokens)
    ]
    
    try:
        print(f"Processing OCR for: {Path(image_path).name}")
        
        # Run the command and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        
        # Log the output
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Return code: {result.returncode}\n")
            f.write(f"STDOUT:\n{result.stdout}\n")
            f.write(f"STDERR:\n{result.stderr}\n")
        
        if result.returncode == 0:
            print(f"✓ OCR completed: {output_image}, {output_text}")
            return True
        else:
            print(f"✗ OCR failed for {Path(image_path).name}")
            return False
            
    except Exception as e:
        print(f"✗ Error processing OCR for {Path(image_path).name}: {e}")
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"Error: {str(e)}\n")
        return False

def create_summary_report(output_folder: str, results: dict, total_images: int):
    """Create a summary report of the batch processing"""
    summary_file = Path(output_folder) / "processing_summary.json"
    
    summary = {
        "total_images": total_images,
        "markdown_success": results.get("markdown_success", 0),
        "markdown_failed": results.get("markdown_failed", 0),
        "ocr_success": results.get("ocr_success", 0),
        "ocr_failed": results.get("ocr_failed", 0),
        "success_rate": {
            "markdown": f"{(results.get('markdown_success', 0) / max(total_images, 1)) * 100:.1f}%",
            "ocr": f"{(results.get('ocr_success', 0) / max(total_images, 1)) * 100:.1f}%"
        }
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total images processed: {total_images}")
    print(f"Markdown - Success: {results.get('markdown_success', 0)}, Failed: {results.get('markdown_failed', 0)}")
    print(f"OCR - Success: {results.get('ocr_success', 0)}, Failed: {results.get('ocr_failed', 0)}")
    print(f"Summary saved to: {summary_file}")

def main():
    args = get_args()
    
    try:
        # Find all images in input folder
        print(f"Scanning for images in: {args.input_folder}")
        image_files = find_images(args.input_folder, args.image_extensions)
        
        if not image_files:
            print(f"No images found in {args.input_folder}")
            sys.exit(1)
        
        print(f"Found {len(image_files)} images")
        
        # Create output folder structure
        markdown_folder, ocr_folder, ocr_text_folder, logs_folder = create_output_structure(args.output_folder)
        
        print(f"Output structure created:")
        print(f"  Markdown: {markdown_folder}")
        print(f"  OCR Images: {ocr_folder}")
        print(f"  OCR Text: {ocr_text_folder}")
        print(f"  Logs: {logs_folder}")
        
        # Process each image
        results = {
            "markdown_success": 0,
            "markdown_failed": 0,
            "ocr_success": 0,
            "ocr_failed": 0
        }
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Processing: {Path(image_path).name}")
            
            # Run markdown task
            if not args.skip_md:
                if run_markdown_task(image_path, markdown_folder, args.device, args.max_tokens, logs_folder):
                    results["markdown_success"] += 1
                else:
                    results["markdown_failed"] += 1
            
            # Run OCR task
            if not args.skip_ocr:
                if run_ocr_task(image_path, ocr_folder, ocr_text_folder, args.device, args.max_tokens, logs_folder):
                    results["ocr_success"] += 1
                else:
                    results["ocr_failed"] += 1
        
        # Create summary report
        create_summary_report(args.output_folder, results, len(image_files))
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()