import re
import torch
import requests
import argparse
import sys
import os
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Kosmos2_5ForConditionalGeneration, infer_device

def get_args():
    parser = argparse.ArgumentParser(description='Perform OCR on image using Kosmos-2.5')
    parser.add_argument('--image', '-i', type=str, required=True, 
                       help='Path to input image file or URL')
    parser.add_argument('--output', '-o', type=str, default='./output.png',
                       help='Output path for the annotated image (default: ./output.png)')
    parser.add_argument('--text_output', '-t', type=str, default=None,
                       help='Output path for the OCR text results (optional)')
    parser.add_argument('--device', '-d', type=str, default='cuda:0', 
                       help='Device to use (default: cuda:0, use cpu for CPU)')
    parser.add_argument('--max_tokens', '-m', type=int, default=1024,
                       help='Maximum number of tokens to generate (default: 1024)')
    parser.add_argument('--no_bbox', action='store_true',
                       help='Skip drawing bounding boxes on the output image')
    return parser.parse_args()

def load_image(image_path):
    """Load image from local path or URL"""
    try:
        if image_path.startswith(('http://', 'https://')):
            # Load from URL
            print(f"Loading image from URL: {image_path}")
            response = requests.get(image_path, stream=True)
            response.raise_for_status()
            image = Image.open(response.raw)
        else:
            # Load from local file
            print(f"Loading image from file: {image_path}")
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            image = Image.open(image_path)
        
        return image.convert('RGB')  # Ensure RGB format
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

def post_process(y, scale_height, scale_width, prompt="<ocr>"):
    y = y.replace(prompt, "")
    if "<md>" in prompt:
        return y
    pattern = r"<bbox><x_\d+><y_\d+><x_\d+><y_\d+></bbox>"
    bboxs_raw = re.findall(pattern, y)
    lines = re.split(pattern, y)[1:]
    bboxs = [re.findall(r"\d+", i) for i in bboxs_raw]
    bboxs = [[int(j) for j in i] for i in bboxs]
    info = ""
    for i in range(len(lines)):
        if i < len(bboxs):
            box = bboxs[i]
            x0, y0, x1, y1 = box
            if not (x0 >= x1 or y0 >= y1):
                x0 = int(x0 * scale_width)
                y0 = int(y0 * scale_height)
                x1 = int(x1 * scale_width)
                y1 = int(y1 * scale_height)
                info += f"{x0},{y0},{x1},{y0},{x1},{y1},{x0},{y1},{lines[i]}\n"
    return info

def draw_bounding_boxes(image, output_text):
    """Draw bounding boxes on the image"""
    draw = ImageDraw.Draw(image)
    lines = output_text.strip().split("\n")
    
    for line in lines:
        # Draw the bounding box
        parts = line.split(",")
        if len(parts) < 8:
            continue
        try:
            coords = list(map(int, parts[:8]))
            draw.polygon(coords, outline="red", width=2)
        except ValueError:
            # Skip lines that don't have valid coordinates
            continue
    
    return image

def save_text_output(text, output_path):
    """Save OCR text results to file"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"OCR text saved to: {output_path}")
    except Exception as e:
        print(f"Error saving text output: {e}")

def main():
    args = get_args()
    
    # Setup device and dtype
    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        device = "cpu"
        dtype = torch.float32
    elif device == "cpu":
        dtype = torch.float32
    else:
        dtype = torch.bfloat16
    
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")
    
    # Load model and processor
    repo = "microsoft/kosmos-2.5"
    try:
        print("Loading model...")
        model = Kosmos2_5ForConditionalGeneration.from_pretrained(
            repo, 
            device_map=device, 
            torch_dtype=dtype
        )
        processor = AutoProcessor.from_pretrained(repo)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Load image
    image = load_image(args.image)
    print(f"Image loaded successfully. Size: {image.size}")
    
    # Process image
    prompt = "<ocr>"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    
    height, width = inputs.pop("height"), inputs.pop("width")
    raw_width, raw_height = image.size
    scale_height = raw_height / height
    scale_width = raw_width / width
    
    # Move inputs to device
    inputs = {k: v.to(device) if v is not None else None for k, v in inputs.items()}
    if "flattened_patches" in inputs and inputs["flattened_patches"] is not None:
        inputs["flattened_patches"] = inputs["flattened_patches"].to(dtype)
    
    # Generate OCR results
    print("Performing OCR...")
    try:
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
            )
        
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        output_text = post_process(generated_text[0], scale_height, scale_width, prompt)
        
        print("\n" + "="*50)
        print("OCR RESULTS:")
        print("="*50)
        print(output_text)
        print("="*50)
        
        # Save text output if specified
        if args.text_output:
            save_text_output(output_text, args.text_output)
        
        # Create output image with bounding boxes (unless disabled)
        if not args.no_bbox:
            # Create a copy of the original image for annotation
            annotated_image = image.copy()
            annotated_image = draw_bounding_boxes(annotated_image, output_text)
            
            # Ensure output directory exists
            output_dir = os.path.dirname(args.output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Save the annotated image
            annotated_image.save(args.output)
            print(f"Annotated image saved to: {args.output}")
        else:
            print("Bounding box drawing skipped (--no_bbox flag used)")
        
    except Exception as e:
        print(f"Error during OCR generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()