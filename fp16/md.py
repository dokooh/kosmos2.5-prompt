import re
import torch
import requests
import argparse
import sys
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Kosmos2_5ForConditionalGeneration, infer_device

def get_args():
    parser = argparse.ArgumentParser(description='Generate markdown from image using Kosmos-2.5')
    parser.add_argument('--image', '-i', type=str, required=True, 
                       help='Path to input image file or URL')
    parser.add_argument('--device', '-d', type=str, default='cuda:0', 
                       help='Device to use (default: cuda:0, use cpu for CPU)')
    parser.add_argument('--max_tokens', '-m', type=int, default=1024,
                       help='Maximum number of tokens to generate (default: 1024)')
    return parser.parse_args()

def load_image(image_path):
    """Load image from local path or URL"""
    try:
        if image_path.startswith(('http://', 'https://')):
            # Load from URL
            response = requests.get(image_path, stream=True)
            response.raise_for_status()
            image = Image.open(response.raw)
        else:
            # Load from local file
            image = Image.open(image_path)
        
        return image.convert('RGB')  # Ensure RGB format
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

def main():
    args = get_args()
    
    # Setup device and dtype
    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        device = "cpu"
        dtype = torch.float32
    else:
        dtype = torch.bfloat16 if device.startswith('cuda') else torch.float32
    
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
    print(f"Loading image from: {args.image}")
    image = load_image(args.image)
    print(f"Image loaded successfully. Size: {image.size}")
    
    # Process image
    prompt = "<md>"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    
    height, width = inputs.pop("height"), inputs.pop("width")
    raw_width, raw_height = image.size
    scale_height = raw_height / height
    scale_width = raw_width / width
    
    # Move inputs to device
    inputs = {k: v.to(device) if v is not None else None for k, v in inputs.items()}
    if "flattened_patches" in inputs and inputs["flattened_patches"] is not None:
        inputs["flattened_patches"] = inputs["flattened_patches"].to(dtype)
    
    # Generate markdown
    print("Generating markdown...")
    try:
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
            )
        
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Extract the generated markdown (remove the prompt)
        result = generated_text[0].replace(prompt, "").strip()
        
        print("\n" + "="*50)
        print("GENERATED MARKDOWN:")
        print("="*50)
        print(result)
        print("="*50)
        
    except Exception as e:
        print(f"Error during generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()