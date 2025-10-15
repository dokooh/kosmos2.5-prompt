#!/usr/bin/env python3
"""
Optimized OCR Inference Module for Kosmos-2.5 with FP16 Quantization

This module provides fast OCR inference using FP16 quantized Kosmos-2.5 model.
Features:
- FP16 quantization for faster inference
- SafeTensors format for faster loading
- Support for local SafeTensor model paths
- Optimized memory usage
- Batch processing support
- Enhanced error handling
"""

import re
import torch
import requests
import argparse
import sys
import os
import time
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForImageTextToText
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedOCRInference:
    def __init__(self, model_name="microsoft/kosmos-2.5", device=None, cache_dir=None, local_model_path=None):
        """
        Initialize OCR inference with support for local SafeTensor models
        
        Args:
            model_name (str): HuggingFace model name or local path
            device (str): Device to use for inference
            cache_dir (str): Cache directory for downloaded models
            local_model_path (str): Path to local SafeTensor model directory
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        self.local_model_path = local_model_path
        self.model = None
        self.processor = None
        
        # Determine the actual model path to use
        self.effective_model_path = self._determine_model_path()
        
        # Use bfloat16 for better performance on modern hardware, fallback to float16
        if self.device.startswith('cuda') and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
            logger.info("Using bfloat16 for optimal performance")
        elif self.device.startswith('cuda'):
            self.dtype = torch.float16
            logger.info("Using float16 (bfloat16 not supported)")
        else:
            self.dtype = torch.float32
            logger.info("Using float32 for CPU")
        
        logger.info(f"Initializing OCR inference on {self.device} with {self.dtype}")
        logger.info(f"Model path: {self.effective_model_path}")
    
    def _determine_model_path(self):
        """Determine which model path to use based on parameters"""
        if self.local_model_path:
            if os.path.exists(self.local_model_path):
                logger.info(f"Using local SafeTensor model from: {self.local_model_path}")
                return self.local_model_path
            else:
                logger.warning(f"Local model path does not exist: {self.local_model_path}")
                logger.info(f"Falling back to: {self.model_name}")
                return self.model_name
        else:
            return self.model_name
    
    def _validate_safetensor_model(self, model_path):
        """Validate that the model path contains SafeTensor files"""
        if not os.path.isdir(model_path):
            return False
        
        # Check for SafeTensor files
        safetensor_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]
        config_files = [f for f in os.listdir(model_path) if f in ['config.json', 'model.safetensors.index.json']]
        
        has_safetensors = len(safetensor_files) > 0
        has_config = len(config_files) > 0
        
        if has_safetensors and has_config:
            logger.info(f"Found {len(safetensor_files)} SafeTensor files in {model_path}")
            return True
        else:
            logger.warning(f"Model path missing required files. SafeTensors: {has_safetensors}, Config: {has_config}")
            return False
        
    def load_model(self):
        """Load model with SafeTensors and optimized parameters for faster loading"""
        if self.model is not None:
            return
            
        logger.info("Loading optimized Kosmos-2.5 model with SafeTensors...")
        
        # Validate local model if specified
        is_local_model = os.path.exists(self.effective_model_path)
        if is_local_model:
            if not self._validate_safetensor_model(self.effective_model_path):
                logger.error(f"Invalid SafeTensor model directory: {self.effective_model_path}")
                raise ValueError("Local model path does not contain valid SafeTensor files")
        
        try:
            # Configure loading parameters based on whether it's local or remote
            loading_kwargs = {
                "torch_dtype": self.dtype,
                "cache_dir": self.cache_dir,
                "trust_remote_code": True,
            }
            
            if is_local_model:
                # For local SafeTensor models
                loading_kwargs.update({
                    "local_files_only": True,        # Only use local files
                    "use_safetensors": True,         # Explicitly use SafeTensors
                    "low_cpu_mem_usage": True,       # Optimize memory usage
                    "device_map": "auto",            # Automatic device placement
                })
                logger.info("Loading from local SafeTensor files...")
            else:
                # For remote models with optimizations
                loading_kwargs.update({
                    "low_cpu_mem_usage": True,       # Reduces CPU memory during loading
                    "use_safetensors": True,         # Prefer SafeTensors format
                    "device_map": "auto",            # Automatic device placement
                    "local_files_only": False,       # Allow downloading if not local
                    "resume_download": True,         # Resume interrupted downloads
                })
                logger.info("Loading from HuggingFace Hub with SafeTensors preference...")
            
            # Add Flash Attention if available
            if hasattr(torch.nn, 'scaled_dot_product_attention'):
                loading_kwargs["attn_implementation"] = "flash_attention_2"
            
            # Load the model
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.effective_model_path,
                **loading_kwargs
            )
            
            # Ensure model is in the correct dtype
            if self.device.startswith('cuda'):
                if self.dtype == torch.bfloat16:
                    self.model = self.model.bfloat16()
                else:
                    self.model = self.model.half()
                    
            # Load processor with optimizations
            processor_kwargs = {
                "cache_dir": self.cache_dir,
                "trust_remote_code": True,
                "use_fast": True,  # Use fast tokenizer when available
            }
            
            if is_local_model:
                processor_kwargs["local_files_only"] = True
            else:
                processor_kwargs.update({
                    "local_files_only": False,
                    "resume_download": True
                })
            
            self.processor = AutoProcessor.from_pretrained(
                self.effective_model_path,
                **processor_kwargs
            )
            
            # Set pad token if not present
            if self.processor.tokenizer.pad_token is None:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
                logger.info("Set pad_token to eos_token")
                
            # Enable model optimizations
            if hasattr(self.model, 'eval'):
                self.model.eval()
                
            # Enable torch.compile for PyTorch 2.0+ (if available)
            if hasattr(torch, 'compile') and self.device.startswith('cuda'):
                try:
                    logger.info("Compiling model with torch.compile for faster inference...")
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                except Exception as e:
                    logger.warning(f"torch.compile failed, continuing without it: {e}")
                
            logger.info("Model loaded successfully with optimizations")
            
            # Print model info
            try:
                num_params = sum(p.numel() for p in self.model.parameters())
                logger.info(f"Model parameters: {num_params:,}")
                
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    logger.info(f"GPU memory available: {gpu_memory:.1f} GB")
            except:
                pass
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Fallback to basic loading
            logger.info("Attempting fallback model loading...")
            try:
                fallback_kwargs = {
                    "torch_dtype": self.dtype,
                    "cache_dir": self.cache_dir,
                    "trust_remote_code": True
                }
                
                if is_local_model:
                    fallback_kwargs["local_files_only"] = True
                
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.effective_model_path,
                    **fallback_kwargs
                )
                
                processor_fallback_kwargs = {
                    "cache_dir": self.cache_dir,
                    "trust_remote_code": True
                }
                
                if is_local_model:
                    processor_fallback_kwargs["local_files_only"] = True
                
                self.processor = AutoProcessor.from_pretrained(
                    self.effective_model_path,
                    **processor_fallback_kwargs
                )
                
                if self.processor.tokenizer.pad_token is None:
                    self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
                    
                logger.info("Fallback model loading successful")
                
            except Exception as e2:
                logger.error(f"Fallback loading also failed: {e2}")
                raise
    
    def load_image(self, image_path):
        """Load image from local path or URL with error handling"""
        try:
            if image_path.startswith(('http://', 'https://')):
                logger.info(f"Loading image from URL: {image_path}")
                response = requests.get(image_path, stream=True, timeout=30)
                response.raise_for_status()
                image = Image.open(response.raw)
            else:
                logger.info(f"Loading image from file: {image_path}")
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                image = Image.open(image_path)
            
            # Convert to RGB and validate
            image = image.convert('RGB')
            logger.info(f"Image loaded successfully. Size: {image.size}")
            return image
            
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise
    
    def post_process_ocr(self, generated_text, scale_height, scale_width, prompt="<ocr>"):
        """Post-process OCR results to extract bounding boxes and text"""
        text = generated_text.replace(prompt, "").strip()
        
        # Pattern to match bounding boxes
        pattern = r"<bbox><x_(\d+)><y_(\d+)><x_(\d+)><y_(\d+)></bbox>"
        
        results = []
        
        for match in re.finditer(pattern, text):
            # Extract coordinates
            x1, y1, x2, y2 = map(int, match.groups())
            
            # Scale coordinates to original image size
            x1_scaled = int(x1 * scale_width)
            y1_scaled = int(y1 * scale_height)
            x2_scaled = int(x2 * scale_width)
            y2_scaled = int(y2 * scale_height)
            
            # Get corresponding text (next non-empty part after bbox)
            text_start = match.end()
            next_bbox = re.search(pattern, text[text_start:])
            
            if next_bbox:
                extracted_text = text[text_start:text_start + next_bbox.start()].strip()
            else:
                extracted_text = text[text_start:].strip()
            
            # Clean up extracted text
            extracted_text = re.sub(r'<[^>]+>', '', extracted_text).strip()
            
            if extracted_text and not (x1_scaled >= x2_scaled or y1_scaled >= y2_scaled):
                results.append({
                    'bbox': [x1_scaled, y1_scaled, x2_scaled, y2_scaled],
                    'text': extracted_text,
                    'confidence': 1.0  # Kosmos doesn't provide confidence scores
                })
        
        return results
    
    def draw_bounding_boxes(self, image, ocr_results, output_path=None):
        """Draw bounding boxes and text on image"""
        # Create a copy for annotation
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        
        # Try to load a font
        try:
            if os.name == 'nt':  # Windows
                font = ImageFont.truetype("arial.ttf", 16)
            else:  # Linux/Mac
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 16)
        except:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 16)
            except:
                font = ImageFont.load_default()
        
        for i, result in enumerate(ocr_results):
            bbox = result['bbox']
            text = result['text']
            
            # Use different colors for different text regions
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
            color = colors[i % len(colors)]
            
            # Draw bounding box
            draw.rectangle(bbox, outline=color, width=2)
            
            # Draw text label
            try:
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            except:
                # Fallback for older PIL versions
                text_width, text_height = draw.textsize(text, font=font)
            
            # Position text above the bounding box
            text_x = bbox[0]
            text_y = max(0, bbox[1] - text_height - 5)
            
            # Draw background for text
            draw.rectangle(
                [text_x, text_y, text_x + text_width, text_y + text_height],
                fill=color, outline=color
            )
            draw.text((text_x, text_y), text, fill="white", font=font)
        
        if output_path:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            annotated_image.save(output_path, quality=95, optimize=True)
            logger.info(f"Annotated image saved to: {output_path}")
        
        return annotated_image
    
    def perform_ocr(self, image_path, max_tokens=1024, save_image=None, save_text=None):
        """Perform OCR on image and return structured results"""
        if self.model is None:
            self.load_model()
        
        # Load and process image
        image = self.load_image(image_path)
        
        prompt = "<ocr>"
        start_time = time.time()
        
        try:
            # Process inputs
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            
            # Extract scaling information
            height = inputs.pop("height")
            width = inputs.pop("width")
            raw_width, raw_height = image.size
            scale_height = raw_height / height
            scale_width = raw_width / width
            
            # Move inputs to device and convert to correct dtype
            inputs = {k: v.to(self.device) if v is not None else None for k, v in inputs.items()}
            
            # Convert flattened_patches to correct dtype
            if "flattened_patches" in inputs and inputs["flattened_patches"] is not None:
                inputs["flattened_patches"] = inputs["flattened_patches"].to(self.dtype)
            
            # Generate OCR results
            logger.info("Performing OCR inference...")
            with torch.no_grad():
                # Use torch.cuda.amp for potential speedup
                if self.device.startswith('cuda'):
                    with torch.cuda.amp.autocast(dtype=self.dtype):
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            do_sample=False,
                            pad_token_id=self.processor.tokenizer.eos_token_id,
                            use_cache=True,
                            num_beams=1,  # Faster than beam search
                        )
                else:
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=False,
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                        use_cache=True,
                        num_beams=1,
                    )
            
            # Decode results
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Post-process to extract structured data
            ocr_results = self.post_process_ocr(generated_text, scale_height, scale_width, prompt)
            
            inference_time = time.time() - start_time
            logger.info(f"OCR completed in {inference_time:.2f}s. Found {len(ocr_results)} text regions.")
            
            # Save outputs if requested
            if save_text:
                self.save_text_results(ocr_results, save_text)
            
            if save_image:
                self.draw_bounding_boxes(image, ocr_results, save_image)
            
            return {
                'results': ocr_results,
                'inference_time': inference_time,
                'raw_output': generated_text
            }
            
        except Exception as e:
            logger.error(f"Error during OCR inference: {e}")
            raise
    
    def save_text_results(self, ocr_results, output_path):
        """Save OCR results as structured text"""
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("OCR Results\n")
                f.write("=" * 50 + "\n\n")
                
                for i, result in enumerate(ocr_results, 1):
                    bbox = result['bbox']
                    text = result['text']
                    f.write(f"Region {i}:\n")
                    f.write(f"  Bounding Box: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]\n")
                    f.write(f"  Text: {text}\n\n")
            
            logger.info(f"OCR text results saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving text results: {e}")
    
    def batch_process(self, image_paths, output_dir, max_tokens=1024):
        """Process multiple images in batch"""
        if self.model is None:
            self.load_model()
        
        results = []
        os.makedirs(output_dir, exist_ok=True)
        
        for i, image_path in enumerate(image_paths, 1):
            logger.info(f"Processing image {i}/{len(image_paths)}: {image_path}")
            
            try:
                # Generate output filenames
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_image = os.path.join(output_dir, f"{base_name}_ocr.png")
                output_text = os.path.join(output_dir, f"{base_name}_ocr.txt")
                
                # Perform OCR
                result = self.perform_ocr(
                    image_path=image_path,
                    max_tokens=max_tokens,
                    save_image=output_image,
                    save_text=output_text
                )
                
                result['input_path'] = image_path
                result['output_image'] = output_image
                result['output_text'] = output_text
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                results.append({
                    'input_path': image_path,
                    'error': str(e),
                    'inference_time': 0
                })
        
        return results

def get_args():
    parser = argparse.ArgumentParser(description='Optimized OCR inference using Kosmos-2.5 with SafeTensors')
    parser.add_argument('--image', '-i', type=str, required=True,
                       help='Path to input image file or URL')
    parser.add_argument('--output_image', '-o', type=str, default='./ocr_output.png',
                       help='Output path for annotated image')
    parser.add_argument('--output_text', '-t', type=str, default=None,
                       help='Output path for OCR text results')
    parser.add_argument('--device', '-d', type=str, default=None,
                       help='Device to use (auto-detected if not specified)')
    parser.add_argument('--max_tokens', '-m', type=int, default=1024,
                       help='Maximum tokens to generate')
    parser.add_argument('--cache_dir', type=str, default=None,
                       help='Cache directory for model files')
    parser.add_argument('--local_model_path', type=str, default=None,
                       help='Path to local SafeTensor model directory')
    parser.add_argument('--model_name', type=str, default='microsoft/kosmos-2.5',
                       help='HuggingFace model name (used if local_model_path not provided)')
    parser.add_argument('--no_image_output', action='store_true',
                       help='Skip saving annotated image')
    parser.add_argument('--batch', action='store_true',
                       help='Process multiple images (image should be a directory)')
    
    return parser.parse_args()

def main():
    args = get_args()
    
    # Initialize OCR inference
    ocr_engine = OptimizedOCRInference(
        model_name=args.model_name,
        device=args.device,
        cache_dir=args.cache_dir,
        local_model_path=args.local_model_path
    )
    
    try:
        if args.batch and os.path.isdir(args.image):
            # Batch processing
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
            image_paths = [
                os.path.join(args.image, f) for f in os.listdir(args.image)
                if any(f.lower().endswith(ext) for ext in image_extensions)
            ]
            
            if not image_paths:
                logger.error(f"No images found in directory: {args.image}")
                sys.exit(1)
            
            logger.info(f"Processing {len(image_paths)} images in batch mode")
            results = ocr_engine.batch_process(
                image_paths=image_paths,
                output_dir=args.output_image,  # Use as output directory
                max_tokens=args.max_tokens
            )
            
            # Print summary
            successful = sum(1 for r in results if 'error' not in r)
            total_time = sum(r.get('inference_time', 0) for r in results)
            total_regions = sum(len(r.get('results', [])) for r in results if 'error' not in r)
            
            print(f"\n{'='*60}")
            print("BATCH OCR PROCESSING SUMMARY")
            print(f"{'='*60}")
            print(f"Total images processed: {len(results)}")
            print(f"Successful: {successful}")
            print(f"Failed: {len(results) - successful}")
            print(f"Total text regions found: {total_regions}")
            print(f"Total processing time: {total_time:.2f}s")
            print(f"Average time per image: {total_time/len(results):.2f}s")
            print(f"{'='*60}")
            
        else:
            # Single image processing
            results = ocr_engine.perform_ocr(
                image_path=args.image,
                max_tokens=args.max_tokens,
                save_image=None if args.no_image_output else args.output_image,
                save_text=args.output_text
            )
            
            # Print results summary
            print(f"\n{'='*60}")
            print("OCR RESULTS SUMMARY")
            print(f"{'='*60}")
            print(f"Processing time: {results['inference_time']:.2f}s")
            print(f"Text regions found: {len(results['results'])}")
            print(f"{'='*60}")
            
            for i, result in enumerate(results['results'], 1):
                print(f"Region {i}: {result['text']}")
            
            print(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"OCR inference failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
