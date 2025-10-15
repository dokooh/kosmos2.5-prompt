#!/usr/bin/env python3
"""
Optimized Markdown Generation Inference Module for Kosmos-2.5 with SafeTensors

This module provides fast markdown generation using optimized Kosmos-2.5 model.
Features:
- SafeTensors format for faster loading
- Support for local SafeTensor model paths
- bfloat16/float16 quantization for optimal performance
- Optimized memory usage
- Enhanced markdown post-processing
- Batch processing support
"""

import torch
import requests
import argparse
import sys
import os
import time
import re
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedMarkdownInference:
    def __init__(self, model_name="microsoft/kosmos-2.5", device=None, cache_dir=None, local_model_path=None):
        """
        Initialize Markdown inference with support for local SafeTensor models
        
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
        
        logger.info(f"Initializing Markdown inference on {self.device} with {self.dtype}")
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
    
    def post_process_markdown(self, generated_text, prompt="<md>"):
        """Post-process and clean up generated markdown"""
        # Remove the prompt
        markdown = generated_text.replace(prompt, "").strip()
        
        # Clean up common issues
        markdown = self.clean_markdown(markdown)
        
        return markdown
    
    def clean_markdown(self, text):
        """Clean and format markdown text"""
        # Remove extra whitespace and clean up
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Remove any remaining HTML-like tags
            line = re.sub(r'<[^>]+>', '', line)
            if line:  # Skip empty lines initially
                cleaned_lines.append(line)
        
        # Join lines back
        text = '\n'.join(cleaned_lines)
        
        # Fix common markdown formatting issues
        # Fix headers - ensure proper spacing
        text = re.sub(r'^#{1,6}\s*', lambda m: m.group(0).rstrip() + ' ', text, flags=re.MULTILINE)
        
        # Ensure proper spacing around headers
        text = re.sub(r'(^#{1,6}.*$)', r'\n\1\n', text, flags=re.MULTILINE)
        
        # Fix list items
        text = re.sub(r'^[\*\-\+]\s+', '- ', text, flags=re.MULTILINE)
        text = re.sub(r'^\d+\.\s+', lambda m: m.group(0), text, flags=re.MULTILINE)
        
        # Fix table formatting
        text = re.sub(r'\|([^|]+)\|', lambda m: '|' + m.group(1).strip() + '|', text)
        
        # Fix emphasis and strong formatting
        text = re.sub(r'\*\*([^*]+)\*\*', r'**\1**', text)
        text = re.sub(r'\*([^*]+)\*', r'*\1*', text)
        
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Ensure text starts and ends cleanly
        text = text.strip()
        
        return text
    
    def generate_markdown(self, image_path, max_tokens=2048, temperature=0.1, save_output=None):
        """Generate markdown from image"""
        if self.model is None:
            self.load_model()
        
        # Load and process image
        image = self.load_image(image_path)
        
        prompt = "<md>"
        start_time = time.time()
        
        try:
            # Process inputs
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            
            # Remove height/width info (not needed for generation)
            inputs.pop("height", None)
            inputs.pop("width", None)
            
            # Move inputs to device and convert to correct dtype
            inputs = {k: v.to(self.device) if v is not None else None for k, v in inputs.items()}
            
            # Convert flattened_patches to correct dtype
            if "flattened_patches" in inputs and inputs["flattened_patches"] is not None:
                inputs["flattened_patches"] = inputs["flattened_patches"].to(self.dtype)
            
            # Generate markdown
            logger.info("Generating markdown...")
            with torch.no_grad():
                # Use torch.cuda.amp for potential speedup
                if self.device.startswith('cuda'):
                    with torch.cuda.amp.autocast(dtype=self.dtype):
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            do_sample=temperature > 0,
                            temperature=temperature if temperature > 0 else None,
                            pad_token_id=self.processor.tokenizer.eos_token_id,
                            eos_token_id=self.processor.tokenizer.eos_token_id,
                            repetition_penalty=1.1,
                            length_penalty=1.0,
                            use_cache=True,
                            num_beams=1 if temperature > 0 else 1,  # Use greedy for deterministic
                        )
                else:
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=temperature > 0,
                        temperature=temperature if temperature > 0 else None,
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                        repetition_penalty=1.1,
                        length_penalty=1.0,
                        use_cache=True,
                        num_beams=1,
                    )
            
            # Decode results
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Post-process markdown
            markdown_output = self.post_process_markdown(generated_text, prompt)
            
            inference_time = time.time() - start_time
            logger.info(f"Markdown generation completed in {inference_time:.2f}s")
            
            # Save output if requested
            if save_output:
                self.save_markdown(markdown_output, save_output)
            
            return {
                'markdown': markdown_output,
                'inference_time': inference_time,
                'raw_output': generated_text,
                'word_count': len(markdown_output.split()),
                'char_count': len(markdown_output)
            }
            
        except Exception as e:
            logger.error(f"Error during markdown generation: {e}")
            raise
    
    def save_markdown(self, markdown_text, output_path):
        """Save markdown to file"""
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_text)
            
            logger.info(f"Markdown saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving markdown: {e}")
    
    def batch_process(self, image_paths, output_dir, max_tokens=2048, temperature=0.1):
        """Process multiple images in batch"""
        if self.model is None:
            self.load_model()
        
        results = []
        os.makedirs(output_dir, exist_ok=True)
        
        for i, image_path in enumerate(image_paths, 1):
            logger.info(f"Processing image {i}/{len(image_paths)}: {image_path}")
            
            try:
                # Generate output filename
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_path = os.path.join(output_dir, f"{base_name}_markdown.md")
                
                # Generate markdown
                result = self.generate_markdown(
                    image_path=image_path,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    save_output=output_path
                )
                
                result['input_path'] = image_path
                result['output_path'] = output_path
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
    parser = argparse.ArgumentParser(description='Optimized Markdown generation using Kosmos-2.5 with SafeTensors')
    parser.add_argument('--image', '-i', type=str, required=True,
                       help='Path to input image file or URL')
    parser.add_argument('--output', '-o', type=str, default='./output.md',
                       help='Output path for generated markdown')
    parser.add_argument('--device', '-d', type=str, default=None,
                       help='Device to use (auto-detected if not specified)')
    parser.add_argument('--max_tokens', '-m', type=int, default=2048,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', '-t', type=float, default=0.1,
                       help='Sampling temperature (0 for deterministic)')
    parser.add_argument('--cache_dir', type=str, default=None,
                       help='Cache directory for model files')
    parser.add_argument('--local_model_path', type=str, default=None,
                       help='Path to local SafeTensor model directory')
    parser.add_argument('--model_name', type=str, default='microsoft/kosmos-2.5',
                       help='HuggingFace model name (used if local_model_path not provided)')
    parser.add_argument('--batch', action='store_true',
                       help='Process multiple images (image should be a directory)')
    parser.add_argument('--print_output', '-p', action='store_true',
                       help='Print generated markdown to console')
    
    return parser.parse_args()

def main():
    args = get_args()
    
    # Initialize markdown inference
    md_engine = OptimizedMarkdownInference(
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
            results = md_engine.batch_process(
                image_paths=image_paths,
                output_dir=args.output,
                max_tokens=args.max_tokens,
                temperature=args.temperature
            )
            
            # Print summary
            successful = sum(1 for r in results if 'error' not in r)
            total_time = sum(r.get('inference_time', 0) for r in results)
            total_words = sum(r.get('word_count', 0) for r in results if 'error' not in r)
            
            print(f"\n{'='*60}")
            print("BATCH MARKDOWN PROCESSING SUMMARY")
            print(f"{'='*60}")
            print(f"Total images processed: {len(results)}")
            print(f"Successful: {successful}")
            print(f"Failed: {len(results) - successful}")
            print(f"Total words generated: {total_words}")
            print(f"Total processing time: {total_time:.2f}s")
            print(f"Average time per image: {total_time/len(results):.2f}s")
            print(f"{'='*60}")
            
        else:
            # Single image processing
            result = md_engine.generate_markdown(
                image_path=args.image,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                save_output=args.output
            )
            
            # Print results summary
            print(f"\n{'='*60}")
            print("MARKDOWN GENERATION SUMMARY")
            print(f"{'='*60}")
            print(f"Processing time: {result['inference_time']:.2f}s")
            print(f"Word count: {result['word_count']}")
            print(f"Character count: {result['char_count']}")
            print(f"Output saved to: {args.output}")
            print(f"{'='*60}")
            
            if args.print_output:
                print("\nGENERATED MARKDOWN:")
                print("=" * 60)
                print(result['markdown'])
                print("=" * 60)
        
    except Exception as e:
        logger.error(f"Markdown generation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
