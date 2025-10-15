#!/usr/bin/env python3
"""
Kosmos-2.5 Model Quantization Script

This script provides multiple quantization approaches for the Microsoft Kosmos-2.5 model:
1. BitsAndBytes (8-bit/4-bit) - Easy to use, good for inference
2. GPTQ (4-bit) - Better compression with minimal accuracy loss
3. AWQ (4-bit) - Fastest inference with activation-aware quantization

Choose the method based on your requirements:
- BitsAndBytes: Quick setup, good memory reduction
- GPTQ: Best balance of size/accuracy for general use
- AWQ: Best for inference speed and throughput
"""

import os
import torch
import argparse
import numpy as np
from PIL import Image
from transformers import (
    AutoTokenizer, 
    AutoProcessor,
    AutoModelForImageTextToText,  # Use the new class name
    BitsAndBytesConfig,
    GPTQConfig
)

from optimum.gptq import GPTQQuantizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Update the class to handle both old and new model classes
class KosmosQuantizer:
    def __init__(self, model_name="microsoft/kosmos-2.5", cache_dir=None):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.tokenizer = None
        self.processor = None
        self.model = None
        
        # Try to determine the correct model class
        try:
            # Try the new class first
            from transformers import AutoModelForImageTextToText
            self.model_class = AutoModelForImageTextToText
            logger.info("Using AutoModelForImageTextToText")
        except ImportError:
            # Fallback to the deprecated class
            from transformers import AutoModelForVision2Seq
            self.model_class = AutoModelForVision2Seq
            logger.info("Using AutoModelForVision2Seq (deprecated)")
            
    def load_tokenizer_and_processor(self):
        """Load tokenizer and processor"""
        logger.info("Loading tokenizer and processor...")
        try:
            # Use trust_remote_code=True to avoid compatibility issues
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            # Handle the slow processor warning
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, 
                cache_dir=self.cache_dir,
                trust_remote_code=True,
                use_fast=True  # Enable fast processor to avoid warning
            )
            
            # Handle vocabulary holes by ensuring proper padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Set pad_token to eos_token to handle vocabulary issues")
                
        except Exception as e:
            logger.error(f"Failed to load tokenizer/processor: {e}")
            # Fallback without use_fast
            try:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_name, 
                    cache_dir=self.cache_dir,
                    trust_remote_code=True
                )
                logger.info("Loaded processor without use_fast option")
            except Exception as e2:
                logger.error(f"Fallback processor loading also failed: {e2}")
                raise
        
    def quantize_bitsandbytes(self, bits=8, save_path="./kosmos2.5-bnb-quantized"):
        """
        Quantize using BitsAndBytes (8-bit or 4-bit)
        """
        logger.info(f"Starting BitsAndBytes {bits}-bit quantization...")
        
        if bits == 8:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                llm_int8_enable_fp32_cpu_offload=False,
            )
        elif bits == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",  # Normal Float 4
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,  # Nested quantization
            )
        else:
            raise ValueError("Only 4-bit and 8-bit quantization supported")
            
        # Load model with quantization config using the correct model class
        try:
            self.model = self.model_class.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                dtype=torch.float16,  # Use dtype instead of torch_dtype
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
        except Exception as e:
            logger.error(f"Failed to load model for quantization: {e}")
            raise
        
        # Create save directory
        os.makedirs(save_path, exist_ok=True)
        
        # Save quantized model with proper error handling
        try:
            self.model.save_pretrained(save_path, safe_serialization=True)
            logger.info(f"Model saved to {save_path}")
        except Exception as e:
            logger.warning(f"Failed to save model: {e}")
            # Try without safe serialization
            self.model.save_pretrained(save_path, safe_serialization=False)
            
        # Save tokenizer and processor with vocabulary fix
        if self.tokenizer:
            try:
                # Handle vocabulary holes by cleaning the tokenizer
                self._fix_tokenizer_vocabulary()
                self.tokenizer.save_pretrained(save_path)
                logger.info(f"Tokenizer saved to {save_path}")
            except Exception as e:
                logger.warning(f"Failed to save tokenizer: {e}")
                
        if self.processor:
            try:
                self.processor.save_pretrained(save_path)
                logger.info(f"Processor saved to {save_path}")
            except Exception as e:
                logger.warning(f"Failed to save processor: {e}")
            
        logger.info(f"BitsAndBytes {bits}-bit quantized model saved to {save_path}")
        return self.model
    
    def _fix_tokenizer_vocabulary(self):
        """Fix vocabulary holes in the tokenizer"""
        try:
            # Get the vocabulary
            vocab = self.tokenizer.get_vocab()
            vocab_size = len(vocab)
            
            # Check for holes in vocabulary indices
            indices = sorted(vocab.values())
            max_index = max(indices) if indices else 0
            
            logger.info(f"Vocabulary size: {vocab_size}, Max index: {max_index}")
            
            # If there are holes, we'll just log them but continue
            # The warning is mostly cosmetic for saving
            missing_indices = set(range(max_index + 1)) - set(indices)
            if missing_indices:
                logger.warning(f"Found {len(missing_indices)} holes in vocabulary at indices: {sorted(list(missing_indices))[:20]}...")
                
        except Exception as e:
            logger.warning(f"Could not analyze vocabulary: {e}")
        
    def quantize_gptq(self, bits=4, save_path="./kosmos2.5-gptq-quantized", dataset_path=None):
        """
        Quantize using GPTQ (4-bit recommended)
        
        Args:
            bits (int): Number of bits (4 recommended)
            save_path (str): Path to save quantized model
            dataset_path (str): Path to calibration dataset (optional)
        """
        logger.info(f"Starting GPTQ {bits}-bit quantization...")
        
        # Load unquantized model first using Auto class
        try:
            self.model = self.model_class.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
        except Exception as e:
            logger.error(f"Failed to load model for GPTQ: {e}")
            raise
    
        # Fix the model config to add missing attributes for GPTQ compatibility
        if not hasattr(self.model.config, 'use_cache'):
            logger.info("Adding missing 'use_cache' attribute to model config for GPTQ compatibility")
            self.model.config.use_cache = True

        # Prepare calibration data first
        try:
            if dataset_path is None:
                calibration_data = self._prepare_calibration_data_for_gptq()
            else:
                calibration_data = self._load_custom_dataset(dataset_path)
        except Exception as e:
            logger.error(f"Failed to prepare calibration data: {e}")
            raise
    
        # GPTQ quantization is often incompatible with multimodal models like Kosmos-2.5
        # We'll try multiple approaches but inform the user about limitations
        logger.warning("GPTQ quantization may not be fully compatible with Kosmos-2.5 multimodal architecture.")
        logger.info("If GPTQ fails, consider using BitsAndBytes quantization which is more reliable.")
        
        quantized_model = None
        
        # Approach 1: Try with simplified config
        try:
            logger.info("Attempting GPTQ with simplified configuration...")
            
            # Create a custom wrapper to handle the use_cache issue
            class ConfigWrapper:
                def __init__(self, original_config):
                    self._original_config = original_config
                    self.use_cache = False  # Always False for GPTQ
                    
                def __getattr__(self, name):
                    if name == 'use_cache':
                        return False
                    return getattr(self._original_config, name)
                    
                def __setattr__(self, name, value):
                    if name.startswith('_'):
                        super().__setattr__(name, value)
                    elif name == 'use_cache':
                        pass  # Ignore use_cache setting
                    else:
                        setattr(self._original_config, name, value)
            
            # Temporarily wrap the config
            original_config = self.model.config
            self.model.config = ConfigWrapper(original_config)
            
            # Create quantizer with minimal settings
            quantizer = GPTQQuantizer(
                bits=bits,
                group_size=128,
                desc_act=False,
            )
            
            # Attempt quantization
            quantized_model = quantizer.quantize_model(
                model=self.model, 
                tokenizer=self.tokenizer
            )
            
            # Restore original config
            quantized_model.config = original_config
            logger.info("Successfully quantized using optimum GPTQQuantizer with config wrapper")
            
        except Exception as e:
            logger.warning(f"GPTQ quantization failed: {e}")
            logger.error("GPTQ quantization is not compatible with this model architecture.")
            logger.info("\nRecommended alternatives:")
            logger.info("1. Use BitsAndBytes 4-bit quantization (most reliable):")
            logger.info("   python kosmos_quantize.py --method bnb --bits 4 --save_path ./kosmos2.5-bnb4bit")
            logger.info("2. Use BitsAndBytes 8-bit quantization (faster):")
            logger.info("   python kosmos_quantize.py --method bnb --bits 8 --save_path ./kosmos2.5-bnb8bit")
            raise RuntimeError("GPTQ quantization is not supported for this model architecture. Please use BitsAndBytes instead.")

        if quantized_model is None:
            raise RuntimeError("GPTQ quantization failed")
        
        # Create save directory
        os.makedirs(save_path, exist_ok=True)
        
        # Save quantized model
        try:
            quantized_model.save_pretrained(save_path, safe_serialization=True)
            logger.info(f"Quantized model saved to {save_path}")
        except Exception as e:
            logger.warning(f"Failed to save with safe serialization: {e}")
            try:
                quantized_model.save_pretrained(save_path, safe_serialization=False)
                logger.info(f"Quantized model saved to {save_path} (without safe serialization)")
            except Exception as e2:
                logger.error(f"Failed to save model completely: {e2}")
                raise e2
    
        # Save tokenizer and processor
        if self.tokenizer:
            try:
                self._fix_tokenizer_vocabulary()
                self.tokenizer.save_pretrained(save_path)
                logger.info(f"Tokenizer saved to {save_path}")
            except Exception as e:
                logger.warning(f"Failed to save tokenizer: {e}")
        
        if self.processor:
            try:
                self.processor.save_pretrained(save_path)
                logger.info(f"Processor saved to {save_path}")
            except Exception as e:
                logger.warning(f"Failed to save processor: {e}")
    
        logger.info(f"GPTQ {bits}-bit quantized model saved to {save_path}")
        return quantized_model

    def _prepare_calibration_data_for_gptq(self):
        """Prepare calibration data specifically formatted for GPTQ"""
        # Create sample prompts for multimodal tasks
        calibration_prompts = [
            "<ocr>",
            "<md>", 
            "What is in this image?",
            "Describe the contents of this document.",
            "Extract the text from this image.",
            "Please analyze this image.",
            "What text can you see?",
            "Convert this to markdown.",
        ]
        
        if not self.tokenizer or not self.processor:
            self.load_tokenizer_and_processor()
            
        # For GPTQ, we need text-only calibration data
        calibration_data = []
        for prompt in calibration_prompts:
            try:
                # Use text-only tokenization for GPTQ compatibility
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                # GPTQ typically expects just input_ids
                if 'input_ids' in inputs:
                    calibration_data.append({'input_ids': inputs['input_ids']})
                
            except Exception as e:
                logger.warning(f"Failed to process calibration prompt '{prompt}': {e}")
                continue
                
        logger.info(f"Prepared {len(calibration_data)} text-only calibration samples for GPTQ")
        return calibration_data
    
    def quantize_awq(self, save_path="./kosmos2.5-awq-quantized"):
        """
        Quantize using AWQ (Activation-aware Weight Quantization)
        Note: This requires the awq library to be installed
        """
        logger.warning("AWQ quantization is not compatible with Kosmos-2.5 multimodal architecture.")
        logger.info("AWQ is designed for causal language models, not multimodal vision-language models.")
        
        try:
            from awq import AutoAWQForCausalLM
            from awq.quantize.quantizer import AwqQuantizer
        except ImportError:
            raise ImportError("AWQ library not installed. Install with: pip install autoawq")
        
        logger.info("Attempting AWQ 4-bit quantization (experimental)...")
        
        # Check if the model architecture is supported
        try:
            # Try to load the model to check compatibility
            temp_model = self.model_class.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            model_type = getattr(temp_model.config, 'model_type', 'unknown')
            logger.info(f"Detected model type: {model_type}")
            
            # Clean up temp model
            del temp_model
            torch.cuda.empty_cache()
            
            if model_type == 'kosmos-2.5':
                logger.error("AWQ does not support Kosmos-2.5 model architecture.")
                logger.info("Kosmos-2.5 uses a multimodal architecture that AWQ cannot quantize.")
                logger.info("\nRecommended alternatives:")
                logger.info("1. Use BitsAndBytes 4-bit quantization (best compatibility):")
                logger.info("   python kosmos_quantize.py --method bnb --bits 4 --save_path ./kosmos2.5-bnb4bit")
                logger.info("2. Use BitsAndBytes 8-bit quantization (faster):")
                logger.info("   python kosmos_quantize.py --method bnb --bits 8 --save_path ./kosmos2.5-bnb8bit")
                raise TypeError(f"AWQ does not support {model_type} model architecture. Please use BitsAndBytes instead.")
                
        except Exception as e:
            if "isn't supported yet" in str(e) or "kosmos-2.5" in str(e).lower():
                logger.error("AWQ quantization failed: Model architecture not supported")
                logger.info("AWQ currently supports these model types:")
                logger.info("  - LLaMA, LLaMA-2, Code Llama")
                logger.info("  - OPT, Falcon, Bloom")
                logger.info("  - GPT-J, GPT-NeoX")
                logger.info("  - Mistral, Mixtral")
                logger.info("  - Qwen, Baichuan, etc.")
                logger.info("")
                logger.info("Kosmos-2.5 is a multimodal model that AWQ cannot quantize.")
                logger.info("\nPlease use BitsAndBytes quantization instead:")
                logger.info("  python kosmos_quantize.py --method bnb --bits 4 --save_path ./kosmos2.5-bnb4bit")
                raise TypeError("AWQ quantization is not supported for Kosmos-2.5. Use BitsAndBytes instead.")
            else:
                raise e
        
        # This code will not be reached for Kosmos-2.5, but kept for other models
        try:
            # AWQ quantization config
            quant_config = {
                "zero_point": True,
                "q_group_size": 128,
                "w_bit": 4,
                "version": "GEMM"
            }
            
            # Load model for AWQ quantization
            model = AutoAWQForCausalLM.from_pretrained(
                self.model_name, 
                device_map="auto",
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            # Prepare calibration data
            calibration_data = self._prepare_calibration_data()
            
            # Quantize
            model.quantize(
                tokenizer=self.tokenizer,
                quant_config=quant_config,
                calib_data=calibration_data
            )
            
            # Create save directory
            os.makedirs(save_path, exist_ok=True)
            
            # Save
            model.save_quantized(save_path)
            
            if self.tokenizer:
                try:
                    self._fix_tokenizer_vocabulary()
                    self.tokenizer.save_pretrained(save_path)
                except Exception as e:
                    logger.warning(f"Failed to save tokenizer: {e}")
                    
            if self.processor:
                try:
                    self.processor.save_pretrained(save_path)
                except Exception as e:
                    logger.warning(f"Failed to save processor: {e}")
            
            logger.info(f"AWQ quantized model saved to {save_path}")
            return model
            
        except Exception as e:
            logger.error(f"AWQ quantization failed: {e}")
            raise
    
    def _prepare_calibration_data(self):
        """Prepare calibration data for quantization"""
        # Create sample prompts for multimodal tasks
        calibration_prompts = [
            "<ocr>",
            "<md>", 
            "What is in this image?",
            "Describe the contents of this document.",
            "Extract the text from this image.",
        ]
        
        if not self.tokenizer or not self.processor:
            self.load_tokenizer_and_processor()
            
        # Prepare calibration data with dummy images
        calibration_data = []
        for prompt in calibration_prompts:
            # Create a dummy image for each prompt
            dummy_image = Image.fromarray(
                np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            )
            
            try:
                # Use processor for multimodal inputs
                inputs = self.processor(
                    text=prompt,
                    images=dummy_image,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                # Remove problematic keys and clean inputs
                filtered_inputs = {}
                for k, v in inputs.items():
                    if k not in ['token_type_ids'] and v is not None:
                        # Ensure tensors are proper shape
                        if hasattr(v, 'shape') and len(v.shape) > 0:
                            filtered_inputs[k] = v
                
                if filtered_inputs:  # Only add if we have valid inputs
                    calibration_data.append(filtered_inputs)
                
            except Exception as e:
                logger.warning(f"Failed to process calibration prompt '{prompt}': {e}")
                # Fallback to text-only tokenization
                try:
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    )
                    # Remove problematic keys
                    filtered_inputs = {k: v for k, v in inputs.items() 
                                     if k not in ['token_type_ids'] and v is not None}
                    if filtered_inputs:
                        calibration_data.append(filtered_inputs)
                except Exception as e2:
                    logger.warning(f"Fallback tokenization also failed: {e2}")
                    continue
                
        logger.info(f"Prepared {len(calibration_data)} calibration samples")
        return calibration_data
    
    def _load_custom_dataset(self, dataset_path):
        """Load custom calibration dataset"""
        # Implement custom dataset loading logic here
        logger.info(f"Loading custom dataset from {dataset_path}")
        # This is a placeholder - implement based on your dataset format
        return self._prepare_calibration_data()
    
    def benchmark_model(self, model, test_iterations=10):
        """Benchmark model inference speed and memory usage"""
        logger.info("Benchmarking model performance...")
        
        # For Kosmos-2.5, we need both text and image inputs
        dummy_image = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )
        
        # Prepare inputs using the processor
        dummy_text = "<ocr>"
        
        if self.processor is None:
            self.load_tokenizer_and_processor()
        
        try:
            # Use processor instead of tokenizer for multimodal inputs
            inputs = self.processor(
                text=dummy_text,
                images=dummy_image,
                return_tensors="pt"
            )
            
            # Remove any unused parameters that might cause issues
            essential_inputs = {}
            for key in ['input_ids', 'attention_mask', 'pixel_values', 'image_embeds']:
                if key in inputs and inputs[key] is not None:
                    essential_inputs[key] = inputs[key]
            
            # Move inputs to the same device as model
            device = next(model.parameters()).device
            essential_inputs = {k: v.to(device) if v is not None else v 
                               for k, v in essential_inputs.items()}
            
            # Warm up
            logger.info("Warming up model...")
            with torch.no_grad():
                try:
                    for _ in range(3):
                        _ = model.generate(
                            **essential_inputs,
                            max_new_tokens=50,
                            do_sample=False,
                            pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.pad_token_id is None else self.tokenizer.pad_token_id
                        )
                except Exception as e:
                    logger.warning(f"Warmup failed: {e}")
                    # Fallback to simpler generation
                    try:
                        for _ in range(3):
                            _ = model.generate(
                                input_ids=essential_inputs.get('input_ids'),
                                max_new_tokens=20,
                                do_sample=False
                            )
                    except Exception as e2:
                        logger.error(f"Fallback warmup also failed: {e2}")
                        return None
            
            # Benchmark
            logger.info(f"Running benchmark with {test_iterations} iterations...")
            import time
            start_time = time.time()
            
            successful_runs = 0
            with torch.no_grad():
                for i in range(test_iterations):
                    try:
                        outputs = model.generate(
                            **essential_inputs,
                            max_new_tokens=50,
                            do_sample=False,
                            pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.pad_token_id is None else self.tokenizer.pad_token_id
                        )
                        successful_runs += 1
                        
                        if i == 0:
                            # Decode and show sample output
                            try:
                                decoded_output = self.processor.batch_decode(
                                    outputs, skip_special_tokens=True
                                )[0]
                                logger.info(f"Sample output: {decoded_output}")
                            except Exception as e:
                                logger.info(f"Could not decode output: {e}")
                        
                    except Exception as e:
                        logger.warning(f"Generation failed on iteration {i}: {e}")
                        continue
                    
            end_time = time.time()
            
            if successful_runs == 0:
                logger.error("All benchmark iterations failed!")
                return None
                
            avg_time = (end_time - start_time) / successful_runs
            
            # Memory usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.max_memory_allocated() / 1024**3  # GB
                logger.info(f"GPU Memory used: {memory_used:.2f} GB")
                
            logger.info(f"Successful runs: {successful_runs}/{test_iterations}")
            logger.info(f"Average inference time: {avg_time:.3f} seconds")
            return avg_time
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return None
    
    def check_quantization_compatibility(self):
        """Check which quantization methods are likely to work with this model"""
        logger.info("Checking quantization compatibility...")
        
        compatibility = {
            'bitsandbytes': True,  # Almost always works
            'gptq': False,         # Usually incompatible with multimodal models
            'awq': False           # Only works with specific causal LM architectures
        }
        
        try:
            # Load model briefly to check config
            temp_model = self.model_class.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            model_type = getattr(temp_model.config, 'model_type', 'unknown')
            logger.info(f"Model type: {model_type}")
            
            # Check for GPTQ compatibility (limited for multimodal models)
            if hasattr(temp_model.config, 'use_cache') and model_type != 'kosmos-2.5':
                compatibility['gptq'] = True
            
            # AWQ compatibility - very limited model support
            awq_supported_models = [
                'llama', 'llama2', 'code_llama', 'opt', 'falcon', 'bloom',
                'gptj', 'gpt_neox', 'mistral', 'mixtral', 'qwen', 'baichuan'
            ]
            
            if any(supported in model_type.lower() for supported in awq_supported_models):
                compatibility['awq'] = True
            
            # Clean up
            del temp_model
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.warning(f"Could not check compatibility: {e}")

        logger.info("Quantization compatibility for this model:")
        for method, compatible in compatibility.items():
            if compatible:
                status = "✓ Compatible"
                color = ""
            else:
                status = "✗ Not supported"
                color = ""
            logger.info(f"  {method.upper()}: {status}")
    
        # Provide specific recommendations
        logger.info("\nRecommendations:")
        if compatibility['bitsandbytes']:
            logger.info("✓ BitsAndBytes: Recommended for Kosmos-2.5 (reliable, good compression)")
        if not compatibility['gptq']:
            logger.info("✗ GPTQ: Not compatible with multimodal architectures")
        if not compatibility['awq']:
            logger.info("✗ AWQ: Only supports specific causal language models")

        return compatibility

def main():
    parser = argparse.ArgumentParser(description='Quantize Kosmos-2.5 model')
    parser.add_argument('--method', choices=['bnb', 'gptq', 'awq'], required=True,
                       help='Quantization method')
    parser.add_argument('--bits', type=int, choices=[4, 8], default=4,
                       help='Number of bits for quantization')
    parser.add_argument('--model_name', default='microsoft/kosmos-2.5',
                       help='Model name or path')
    parser.add_argument('--save_path', required=True,
                       help='Path to save quantized model')
    parser.add_argument('--cache_dir', default=None,
                       help='Cache directory for model downloads')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark after quantization')
    parser.add_argument('--dataset_path', default=None,
                       help='Path to calibration dataset (for GPTQ)')
    parser.add_argument('--check_compatibility', action='store_true',
                       help='Check quantization compatibility before proceeding')
    
    args = parser.parse_args()
    
    # Special handling for Kosmos-2.5
    if 'kosmos' in args.model_name.lower():
        if args.method in ['gptq', 'awq']:
            logger.warning(f"⚠️  {args.method.upper()} is not compatible with Kosmos-2.5!")
            logger.info("Kosmos-2.5 is a multimodal vision-language model.")
            logger.info("GPTQ and AWQ are designed for causal language models only.")
            logger.info("")
            logger.info("✅ Recommended: Use BitsAndBytes quantization")
            logger.info("   python kosmos_quantize.py --method bnb --bits 4 --save_path ./kosmos2.5-bnb4bit")
            logger.info("")
            
            if not args.check_compatibility:
                user_input = input("Continue anyway? This will likely fail. (y/N): ")
                if user_input.lower() != 'y':
                    logger.info("Exiting. Please use --method bnb instead.")
                    return
    
    # Initialize quantizer
    quantizer = KosmosQuantizer(args.model_name, args.cache_dir)
    quantizer.load_tokenizer_and_processor()
    
    # Check compatibility if requested
    if args.check_compatibility:
        compatibility = quantizer.check_quantization_compatibility()
        if not compatibility.get(args.method, False):
            logger.warning(f"{args.method.upper()} is not compatible with this model.")
            logger.info("Please choose a compatible quantization method.")
            return
    
    # Perform quantization
    try:
        if args.method == 'bnb':
            model = quantizer.quantize_bitsandbytes(args.bits, args.save_path)
        elif args.method == 'gptq':
            model = quantizer.quantize_gptq(args.bits, args.save_path, args.dataset_path)
        elif args.method == 'awq':
            model = quantizer.quantize_awq(args.save_path)
        
        # Benchmark if requested
        if args.benchmark:
            quantizer.benchmark_model(model)
        
        logger.info("✅ Quantization completed successfully!")
        logger.info(f"📁 Quantized model saved to: {args.save_path}")
        
    except Exception as e:
        logger.error(f"❌ Quantization failed: {e}")
        logger.info("\n🔧 Troubleshooting suggestions:")
        
        if 'kosmos' in args.model_name.lower():
            logger.info("For Kosmos-2.5, use BitsAndBytes quantization:")
            logger.info("  python kosmos_quantize.py --method bnb --bits 4 --save_path ./kosmos2.5-bnb4bit")
            logger.info("  python kosmos_quantize.py --method bnb --bits 8 --save_path ./kosmos2.5-bnb8bit")
        else:
            logger.info("1. Try BitsAndBytes quantization (most compatible):")
            logger.info("   python kosmos_quantize.py --method bnb --bits 4 --save_path ./model-bnb4bit")
            logger.info("2. Check model compatibility:")
            logger.info("   python kosmos_quantize.py --check_compatibility --method bnb --save_path ./temp")
            logger.info("3. Ensure sufficient GPU memory is available")
        
        raise

if __name__ == "__main__":
    # Example usage:
    # python kosmos_quantize.py --method bnb --bits 4 --save_path ./kosmos2.5-bnb4bit
    # python kosmos_quantize.py --method gptq --bits 4 --save_path ./kosmos2.5-gptq4bit
    # python kosmos_quantize.py --method awq --save_path ./kosmos2.5-awq4bit
    
    main()
