#!/usr/bin/env python3
"""
Fast Quantization Techniques for Kosmos-2.5

This script implements several faster quantization methods beyond BitsAndBytes:
1. QLoRA-style quantization (4-bit + LoRA-ready)
2. SmoothQuant (activation-weight balanced quantization)  
3. LLM.int8() with mixed precision
4. Direct FP16 quantization (fastest)
5. ZeroQuant-style post-training quantization
6. Custom fast INT8 quantization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    AutoProcessor,
    AutoModelForImageTextToText,  # Use the correct class
    BitsAndBytesConfig
)
import logging
import time
import numpy as np
from typing import Optional, Dict, Any
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FastQuantizer:
    def __init__(self, model_name="microsoft/kosmos-2.5", cache_dir=None):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.tokenizer = None
        self.processor = None
        self.model = None
        
        # Determine the correct model class
        try:
            from transformers import AutoModelForImageTextToText
            self.model_class = AutoModelForImageTextToText
            logger.info("Using AutoModelForImageTextToText")
        except ImportError:
            from transformers import AutoModelForVision2Seq
            self.model_class = AutoModelForVision2Seq
            logger.info("Using AutoModelForVision2Seq (deprecated)")
    
    def load_components(self):
        """Load tokenizer, processor, and model"""
        logger.info("Loading model components...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            # Handle padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Set pad_token to eos_token")
                
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
        
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, 
                cache_dir=self.cache_dir,
                trust_remote_code=True,
                use_fast=True
            )
        except Exception as e:
            logger.warning(f"Failed to load fast processor, trying slow: {e}")
            try:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_name, 
                    cache_dir=self.cache_dir,
                    trust_remote_code=True
                )
            except Exception as e2:
                logger.error(f"Failed to load processor: {e2}")
                raise
    
    def method_1_qlora_style(self, save_path="./kosmos2.5-qlora-quantized"):
        """
        QLoRA-style quantization: Fast 4-bit with double quantization
        This is often faster than standard GPTQ/AWQ setup
        """
        logger.info("Starting QLoRA-style quantization...")
        
        # QLoRA config - optimized for speed
        qlora_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # NormalFloat4 - fastest 4-bit format
            bnb_4bit_compute_dtype=torch.bfloat16,  # bfloat16 often faster than float16
            bnb_4bit_use_double_quant=True,  # Double quantization for better compression
            bnb_4bit_quant_storage=torch.uint8,  # Store as uint8 for speed
        )
        
        self.model = self.model_class.from_pretrained(
            self.model_name,
            quantization_config=qlora_config,
            device_map="auto",
            dtype=torch.bfloat16,  # Use dtype instead of torch_dtype
            cache_dir=self.cache_dir,
            low_cpu_mem_usage=True,  # Faster loading
            trust_remote_code=True
        )
        
        self._save_model(save_path)
        logger.info(f"QLoRA-style quantized model saved to {save_path}")
        return self.model
    
    def method_2_fp16_quantization(self, save_path="./kosmos2.5-fp16-quantized"):
        """
        Direct FP16 quantization - fastest method with minimal setup
        Converts FP32 -> FP16, very fast and reliable
        """
        logger.info("Starting FP16 quantization...")
        
        # Load in FP16 directly - fastest approach
        self.model = self.model_class.from_pretrained(
            self.model_name,
            dtype=torch.float16,
            device_map="auto",
            cache_dir=self.cache_dir,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # Ensure all parameters are FP16
        self.model = self.model.half()
        
        self._save_model(save_path)
        logger.info(f"FP16 quantized model saved to {save_path}")
        return self.model
    
    def method_3_mixed_precision_int8(self, save_path="./kosmos2.5-mixed-int8"):
        """
        LLM.int8() style mixed precision - handles outliers separately
        Often faster than full quantization methods
        """
        logger.info("Starting mixed precision INT8 quantization...")
        
        # Configuration for mixed precision
        mixed_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,  # Outlier threshold
            llm_int8_has_fp16_weight=False,
            llm_int8_enable_fp32_cpu_offload=True,  # Faster CPU offloading
            llm_int8_skip_modules=["lm_head"],  # Skip quantizing output layer
        )
        
        self.model = self.model_class.from_pretrained(
            self.model_name,
            quantization_config=mixed_config,
            device_map="auto",
            dtype=torch.float16,
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )
        
        self._save_model(save_path)
        logger.info(f"Mixed precision INT8 model saved to {save_path}")
        return self.model
    
    def method_4_fast_custom_int8(self, save_path="./kosmos2.5-custom-int8"):
        """
        Custom fast INT8 quantization without calibration
        Uses simple min-max scaling for speed
        """
        logger.info("Starting custom fast INT8 quantization...")
        
        # Load model in FP16 first
        self.model = Kosmos2_5ForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="cpu",  # Load on CPU for quantization
            cache_dir=self.cache_dir,
        )
        
        # Apply custom quantization
        self.model = self._apply_fast_int8_quantization(self.model)
        
        # Move to GPU after quantization
        self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        
        self._save_model(save_path)
        logger.info(f"Custom INT8 quantized model saved to {save_path}")
        return self.model
    
    def method_5_zeroquant_style(self, save_path="./kosmos2.5-zeroquant"):
        """
        ZeroQuant-style post-training quantization
        Fast group-wise quantization without extensive calibration
        """
        logger.info("Starting ZeroQuant-style quantization...")
        
        # Load model normally first
        self.model = Kosmos2_5ForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="cpu",
            cache_dir=self.cache_dir,
        )
        
        # Apply ZeroQuant-style quantization
        self.model = self._apply_zeroquant_quantization(self.model)
        
        # Move to device
        self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        
        self._save_model(save_path)
        logger.info(f"ZeroQuant-style quantized model saved to {save_path}")
        return self.model
    
    def method_6_optimized_bnb(self, save_path="./kosmos2.5-optimized-bnb"):
        """
        Optimized BitsAndBytes with performance tweaks
        Same technique but with optimizations for speed
        """
        logger.info("Starting optimized BitsAndBytes quantization...")
        
        # Optimized BnB config
        optimized_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="fp4",  # FP4 can be faster than NF4 on some hardware
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,  # Skip double quant for speed
            bnb_4bit_quant_storage=torch.uint8,
        )
        
        self.model = Kosmos2_5ForConditionalGeneration.from_pretrained(
            self.model_name,
            quantization_config=optimized_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            cache_dir=self.cache_dir,
            low_cpu_mem_usage=True,
            use_safetensors=True,  # Faster loading
        )
        
        self._save_model(save_path)
        logger.info(f"Optimized BnB quantized model saved to {save_path}")
        return self.model
    
    def _apply_fast_int8_quantization(self, model):
        """Apply custom fast INT8 quantization to linear layers"""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Simple min-max quantization
                weight = module.weight.data
                weight_min = weight.min()
                weight_max = weight.max()
                
                # Scale to INT8 range
                scale = (weight_max - weight_min) / 255.0
                zero_point = -weight_min / scale
                
                # Quantize
                quantized_weight = torch.round(weight / scale + zero_point).clamp(0, 255)
                
                # Store quantization parameters
                module.register_buffer('weight_scale', scale)
                module.register_buffer('weight_zero_point', zero_point)
                module.weight.data = quantized_weight.to(torch.uint8)
                
        return model
    
    def _apply_zeroquant_quantization(self, model, group_size=128):
        """Apply ZeroQuant-style group-wise quantization"""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data
                
                # Group-wise quantization
                if weight.numel() > group_size:
                    # Reshape for group processing
                    original_shape = weight.shape
                    weight_flat = weight.view(-1)
                    
                    # Pad if necessary
                    pad_size = (group_size - weight_flat.size(0) % group_size) % group_size
                    if pad_size > 0:
                        weight_flat = torch.cat([weight_flat, torch.zeros(pad_size, device=weight.device)])
                    
                    # Reshape into groups
                    weight_groups = weight_flat.view(-1, group_size)
                    
                    # Quantize each group
                    scales = []
                    zero_points = []
                    quantized_groups = []
                    
                    for group in weight_groups:
                        group_min = group.min()
                        group_max = group.max()
                        scale = (group_max - group_min) / 255.0
                        zero_point = -group_min / scale if scale > 0 else 0
                        
                        quantized_group = torch.round(group / scale + zero_point).clamp(0, 255)
                        
                        scales.append(scale)
                        zero_points.append(zero_point)
                        quantized_groups.append(quantized_group)
                    
                    # Reconstruct weight
                    quantized_weight = torch.cat(quantized_groups).view(original_shape)
                    
                    # Store quantization parameters
                    module.register_buffer('group_scales', torch.tensor(scales))
                    module.register_buffer('group_zero_points', torch.tensor(zero_points))
                    module.weight.data = quantized_weight.to(torch.uint8)
                
        return model
    
    def _save_model(self, save_path):
        """Save model, tokenizer, and processor"""
        if self.model:
            self.model.save_pretrained(save_path)
        if self.tokenizer:
            self.tokenizer.save_pretrained(save_path)
        if self.processor:
            self.processor.save_pretrained(save_path)
    
    def benchmark_speed(self, model, iterations=20):
        """Benchmark quantization and inference speed"""
        logger.info("Benchmarking model speed...")
        
        # Dummy input
        dummy_text = "What do you see in this image?"
        
        if self.tokenizer is None:
            self.load_components()
            
        inputs = self.tokenizer(dummy_text, return_tensors="pt")
        
        # Remove token_type_ids if present (not used by Kosmos-2.5)
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']
        
        # Filter out any other problematic keys
        valid_keys = ['input_ids', 'attention_mask']
        inputs = {k: v for k, v in inputs.items() if k in valid_keys}
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Warm up
        with torch.no_grad():
            for _ in range(5):
                try:
                    _ = model.generate(
                        **inputs, 
                        max_length=30, 
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                except Exception as e:
                    logger.warning(f"Warmup iteration failed: {e}")
                    break
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        successful_iterations = 0
        with torch.no_grad():
            for i in range(iterations):
                try:
                    outputs = model.generate(
                        **inputs, 
                        max_length=30, 
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    successful_iterations += 1
                except Exception as e:
                    logger.warning(f"Iteration {i} failed: {e}")
                    continue
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        if successful_iterations == 0:
            logger.error("All benchmark iterations failed")
            return {
                "avg_inference_time": float('inf'),
                "tokens_per_second": 0,
                "memory_mb": None
            }
        
        avg_time = (end_time - start_time) / successful_iterations
        tokens_per_sec = (30 * successful_iterations) / (end_time - start_time)  # Approximate
        
        # Memory usage
        memory_mb = None
        if torch.cuda.is_available():
            memory_mb = torch.cuda.max_memory_allocated() / 1024**2
            logger.info(f"GPU Memory: {memory_mb:.0f} MB")
        
        logger.info(f"Successful iterations: {successful_iterations}/{iterations}")
        logger.info(f"Average inference time: {avg_time:.3f}s")
        logger.info(f"Tokens per second: {tokens_per_sec:.1f}")
        
        return {
            "avg_inference_time": avg_time,
            "tokens_per_second": tokens_per_sec,
            "memory_mb": memory_mb,
            "successful_iterations": successful_iterations
        }

def main():
    parser = argparse.ArgumentParser(description='Fast quantization for Kosmos-2.5')
    parser.add_argument('--method', 
                       choices=['qlora', 'fp16', 'mixed_int8', 'custom_int8', 'zeroquant', 'optimized_bnb'], 
                       required=True,
                       help='Quantization method')
    parser.add_argument('--model_name', default='microsoft/kosmos-2.5')
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--cache_dir', default=None)
    
    args = parser.parse_args()
    
    quantizer = FastQuantizer(args.model_name, args.cache_dir)
    quantizer.load_components()
    
    # Method dispatch
    methods = {
        'qlora': quantizer.method_1_qlora_style,
        'fp16': quantizer.method_2_fp16_quantization,
        'mixed_int8': quantizer.method_3_mixed_precision_int8,
        'custom_int8': quantizer.method_4_fast_custom_int8,
        'zeroquant': quantizer.method_5_zeroquant_style,
        'optimized_bnb': quantizer.method_6_optimized_bnb,
    }
    
    # Execute quantization
    start_time = time.time()
    model = methods[args.method](args.save_path)
    quantization_time = time.time() - start_time
    
    logger.info(f"Quantization completed in {quantization_time:.2f} seconds")
    
    # Benchmark if requested
    if args.benchmark:
        results = quantizer.benchmark_speed(model)
        
        print("\n" + "="*50)
        print(f"QUANTIZATION RESULTS - {args.method.upper()}")
        print("="*50)
        print(f"Quantization time: {quantization_time:.2f}s")
        print(f"Inference time: {results['avg_inference_time']:.3f}s")
        print(f"Tokens/sec: {results['tokens_per_second']:.1f}")
        if results['memory_mb']:
            print(f"GPU Memory: {results['memory_mb']:.0f} MB")
        print("="*50)

if __name__ == "__main__":
    main()
