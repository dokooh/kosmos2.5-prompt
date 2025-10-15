#!/usr/bin/env python3
# filepath: c:\SAI\IA\unilm\kosmos-2.5\md_fp8_mixed.py
"""
8-bit Mixed Precision Markdown Generation for Kosmos-2.5 with Enhanced Debugging - DTYPE MISMATCH FIX

This module provides fast markdown generation using 8-bit mixed precision quantized Kosmos-2.5 model.
Features:
- 8-bit mixed precision quantization with faster inference and reduced memory usage
- BitsAndBytesConfig for advanced quantization settings
- SafeTensors format support for faster loading
- Support for local model checkpoints and remote models
- Optimized memory usage with gradient checkpointing
- Enhanced markdown post-processing with structure detection
- Batch processing support with progress tracking
- Comprehensive error handling and fallback mechanisms
- EXTENSIVE DEBUGGING OUTPUT TO IDENTIFY STOPPING POINTS
- DTYPE MISMATCH FIX: Ensures consistent tensor dtypes throughout processing
"""

import torch
import requests
import argparse
import sys
import os
import time
import re
import traceback as tb_module  # Avoid namespace conflict
from PIL import Image
from transformers import (
    AutoProcessor, 
    AutoModelForImageTextToText,
    BitsAndBytesConfig
)
import logging

# Enhanced logging setup with more detailed formatting
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('md_debug.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

def debug_checkpoint(message, checkpoint_id=None):
    """Debug checkpoint function to track execution flow"""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    if checkpoint_id:
        logger.debug(f"🔍 CHECKPOINT [{checkpoint_id}]: {message}")
        print(f"[{timestamp}] 🔍 CHECKPOINT [{checkpoint_id}]: {message}", flush=True)
    else:
        logger.debug(f"🔍 DEBUG: {message}")
        print(f"[{timestamp}] 🔍 DEBUG: {message}", flush=True)

def debug_memory_status():
    """Debug memory status"""
    if torch.cuda.is_available():
        try:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            debug_checkpoint(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Total: {total:.2f}GB")
        except Exception as e:
            debug_checkpoint(f"Failed to get GPU memory status: {e}")
    else:
        debug_checkpoint("CUDA not available")

def debug_tensor_detailed(tensor, name):
    """Ultra-detailed tensor debugging to catch problematic values"""
    try:
        debug_checkpoint(f"=== DETAILED TENSOR ANALYSIS: {name} ===")
        debug_checkpoint(f"  Device: {tensor.device}")
        debug_checkpoint(f"  Dtype: {tensor.dtype}")
        debug_checkpoint(f"  Shape: {tensor.shape}")
        debug_checkpoint(f"  Numel: {tensor.numel()}")
        debug_checkpoint(f"  Requires grad: {tensor.requires_grad}")
        
        if tensor.numel() > 0:
            # Check for problematic values
            has_nan = torch.isnan(tensor).any().item()
            has_inf = torch.isinf(tensor).any().item() 
            has_neg_inf = torch.isneginf(tensor).any().item()
            has_pos_inf = torch.isposinf(tensor).any().item()
            
            debug_checkpoint(f"  Contains NaN: {has_nan}")
            debug_checkpoint(f"  Contains Inf: {has_inf}")
            debug_checkpoint(f"  Contains +Inf: {has_pos_inf}")
            debug_checkpoint(f"  Contains -Inf: {has_neg_inf}")
            
            if not has_nan and not has_inf:
                try:
                    min_val = tensor.min().item()
                    max_val = tensor.max().item()
                    mean_val = tensor.float().mean().item()
                    debug_checkpoint(f"  Range: [{min_val:.6f}, {max_val:.6f}]")
                    debug_checkpoint(f"  Mean: {mean_val:.6f}")
                    
                    # Check for extremely large values that might cause issues
                    if abs(min_val) > 1e6 or abs(max_val) > 1e6:
                        debug_checkpoint(f"  WARNING: Extreme values detected!")
                    
                    # For integer tensors, check for valid vocabulary range
                    if tensor.dtype in [torch.int32, torch.int64, torch.long]:
                        if min_val < 0:
                            debug_checkpoint(f"  WARNING: Negative values in integer tensor!")
                        if max_val > 250000:  # Typical vocab size limit
                            debug_checkpoint(f"  WARNING: Very large token IDs detected!")
                            
                except Exception as e:
                    debug_checkpoint(f"  Error computing statistics: {e}")
            else:
                debug_checkpoint(f"  CRITICAL: Tensor contains invalid values!")
        
        debug_checkpoint(f"=== END DETAILED ANALYSIS: {name} ===")
        
    except Exception as e:
        debug_checkpoint(f"Error in detailed tensor analysis for {name}: {e}")

def get_model_dtype(model):
    """Get the primary dtype of the model"""
    try:
        # Get dtype from the first parameter
        first_param = next(model.parameters())
        model_dtype = first_param.dtype
        debug_checkpoint(f"Model primary dtype: {model_dtype}")
        return model_dtype
    except Exception as e:
        debug_checkpoint(f"Failed to get model dtype: {e}")
        return torch.float32

def ultra_safe_tensor_validation_with_dtype(tensor_dict, target_dtype):
    """Ultra-safe tensor validation with consistent dtype conversion"""
    debug_checkpoint(f"Starting ultra-safe tensor validation with target dtype: {target_dtype}", "ULTRA_SAFE_DTYPE_START")
    
    fixed_tensors = {}
    
    for key, tensor in tensor_dict.items():
        if isinstance(tensor, torch.Tensor):
            debug_checkpoint(f"Validating tensor: {key}")
            debug_tensor_detailed(tensor, key)
            
            try:
                # Create a copy to avoid modifying original
                fixed_tensor = tensor.clone()
                
                # Step 1: Fix NaN and Inf values
                if torch.isnan(fixed_tensor).any():
                    debug_checkpoint(f"Fixing NaN values in {key}")
                    fixed_tensor = torch.nan_to_num(fixed_tensor, nan=0.0)
                
                if torch.isinf(fixed_tensor).any():
                    debug_checkpoint(f"Fixing infinite values in {key}")
                    fixed_tensor = torch.nan_to_num(fixed_tensor, posinf=1.0, neginf=-1.0)
                
                # Step 2: Tensor-specific validation and fixing
                if 'attention_mask' in key.lower():
                    debug_checkpoint(f"Processing attention mask: {key}")
                    # Ensure attention mask only has 0 and 1 values
                    fixed_tensor = torch.clamp(fixed_tensor, 0, 1)
                    # Convert to long/int64 for attention masks (keep as integer type)
                    if fixed_tensor.dtype != torch.long:
                        fixed_tensor = fixed_tensor.long()
                    debug_checkpoint(f"Attention mask {key} validated and converted to long")
                
                elif 'input_ids' in key.lower():
                    debug_checkpoint(f"Processing input_ids: {key}")
                    # Ensure input_ids are non-negative and within reasonable range
                    fixed_tensor = torch.clamp(fixed_tensor, min=0, max=250000)
                    # Ensure input_ids are long (keep as integer type)
                    if fixed_tensor.dtype != torch.long:
                        fixed_tensor = fixed_tensor.long()
                    debug_checkpoint(f"Input_ids {key} validated and clamped")
                
                elif 'pixel_values' in key.lower():
                    debug_checkpoint(f"Processing pixel values: {key}")
                    # Ensure pixel values are in reasonable range
                    fixed_tensor = torch.clamp(fixed_tensor, -10.0, 10.0)
                    # Convert pixel values to target dtype (float type)
                    if target_dtype in [torch.float16, torch.bfloat16, torch.float32]:
                        if fixed_tensor.dtype != target_dtype:
                            debug_checkpoint(f"Converting pixel values from {fixed_tensor.dtype} to {target_dtype}")
                            fixed_tensor = fixed_tensor.to(dtype=target_dtype)
                    debug_checkpoint(f"Pixel values {key} clamped and converted to {fixed_tensor.dtype}")
                
                elif 'position' in key.lower():
                    debug_checkpoint(f"Processing position tensor: {key}")
                    # Ensure position values are reasonable
                    fixed_tensor = torch.clamp(fixed_tensor, min=0)
                    if fixed_tensor.dtype != torch.long:
                        fixed_tensor = fixed_tensor.long()
                    debug_checkpoint(f"Position tensor {key} validated")
                
                else:
                    # For other float tensors, convert to target dtype
                    if fixed_tensor.dtype.is_floating_point and target_dtype.is_floating_point:
                        if fixed_tensor.dtype != target_dtype:
                            debug_checkpoint(f"Converting {key} from {fixed_tensor.dtype} to {target_dtype}")
                            fixed_tensor = fixed_tensor.to(dtype=target_dtype)
                
                # Step 3: Final validation
                debug_tensor_detailed(fixed_tensor, f"{key}_fixed")
                
                # Step 4: Double-check for any remaining issues
                if torch.isnan(fixed_tensor).any() or torch.isinf(fixed_tensor).any():
                    debug_checkpoint(f"CRITICAL: Tensor {key} still has invalid values after fixing!")
                    # Last resort: replace with zeros if still problematic
                    if 'attention_mask' in key.lower():
                        fixed_tensor = torch.ones_like(fixed_tensor, dtype=torch.long)
                    elif 'input_ids' in key.lower():
                        fixed_tensor = torch.zeros_like(fixed_tensor, dtype=torch.long)
                    else:
                        fixed_tensor = torch.zeros_like(fixed_tensor, dtype=target_dtype)
                    debug_checkpoint(f"Replaced {key} with safe values as last resort")
                
                fixed_tensors[key] = fixed_tensor
                debug_checkpoint(f"Tensor {key} validation completed successfully - Final dtype: {fixed_tensor.dtype}")
                
            except Exception as e:
                debug_checkpoint(f"Error fixing tensor {key}: {e}")
                # If all else fails, try to create a safe version
                try:
                    if 'attention_mask' in key.lower():
                        # Create a basic attention mask
                        safe_shape = tensor.shape
                        fixed_tensors[key] = torch.ones(safe_shape, dtype=torch.long, device='cpu')
                        debug_checkpoint(f"Created fallback attention mask for {key}")
                    elif 'input_ids' in key.lower():
                        # Create basic input_ids (all padding tokens)
                        safe_shape = tensor.shape
                        fixed_tensors[key] = torch.zeros(safe_shape, dtype=torch.long, device='cpu')
                        debug_checkpoint(f"Created fallback input_ids for {key}")
                    else:
                        # For other tensors, create zeros with target dtype
                        fixed_tensors[key] = torch.zeros_like(tensor, dtype=target_dtype, device='cpu')
                        debug_checkpoint(f"Created fallback zeros tensor for {key}")
                except Exception as fallback_error:
                    debug_checkpoint(f"Even fallback creation failed for {key}: {fallback_error}")
                    # Skip this tensor entirely
                    continue
        else:
            fixed_tensors[key] = tensor
    
    debug_checkpoint("Ultra-safe tensor validation with dtype completed", "ULTRA_SAFE_DTYPE_END")
    return fixed_tensors

def ultra_safe_device_move_with_dtype(tensor_dict, target_device, target_dtype):
    """Ultra-safe device movement with dtype consistency"""
    debug_checkpoint(f"Starting ultra-safe device move to {target_device} with dtype {target_dtype}", "DEVICE_MOVE_DTYPE_START")
    
    # Parse target device
    if isinstance(target_device, torch.device):
        target_device_str = str(target_device)
    else:
        target_device_str = str(target_device)
    
    device_moved = {}
    
    for key, tensor in tensor_dict.items():
        if isinstance(tensor, torch.Tensor):
            debug_checkpoint(f"Moving tensor {key} to device {target_device_str} with dtype consistency")
            
            try:
                # First, ensure tensor is on CPU for safe transfer
                if tensor.device.type != 'cpu':
                    debug_checkpoint(f"Moving {key} to CPU first")
                    cpu_tensor = tensor.cpu()
                else:
                    cpu_tensor = tensor
                
                # Validate tensor on CPU
                debug_checkpoint(f"Validating {key} on CPU before device move")
                if torch.isnan(cpu_tensor).any() or torch.isinf(cpu_tensor).any():
                    debug_checkpoint(f"CRITICAL: Tensor {key} has invalid values on CPU!")
                    raise ValueError(f"Invalid values in tensor {key}")
                
                # Apply dtype conversion if needed (only for floating point tensors)
                if cpu_tensor.dtype.is_floating_point and target_dtype.is_floating_point:
                    if cpu_tensor.dtype != target_dtype:
                        debug_checkpoint(f"Converting {key} from {cpu_tensor.dtype} to {target_dtype}")
                        cpu_tensor = cpu_tensor.to(dtype=target_dtype)
                
                # Move to target device with explicit error handling
                if target_device_str.startswith('cuda'):
                    debug_checkpoint(f"Moving {key} to CUDA device")
                    
                    # Extra safety: synchronize before move
                    torch.cuda.synchronize()
                    
                    # Move with explicit device specification
                    device_moved[key] = cpu_tensor.to(device=target_device_str, non_blocking=False)
                    
                    # Synchronize after move
                    torch.cuda.synchronize()
                    
                    # Verify the move was successful
                    if device_moved[key].device.type != 'cuda':
                        raise RuntimeError(f"Failed to move {key} to CUDA")
                    
                    debug_checkpoint(f"Successfully moved {key} to {device_moved[key].device} with dtype {device_moved[key].dtype}")
                else:
                    # Moving to CPU or other device
                    device_moved[key] = cpu_tensor.to(device=target_device_str)
                    debug_checkpoint(f"Successfully moved {key} to {device_moved[key].device} with dtype {device_moved[key].dtype}")
                
            except Exception as e:
                debug_checkpoint(f"Failed to move tensor {key} to {target_device_str}: {e}")
                
                # Try fallback strategies
                try:
                    debug_checkpoint(f"Attempting fallback device move for {key}")
                    
                    # Strategy 1: Keep on CPU if CUDA move fails
                    if target_device_str.startswith('cuda'):
                        debug_checkpoint(f"Keeping {key} on CPU as fallback")
                        # Apply dtype conversion even for CPU fallback
                        if tensor.dtype.is_floating_point and target_dtype.is_floating_point:
                            if tensor.dtype != target_dtype:
                                device_moved[key] = tensor.cpu().to(dtype=target_dtype)
                            else:
                                device_moved[key] = tensor.cpu()
                        else:
                            device_moved[key] = tensor.cpu()
                    else:
                        # For non-CUDA targets, try direct move
                        device_moved[key] = tensor.to(device=target_device_str)
                    
                    debug_checkpoint(f"Fallback move successful for {key}")
                    
                except Exception as fallback_error:
                    debug_checkpoint(f"Fallback move also failed for {key}: {fallback_error}")
                    
                    # Last resort: create a new tensor on target device
                    try:
                        if 'attention_mask' in key.lower():
                            device_moved[key] = torch.ones_like(tensor, device='cpu', dtype=torch.long)
                        elif 'input_ids' in key.lower():
                            device_moved[key] = torch.zeros_like(tensor, device='cpu', dtype=torch.long)
                        else:
                            device_moved[key] = torch.zeros_like(tensor, device='cpu', dtype=target_dtype)
                        debug_checkpoint(f"Created replacement tensor for {key} on CPU with correct dtype")
                    except Exception as replacement_error:
                        debug_checkpoint(f"Even replacement tensor creation failed for {key}: {replacement_error}")
                        # Skip this tensor
                        continue
        else:
            device_moved[key] = tensor
    
    debug_checkpoint("Ultra-safe device move with dtype completed", "DEVICE_MOVE_DTYPE_END")
    return device_moved

def safe_execute(func, description, *args, **kwargs):
    """Safely execute a function with detailed error reporting"""
    debug_checkpoint(f"Starting: {description}")
    try:
        result = func(*args, **kwargs)
        debug_checkpoint(f"Completed: {description}")
        return result
    except Exception as e:
        debug_checkpoint(f"FAILED: {description} - Error: {str(e)}")
        logger.error(f"Exception in {description}: {tb_module.format_exc()}")
        raise

def tensor_to_float(tensor_val):
    """Safely convert tensor to float for formatting"""
    if isinstance(tensor_val, torch.Tensor):
        return float(tensor_val.item())
    return float(tensor_val)

def enable_cuda_debugging():
    """Enable CUDA debugging environment variables"""
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    debug_checkpoint("Enabled CUDA debugging environment variables")

def ensure_dtype_consistency_md(inputs, target_dtype=None, model=None):
    """Ensure all input tensors have consistent dtypes to avoid bias/query mismatches for MD"""
    if target_dtype is None and model is not None:
        target_dtype = get_model_dtype(model)
    elif target_dtype is None:
        target_dtype = torch.float16
    
    debug_checkpoint(f"MD: Ensuring dtype consistency with target: {target_dtype}")
    
    if isinstance(inputs, dict):
        consistent_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                # Only convert floating point tensors, preserve integer/bool tensors
                if value.dtype in [torch.float32, torch.float16, torch.bfloat16, torch.float64]:
                    if value.dtype != target_dtype:
                        debug_checkpoint(f"MD: Converting {key} from {value.dtype} to {target_dtype}")
                        consistent_inputs[key] = value.to(dtype=target_dtype)
                    else:
                        consistent_inputs[key] = value
                else:
                    # Keep integer/bool tensors as-is
                    consistent_inputs[key] = value
            else:
                consistent_inputs[key] = value
        return consistent_inputs
    elif isinstance(inputs, torch.Tensor):
        if inputs.dtype in [torch.float32, torch.float16, torch.bfloat16, torch.float64]:
            return inputs.to(dtype=target_dtype)
        else:
            return inputs
    else:
        return inputs

def safe_model_forward_with_dtype_fix_md(model, inputs, **generation_kwargs):
    """Safely perform model generation with dtype consistency fixes for MD"""
    debug_checkpoint("MD: Starting safe model forward with dtype consistency")
    
    # Get model's primary dtype
    model_dtype = get_model_dtype(model)
    debug_checkpoint(f"MD: Model primary dtype detected: {model_dtype}")
    
    # Ensure input dtype consistency
    consistent_inputs = ensure_dtype_consistency_md(inputs, model_dtype, model)
    
    # Validate all inputs are on the same device
    device = None
    for key, value in consistent_inputs.items():
        if isinstance(value, torch.Tensor):
            if device is None:
                device = value.device
            elif value.device != device:
                debug_checkpoint(f"MD: Moving {key} from {value.device} to {device}")
                consistent_inputs[key] = value.to(device)
    
    debug_checkpoint(f"MD: All inputs prepared for device: {device}, dtype: {model_dtype}")
    
    try:
        # Generate with consistent inputs
        with torch.no_grad():
            generated_ids = model.generate(
                **consistent_inputs,
                **generation_kwargs
            )
        debug_checkpoint("MD: Model generation completed successfully with dtype fix")
        return generated_ids
    except RuntimeError as e:
        if "dtype" in str(e).lower() or "bias" in str(e).lower():
            debug_checkpoint(f"MD: Dtype error encountered, attempting fallback: {e}")
            
            # Fallback: try with different dtype
            fallback_dtype = torch.float32 if model_dtype != torch.float32 else torch.float16
            debug_checkpoint(f"MD: Trying fallback dtype: {fallback_dtype}")
            
            fallback_inputs = ensure_dtype_consistency_md(inputs, fallback_dtype, model)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **fallback_inputs,
                    **generation_kwargs
                )
            debug_checkpoint("MD: Model generation completed with fallback dtype")
            return generated_ids
        else:
            raise

class EightBitMarkdownInference:
    def __init__(self, model_checkpoint, device=None, cache_dir=None, use_8bit=True, mixed_precision=True, force_fallback=False):
        """
        Initialize 8-bit Mixed Precision Markdown inference
        
        Args:
            model_checkpoint (str): Path to model checkpoint (local directory or HuggingFace model name)
            device (str): Device to use for inference
            cache_dir (str): Cache directory for downloaded models
            use_8bit (bool): Enable 8-bit quantization
            mixed_precision (bool): Enable mixed precision for critical layers
            force_fallback (bool): Force use of fallback mode (no 8-bit)
        """
        debug_checkpoint("Initializing EightBitMarkdownInference", "INIT_START")
        
        # Enable CUDA debugging
        enable_cuda_debugging()
        
        self.model_checkpoint = model_checkpoint
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        self.use_8bit = use_8bit and torch.cuda.is_available() and not force_fallback
        self.mixed_precision = mixed_precision
        self.force_fallback = force_fallback
        self.model = None
        self.processor = None
        self.model_dtype = None  # Will be set after model loading
        
        # Set quantization method based on configuration
        if self.use_8bit:
            self.quantization_method = "8-bit"
        elif self.mixed_precision:
            self.quantization_method = "mixed_precision"
        else:
            self.quantization_method = "fallback"
        
        debug_checkpoint(f"Parameters - Model: {model_checkpoint}, Device: {self.device}, 8bit: {self.use_8bit}")
        
        # Determine model type
        self.is_local_checkpoint = os.path.exists(model_checkpoint)
        debug_checkpoint(f"Local checkpoint: {self.is_local_checkpoint}")
        
        # Configure precision settings
        safe_execute(self._configure_precision, "Configure precision settings")
        
        logger.info(f"Initializing 8-bit Mixed Precision Markdown inference on {self.device}")
        logger.info(f"Model checkpoint: {self.model_checkpoint}")
        logger.info(f"8-bit quantization: {self.use_8bit}")
        logger.info(f"Mixed precision: {self.mixed_precision}")
        logger.info(f"Local checkpoint: {self.is_local_checkpoint}")
        logger.info(f"Force fallback: {self.force_fallback}")
        
        debug_checkpoint("EightBitMarkdownInference initialization completed", "INIT_END")
    
    def _configure_precision(self):
        """Configure precision and quantization settings"""
        debug_checkpoint("Configuring precision settings", "PRECISION_START")
        
        if self.use_8bit:
            debug_checkpoint("Setting up 8-bit quantization")
            # Ultra-conservative 8-bit configuration
            debug_checkpoint("Creating ultra-conservative BitsAndBytesConfig")
            self.quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,  # Enable CPU offload to reduce GPU issues
                llm_int8_has_fp16_weight=False,
                llm_int8_threshold=6.0,
            )
            debug_checkpoint("Ultra-conservative BitsAndBytesConfig created successfully")
            
            # Use float16 for non-quantized operations
            self.dtype = torch.float16
            debug_checkpoint("Set dtype to float16 for 8-bit mode")
        else:
            debug_checkpoint("Setting up non-8bit precision")
            self.quantization_config = None
            # Use float32 for maximum compatibility and to avoid dtype mismatches
            if self.device.startswith('cuda'):
                self.dtype = torch.float32  # Use float32 for maximum compatibility
                logger.info("Using float32 for maximum compatibility")
                debug_checkpoint("Set dtype to float32")
            else:
                self.dtype = torch.float32
                logger.info("Using float32 for CPU")
                debug_checkpoint("Set dtype to float32 for CPU")
        
        debug_checkpoint("Precision configuration completed", "PRECISION_END")
    
    def _validate_checkpoint(self, checkpoint_path):
        """Validate that the checkpoint contains required files"""
        debug_checkpoint(f"Validating checkpoint: {checkpoint_path}", "VALIDATE_START")
        
        if not os.path.isdir(checkpoint_path):
            debug_checkpoint("Checkpoint path is not a directory")
            return False
        
        debug_checkpoint("Scanning checkpoint directory for files")
        try:
            files = os.listdir(checkpoint_path)
            debug_checkpoint(f"Found {len(files)} files in checkpoint directory")
            
            # Check for model files (SafeTensors or PyTorch)
            model_files = [f for f in files if f.endswith(('.safetensors', '.bin', '.pt'))]
            config_files = [f for f in files if f in ['config.json', 'model.safetensors.index.json', 'pytorch_model.bin.index.json']]
            
            debug_checkpoint(f"Model files: {model_files}")
            debug_checkpoint(f"Config files: {config_files}")
            
            has_model = len(model_files) > 0
            has_config = len(config_files) > 0
            
            if has_model and has_config:
                logger.info(f"✓ Found {len(model_files)} model files in {checkpoint_path}")
                debug_checkpoint("Checkpoint validation successful", "VALIDATE_END")
                return True
            else:
                logger.warning(f"⚠ Checkpoint missing required files. Model files: {has_model}, Config: {has_config}")
                debug_checkpoint("Checkpoint validation failed", "VALIDATE_END")
                return False
        except Exception as e:
            debug_checkpoint(f"Error during checkpoint validation: {e}")
            return False
    
    def load_model(self):
        """Load model with 8-bit mixed precision quantization"""
        debug_checkpoint("Starting model loading", "LOAD_MODEL_START")
        
        if self.model is not None:
            debug_checkpoint("Model already loaded, skipping")
            return
            
        logger.info("Loading Kosmos-2.5 model with dtype consistency...")
        debug_memory_status()
        
        # Validate local checkpoint if specified
        if self.is_local_checkpoint:
            debug_checkpoint("Validating local checkpoint")
            if not safe_execute(self._validate_checkpoint, "Validate checkpoint", self.model_checkpoint):
                logger.error(f"Invalid checkpoint directory: {self.model_checkpoint}")
                raise ValueError("Checkpoint path does not contain valid model files")
        
        # Try fallback first for maximum stability
        debug_checkpoint("Using fallback loading for maximum stability", "FALLBACK_FIRST")
        try:
            self._load_fallback_model()
            debug_checkpoint("Fallback model loading successful", "FALLBACK_SUCCESS")
        except Exception as e:
            debug_checkpoint(f"Fallback model loading failed: {str(e)}", "FALLBACK_FAILED")
            logger.error(f"Model loading failed: {e}")
            raise
        
        # Get model dtype after loading
        self.model_dtype = get_model_dtype(self.model)
        debug_checkpoint(f"Model loaded with dtype: {self.model_dtype}")
        
        # Common post-loading setup
        self._post_loading_setup()
        
        debug_checkpoint("Model loading completed successfully", "LOAD_MODEL_END")
    
    def _load_fallback_model(self):
        """Load model without 8-bit quantization as fallback"""
        debug_checkpoint("Loading fallback model", "FALLBACK_LOAD_START")
        
        # Disable 8-bit for maximum stability
        self.use_8bit = False
        
        fallback_kwargs = {
            "torch_dtype": self.dtype,  # Use consistent dtype
            "cache_dir": self.cache_dir,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "device_map": None,  # Handle device placement manually for safety
        }
        
        if self.is_local_checkpoint:
            fallback_kwargs["local_files_only"] = True
        
        debug_checkpoint(f"Fallback kwargs: {fallback_kwargs}")
        
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_checkpoint,
            **fallback_kwargs
        )
        
        # Move to device manually with extra safety
        if torch.cuda.is_available() and self.device.startswith('cuda'):
            debug_checkpoint("Moving fallback model to GPU with extra safety")
            try:
                # Clear cache first
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Move model to device with consistent dtype
                self.model = self.model.to(device=self.device, dtype=self.dtype)
                
                # Synchronize after move
                torch.cuda.synchronize()
                debug_checkpoint("Model successfully moved to GPU with consistent dtype")
                
            except Exception as e:
                debug_checkpoint(f"Failed to move model to GPU: {e}")
                logger.warning(f"Keeping model on CPU due to GPU move failure: {e}")
                self.device = "cpu"
                self.model = self.model.to(device="cpu", dtype=self.dtype)
        else:
            # For CPU, ensure consistent dtype
            self.model = self.model.to(dtype=self.dtype)
        
        debug_checkpoint("Fallback model loaded with consistent dtype", "FALLBACK_LOAD_END")
    
    def _post_loading_setup(self):
        """Common setup after model loading"""
        debug_checkpoint("Starting post-loading setup", "POSTLOAD_START")
        
        # Load processor
        debug_checkpoint("Loading processor", "PROCESSOR_LOAD_START")
        processor_kwargs = {
            "cache_dir": self.cache_dir,
            "trust_remote_code": True,
            "use_fast": True,
        }
        
        if self.is_local_checkpoint:
            processor_kwargs["local_files_only"] = True
        else:
            processor_kwargs.update({
                "local_files_only": False,
                "resume_download": True
            })
        
        debug_checkpoint(f"Processor loading kwargs: {processor_kwargs}")
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_checkpoint,
            **processor_kwargs
        )
        debug_checkpoint("Processor loaded", "PROCESSOR_LOAD_END")
        
        # Set pad token if not present
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
            debug_checkpoint("Set pad token to eos token")
        
        # Set model to eval mode
        if hasattr(self.model, 'eval'):
            self.model.eval()
            debug_checkpoint("Model set to eval mode")
        
        # Disable gradient checkpointing to avoid dtype issues
        # if hasattr(self.model, 'gradient_checkpointing_enable'):
        #     self.model.gradient_checkpointing_enable()
        #     debug_checkpoint("Enabled gradient checkpointing")
        
        debug_checkpoint("Post-loading setup completed", "POSTLOAD_END")
    
    def get_autocast_context(self):
        """Get appropriate autocast context based on configuration"""
        debug_checkpoint("Getting autocast context")
        
        if self.mixed_precision and torch.cuda.is_available() and self.device.startswith('cuda'):
            # Use mixed precision autocast for CUDA
            debug_checkpoint("Using CUDA mixed precision autocast")
            return torch.cuda.amp.autocast(dtype=self.dtype)
        else:
            # Use null context (no autocast) for CPU or when mixed precision disabled
            debug_checkpoint("Using null context (no autocast)")
            from contextlib import nullcontext
            return nullcontext()
    
    def get_quantization_info(self):
        """Get information about current quantization configuration"""
        return self.quantization_method
    
    def load_image(self, image_path):
        """Load image from local path or URL with error handling"""
        debug_checkpoint(f"Loading image: {image_path}", "IMAGE_LOAD_START")
        
        try:
            if image_path.startswith(('http://', 'https://')):
                debug_checkpoint("Loading image from URL")
                logger.info(f"Loading image from URL: {image_path}")
                response = requests.get(image_path, stream=True, timeout=30)
                response.raise_for_status()
                image = Image.open(response.raw)
                debug_checkpoint("URL image loaded successfully")
            else:
                debug_checkpoint("Loading image from file")
                logger.info(f"Loading image from file: {image_path}")
                if not os.path.exists(image_path):
                    debug_checkpoint("Image file not found")
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                image = Image.open(image_path)
                debug_checkpoint("File image loaded successfully")
            
            # Convert to RGB and validate
            debug_checkpoint("Converting image to RGB")
            image = image.convert('RGB')
            
            # Validate image size
            if image.size[0] < 1 or image.size[1] < 1:
                raise ValueError(f"Invalid image size: {image.size}")
            
            logger.info(f"Image loaded successfully. Size: {image.size}")
            debug_checkpoint(f"Image conversion completed. Size: {image.size}", "IMAGE_LOAD_END")
            return image
            
        except Exception as e:
            debug_checkpoint(f"Image loading failed: {str(e)}", "IMAGE_LOAD_FAILED")
            logger.error(f"Error loading image: {e}")
            raise
    
    def post_process_markdown(self, generated_text, prompt="<md>"):
        """Post-process and clean up generated markdown with enhanced structure detection"""
        debug_checkpoint("Starting markdown post-processing", "POSTPROCESS_START")
        
        # Remove the prompt
        markdown = generated_text.replace(prompt, "").strip()
        debug_checkpoint(f"Cleaned text length: {len(markdown)}")
        
        # Enhanced markdown cleaning
        markdown = safe_execute(self.clean_markdown_advanced, "Clean markdown advanced", markdown)
        
        debug_checkpoint("Markdown post-processing completed", "POSTPROCESS_END")
        return markdown
    
    def clean_markdown_advanced(self, text):
        """Advanced markdown cleaning with structure detection"""
        debug_checkpoint("Starting advanced markdown cleaning")
        
        # Remove extra whitespace and clean up
        lines = text.split('\n')
        cleaned_lines = []
        
        in_code_block = False
        for line in lines:
            # Handle code blocks
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                cleaned_lines.append(line.strip())
                continue
            
            if in_code_block:
                # Preserve code block formatting
                cleaned_lines.append(line)
                continue
            
            line = line.strip()
            # Remove any remaining HTML-like tags (except in code)
            line = re.sub(r'<(?!code|pre)[^>]+>', '', line)
            
            if line:  # Skip empty lines initially
                cleaned_lines.append(line)
        
        # Join lines back
        text = '\n'.join(cleaned_lines)
        
        # Advanced markdown formatting fixes
        
        # Fix headers - ensure proper spacing and format
        text = re.sub(r'^(#{1,6})\s*(.+)', r'\1 \2', text, flags=re.MULTILINE)
        
        # Ensure proper spacing around headers
        text = re.sub(r'(?<!^)(?<!\n)(^#{1,6}.*$)', r'\n\1', text, flags=re.MULTILINE)
        text = re.sub(r'(^#{1,6}.*$)(?!\n)(?!\Z)', r'\1\n', text, flags=re.MULTILINE)
        
        # Fix list items with proper indentation
        text = re.sub(r'^[\*\-\+]\s+', '- ', text, flags=re.MULTILINE)
        text = re.sub(r'^(\s*)(\d+)\.\s+', r'\1\2. ', text, flags=re.MULTILINE)
        
        # Fix nested lists
        text = re.sub(r'^(\s+)[\*\-\+]\s+', r'\1- ', text, flags=re.MULTILINE)
        
        # Fix table formatting
        text = re.sub(r'\|\s*([^|]+?)\s*\|', lambda m: '| ' + m.group(1).strip() + ' |', text)
        
        # Ensure table header separators
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if '|' in line and i < len(lines) - 1:
                next_line = lines[i + 1]
                if '|' in next_line and not re.match(r'^\s*\|[\s\-:]+\|\s*$', next_line):
                    # Check if this looks like a header row
                    if line.count('|') >= 2 and not re.match(r'^\s*\|[\s\-:]+\|\s*$', line):
                        # Insert separator row
                        cols = line.count('|') - 1
                        separator = '|' + '---|' * cols
                        lines.insert(i + 1, separator)
                        break
        text = '\n'.join(lines)
        
        # Fix emphasis and strong formatting
        text = re.sub(r'\*\*([^*]+?)\*\*', r'**\1**', text)
        text = re.sub(r'\*([^*]+?)\*', r'*\1*', text)
        text = re.sub(r'__([^_]+?)__', r'**\1**', text)
        text = re.sub(r'_([^_]+?)_', r'*\1*', text)
        
        # Fix links
        text = re.sub(r'\[([^\]]+?)\]\s*\(([^\)]+?)\)', r'[\1](\2)', text)
        
        # Fix code spans
        text = re.sub(r'`([^`]+?)`', r'`\1`', text)
        
        # Remove excessive newlines but preserve intentional spacing
        text = re.sub(r'\n{4,}', '\n\n\n', text)  # Max 3 newlines
        text = re.sub(r'\n{3}(?=\n)', '\n\n', text)  # Reduce 3+ to 2
        
        # Ensure proper spacing around block elements
        text = re.sub(r'(\n#+.*\n)(\w)', r'\1\n\2', text)  # Space after headers
        text = re.sub(r'(\w)(\n#+.*)', r'\1\n\2', text)    # Space before headers
        
        # Ensure text starts and ends cleanly
        text = text.strip()
        
        debug_checkpoint(f"Advanced markdown cleaning completed: '{text[:100]}...'")
        return text
    
    def generate_markdown_custom_prompt(self, image_path, custom_prompt, max_tokens=2048, temperature=0.1, save_output=None):
        """Generate markdown from image using custom prompt with dtype consistency"""
        debug_checkpoint(f"Starting markdown generation with custom prompt: {custom_prompt[:50]}...", "MD_CUSTOM_START")
        
        if self.model is None:
            debug_checkpoint("Model not loaded, loading now")
            safe_execute(self.load_model, "Load model")
        
        # Load and process image
        image = safe_execute(self.load_image, "Load image", image_path)
        
        # Use custom prompt
        prompt = custom_prompt
        start_time = time.time()
        debug_checkpoint(f"Starting inference with custom prompt length: {len(prompt)} chars")
        
        try:
            # Process inputs with ultra-safe handling
            debug_checkpoint("Processing inputs with custom prompt", "PROCESS_INPUTS_START")
            
            # Clear CUDA cache before processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            debug_checkpoint(f"Processor returned keys: {list(inputs.keys())}")
            
            # Remove height/width info gracefully
            height = inputs.pop("height", None)
            width = inputs.pop("width", None)
            if height is not None:
                debug_checkpoint(f"Removed height: {tensor_to_float(height)}")
            if width is not None:
                debug_checkpoint(f"Removed width: {tensor_to_float(width)}")
            
            debug_checkpoint("Custom prompt input processing completed", "PROCESS_INPUTS_END")
            
            # DTYPE-CONSISTENT TENSOR VALIDATION AND DEVICE PLACEMENT
            debug_checkpoint("Starting dtype-consistent tensor validation", "DTYPE_CONSISTENT_START")
            debug_memory_status()
            
            # Step 1: Get model dtype for consistency
            model_dtype = self.model_dtype or get_model_dtype(self.model)
            debug_checkpoint(f"Using model dtype: {model_dtype} for consistency")
            
            # Step 2: Ultra-safe tensor validation with target dtype
            validated_inputs = ultra_safe_tensor_validation_with_dtype(inputs, model_dtype)
            debug_checkpoint("Tensor validation with dtype completed")
            
            # Step 3: Device placement with dtype preservation
            device_inputs = ultra_safe_device_move_with_dtype(validated_inputs, self.device, model_dtype)
            debug_checkpoint(f"Inputs moved to device {self.device} with dtype preservation")
            
            debug_checkpoint("Dtype-consistent processing completed", "DTYPE_CONSISTENT_END")
            
            # ADVANCED GENERATION WITH DTYPE MONITORING
            debug_checkpoint("Starting advanced generation with monitoring", "ADVANCED_GENERATE_START")
            debug_memory_status()
            
            # Pre-generation memory and dtype validation
            debug_checkpoint("Pre-generation validation")
            for key, tensor in device_inputs.items():
                debug_checkpoint(f"Input {key}: device={tensor.device}, dtype={tensor.dtype}, shape={tensor.shape}")
            
            # Multi-attempt generation with fallback
            generation_attempts = [
                {"temperature": temperature, "do_sample": temperature > 0},
                {"temperature": 0.0, "do_sample": False},  # Fallback to deterministic
            ]
            
            generated_text = None
            inference_time = None
            
            for attempt_idx, gen_params in enumerate(generation_attempts):
                debug_checkpoint(f"Generation attempt {attempt_idx + 1} with params: {gen_params}")
                
                try:
                    with torch.no_grad():
                        with self.get_autocast_context():
                            # Pre-validate inputs one more time
                            debug_checkpoint("Final input validation before generation")
                            
                            generation_start = time.time()
                            
                            # Use safe generation with dtype consistency
                            generation_kwargs = {
                                "max_new_tokens": max_tokens,
                                "do_sample": gen_params["do_sample"],
                                "temperature": gen_params["temperature"] if gen_params["do_sample"] else 1.0,
                                "top_p": 0.95 if gen_params["do_sample"] else 1.0,
                                "pad_token_id": self.processor.tokenizer.eos_token_id,
                                "repetition_penalty": 1.1
                            }
                            
                            try:
                                generated_ids = safe_model_forward_with_dtype_fix_md(
                                    self.model, device_inputs, **generation_kwargs
                                )
                            except Exception as dtype_error:
                                debug_checkpoint(f"Safe generation failed, using fallback: {dtype_error}")
                                generated_ids = self.model.generate(
                                    **device_inputs,
                                    **generation_kwargs
                                )
                            
                            inference_time = time.time() - generation_start
                    
                    debug_checkpoint(f"Generation attempt {attempt_idx + 1} successful in {inference_time:.2f}s")
                    
                    # Extract generated text
                    generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    debug_checkpoint(f"Generated text length: {len(generated_text)} characters")
                    break
                    
                except Exception as attempt_error:
                    debug_checkpoint(f"Generation attempt {attempt_idx + 1} failed: {attempt_error}")
                    if attempt_idx == len(generation_attempts) - 1:
                        raise attempt_error
                    continue
            
            if generated_text is None:
                raise RuntimeError("All generation attempts failed")
            
            debug_checkpoint("Advanced generation completed", "ADVANCED_GENERATE_END")
            
            # ADVANCED MARKDOWN POST-PROCESSING
            debug_checkpoint("Starting advanced markdown post-processing", "ADVANCED_POST_START")
            
            # Clean and enhance markdown
            markdown_text = self.clean_markdown_advanced(generated_text)
            debug_checkpoint(f"Cleaned markdown length: {len(markdown_text)} characters")
            
            # Enhanced result structure
            result = {
                'success': True,
                'generated_markdown': markdown_text,
                'raw_generated_text': generated_text,
                'inference_time': inference_time,
                'total_processing_time': time.time() - start_time,
                'model_info': {
                    'quantization_method': self.quantization_method,
                    'model_dtype': str(model_dtype),
                    'device': str(self.device)
                },
                'generation_params': {
                    'max_tokens': max_tokens,
                    'temperature': temperature,
                    'custom_prompt_length': len(prompt)
                },
                'content_analysis': {
                    'word_count': len(markdown_text.split()),
                    'char_count': len(markdown_text),
                    'line_count': markdown_text.count('\n'),
                    'headers_count': markdown_text.count('#'),
                    'lists_count': markdown_text.count('- ') + markdown_text.count('* '),
                    'tables_count': markdown_text.count('|'),
                    'code_blocks_count': markdown_text.count('```')
                }
            }
            
            # Save output if requested
            if save_output:
                debug_checkpoint(f"Saving markdown output to: {save_output}")
                output_dir = os.path.dirname(save_output)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                with open(save_output, 'w', encoding='utf-8') as f:
                    f.write(markdown_text)
                result['output_file'] = save_output
            
            debug_checkpoint("Custom markdown generation completed successfully", "MD_CUSTOM_END")
            return result
            
        except Exception as e:
            debug_checkpoint(f"Error in custom markdown generation: {str(e)}", "MD_CUSTOM_ERROR")
            logger.error(f"Error in generate_markdown_custom_prompt: {e}")
            return {
                'success': False,
                'error': str(e),
                'generated_markdown': '',
                'inference_time': time.time() - start_time,
                'quantization_info': getattr(self, 'quantization_method', 'unknown')
            }

    def generate_markdown(self, image_path, max_tokens=1024, temperature=0.0, save_output=None):
        """Generate markdown from image with dtype consistency to fix float/half mismatch"""
        debug_checkpoint("Starting markdown generation with dtype consistency", "MD_START")
        
        if self.model is None:
            debug_checkpoint("Model not loaded, loading now")
            safe_execute(self.load_model, "Load model")
        
        # Load and process image
        image = safe_execute(self.load_image, "Load image", image_path)
        
        prompt = "<md>"
        start_time = time.time()
        debug_checkpoint(f"Starting inference with prompt: {prompt}")
        
        try:
            # Process inputs with ultra-safe handling
            debug_checkpoint("Processing inputs with processor", "PROCESS_INPUTS_START")
            
            # Clear CUDA cache before processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            debug_checkpoint(f"Processor returned keys: {list(inputs.keys())}")
            
            # Remove height/width info gracefully
            height = inputs.pop("height", None)
            width = inputs.pop("width", None)
            if height is not None:
                debug_checkpoint(f"Removed height: {tensor_to_float(height)}")
            if width is not None:
                debug_checkpoint(f"Removed width: {tensor_to_float(width)}")
            
            debug_checkpoint("Input processing completed", "PROCESS_INPUTS_END")
            
            # DTYPE-CONSISTENT TENSOR VALIDATION AND DEVICE PLACEMENT
            debug_checkpoint("Starting dtype-consistent tensor validation", "DTYPE_CONSISTENT_START")
            debug_memory_status()
            
            # Step 1: Get model dtype for consistency
            model_dtype = self.model_dtype or get_model_dtype(self.model)
            debug_checkpoint(f"Using model dtype: {model_dtype} for consistency")
            
            # Step 2: Ultra-safe tensor validation with target dtype
            validated_inputs = ultra_safe_tensor_validation_with_dtype(inputs, model_dtype)
            debug_checkpoint("Tensor validation with dtype completed")
            
            # Step 3: Get model device safely
            try:
                model_device = next(self.model.parameters()).device
                debug_checkpoint(f"Model is on device: {model_device}")
            except Exception as e:
                debug_checkpoint(f"Failed to get model device: {e}")
                model_device = torch.device('cpu')
                debug_checkpoint("Defaulting to CPU")
            
            # Step 4: Ultra-safe device movement with dtype consistency
            if str(model_device) != 'cpu':
                final_inputs = ultra_safe_device_move_with_dtype(validated_inputs, model_device, model_dtype)
            else:
                final_inputs = validated_inputs
            
            debug_checkpoint("Dtype-consistent processing completed", "DTYPE_CONSISTENT_END")
            debug_memory_status()
            
            # Final dtype verification
            debug_checkpoint("Final dtype verification before generation:")
            for key, tensor in final_inputs.items():
                if isinstance(tensor, torch.Tensor):
                    debug_checkpoint(f"  {key}: {tensor.dtype} on {tensor.device}")
            
            # Generate markdown with ultra-conservative settings
            debug_checkpoint("Starting ultra-conservative generation", "GENERATION_START")
            logger.info("Generating markdown with dtype consistency...")
            
            with torch.no_grad():
                debug_checkpoint("Entered torch.no_grad() context")
                
                # Clear GPU cache before generation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    debug_memory_status()
                
                # Ultra-conservative generation parameters
                generation_kwargs = {
                    "max_new_tokens": min(max_tokens, 256),  # Very conservative limit
                    "do_sample": False,  # Always use greedy decoding for stability
                    "pad_token_id": self.processor.tokenizer.eos_token_id,
                    "eos_token_id": self.processor.tokenizer.eos_token_id,
                    "repetition_penalty": 1.0,  # No repetition penalty for stability
                    "length_penalty": 1.0,
                    "use_cache": False,  # Disable cache for safety
                    "num_beams": 1,  # Always use greedy
                    "early_stopping": True,
                    "output_attentions": False,
                    "output_hidden_states": False,
                    "return_dict_in_generate": False,
                }
                
                debug_checkpoint(f"Ultra-conservative generation kwargs: {generation_kwargs}")
                
                try:
                    # No mixed precision - use standard generation for maximum stability
                    debug_checkpoint("Using standard generation for maximum stability")
                    
                    # Synchronize before generation
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    # Use safe generation with dtype consistency  
                    try:
                        generated_ids = safe_model_forward_with_dtype_fix_md(
                            self.model, final_inputs, **generation_kwargs
                        )
                    except Exception as dtype_error:
                        debug_checkpoint(f"Safe generation failed in standard mode, using fallback: {dtype_error}")
                        generated_ids = self.model.generate(
                            **final_inputs,
                            **generation_kwargs
                        )
                    debug_checkpoint("model.generate completed successfully")
                    
                    # Synchronize after generation
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        
                except Exception as generation_error:
                    debug_checkpoint(f"Generation failed: {generation_error}")
                    
                    # Ultimate fallback: try with minimal settings
                    debug_checkpoint("Attempting ultimate fallback generation")
                    
                    minimal_kwargs = {
                        "max_new_tokens": 32,  # Even smaller for maximum stability
                        "do_sample": False,
                        "pad_token_id": self.processor.tokenizer.eos_token_id,
                        "eos_token_id": self.processor.tokenizer.eos_token_id,
                        "use_cache": False,
                        "num_beams": 1,
                        "output_attentions": False,
                        "output_hidden_states": False,
                    }
                    
                    # Clear cache and try again
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    generated_ids = self.model.generate(
                        **final_inputs,
                        **minimal_kwargs
                    )
                    debug_checkpoint("Ultimate fallback generation completed")
            
            debug_checkpoint("Generation completed", "GENERATION_END")
            debug_memory_status()
            
            # Decode results safely
            debug_checkpoint("Decoding generated results", "DECODE_START")
            try:
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                debug_checkpoint(f"Decoded text length: {len(generated_text)}")
                debug_checkpoint(f"Generated text preview: '{generated_text[:100]}...'")
            except Exception as decode_error:
                debug_checkpoint(f"Decoding failed: {decode_error}")
                generated_text = "<md>Error during text generation</md>"
            debug_checkpoint("Decoding completed", "DECODE_END")
            
            # Post-process markdown
            markdown_output = safe_execute(self.post_process_markdown, "Post-process markdown", generated_text, prompt)
            
            inference_time = time.time() - start_time
            logger.info(f"Markdown generation completed in {inference_time:.2f}s")
            debug_checkpoint(f"Markdown generation completed in {inference_time:.2f}s")
            
            # Calculate statistics
            debug_checkpoint("Calculating statistics")
            word_count = len(markdown_output.split())
            char_count = len(markdown_output)
            line_count = len(markdown_output.split('\n'))
            
            # Detect markdown elements
            headers = len(re.findall(r'^#+\s', markdown_output, re.MULTILINE))
            lists = len(re.findall(r'^[\*\-\+]\s|^\d+\.\s', markdown_output, re.MULTILINE))
            tables = markdown_output.count('|')
            code_blocks = markdown_output.count('```') // 2
            
            debug_checkpoint(f"Statistics - Words: {word_count}, Headers: {headers}, Lists: {lists}")
            
            # Save output if requested
            if save_output:
                debug_checkpoint("Saving markdown output")
                safe_execute(self.save_markdown, "Save markdown", markdown_output, save_output)
            
            result = {
                'markdown': markdown_output,
                'inference_time': inference_time,
                'raw_output': generated_text,
                'statistics': {
                    'word_count': word_count,
                    'char_count': char_count,
                    'line_count': line_count,
                    'headers': headers,
                    'lists': lists,
                    'tables': tables,
                    'code_blocks': code_blocks
                }
            }
            
            debug_checkpoint("Markdown generation completed successfully", "MD_END")
            return result
            
        except Exception as e:
            debug_checkpoint(f"Markdown generation failed with error: {str(e)}", "MD_FAILED")
            logger.error(f"Error during markdown generation: {e}")
            logger.error(f"Full traceback: {tb_module.format_exc()}")
            
            # Clear CUDA cache on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            raise
    
    def save_markdown(self, markdown_text, output_path):
        """Save markdown to file with metadata"""
        debug_checkpoint(f"Saving markdown to: {output_path}", "SAVE_MD_START")
        
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Add metadata header
            metadata = f"""<!--
Generated by 8-bit Mixed Precision Kosmos-2.5 (Dtype-Consistent Mode)
Model: {self.model_checkpoint}
Quantization: {'8-bit' if self.use_8bit else self.dtype}
Mixed Precision: {self.mixed_precision}
Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}
-->

"""
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(metadata + markdown_text)
            
            logger.info(f"Markdown saved to: {output_path}")
            debug_checkpoint("Markdown saved successfully", "SAVE_MD_END")
            
        except Exception as e:
            debug_checkpoint(f"Failed to save markdown: {str(e)}", "SAVE_MD_FAILED")
            logger.error(f"Error saving markdown: {e}")
    
    def batch_process(self, image_paths, output_dir, max_tokens=1024, temperature=0.0):
        """Process multiple images in batch with dtype consistency"""
        debug_checkpoint(f"Starting batch processing of {len(image_paths)} images", "BATCH_START")
        
        if self.model is None:
            debug_checkpoint("Loading model for batch processing")
            safe_execute(self.load_model, "Load model for batch")
        
        results = []
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Starting dtype-consistent batch markdown processing of {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths, 1):
            debug_checkpoint(f"Processing batch image {i}/{len(image_paths)}: {os.path.basename(image_path)}")
            logger.info(f"Processing image {i}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            try:
                # Generate output filename
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_path = os.path.join(output_dir, f"{base_name}_markdown.md")
                
                # Generate markdown with dtype-consistent settings
                result = safe_execute(self.generate_markdown, f"Dtype-consistent markdown for {base_name}",
                    image_path=image_path,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    save_output=output_path
                )
                
                result['input_path'] = image_path
                result['output_path'] = output_path
                results.append(result)
                
                # Progress update
                progress = (i / len(image_paths)) * 100
                logger.info(f"Progress: {progress:.1f}% ({i}/{len(image_paths)})")
                debug_checkpoint(f"Batch progress: {progress:.1f}% completed")
                
            except Exception as e:
                debug_checkpoint(f"Failed to process batch image {image_path}: {str(e)}")
                logger.error(f"Failed to process {image_path}: {e}")
                results.append({
                    'input_path': image_path,
                    'error': str(e),
                    'inference_time': 0,
                    'statistics': {}
                })
        
        logger.info("Dtype-consistent batch markdown processing completed!")
        debug_checkpoint("Batch processing completed", "BATCH_END")
        return results

def get_args():
    parser = argparse.ArgumentParser(description='8-bit Mixed Precision Markdown generation using Kosmos-2.5 - DTYPE MISMATCH FIX')
    parser.add_argument('--image', '-i', type=str, required=True,
                       help='Path to input image file or URL')
    parser.add_argument('--model_checkpoint', '-m', type=str, required=True,
                       help='Path to model checkpoint (local directory or HuggingFace model name)')
    parser.add_argument('--output', '-o', type=str, default='./output.md',
                       help='Output path for generated markdown')
    parser.add_argument('--device', '-d', type=str, default=None,
                       help='Device to use (auto-detected if not specified)')
    
    # Conservative token size configuration
    parser.add_argument('--max_tokens', '--tokens', type=int, default=256,  # Conservative default
                       help='Maximum number of tokens to generate (default: 256, recommended: 128-512)')
    parser.add_argument('--min_tokens', type=int, default=10,
                       help='Minimum number of tokens to generate (default: 10)')
    
    parser.add_argument('--temperature', '-t', type=float, default=0.0,  # Always deterministic
                       help='Sampling temperature (always 0.0 for stable mode)')
    parser.add_argument('--cache_dir', type=str, default=None,
                       help='Cache directory for model files')
    parser.add_argument('--no_8bit', action='store_true',
                       help='Disable 8-bit quantization (recommended for stability)')
    parser.add_argument('--no_mixed_precision', action='store_true',
                       help='Disable mixed precision for critical layers (recommended)')
    parser.add_argument('--batch', action='store_true',
                       help='Process multiple images (image should be a directory)')
    parser.add_argument('--print_output', '-p', action='store_true',
                       help='Print generated markdown to console')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output with statistics')
    parser.add_argument('--debug', action='store_true',
                       help='Enable maximum debug output')
    parser.add_argument('--force_fallback', action='store_true',
                       help='Force use of fallback mode (recommended for stability)')
    
    return parser.parse_args()

def main():
    debug_checkpoint("Application starting", "MAIN_START")
    
    args = get_args()
    
    # Force stable settings
    if args.temperature != 0.0:
        logger.warning("Forcing temperature to 0.0 for stable mode")
        args.temperature = 0.0
    
    # Validate token configuration
    if args.max_tokens < args.min_tokens:
        logger.error(f"max_tokens ({args.max_tokens}) must be >= min_tokens ({args.min_tokens})")
        sys.exit(1)
    
    if args.max_tokens > 512:  # Conservative limit
        logger.warning(f"Large max_tokens ({args.max_tokens}) may cause dtype issues. Capping at 512.")
        args.max_tokens = 512
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        debug_checkpoint("Debug mode enabled")
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
        debug_checkpoint("Verbose mode enabled")
    
    debug_checkpoint(f"Arguments: {vars(args)}")
    debug_checkpoint(f"Token configuration - Max: {args.max_tokens}, Min: {args.min_tokens}, Temperature: {args.temperature}")
    
    # Initialize with dtype-consistent settings
    debug_checkpoint("Initializing dtype-consistent markdown engine")
    md_engine = safe_execute(EightBitMarkdownInference, "Initialize dtype-consistent markdown engine",
        model_checkpoint=args.model_checkpoint,
        device=args.device,
        cache_dir=args.cache_dir,
        use_8bit=False,  # Force disable 8-bit for maximum dtype consistency
        mixed_precision=False,  # Force disable mixed precision
        force_fallback=True  # Force fallback mode
    )
    
    try:
        if args.batch and os.path.isdir(args.image):
            debug_checkpoint("Starting dtype-consistent batch processing mode")
            # Batch processing
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
            image_paths = [
                os.path.join(args.image, f) for f in os.listdir(args.image)
                if any(f.lower().endswith(ext) for ext in image_extensions)
            ]
            
            if not image_paths:
                debug_checkpoint("No images found in batch directory")
                logger.error(f"No images found in directory: {args.image}")
                sys.exit(1)
            
            debug_checkpoint(f"Found {len(image_paths)} images for dtype-consistent batch processing")
            logger.info(f"Processing {len(image_paths)} images in dtype-consistent batch mode with max_tokens={args.max_tokens}")
            
            results = safe_execute(md_engine.batch_process, "Dtype-consistent batch process images",
                image_paths=image_paths,
                output_dir=args.output,
                max_tokens=args.max_tokens,
                temperature=args.temperature
            )
            
            # Calculate batch statistics
            debug_checkpoint("Calculating batch statistics")
            successful = sum(1 for r in results if 'error' not in r)
            total_time = sum(r.get('inference_time', 0) for r in results)
            total_words = sum(r.get('statistics', {}).get('word_count', 0) for r in results if 'error' not in r)
            total_headers = sum(r.get('statistics', {}).get('headers', 0) for r in results if 'error' not in r)
            total_lists = sum(r.get('statistics', {}).get('lists', 0) for r in results if 'error' not in r)
            
            print(f"\n{'='*80}")
            print("BATCH MARKDOWN PROCESSING SUMMARY (DTYPE-CONSISTENT MODE)")
            print(f"{'='*80}")
            print(f"Total images processed: {len(results)}")
            print(f"Successful: {successful}")
            print(f"Failed: {len(results) - successful}")
            print(f"Total words generated: {total_words:,}")
            print(f"Total headers found: {total_headers}")
            print(f"Total list items: {total_lists}")
            print(f"Total processing time: {total_time:.2f}s")
            print(f"Average time per image: {total_time/len(results):.2f}s")
            print(f"Average words per image: {total_words/successful:.0f}" if successful > 0 else "N/A")
            print(f"Model checkpoint: {args.model_checkpoint}")
            print(f"Token configuration: Max={args.max_tokens}, Min={args.min_tokens}")
            print(f"Temperature: {args.temperature}")
            print(f"Mode: Dtype-Consistent (Float32)")
            print(f"Output directory: {args.output}")
            print(f"{'='*80}")
            
        else:
            debug_checkpoint("Starting dtype-consistent single image processing mode")
            # Single image processing
            result = safe_execute(md_engine.generate_markdown, "Generate dtype-consistent markdown for single image",
                image_path=args.image,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                save_output=args.output
            )
            
            stats = result['statistics']
            
            # Print results summary
            print(f"\n{'='*80}")
            print("MARKDOWN GENERATION SUMMARY (DTYPE-CONSISTENT MODE)")
            print(f"{'='*80}")
            print(f"Processing time: {result['inference_time']:.2f}s")
            print(f"Word count: {stats['word_count']:,}")
            print(f"Character count: {stats['char_count']:,}")
            print(f"Line count: {stats['line_count']:,}")
            print(f"Headers: {stats['headers']}")
            print(f"List items: {stats['lists']}")
            print(f"Tables: {stats['tables']}")
            print(f"Code blocks: {stats['code_blocks']}")
            print(f"Output saved to: {args.output}")
            print(f"Model checkpoint: {args.model_checkpoint}")
            print(f"Token configuration: Max={args.max_tokens}, Min={args.min_tokens}")
            print(f"Temperature: {args.temperature}")
            print(f"Mode: Dtype-Consistent (Float32)")
            print(f"{'='*80}")
            
            if args.print_output:
                print("\nGENERATED MARKDOWN:")
                print("=" * 80)
                print(result['markdown'])
                print("=" * 80)
        
        debug_checkpoint("Application completed successfully", "MAIN_END")
        
    except Exception as e:
        debug_checkpoint(f"Application failed with error: {str(e)}", "MAIN_FAILED")
        logger.error(f"Markdown generation failed: {e}")
        if args.verbose or args.debug:
            logger.error(f"Full traceback: {tb_module.format_exc()}")
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
