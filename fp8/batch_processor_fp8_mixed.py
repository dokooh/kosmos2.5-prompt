#!/usr/bin/env python3
"""
Enhanced Batch Processor for Kosmos-2.5 8-bit Mixed Precision with Complete Device Management

This module provides comprehensive batch processing capabilities for both OCR and Markdown generation
using the optimized 8-bit mixed precision modules (ocr_fp8_mixed.py and md_fp8_mixed.py).

Features:
- 8-bit mixed precision quantization with automatic fallback
- Complete device placement management
- Enhanced debugging and error handling
- Parallel and sequential processing modes
- Comprehensive progress tracking and reporting
- SafeTensors format support
- Memory optimization with gradient checkpointing
- Robust error recovery and fallback mechanisms
"""

import os
import sys
import argparse
import json
import time
import logging
import traceback
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import threading
import queue
import psutil

# Import the 8-bit mixed precision inference modules
try:
    from ocr_fp8_mixed import EightBitOCRInference, debug_checkpoint, debug_memory_status
    from md_fp8_mixed import EightBitMarkdownInference
except ImportError as e:
    print(f"Error importing 8-bit mixed precision modules: {e}")
    print("Make sure ocr_fp8_mixed.py and md_fp8_mixed.py are in the same directory")
    sys.exit(1)

# Enhanced logging setup with more detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('batch_fp8_debug.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

class EightBitBatchProcessor:
    """Enhanced batch processor for 8-bit mixed precision Kosmos-2.5 models"""
    
    def __init__(self, 
                 model_checkpoint: str,
                 device: str = None,
                 cache_dir: str = None,
                 use_8bit: bool = True,
                 mixed_precision: bool = True,
                 force_fallback: bool = False,
                 max_workers: int = 1,
                 use_process_pool: bool = False):
        """
        Initialize the 8-bit mixed precision batch processor
        
        Args:
            model_checkpoint (str): Path to model checkpoint (local directory or HuggingFace model name)
            device (str): Device to use for inference
            cache_dir (str): Cache directory for models
            use_8bit (bool): Enable 8-bit quantization
            mixed_precision (bool): Enable mixed precision for critical layers
            force_fallback (bool): Force use of fallback mode (no 8-bit)
            max_workers (int): Maximum number of parallel workers
            use_process_pool (bool): Use process pool instead of thread pool for true parallelism
        """
        debug_checkpoint("Initializing EightBitBatchProcessor", "BATCH_INIT_START")
        
        self.model_checkpoint = model_checkpoint
        self.device = device
        self.cache_dir = cache_dir
        self.use_8bit = use_8bit
        self.mixed_precision = mixed_precision
        self.force_fallback = force_fallback
        self.max_workers = max_workers
        self.use_process_pool = use_process_pool
        
        # Validate model checkpoint
        self.is_local_checkpoint = os.path.exists(model_checkpoint)
        
        # Thread-local storage for inference engines
        self._local = threading.local()
        
        # Progress tracking
        self.progress_queue = queue.Queue()
        self.total_tasks = 0
        self.completed_tasks = 0
        
        # Performance metrics
        self.start_time = None
        self.system_info = self._get_system_info()
        
        logger.info(f"Initialized 8-bit Mixed Precision Batch Processor")
        logger.info(f"Model checkpoint: {self.model_checkpoint}")
        logger.info(f"Local checkpoint: {self.is_local_checkpoint}")
        logger.info(f"Device: {self.device}")
        logger.info(f"8-bit quantization: {self.use_8bit}")
        logger.info(f"Mixed precision: {self.mixed_precision}")
        logger.info(f"Force fallback: {self.force_fallback}")
        logger.info(f"Max workers: {self.max_workers}")
        logger.info(f"Use process pool: {self.use_process_pool}")
        
        debug_checkpoint("EightBitBatchProcessor initialization completed", "BATCH_INIT_END")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for performance analysis"""
        try:
            import torch
            return {
                'cpu_count': psutil.cpu_count(logical=False),
                'cpu_count_logical': psutil.cpu_count(logical=True),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'cuda_available': torch.cuda.is_available(),
                'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'cuda_memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
            }
        except Exception as e:
            debug_checkpoint(f"Failed to get system info: {e}")
            return {}
    
    def _get_ocr_engine(self) -> EightBitOCRInference:
        """Get thread-local OCR engine with proper initialization"""
        if not hasattr(self._local, 'ocr_engine') or self._local.ocr_engine is None:
            debug_checkpoint("Creating thread-local OCR engine")
            try:
                self._local.ocr_engine = EightBitOCRInference(
                    model_checkpoint=self.model_checkpoint,
                    device=self.device,
                    cache_dir=self.cache_dir,
                    use_8bit=self.use_8bit,
                    mixed_precision=self.mixed_precision,
                    force_fallback=self.force_fallback
                )
                debug_checkpoint("Thread-local OCR engine created successfully")
            except Exception as e:
                debug_checkpoint(f"Failed to create OCR engine: {e}")
                raise
        return self._local.ocr_engine
    
    def _get_md_engine(self) -> EightBitMarkdownInference:
        """Get thread-local Markdown engine with proper initialization"""
        if not hasattr(self._local, 'md_engine') or self._local.md_engine is None:
            debug_checkpoint("Creating thread-local Markdown engine")
            try:
                self._local.md_engine = EightBitMarkdownInference(
                    model_checkpoint=self.model_checkpoint,
                    device=self.device,
                    cache_dir=self.cache_dir,
                    use_8bit=self.use_8bit,
                    mixed_precision=self.mixed_precision,
                    force_fallback=self.force_fallback
                )
                debug_checkpoint("Thread-local Markdown engine created successfully")
            except Exception as e:
                debug_checkpoint(f"Failed to create Markdown engine: {e}")
                raise
        return self._local.md_engine
    
    def find_images(self, input_folder: str, extensions: List[str]) -> List[str]:
        """Find all image files in the input folder with enhanced filtering"""
        debug_checkpoint(f"Scanning for images in: {input_folder}", "FIND_IMAGES_START")
        
        image_files = []
        input_path = Path(input_folder)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input folder not found: {input_folder}")
        
        if not input_path.is_dir():
            raise ValueError(f"Input path is not a directory: {input_folder}")
        
        # Search for images with case-insensitive extensions
        for ext in extensions:
            # Search for both lowercase and uppercase extensions
            for case_ext in [ext.lower(), ext.upper()]:
                pattern = f"*{case_ext}"
                found_files = list(input_path.glob(pattern))
                image_files.extend(found_files)
        
        # Remove duplicates and convert to strings
        image_files = list(set([str(f) for f in image_files]))
        image_files.sort()
        
        # Validate image files
        valid_images = []
        for img_file in image_files:
            try:
                from PIL import Image
                with Image.open(img_file) as img:
                    # Quick validation
                    img.verify()
                valid_images.append(img_file)
            except Exception as e:
                logger.warning(f"Skipping invalid image {Path(img_file).name}: {e}")
        
        logger.info(f"Found {len(valid_images)} valid images in {input_folder}")
        debug_checkpoint(f"Image scanning completed. Found {len(valid_images)} valid images", "FIND_IMAGES_END")
        
        return valid_images
    
    def create_output_structure(self, output_folder: str) -> Dict[str, str]:
        """Create comprehensive output folder structure with detailed organization"""
        debug_checkpoint(f"Creating output structure: {output_folder}", "OUTPUT_STRUCT_START")
        
        output_path = Path(output_folder)
        
        # Create main output folder
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create comprehensive subfolders
        folders = {
            'markdown': output_path / "markdown_output",
            'ocr_images': output_path / "ocr_annotated_images", 
            'ocr_text': output_path / "ocr_text_results",
            'logs': output_path / "processing_logs",
            'reports': output_path / "analysis_reports",
            'debug': output_path / "debug_info",
            'errors': output_path / "error_logs",
            'performance': output_path / "performance_metrics"
        }
        
        for folder_name, folder_path in folders.items():
            folder_path.mkdir(exist_ok=True)
            debug_checkpoint(f"Created folder: {folder_name} -> {folder_path}")
        
        # Convert to strings for easier handling
        folder_paths = {k: str(v) for k, v in folders.items()}
        
        logger.info("Created comprehensive output folder structure:")
        for name, path in folder_paths.items():
            logger.info(f"  {name}: {path}")
        
        debug_checkpoint("Output structure creation completed", "OUTPUT_STRUCT_END")
        return folder_paths
    
    def process_single_image_ocr(self, 
                                image_path: str, 
                                output_folders: Dict[str, str],
                                max_tokens: int,
                                task_id: int = 0) -> Dict[str, Any]:
        """Process a single image for OCR with comprehensive error handling"""
        image_name = Path(image_path).stem
        start_time = time.time()
        
        debug_checkpoint(f"Starting OCR processing for: {image_name}", f"OCR_TASK_{task_id}_START")
        
        try:
            # Get thread-local OCR engine
            ocr_engine = self._get_ocr_engine()
            
            # Generate output paths
            output_image = Path(output_folders['ocr_images']) / f"{image_name}_ocr_annotated.png"
            output_text = Path(output_folders['ocr_text']) / f"{image_name}_ocr_results.txt"
            
            logger.info(f"[Task {task_id}] Processing OCR for: {Path(image_path).name}")
            debug_memory_status()
            
            # Perform OCR with comprehensive error handling
            result = ocr_engine.perform_ocr(
                image_path=image_path,
                max_tokens=max_tokens,
                save_image=str(output_image),
                save_text=str(output_text)
            )
            
            processing_time = time.time() - start_time
            
            # Extract comprehensive statistics
            stats = result.get('statistics', {})
            
            success_result = {
                'success': True,
                'task_id': task_id,
                'image_path': image_path,
                'image_name': image_name,
                'output_image': str(output_image),
                'output_text': str(output_text),
                'text_regions': stats.get('total_regions', 0),
                'total_text_length': stats.get('total_text_length', 0),
                'avg_confidence': stats.get('avg_confidence', 0.0),
                'image_size': stats.get('image_size', [0, 0]),
                'processing_time': processing_time,
                'inference_time': result.get('inference_time', 0),
                'quantization_used': '8-bit' if ocr_engine.use_8bit else 'FP16/BF16',
                'results': result.get('results', []),
                'raw_output_length': len(result.get('raw_output', ''))
            }
            
            debug_checkpoint(f"OCR processing completed successfully for: {image_name}", f"OCR_TASK_{task_id}_SUCCESS")
            return success_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"OCR failed for {Path(image_path).name}: {str(e)}"
            logger.error(error_msg)
            
            # Save detailed error information
            error_file = Path(output_folders['errors']) / f"{image_name}_ocr_error.txt"
            try:
                with open(error_file, 'w', encoding='utf-8') as f:
                    f.write(f"OCR Processing Error Report\n")
                    f.write(f"{'='*50}\n")
                    f.write(f"Image: {image_path}\n")
                    f.write(f"Error: {str(e)}\n")
                    f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Processing time: {processing_time:.2f}s\n\n")
                    f.write(f"Full traceback:\n")
                    f.write(traceback.format_exc())
            except Exception:
                pass  # Don't fail on error logging
            
            error_result = {
                'success': False,
                'task_id': task_id,
                'image_path': image_path,
                'image_name': image_name,
                'error': str(e),
                'error_type': type(e).__name__,
                'processing_time': processing_time,
                'error_file': str(error_file)
            }
            
            debug_checkpoint(f"OCR processing failed for: {image_name}", f"OCR_TASK_{task_id}_FAILED")
            return error_result
    
    def process_single_image_markdown(self, 
                                    image_path: str, 
                                    output_folders: Dict[str, str],
                                    max_tokens: int,
                                    temperature: float,
                                    task_id: int = 0) -> Dict[str, Any]:
        """Process a single image for Markdown generation with comprehensive error handling"""
        image_name = Path(image_path).stem
        start_time = time.time()
        
        debug_checkpoint(f"Starting Markdown processing for: {image_name}", f"MD_TASK_{task_id}_START")
        
        try:
            # Get thread-local Markdown engine
            md_engine = self._get_md_engine()
            
            # Generate output path
            output_file = Path(output_folders['markdown']) / f"{image_name}_document.md"
            
            logger.info(f"[Task {task_id}] Processing Markdown for: {Path(image_path).name}")
            debug_memory_status()
            
            # Generate markdown with comprehensive error handling
            result = md_engine.generate_markdown(
                image_path=image_path,
                max_tokens=max_tokens,
                temperature=temperature,
                save_output=str(output_file)
            )
            
            processing_time = time.time() - start_time
            
            # Extract comprehensive statistics
            stats = result.get('statistics', {})
            
            success_result = {
                'success': True,
                'task_id': task_id,
                'image_path': image_path,
                'image_name': image_name,
                'output_file': str(output_file),
                'word_count': stats.get('word_count', 0),
                'char_count': stats.get('char_count', 0),
                'line_count': stats.get('line_count', 0),
                'headers': stats.get('headers', 0),
                'lists': stats.get('lists', 0),
                'tables': stats.get('tables', 0),
                'code_blocks': stats.get('code_blocks', 0),
                'processing_time': processing_time,
                'inference_time': result.get('inference_time', 0),
                'quantization_used': '8-bit' if md_engine.use_8bit else 'FP16/BF16',
                'markdown_length': len(result.get('markdown', '')),
                'raw_output_length': len(result.get('raw_output', ''))
            }
            
            debug_checkpoint(f"Markdown processing completed successfully for: {image_name}", f"MD_TASK_{task_id}_SUCCESS")
            return success_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Markdown generation failed for {Path(image_path).name}: {str(e)}"
            logger.error(error_msg)
            
            # Save detailed error information
            error_file = Path(output_folders['errors']) / f"{image_name}_md_error.txt"
            try:
                with open(error_file, 'w', encoding='utf-8') as f:
                    f.write(f"Markdown Generation Error Report\n")
                    f.write(f"{'='*50}\n")
                    f.write(f"Image: {image_path}\n")
                    f.write(f"Error: {str(e)}\n")
                    f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Processing time: {processing_time:.2f}s\n\n")
                    f.write(f"Full traceback:\n")
                    f.write(traceback.format_exc())
            except Exception:
                pass  # Don't fail on error logging
            
            error_result = {
                'success': False,
                'task_id': task_id,
                'image_path': image_path,
                'image_name': image_name,
                'error': str(e),
                'error_type': type(e).__name__,
                'processing_time': processing_time,
                'error_file': str(error_file)
            }
            
            debug_checkpoint(f"Markdown processing failed for: {image_name}", f"MD_TASK_{task_id}_FAILED")
            return error_result
    
    def process_batch_sequential(self, 
                               image_paths: List[str],
                               output_folders: Dict[str, str],
                               process_ocr: bool,
                               process_md: bool,
                               ocr_max_tokens: int,
                               md_max_tokens: int,
                               temperature: float) -> Dict[str, Any]:
        """Process images sequentially with detailed progress tracking"""
        debug_checkpoint("Starting sequential batch processing", "BATCH_SEQ_START")
        
        results = {
            'ocr_results': [],
            'md_results': [],
            'total_images': len(image_paths),
            'start_time': time.time(),
            'processing_mode': 'sequential'
        }
        
        self.total_tasks = len(image_paths) * (int(process_ocr) + int(process_md))
        self.completed_tasks = 0
        
        for i, image_path in enumerate(image_paths, 1):
            image_name = Path(image_path).name
            logger.info(f"Processing image {i}/{len(image_paths)}: {image_name}")
            
            # Process OCR if requested
            if process_ocr:
                debug_checkpoint(f"Processing OCR for image {i}: {image_name}")
                ocr_result = self.process_single_image_ocr(
                    image_path, output_folders, ocr_max_tokens, task_id=i
                )
                results['ocr_results'].append(ocr_result)
                self.completed_tasks += 1
                
                # Progress update
                progress = (self.completed_tasks / self.total_tasks) * 100
                logger.info(f"Progress: {progress:.1f}% ({self.completed_tasks}/{self.total_tasks})")
            
            # Process Markdown if requested
            if process_md:
                debug_checkpoint(f"Processing Markdown for image {i}: {image_name}")
                md_result = self.process_single_image_markdown(
                    image_path, output_folders, md_max_tokens, temperature, task_id=i
                )
                results['md_results'].append(md_result)
                self.completed_tasks += 1
                
                # Progress update
                progress = (self.completed_tasks / self.total_tasks) * 100
                logger.info(f"Progress: {progress:.1f}% ({self.completed_tasks}/{self.total_tasks})")
        
        results['end_time'] = time.time()
        results['total_time'] = results['end_time'] - results['start_time']
        
        debug_checkpoint("Sequential batch processing completed", "BATCH_SEQ_END")
        return results
    
    def process_batch_parallel(self, 
                             image_paths: List[str],
                             output_folders: Dict[str, str],
                             process_ocr: bool,
                             process_md: bool,
                             ocr_max_tokens: int,
                             md_max_tokens: int,
                             temperature: float) -> Dict[str, Any]:
        """Process images in parallel with enhanced resource management"""
        debug_checkpoint("Starting parallel batch processing", "BATCH_PAR_START")
        
        results = {
            'ocr_results': [],
            'md_results': [],
            'total_images': len(image_paths),
            'start_time': time.time(),
            'processing_mode': 'parallel'
        }
        
        self.total_tasks = len(image_paths) * (int(process_ocr) + int(process_md))
        self.completed_tasks = 0
        
        # Choose executor type based on configuration
        if self.use_process_pool:
            executor_class = ProcessPoolExecutor
            logger.info(f"Using ProcessPoolExecutor with {self.max_workers} workers")
        else:
            executor_class = ThreadPoolExecutor
            logger.info(f"Using ThreadPoolExecutor with {self.max_workers} workers")
        
        with executor_class(max_workers=self.max_workers) as executor:
            futures = []
            task_id = 0
            
            # Submit OCR tasks
            if process_ocr:
                for image_path in image_paths:
                    task_id += 1
                    future = executor.submit(
                        self.process_single_image_ocr,
                        image_path, output_folders, ocr_max_tokens, task_id
                    )
                    futures.append(('ocr', future, task_id))
            
            # Submit Markdown tasks
            if process_md:
                for image_path in image_paths:
                    task_id += 1
                    future = executor.submit(
                        self.process_single_image_markdown,
                        image_path, output_folders, md_max_tokens, temperature, task_id
                    )
                    futures.append(('md', future, task_id))
            
            # Collect results with progress tracking
            for task_type, future in as_completed([f[1] for f in futures]):
                self.completed_tasks += 1
                progress = (self.completed_tasks / self.total_tasks) * 100
                logger.info(f"Completed task {self.completed_tasks}/{self.total_tasks} ({progress:.1f}%)")
                
                try:
                    result = future.result()
                    if task_type == 'ocr':
                        results['ocr_results'].append(result)
                    else:
                        results['md_results'].append(result)
                        
                    # Log success/failure
                    if result.get('success', False):
                        logger.info(f"✓ {task_type.upper()} task completed: {result.get('image_name', 'unknown')}")
                    else:
                        logger.error(f"✗ {task_type.upper()} task failed: {result.get('image_name', 'unknown')} - {result.get('error', 'unknown error')}")
                        
                except Exception as e:
                    logger.error(f"Task execution failed: {e}")
        
        results['end_time'] = time.time()
        results['total_time'] = results['end_time'] - results['start_time']
        
        debug_checkpoint("Parallel batch processing completed", "BATCH_PAR_END")
        return results
    
    def create_performance_report(self, 
                                results: Dict[str, Any], 
                                output_folders: Dict[str, str]) -> str:
        """Create detailed performance analysis report"""
        debug_checkpoint("Creating performance report", "PERF_REPORT_START")
        
        performance_file = Path(output_folders['performance']) / "performance_analysis.json"
        
        # Extract performance metrics
        ocr_results = results.get('ocr_results', [])
        md_results = results.get('md_results', [])
        
        ocr_successful = [r for r in ocr_results if r.get('success', False)]
        md_successful = [r for r in md_results if r.get('success', False)]
        
        # Calculate performance statistics
        performance_data = {
            'system_info': self.system_info,
            'configuration': {
                'model_checkpoint': self.model_checkpoint,
                'is_local_checkpoint': self.is_local_checkpoint,
                'use_8bit': self.use_8bit,
                'mixed_precision': self.mixed_precision,
                'force_fallback': self.force_fallback,
                'max_workers': self.max_workers,
                'use_process_pool': self.use_process_pool,
                'processing_mode': results.get('processing_mode', 'unknown')
            },
            'timing_analysis': {
                'total_processing_time': results.get('total_time', 0),
                'images_processed': results.get('total_images', 0),
                'average_time_per_image': results.get('total_time', 0) / max(results.get('total_images', 1), 1),
                'ocr_processing_times': [r.get('processing_time', 0) for r in ocr_successful],
                'md_processing_times': [r.get('processing_time', 0) for r in md_successful],
                'ocr_inference_times': [r.get('inference_time', 0) for r in ocr_successful],
                'md_inference_times': [r.get('inference_time', 0) for r in md_successful]
            },
            'quantization_analysis': {
                'ocr_quantization_used': [r.get('quantization_used', 'unknown') for r in ocr_successful],
                'md_quantization_used': [r.get('quantization_used', 'unknown') for r in md_successful],
            },
            'error_analysis': {
                'ocr_errors': [r for r in ocr_results if not r.get('success', False)],
                'md_errors': [r for r in md_results if not r.get('success', False)],
                'ocr_error_rate': len([r for r in ocr_results if not r.get('success', False)]) / max(len(ocr_results), 1),
                'md_error_rate': len([r for r in md_results if not r.get('success', False)]) / max(len(md_results), 1)
            },
            'throughput_metrics': {
                'images_per_second': results.get('total_images', 0) / max(results.get('total_time', 1), 1),
                'ocr_regions_per_second': sum(r.get('text_regions', 0) for r in ocr_successful) / max(results.get('total_time', 1), 1),
                'md_words_per_second': sum(r.get('word_count', 0) for r in md_successful) / max(results.get('total_time', 1), 1)
            }
        }
        
        # Save performance report
        with open(performance_file, 'w', encoding='utf-8') as f:
            json.dump(performance_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Performance analysis saved to: {performance_file}")
        debug_checkpoint("Performance report creation completed", "PERF_REPORT_END")
        
        return str(performance_file)
    
    def create_comprehensive_report(self, 
                                  results: Dict[str, Any], 
                                  output_folders: Dict[str, str],
                                  config: Dict[str, Any]) -> str:
        """Create comprehensive processing report with detailed analysis"""
        debug_checkpoint("Creating comprehensive report", "COMP_REPORT_START")
        
        report_file = Path(output_folders['reports']) / "comprehensive_processing_report.json"
        
        # Extract results
        ocr_results = results.get('ocr_results', [])
        md_results = results.get('md_results', [])
        
        ocr_successful = [r for r in ocr_results if r.get('success', False)]
        ocr_failed = [r for r in ocr_results if not r.get('success', False)]
        md_successful = [r for r in md_results if r.get('success', False)]
        md_failed = [r for r in md_results if not r.get('success', False)]
        
        # Calculate comprehensive statistics
        total_text_regions = sum(r.get('text_regions', 0) for r in ocr_successful)
        total_words = sum(r.get('word_count', 0) for r in md_successful)
        total_chars = sum(r.get('char_count', 0) for r in md_successful)
        
        # Create comprehensive report
        report = {
            'metadata': {
                'report_generated': time.strftime('%Y-%m-%d %H:%M:%S'),
                'processor_version': '8-bit Mixed Precision Batch Processor v1.0',
                'system_info': self.system_info
            },
            'configuration': config,
            'processing_summary': {
                'total_images': results['total_images'],
                'total_processing_time': results['total_time'],
                'average_time_per_image': results['total_time'] / max(results['total_images'], 1),
                'processing_mode': results.get('processing_mode', 'unknown'),
                'tasks_completed': len(ocr_results) + len(md_results),
                'overall_success_rate': len(ocr_successful + md_successful) / max(len(ocr_results + md_results), 1)
            },
            'ocr_analysis': {
                'total_processed': len(ocr_results),
                'successful': len(ocr_successful),
                'failed': len(ocr_failed),
                'success_rate': len(ocr_successful) / max(len(ocr_results), 1),
                'total_text_regions_found': total_text_regions,
                'average_regions_per_image': total_text_regions / max(len(ocr_successful), 1),
                'average_processing_time': sum(r.get('processing_time', 0) for r in ocr_successful) / max(len(ocr_successful), 1),
                'average_inference_time': sum(r.get('inference_time', 0) for r in ocr_successful) / max(len(ocr_successful), 1),
                'quantization_distribution': {},
                'error_types': {}
            },
            'markdown_analysis': {
                'total_processed': len(md_results),
                'successful': len(md_successful),
                'failed': len(md_failed),
                'success_rate': len(md_successful) / max(len(md_results), 1),
                'total_words_generated': total_words,
                'total_characters_generated': total_chars,
                'average_words_per_image': total_words / max(len(md_successful), 1),
                'average_processing_time': sum(r.get('processing_time', 0) for r in md_successful) / max(len(md_successful), 1),
                'average_inference_time': sum(r.get('inference_time', 0) for r in md_successful) / max(len(md_successful), 1),
                'content_analysis': {
                    'total_headers': sum(r.get('headers', 0) for r in md_successful),
                    'total_lists': sum(r.get('lists', 0) for r in md_successful),
                    'total_tables': sum(r.get('tables', 0) for r in md_successful),
                    'total_code_blocks': sum(r.get('code_blocks', 0) for r in md_successful)
                },
                'quantization_distribution': {},
                'error_types': {}
            },
            'detailed_results': {
                'ocr_results': ocr_results,
                'markdown_results': md_results
            }
        }
        
        # Analyze quantization usage
        for r in ocr_successful:
            quant = r.get('quantization_used', 'unknown')
            report['ocr_analysis']['quantization_distribution'][quant] = report['ocr_analysis']['quantization_distribution'].get(quant, 0) + 1
        
        for r in md_successful:
            quant = r.get('quantization_used', 'unknown')
            report['markdown_analysis']['quantization_distribution'][quant] = report['markdown_analysis']['quantization_distribution'].get(quant, 0) + 1
        
        # Analyze error types
        for r in ocr_failed:
            error_type = r.get('error_type', 'unknown')
            report['ocr_analysis']['error_types'][error_type] = report['ocr_analysis']['error_types'].get(error_type, 0) + 1
        
        for r in md_failed:
            error_type = r.get('error_type', 'unknown')
            report['markdown_analysis']['error_types'][error_type] = report['markdown_analysis']['error_types'].get(error_type, 0) + 1
        
        # Save comprehensive report
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Comprehensive report saved to: {report_file}")
        debug_checkpoint("Comprehensive report creation completed", "COMP_REPORT_END")
        
        return str(report_file)
    
    def print_detailed_summary(self, results: Dict[str, Any], config: Dict[str, Any]):
        """Print detailed, formatted summary of processing results"""
        debug_checkpoint("Generating detailed summary", "SUMMARY_START")
        
        ocr_results = results.get('ocr_results', [])
        md_results = results.get('md_results', [])
        
        ocr_successful = [r for r in ocr_results if r.get('success', False)]
        md_successful = [r for r in md_results if r.get('success', False)]
        
        total_text_regions = sum(r.get('text_regions', 0) for r in ocr_successful)
        total_words = sum(r.get('word_count', 0) for r in md_successful)
        
        # Print comprehensive summary
        print(f"\n{'='*100}")
        print("8-BIT MIXED PRECISION BATCH PROCESSING SUMMARY")
        print(f"{'='*100}")
        
        print(f"\n📋 CONFIGURATION:")
        print(f"  Model Checkpoint: {config['model_checkpoint']}")
        print(f"  Local Model: {'Yes' if self.is_local_checkpoint else 'No'}")
        print(f"  Device: {config['device'] or 'Auto-detected'}")
        print(f"  8-bit Quantization: {'Enabled' if config['use_8bit'] else 'Disabled'}")
        print(f"  Mixed Precision: {'Enabled' if config['mixed_precision'] else 'Disabled'}")
        print(f"  Force Fallback: {'Yes' if config['force_fallback'] else 'No'}")
        print(f"  Max Workers: {config['max_workers']}")
        print(f"  Processing Mode: {results.get('processing_mode', 'Unknown').title()}")
        print(f"  Executor Type: {'Process Pool' if self.use_process_pool else 'Thread Pool'}")
        
        print(f"\n⏱️  PERFORMANCE METRICS:")
        print(f"  Total Images: {results['total_images']}")
        print(f"  Total Processing Time: {results['total_time']:.2f}s")
        print(f"  Average Time per Image: {results['total_time']/max(results['total_images'], 1):.2f}s")
        print(f"  Images per Second: {results['total_images']/max(results['total_time'], 1):.2f}")
        
        if ocr_results:
            print(f"\n🔍 OCR PROCESSING RESULTS:")
            print(f"  Total Processed: {len(ocr_results)}")
            print(f"  Successful: {len(ocr_successful)}")
            print(f"  Failed: {len(ocr_results) - len(ocr_successful)}")
            print(f"  Success Rate: {(len(ocr_successful)/max(len(ocr_results), 1))*100:.1f}%")
            print(f"  Total Text Regions Found: {total_text_regions:,}")
            print(f"  Average Regions per Image: {total_text_regions/max(len(ocr_successful), 1):.1f}")
            
            if ocr_successful:
                avg_proc_time = sum(r.get('processing_time', 0) for r in ocr_successful) / len(ocr_successful)
                avg_inf_time = sum(r.get('inference_time', 0) for r in ocr_successful) / len(ocr_successful)
                print(f"  Average Processing Time: {avg_proc_time:.2f}s")
                print(f"  Average Inference Time: {avg_inf_time:.2f}s")
                
                # Quantization usage
                quant_usage = {}
                for r in ocr_successful:
                    quant = r.get('quantization_used', 'unknown')
                    quant_usage[quant] = quant_usage.get(quant, 0) + 1
                print(f"  Quantization Usage: {dict(quant_usage)}")
        
        if md_results:
            print(f"\n📝 MARKDOWN GENERATION RESULTS:")
            print(f"  Total Processed: {len(md_results)}")
            print(f"  Successful: {len(md_successful)}")
            print(f"  Failed: {len(md_results) - len(md_successful)}")
            print(f"  Success Rate: {(len(md_successful)/max(len(md_results), 1))*100:.1f}%")
            print(f"  Total Words Generated: {total_words:,}")
            print(f"  Average Words per Image: {total_words/max(len(md_successful), 1):.0f}")
            
            if md_successful:
                total_headers = sum(r.get('headers', 0) for r in md_successful)
                total_lists = sum(r.get('lists', 0) for r in md_successful)
                total_tables = sum(r.get('tables', 0) for r in md_successful)
                total_code_blocks = sum(r.get('code_blocks', 0) for r in md_successful)
                
                avg_proc_time = sum(r.get('processing_time', 0) for r in md_successful) / len(md_successful)
                avg_inf_time = sum(r.get('inference_time', 0) for r in md_successful) / len(md_successful)
                print(f"  Average Processing Time: {avg_proc_time:.2f}s")
                print(f"  Average Inference Time: {avg_inf_time:.2f}s")
                print(f"  Content Elements:")
                print(f"    Headers: {total_headers}")
                print(f"    Lists: {total_lists}")
                print(f"    Tables: {total_tables}")
                print(f"    Code Blocks: {total_code_blocks}")
                
                # Quantization usage
                quant_usage = {}
                for r in md_successful:
                    quant = r.get('quantization_used', 'unknown')
                    quant_usage[quant] = quant_usage.get(quant, 0) + 1
                print(f"  Quantization Usage: {dict(quant_usage)}")
        
        # System information
        if self.system_info:
            print(f"\n💻 SYSTEM INFORMATION:")
            print(f"  CPU Cores: {self.system_info.get('cpu_count', 'Unknown')} physical, {self.system_info.get('cpu_count_logical', 'Unknown')} logical")
            print(f"  System Memory: {self.system_info.get('memory_total_gb', 0):.1f} GB")
            print(f"  CUDA Available: {'Yes' if self.system_info.get('cuda_available', False) else 'No'}")
            if self.system_info.get('cuda_available', False):
                print(f"  CUDA Devices: {self.system_info.get('cuda_device_count', 0)}")
                print(f"  GPU Memory: {self.system_info.get('cuda_memory_gb', 0):.1f} GB")
        
        print(f"\n{'='*100}")
        
        debug_checkpoint("Detailed summary generation completed", "SUMMARY_END")

def get_args():
    """Parse command line arguments with comprehensive options"""
    parser = argparse.ArgumentParser(
        description='8-bit Mixed Precision Batch Processor for Kosmos-2.5 with Complete Device Management',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output arguments
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Path to input folder containing images')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Path to output folder for results')
    
    # Model configuration
    parser.add_argument('--model_checkpoint', '-m', type=str, required=True,
                       help='Path to model checkpoint (local directory or HuggingFace model name)')
    parser.add_argument('--cache_dir', type=str, default=None,
                       help='Cache directory for model files')
    
    # Quantization and precision settings
    parser.add_argument('--no_8bit', action='store_true',
                       help='Disable 8-bit quantization')
    parser.add_argument('--no_mixed_precision', action='store_true',
                       help='Disable mixed precision for critical layers')
    parser.add_argument('--force_fallback', action='store_true',
                       help='Force use of fallback mode (no 8-bit quantization)')
    
    # Processing configuration
    parser.add_argument('--device', '-d', type=str, default=None,
                       help='Device to use (auto-detected if not specified)')
    
    # Enhanced token size configuration
    parser.add_argument('--max_tokens', '--tokens', type=int, default=1024,
                       help='Maximum number of tokens to generate (default: 1024 for OCR, 2048 for markdown)')
    parser.add_argument('--min_tokens', type=int, default=10,
                       help='Minimum number of tokens to generate (default: 10)')
    parser.add_argument('--ocr_tokens', type=int, default=None,
                       help='Specific token limit for OCR tasks (overrides max_tokens for OCR)')
    parser.add_argument('--md_tokens', type=int, default=None,
                       help='Specific token limit for markdown tasks (overrides max_tokens for markdown)')
    
    parser.add_argument('--temperature', '-t', type=float, default=0.1,
                       help='Sampling temperature for markdown generation (0.0-1.0)')
    
    # Task selection
    parser.add_argument('--skip_ocr', action='store_true',
                       help='Skip OCR processing')
    parser.add_argument('--skip_md', action='store_true',
                       help='Skip markdown generation')
    
    # Performance configuration
    parser.add_argument('--max_workers', type=int, default=1,
                       help='Maximum number of parallel workers (1 for sequential)')
    parser.add_argument('--parallel', action='store_true',
                       help='Enable parallel processing (requires max_workers > 1)')
    parser.add_argument('--use_process_pool', action='store_true',
                       help='Use process pool instead of thread pool for true parallelism')
    
    # File handling
    parser.add_argument('--image_extensions', type=str, nargs='+', 
                       default=['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'],
                       help='Image file extensions to process')
    
    # Debug and logging
    parser.add_argument('--debug', action='store_true',
                       help='Enable maximum debug output')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    return parser.parse_args()

def main():
    """Main execution function with comprehensive error handling"""
    debug_checkpoint("8-bit Mixed Precision Batch Processor starting", "MAIN_START")
    
    args = get_args()
    
    # Validate and configure token settings
    if args.max_tokens < args.min_tokens:
        logger.error(f"max_tokens ({args.max_tokens}) must be >= min_tokens ({args.min_tokens})")
        sys.exit(1)
    
    # Set task-specific token limits
    ocr_max_tokens = args.ocr_tokens if args.ocr_tokens is not None else args.max_tokens
    md_max_tokens = args.md_tokens if args.md_tokens is not None else (args.max_tokens if args.max_tokens != 1024 else 2048)
    
    # Validate token limits
    if ocr_max_tokens > 4096:
        logger.warning(f"Large OCR token limit ({ocr_max_tokens}) may cause memory issues")
    if md_max_tokens > 8192:
        logger.warning(f"Very large markdown token limit ({md_max_tokens}) may cause memory issues")
    
    if args.temperature < 0 or args.temperature > 1.0:
        logger.error(f"Temperature must be between 0.0 and 1.0, got {args.temperature}")
        sys.exit(1)
    
    # Set logging level based on arguments
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        debug_checkpoint("Debug mode enabled")
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
        debug_checkpoint("Verbose mode enabled")
    
    # Validate arguments
    if args.skip_ocr and args.skip_md:
        logger.error("Cannot skip both OCR and markdown processing")
        sys.exit(1)
    
    if args.parallel and args.max_workers <= 1:
        logger.warning("Parallel processing requested but max_workers <= 1. Using sequential processing.")
        args.parallel = False
    
    # Create configuration dict
    config = {
        'model_checkpoint': args.model_checkpoint,
        'cache_dir': args.cache_dir,
        'device': args.device,
        'use_8bit': not args.no_8bit,
        'mixed_precision': not args.no_mixed_precision,
        'force_fallback': args.force_fallback,
        'max_tokens': args.max_tokens,
        'min_tokens': args.min_tokens,
        'ocr_max_tokens': ocr_max_tokens,
        'md_max_tokens': md_max_tokens,
        'temperature': args.temperature,
        'max_workers': args.max_workers,
        'parallel': args.parallel,
        'use_process_pool': args.use_process_pool,
        'process_ocr': not args.skip_ocr,
        'process_md': not args.skip_md,
        'image_extensions': args.image_extensions
    }
    
    debug_checkpoint(f"Configuration: {config}")
    debug_checkpoint(f"Token configuration - OCR: {ocr_max_tokens}, MD: {md_max_tokens}, Min: {args.min_tokens}")
    
    try:
        # Initialize batch processor
        debug_checkpoint("Initializing 8-bit batch processor")
        processor = EightBitBatchProcessor(
            model_checkpoint=args.model_checkpoint,
            device=args.device,
            cache_dir=args.cache_dir,
            use_8bit=not args.no_8bit,
            mixed_precision=not args.no_mixed_precision,
            force_fallback=args.force_fallback,
            max_workers=args.max_workers,
            use_process_pool=args.use_process_pool
        )
        
        # Find images
        logger.info(f"Scanning for images in: {args.input}")
        image_files = processor.find_images(args.input, args.image_extensions)
        
        if not image_files:
            logger.error(f"No valid images found in {args.input}")
            sys.exit(1)
        
        # Create output structure
        output_folders = processor.create_output_structure(args.output)
        
        # Process images
        logger.info(f"Starting 8-bit mixed precision batch processing of {len(image_files)} images")
        logger.info(f"Processing mode: {'Parallel' if args.parallel else 'Sequential'}")
        logger.info(f"Token limits - OCR: {ocr_max_tokens}, Markdown: {md_max_tokens}")
        
        if args.parallel:
            results = processor.process_batch_parallel(
                image_files, output_folders,
                not args.skip_ocr, not args.skip_md,
                ocr_max_tokens, md_max_tokens, args.temperature
            )
        else:
            results = processor.process_batch_sequential(
                image_files, output_folders,
                not args.skip_ocr, not args.skip_md,
                ocr_max_tokens, md_max_tokens, args.temperature
            )
        
        # Create reports
        debug_checkpoint("Generating reports")
        comprehensive_report = processor.create_comprehensive_report(results, output_folders, config)
        performance_report = processor.create_performance_report(results, output_folders)
        
        # Print summary
        processor.print_detailed_summary(results, config)
        
        logger.info(f"8-bit mixed precision batch processing completed successfully!")
        logger.info(f"Comprehensive report: {comprehensive_report}")
        logger.info(f"Performance analysis: {performance_report}")
        
        debug_checkpoint("Application completed successfully", "MAIN_END")
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        debug_checkpoint("Application interrupted by user", "MAIN_INTERRUPTED")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        debug_checkpoint(f"Application failed with error: {str(e)}", "MAIN_FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
