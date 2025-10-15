#!/usr/bin/env python3
"""
Custom Batch Processor for Kosmos-2.5 with Dedicated Prompts (FP16 Version)

This script is based on batch_processor_custom.py but uses FP16 precision instead of FP8.
It includes dedicated prompts for both OCR and Markdown extraction with enhanced 
prompt validation and display.

Features:
- Custom prompts for OCR and Markdown tasks
- Prompt validation and display
- FP16 precision support (bfloat16/float16)
- Enhanced error handling
- Detailed output formatting
"""

import os
import sys
import re
import time
import json
import argparse
import logging
import threading
import torch
import requests
from pathlib import Path
from typing import Dict, Any, Optional, List
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Kosmos2_5ForConditionalGeneration

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FP16OCRInference:
    """FP16 OCR inference engine for KOSMOS-2.5"""
    
    def __init__(self, model_checkpoint: str = "microsoft/kosmos-2.5", device: str = None):
        """
        Initialize FP16 OCR inference engine
        
        Args:
            model_checkpoint: Model repository or local path
            device: Device to use for inference
        """
        self.model_checkpoint = model_checkpoint
        self.device = self._setup_device(device)
        self.dtype = torch.bfloat16 if self.device.startswith('cuda') else torch.float32
        
        # Initialize model and processor
        self._load_model()
        
        logger.info(f"FP16 OCR Engine initialized on {self.device} with {self.dtype}")
    
    def _setup_device(self, device: str = None) -> str:
        """Setup and validate device"""
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        if device.startswith('cuda') and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = "cpu"
        
        return device
    
    def _load_model(self):
        """Load model and processor"""
        try:
            logger.info(f"Loading KOSMOS-2.5 model from {self.model_checkpoint}")
            
            self.model = Kosmos2_5ForConditionalGeneration.from_pretrained(
                self.model_checkpoint,
                device_map=self.device,
                torch_dtype=self.dtype
            )
            
            self.processor = AutoProcessor.from_pretrained(self.model_checkpoint)
            
            logger.info("Model and processor loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def load_image(self, image_path: str) -> Image.Image:
        """Load image from path or URL"""
        try:
            if image_path.startswith(('http://', 'https://')):
                response = requests.get(image_path, stream=True)
                response.raise_for_status()
                image = Image.open(response.raw)
            else:
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                image = Image.open(image_path)
            
            return image.convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise
    
    def post_process_ocr(self, generated_text: str, scale_height: float, scale_width: float, prompt: str = "<ocr>") -> str:
        """Post-process OCR results to extract bounding boxes and text"""
        y = generated_text.replace(prompt, "")
        
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
    
    def perform_ocr_custom_prompt(self, image_path: str, custom_prompt: str, max_tokens: int = 1024) -> Dict[str, Any]:
        """
        Perform OCR with custom prompt
        
        Args:
            image_path: Path to image file
            custom_prompt: Custom prompt for OCR
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with OCR results and metadata
        """
        start_time = time.time()
        
        try:
            # Load and process image
            image = self.load_image(image_path)
            
            # Process inputs
            inputs = self.processor(text=custom_prompt, images=image, return_tensors="pt")
            height, width = inputs.pop("height"), inputs.pop("width")
            raw_width, raw_height = image.size
            scale_height = raw_height / height
            scale_width = raw_width / width
            
            # Move to device and correct dtype
            inputs = {k: v.to(self.device) if v is not None else None for k, v in inputs.items()}
            if "flattened_patches" in inputs and inputs["flattened_patches"] is not None:
                inputs["flattened_patches"] = inputs["flattened_patches"].to(self.dtype)
            
            # Generate OCR results
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode results
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            processed_text = self.post_process_ocr(generated_text, scale_height, scale_width, custom_prompt)
            
            processing_time = time.time() - start_time
            
            # Extract plain text without coordinates
            plain_text_lines = []
            for line in processed_text.strip().split('\n'):
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 9:
                        plain_text_lines.append(','.join(parts[8:]))
            plain_text = '\n'.join(plain_text_lines)
            
            result = {
                'success': True,
                'image_path': image_path,
                'prompt_used': custom_prompt,
                'raw_output': generated_text,
                'processed_output': processed_text,
                'text_extracted': plain_text,
                'image_dimensions': {'width': raw_width, 'height': raw_height},
                'processing_time': processing_time,
                'max_tokens_used': max_tokens,
                'model_info': {
                    'checkpoint': self.model_checkpoint,
                    'device': self.device,
                    'dtype': str(self.dtype)
                }
            }
            
            logger.info(f"OCR completed in {processing_time:.2f}s, extracted {len(plain_text_lines)} text segments")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"OCR failed: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'image_path': image_path,
                'prompt_used': custom_prompt,
                'processing_time': processing_time
            }


class FP16MarkdownInference:
    """FP16 Markdown inference engine for KOSMOS-2.5"""
    
    def __init__(self, model_checkpoint: str = "microsoft/kosmos-2.5", device: str = None):
        """
        Initialize FP16 Markdown inference engine
        
        Args:
            model_checkpoint: Model repository or local path
            device: Device to use for inference
        """
        self.model_checkpoint = model_checkpoint
        self.device = self._setup_device(device)
        self.dtype = torch.bfloat16 if self.device.startswith('cuda') else torch.float32
        
        # Initialize model and processor
        self._load_model()
        
        logger.info(f"FP16 Markdown Engine initialized on {self.device} with {self.dtype}")
    
    def _setup_device(self, device: str = None) -> str:
        """Setup and validate device"""
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        if device.startswith('cuda') and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = "cpu"
        
        return device
    
    def _load_model(self):
        """Load model and processor"""
        try:
            logger.info(f"Loading KOSMOS-2.5 model from {self.model_checkpoint}")
            
            self.model = Kosmos2_5ForConditionalGeneration.from_pretrained(
                self.model_checkpoint,
                device_map=self.device,
                torch_dtype=self.dtype
            )
            
            self.processor = AutoProcessor.from_pretrained(self.model_checkpoint)
            
            logger.info("Model and processor loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def load_image(self, image_path: str) -> Image.Image:
        """Load image from path or URL"""
        try:
            if image_path.startswith(('http://', 'https://')):
                response = requests.get(image_path, stream=True)
                response.raise_for_status()
                image = Image.open(response.raw)
            else:
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                image = Image.open(image_path)
            
            return image.convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise
    
    def generate_markdown_custom_prompt(self, image_path: str, custom_prompt: str, 
                                      max_tokens: int = 2048, temperature: float = 0.1) -> Dict[str, Any]:
        """
        Generate markdown with custom prompt
        
        Args:
            image_path: Path to image file
            custom_prompt: Custom prompt for markdown generation
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            
        Returns:
            Dictionary with markdown results and metadata
        """
        start_time = time.time()
        
        try:
            # Load and process image
            image = self.load_image(image_path)
            
            # Process inputs
            inputs = self.processor(text=custom_prompt, images=image, return_tensors="pt")
            height, width = inputs.pop("height"), inputs.pop("width")
            
            # Move to device and correct dtype
            inputs = {k: v.to(self.device) if v is not None else None for k, v in inputs.items()}
            if "flattened_patches" in inputs and inputs["flattened_patches"] is not None:
                inputs["flattened_patches"] = inputs["flattened_patches"].to(self.dtype)
            
            # Generate markdown
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode results
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Extract markdown (remove prompt)
            markdown_content = generated_text.replace(custom_prompt, "").strip()
            
            processing_time = time.time() - start_time
            
            # Analyze content
            line_count = len(markdown_content.split('\n'))
            word_count = len(markdown_content.split())
            char_count = len(markdown_content)
            
            # Check for markdown elements
            has_headers = bool(re.search(r'^#+\s', markdown_content, re.MULTILINE))
            has_lists = bool(re.search(r'^[\*\-\+]\s|^\d+\.\s', markdown_content, re.MULTILINE))
            has_tables = '|' in markdown_content
            has_code = '```' in markdown_content or '`' in markdown_content
            
            result = {
                'success': True,
                'image_path': image_path,
                'prompt_used': custom_prompt,
                'raw_output': generated_text,
                'generated_markdown': markdown_content,
                'content_analysis': {
                    'line_count': line_count,
                    'word_count': word_count,
                    'character_count': char_count,
                    'has_headers': has_headers,
                    'has_lists': has_lists,
                    'has_tables': has_tables,
                    'has_code_blocks': has_code
                },
                'generation_params': {
                    'max_tokens': max_tokens,
                    'temperature': temperature
                },
                'processing_time': processing_time,
                'model_info': {
                    'checkpoint': self.model_checkpoint,
                    'device': self.device,
                    'dtype': str(self.dtype)
                }
            }
            
            logger.info(f"Markdown generated in {processing_time:.2f}s, {word_count} words, {line_count} lines")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Markdown generation failed: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'image_path': image_path,
                'prompt_used': custom_prompt,
                'processing_time': processing_time
            }


class CustomBatchProcessorFP16:
    """Custom batch processor with dedicated prompts for OCR and Markdown extraction (FP16 version)"""
    
    def __init__(self, 
                 model_checkpoint: str = "microsoft/kosmos-2.5",
                 device: str = None):
        """
        Initialize the custom batch processor
        
        Args:
            model_checkpoint: Path to model checkpoint or repository
            device: Device to use for inference
        """
        self.model_checkpoint = model_checkpoint
        self.device = device
        
        # Thread-local storage for inference engines
        self._local = threading.local()
        
        # Custom prompts for different tasks (same as original)
        self.prompts = {
            'ocr_detailed': """<grounding>Perform comprehensive OCR analysis on this image. Extract all visible text with precise bounding box coordinates. Include:
1. Main headings and titles
2. Body text and paragraphs  
3. Tables, lists, and structured data
4. Form fields and labels
5. Small text and footnotes
6. Any numerical data or codes
Provide exact text extraction with spatial relationships.""",

            'ocr_structured': """<grounding>Extract all text from this document image in a structured format. Identify and categorize:
- Headers and section titles
- Paragraph content
- Bullet points and numbered lists
- Table data with row/column structure
- Captions and labels
- Page numbers and metadata
Maintain the document's hierarchical structure and reading order.""",

            'markdown_comprehensive': """<md>Convert this document image to clean, well-structured Markdown format. Generate comprehensive markdown that includes:

# Document Structure
- Proper heading hierarchy (# ## ### etc.)
- Paragraph formatting with appropriate spacing
- Lists (ordered and unordered) with correct indentation
- Tables with proper alignment
- Code blocks for technical content
- Links and references where applicable

# Content Preservation
- Maintain original document flow and organization
- Preserve technical terminology and formatting
- Include all visible text content
- Represent visual elements appropriately

# Quality Standards
- Use semantic markdown elements
- Ensure proper nesting and hierarchy
- Create readable, professional output
- Follow markdown best practices

Generate complete, publication-ready markdown.""",

            'markdown_technical': """<md>Analyze this technical document and convert it to specialized Markdown optimized for technical content:

## Technical Elements
- Code snippets in appropriate language blocks
- Mathematical expressions and formulas
- Technical diagrams descriptions
- API documentation formatting
- Configuration examples
- Command-line instructions

## Structure
- Clear section hierarchy with descriptive headers
- Technical glossaries and definitions
- Step-by-step procedures as ordered lists
- Important notes and warnings as blockquotes
- Cross-references and internal links

## Formatting
- Consistent code fence languages
- Proper table formatting for specifications  
- Emphasis for key terms and concepts
- Inline code for technical terms
- Professional documentation standards

Create comprehensive technical documentation in markdown format.""",

            'markdown_business': """<md>Transform this business document into professional Markdown suitable for corporate documentation:

## Business Content Structure
- Executive summary sections
- Key findings and recommendations  
- Financial data in well-formatted tables
- Process flows and procedures
- Organizational charts descriptions
- Project timelines and milestones

## Professional Formatting
- Clear heading hierarchy for navigation
- Bullet points for key insights
- Numbered lists for procedures
- Tables for data presentation
- Blockquotes for important statements
- Professional language and tone

## Document Standards
- Consistent formatting throughout
- Appropriate emphasis and highlighting
- Clean, readable layout
- Corporate documentation style
- Comprehensive coverage of all content

Generate polished, business-ready markdown documentation."""
        }
        
        logger.info(f"Custom Batch Processor FP16 initialized")
        logger.info(f"Model: {self.model_checkpoint}")
        logger.info(f"Device: {self.device or 'auto-detect'}")
        logger.info(f"Loaded {len(self.prompts)} custom prompts")
    
    def display_prompts(self):
        """Display all available prompts for validation"""
        print("\n" + "="*80)
        print("CUSTOM PROMPTS VALIDATION (FP16 VERSION)")
        print("="*80)
        
        for prompt_name, prompt_text in self.prompts.items():
            validation = self.validate_prompt(prompt_text, prompt_name)
            
            print(f"\nPrompt: {prompt_name.upper()}")
            print(f"Tag Type: {validation['tag_type']}")
            print(f"Length: {validation['length']} characters")
            print(f"Words: {validation['word_count']}")
            print(f"Has Tag: {validation['has_grounding'] or validation['has_markdown']}")
            print(f"Complexity Score: {validation['complexity_score']}")
            print("-" * 40)
            print(prompt_text[:200] + "..." if len(prompt_text) > 200 else prompt_text)
            print("-" * 80)
    
    def validate_prompt(self, prompt_text: str, prompt_name: str) -> Dict[str, Any]:
        """Validate prompt quality and characteristics"""
        validation = {
            'name': prompt_name,
            'length': len(prompt_text),
            'word_count': len(prompt_text.split()),
            'has_grounding': '<grounding>' in prompt_text,
            'has_markdown': '<md>' in prompt_text,
            'tag_type': 'grounding' if '<grounding>' in prompt_text else 'markdown' if '<md>' in prompt_text else 'none',
            'has_structure': any(keyword in prompt_text.lower() for keyword in ['header', 'section', 'paragraph', 'list']),
            'has_instructions': any(keyword in prompt_text.lower() for keyword in ['extract', 'convert', 'generate', 'analyze']),
            'complexity_score': 0
        }
        
        # Calculate complexity score
        complexity_indicators = [
            'comprehensive', 'detailed', 'structured', 'hierarchical',
            'technical', 'professional', 'quality', 'standards'
        ]
        validation['complexity_score'] = sum(1 for indicator in complexity_indicators 
                                           if indicator in prompt_text.lower())
        
        return validation
    
    def _get_ocr_engine(self) -> FP16OCRInference:
        """Get thread-local OCR engine"""
        if not hasattr(self._local, 'ocr_engine') or self._local.ocr_engine is None:
            logger.info("Creating new FP16 OCR engine for thread")
            self._local.ocr_engine = FP16OCRInference(
                model_checkpoint=self.model_checkpoint,
                device=self.device
            )
        return self._local.ocr_engine
    
    def _get_md_engine(self) -> FP16MarkdownInference:
        """Get thread-local Markdown engine"""
        if not hasattr(self._local, 'md_engine') or self._local.md_engine is None:
            logger.info("Creating new FP16 Markdown engine for thread")
            self._local.md_engine = FP16MarkdownInference(
                model_checkpoint=self.model_checkpoint,
                device=self.device
            )
        return self._local.md_engine
    
    def process_image_with_custom_ocr(self, 
                                    image_path: str, 
                                    prompt_key: str = 'ocr_detailed',
                                    max_tokens: int = 1024) -> Dict[str, Any]:
        """Process image with custom OCR prompt"""
        
        if prompt_key not in self.prompts:
            raise ValueError(f"Unknown prompt key: {prompt_key}. Available: {list(self.prompts.keys())}")
        
        prompt_text = self.prompts[prompt_key]
        image_name = Path(image_path).stem
        
        print(f"\nOCR Processing: {image_name}")
        print(f"Using prompt: {prompt_key}")
        print(f"Prompt preview: {prompt_text[:100]}...")
        
        start_time = time.time()
        
        try:
            ocr_engine = self._get_ocr_engine()
            result = ocr_engine.perform_ocr_custom_prompt(
                image_path=image_path,
                custom_prompt=prompt_text,
                max_tokens=max_tokens
            )
            
            processing_time = time.time() - start_time
            result['total_processing_time'] = processing_time
            
            if result['success']:
                print(f"OCR SUCCESS: {len(result.get('text_extracted', '').split())} words extracted in {processing_time:.2f}s")
            else:
                print(f"OCR FAILED: {result.get('error', 'Unknown error')} in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"OCR processing failed for {image_name}: {e}"
            logger.error(error_msg)
            
            return {
                'success': False,
                'error': error_msg,
                'image_path': image_path,
                'prompt_used': prompt_text,
                'total_processing_time': processing_time
            }
    
    def process_image_with_custom_markdown(self, 
                                         image_path: str, 
                                         prompt_key: str = 'markdown_comprehensive',
                                         max_tokens: int = 2048,
                                         temperature: float = 0.1) -> Dict[str, Any]:
        """Process image with custom Markdown prompt"""
        
        if prompt_key not in self.prompts:
            raise ValueError(f"Unknown prompt key: {prompt_key}. Available: {list(self.prompts.keys())}")
        
        prompt_text = self.prompts[prompt_key]
        image_name = Path(image_path).stem
        
        print(f"\nMarkdown Processing: {image_name}")
        print(f"Using prompt: {prompt_key}")
        print(f"Prompt preview: {prompt_text[:100]}...")
        
        start_time = time.time()
        
        try:
            md_engine = self._get_md_engine()
            result = md_engine.generate_markdown_custom_prompt(
                image_path=image_path,
                custom_prompt=prompt_text,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            processing_time = time.time() - start_time
            result['total_processing_time'] = processing_time
            
            if result['success']:
                word_count = result.get('content_analysis', {}).get('word_count', 0)
                print(f"Markdown SUCCESS: {word_count} words generated in {processing_time:.2f}s")
            else:
                print(f"Markdown FAILED: {result.get('error', 'Unknown error')} in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Markdown processing failed for {image_name}: {e}"
            logger.error(error_msg)
            
            return {
                'success': False,
                'error': error_msg,
                'image_path': image_path,
                'prompt_used': prompt_text,
                'total_processing_time': processing_time
            }
    
    def process_image_dual_mode(self, 
                              image_path: str,
                              ocr_prompt_key: str = 'ocr_detailed',
                              md_prompt_key: str = 'markdown_comprehensive',
                              ocr_max_tokens: int = 1024,
                              md_max_tokens: int = 2048,
                              temperature: float = 0.1) -> Dict[str, Any]:
        """Process image with both OCR and Markdown using custom prompts"""
        
        total_start = time.time()
        image_name = Path(image_path).stem
        
        print(f"\n{'='*80}")
        print(f"DUAL MODE PROCESSING: {image_name}")
        print(f"{'='*80}")
        print(f"OCR Prompt: {ocr_prompt_key}")
        print(f"Markdown Prompt: {md_prompt_key}")
        
        # Process OCR
        print(f"\n{'-'*40} OCR PHASE {'-'*40}")
        ocr_result = self.process_image_with_custom_ocr(
            image_path, ocr_prompt_key, ocr_max_tokens
        )
        
        # Process Markdown
        print(f"\n{'-'*35} MARKDOWN PHASE {'-'*35}")
        md_result = self.process_image_with_custom_markdown(
            image_path, md_prompt_key, md_max_tokens, temperature
        )
        
        total_time = time.time() - total_start
        
        # Create combined result
        result = {
            'image_path': image_path,
            'ocr_result': ocr_result,
            'markdown_result': md_result,
            'processing_summary': {
                'ocr_success': ocr_result.get('success', False),
                'markdown_success': md_result.get('success', False),
                'total_processing_time': total_time,
                'ocr_time': ocr_result.get('total_processing_time', 0),
                'markdown_time': md_result.get('total_processing_time', 0)
            }
        }
        
        print(f"\n{'-'*35} DUAL MODE SUMMARY {'-'*35}")
        print(f"OCR Success: {'YES' if result['processing_summary']['ocr_success'] else 'NO'}")
        print(f"Markdown Success: {'YES' if result['processing_summary']['markdown_success'] else 'NO'}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"{'='*80}")
        
        return result
    
    def save_results(self, results: Dict[str, Any], output_dir: str):
        """Save processing results to files"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        base_name = Path(results['image_path']).stem
        
        # Save OCR results
        if results.get('ocr_result', {}).get('success'):
            ocr_text = results['ocr_result'].get('text_extracted', '')
            if ocr_text:
                ocr_file = os.path.join(output_dir, f"{base_name}_ocr.txt")
                with open(ocr_file, 'w', encoding='utf-8') as f:
                    f.write(ocr_text)
                print(f"OCR text saved: {ocr_file}")
        
        # Save Markdown results
        if results.get('markdown_result', {}).get('success'):
            md_content = results['markdown_result'].get('generated_markdown', '')
            if md_content:
                md_file = os.path.join(output_dir, f"{base_name}_markdown.md")
                with open(md_file, 'w', encoding='utf-8') as f:
                    f.write(md_content)
                print(f"Markdown saved: {md_file}")
        
        # Save combined JSON results
        json_file = os.path.join(output_dir, f"{base_name}_results.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Complete results saved: {json_file}")
    
    def process_batch(self, input_path: str, output_dir: str, 
                     ocr_prompt_key: str = 'ocr_structured',
                     md_prompt_key: str = 'markdown_comprehensive',
                     ocr_max_tokens: int = 2048,
                     md_max_tokens: int = 2048,
                     temperature: float = 0.1,
                     image_extensions: List[str] = None) -> Dict[str, Any]:
        """Process batch of images with custom prompts"""
        
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        # Collect images to process
        input_path_obj = Path(input_path)
        images_to_process = []
        
        if input_path_obj.is_file():
            if input_path_obj.suffix.lower() in image_extensions:
                images_to_process = [str(input_path_obj)]
            else:
                raise ValueError(f"File {input_path} is not a supported image format")
        elif input_path_obj.is_dir():
            for ext in image_extensions:
                images_to_process.extend(input_path_obj.glob(f"*{ext.lower()}"))
                images_to_process.extend(input_path_obj.glob(f"*{ext.upper()}"))
            images_to_process = [str(p) for p in set(images_to_process)]
            images_to_process.sort()
        else:
            raise FileNotFoundError(f"Input path not found: {input_path}")
        
        if not images_to_process:
            raise ValueError(f"No images found to process in {input_path}")
        
        print(f"\n{'='*80}")
        print(f"BATCH PROCESSING STARTED")
        print(f"{'='*80}")
        print(f"Input: {input_path}")
        print(f"Output: {output_dir}")
        print(f"Images found: {len(images_to_process)}")
        print(f"OCR Prompt: {ocr_prompt_key}")
        print(f"Markdown Prompt: {md_prompt_key}")
        print(f"{'='*80}")
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Process each image
        batch_results = {
            'total_images': len(images_to_process),
            'successful_ocr': 0,
            'successful_markdown': 0,
            'failed_images': [],
            'processing_times': [],
            'results': []
        }
        
        for i, image_path in enumerate(images_to_process, 1):
            print(f"\n[{i}/{len(images_to_process)}] Processing: {Path(image_path).name}")
            
            try:
                result = self.process_image_dual_mode(
                    image_path=image_path,
                    ocr_prompt_key=ocr_prompt_key,
                    md_prompt_key=md_prompt_key,
                    ocr_max_tokens=ocr_max_tokens,
                    md_max_tokens=md_max_tokens,
                    temperature=temperature
                )
                
                # Save individual results
                self.save_results(result, output_dir)
                
                # Update batch statistics
                if result['processing_summary']['ocr_success']:
                    batch_results['successful_ocr'] += 1
                if result['processing_summary']['markdown_success']:
                    batch_results['successful_markdown'] += 1
                
                batch_results['processing_times'].append(result['processing_summary']['total_processing_time'])
                batch_results['results'].append({
                    'image_path': image_path,
                    'ocr_success': result['processing_summary']['ocr_success'],
                    'markdown_success': result['processing_summary']['markdown_success'],
                    'processing_time': result['processing_summary']['total_processing_time']
                })
                
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                batch_results['failed_images'].append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        # Print batch summary
        self._print_batch_summary(batch_results)
        
        # Save batch results
        batch_summary_file = os.path.join(output_dir, "batch_summary.json")
        with open(batch_summary_file, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, indent=2, ensure_ascii=False)
        
        return batch_results
    
    def _print_batch_summary(self, batch_results: Dict[str, Any]):
        """Print detailed batch processing summary"""
        total = batch_results['total_images']
        ocr_success = batch_results['successful_ocr']
        md_success = batch_results['successful_markdown']
        failed = len(batch_results['failed_images'])
        processing_times = batch_results['processing_times']
        
        avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
        total_time = sum(processing_times)
        
        print(f"\n{'='*80}")
        print(f"BATCH PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"Total Images: {total}")
        print(f"OCR Success: {ocr_success} ({(ocr_success/total)*100:.1f}%)")
        print(f"Markdown Success: {md_success} ({(md_success/total)*100:.1f}%)")
        print(f"Failed: {failed}")
        print(f"Average Processing Time: {avg_time:.2f}s")
        print(f"Total Processing Time: {total_time:.2f}s")
        print(f"{'='*80}")


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Enhanced Custom Batch Processor for Kosmos-2.5 with dedicated prompts (FP16 Version)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output arguments (required)
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Path to input image file or folder containing images')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Path to output folder for results')
    
    # Model configuration
    parser.add_argument('--model_checkpoint', '-m', type=str, 
                       default="microsoft/kosmos-2.5",
                       help='Path to model checkpoint or repository')
    parser.add_argument('--device', '-d', type=str, default=None,
                       help='Device to use (auto-detected if not specified)')
    
    # Prompt selection
    parser.add_argument('--ocr_prompt', type=str, default='ocr_structured',
                       choices=['ocr_detailed', 'ocr_structured'],
                       help='OCR prompt to use')
    parser.add_argument('--md_prompt', type=str, default='markdown_comprehensive',
                       choices=['markdown_comprehensive', 'markdown_technical', 'markdown_business'],
                       help='Markdown prompt to use')
    
    # Token limits and generation parameters
    parser.add_argument('--ocr_tokens', type=int, default=2048,
                       help='Maximum tokens for OCR generation')
    parser.add_argument('--md_tokens', type=int, default=2048,
                       help='Maximum tokens for Markdown generation')
    parser.add_argument('--temperature', '-t', type=float, default=0.1,
                       help='Temperature for Markdown generation (0.0-1.0)')
    
    # File handling
    parser.add_argument('--image_extensions', type=str, nargs='+',
                       default=['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'],
                       help='Image file extensions to process')
    
    # Display options
    parser.add_argument('--show_prompts', action='store_true',
                       help='Display all custom prompts for validation')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    
    return parser.parse_args()

def main():
    """Main execution function with command line argument support"""
    args = get_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    
    # Validate arguments
    if args.temperature < 0 or args.temperature > 1.0:
        print(f"Error: Temperature must be between 0.0 and 1.0, got {args.temperature}")
        sys.exit(1)
    
    print("Enhanced Custom Batch Processor for Kosmos-2.5 (FP16 Version)")
    print("="*60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Model: {args.model_checkpoint}")
    print(f"OCR Prompt: {args.ocr_prompt}")
    print(f"Markdown Prompt: {args.md_prompt}")
    print(f"Device: {args.device or 'auto-detect'}")
    print(f"Temperature: {args.temperature}")
    print("="*60)
    
    try:
        # Initialize processor
        processor = CustomBatchProcessorFP16(
            model_checkpoint=args.model_checkpoint,
            device=args.device
        )
        
        # Display custom prompts if requested
        if args.show_prompts:
            processor.display_prompts()
            print("\nPrompts displayed. Exiting...")
            return
        
        # Validate input path
        if not os.path.exists(args.input):
            print(f"Error: Input path not found: {args.input}")
            sys.exit(1)
        
        # Process images
        batch_results = processor.process_batch(
            input_path=args.input,
            output_dir=args.output,
            ocr_prompt_key=args.ocr_prompt,
            md_prompt_key=args.md_prompt,
            ocr_max_tokens=args.ocr_tokens,
            md_max_tokens=args.md_tokens,
            temperature=args.temperature,
            image_extensions=args.image_extensions
        )
        
        print(f"\nBatch processing completed successfully!")
        print(f"Results saved to: {args.output}")
        
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()