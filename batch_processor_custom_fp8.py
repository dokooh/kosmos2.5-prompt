#!/usr/bin/env python3
"""
Custom Batch Processor for Kosmos-2.5 with Dedicated Prompts

This script is based on batch_processor_fp8_mixed.py but includes dedicated prompts
for both OCR and Markdown extraction with enhanced prompt validation and display.

Features:
- Custom prompts for OCR and Markdown tasks
- Prompt validation and display
- FP8 mixed precision support
- Enhanced error handling
- Detailed output formatting
"""

import os
import sys
import time
import json
import logging
import threading
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
from PIL import Image

# Import the 8-bit mixed precision inference modules
try:
    sys.path.append(str(Path(__file__).parent / "fp8"))
    from fp8.ocr_fp8_mixed import EightBitOCRInference, debug_checkpoint
    from fp8.md_fp8_mixed import EightBitMarkdownInference
except ImportError as e:
    print(f"❌ Error importing 8-bit mixed precision modules: {e}")
    print("Make sure ocr_fp8_mixed.py and md_fp8_mixed.py are in the fp8/ directory")
    sys.exit(1)

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CustomBatchProcessor:
    """Custom batch processor with dedicated prompts for OCR and Markdown extraction"""
    
    def __init__(self, 
                 model_checkpoint: str,
                 device: str = None,
                 use_8bit: bool = True,
                 mixed_precision: bool = True):
        """
        Initialize the custom batch processor
        
        Args:
            model_checkpoint: Path to model checkpoint
            device: Device to use for inference
            use_8bit: Enable 8-bit quantization
            mixed_precision: Enable mixed precision
        """
        self.model_checkpoint = model_checkpoint
        self.device = device
        self.use_8bit = use_8bit
        self.mixed_precision = mixed_precision
        
        # Thread-local storage for inference engines
        self._local = threading.local()
        
        # Custom prompts for different tasks
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
        
        logger.info(f"🚀 Custom Batch Processor initialized")
        logger.info(f"📁 Model: {self.model_checkpoint}")
        logger.info(f"⚙️  8-bit: {self.use_8bit}, Mixed Precision: {self.mixed_precision}")
        logger.info(f"📝 Loaded {len(self.prompts)} custom prompts")
    
    def display_prompts(self):
        """Display all available prompts for validation"""
        print("\\n" + "="*80)
        print("📝 CUSTOM PROMPTS VALIDATION")
        print("="*80)
        
        for prompt_name, prompt_text in self.prompts.items():
            print(f"\\n🎯 {prompt_name.upper().replace('_', ' ')}:")
            print("-" * 60)
            print(prompt_text)
            print("-" * 60)
            print(f"📏 Length: {len(prompt_text)} characters")
            print(f"📐 Token estimate: ~{len(prompt_text.split())} words")
    
    def validate_prompt(self, prompt_text: str, prompt_name: str) -> Dict[str, Any]:
        """Validate prompt quality and characteristics"""
        validation = {
            'name': prompt_name,
            'length': len(prompt_text),
            'word_count': len(prompt_text.split()),
            'has_grounding': '<grounding>' in prompt_text,
            'has_markdown': '<md>' in prompt_text,
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
    
    def _get_ocr_engine(self) -> EightBitOCRInference:
        """Get thread-local OCR engine"""
        if not hasattr(self._local, 'ocr_engine') or self._local.ocr_engine is None:
            self._local.ocr_engine = EightBitOCRInference(
                model_checkpoint=self.model_checkpoint,
                device=self.device,
                use_8bit=self.use_8bit,
                mixed_precision=self.mixed_precision
            )
        return self._local.ocr_engine
    
    def _get_md_engine(self) -> EightBitMarkdownInference:
        """Get thread-local Markdown engine"""
        if not hasattr(self._local, 'md_engine') or self._local.md_engine is None:
            self._local.md_engine = EightBitMarkdownInference(
                model_checkpoint=self.model_checkpoint,
                device=self.device,
                use_8bit=self.use_8bit,
                mixed_precision=self.mixed_precision
            )
        return self._local.md_engine
    
    def process_image_with_custom_ocr(self, 
                                    image_path: str, 
                                    prompt_key: str = 'ocr_detailed',
                                    max_tokens: int = 1024) -> Dict[str, Any]:
        """Process image with custom OCR prompt"""
        
        if prompt_key not in self.prompts:
            raise ValueError(f"Unknown prompt key: {prompt_key}")
        
        prompt_text = self.prompts[prompt_key]
        image_name = Path(image_path).stem
        
        print(f"\\n🔍 OCR Processing: {image_name}")
        print(f"📝 Using prompt: {prompt_key}")
        print(f"🎯 Prompt preview: {prompt_text[:100]}...")
        
        start_time = time.time()
        
        try:
            # Validate prompt
            validation = self.validate_prompt(prompt_text, prompt_key)
            
            # Get OCR engine
            ocr_engine = self._get_ocr_engine()
            
            # Load and validate image
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Process with custom prompt
            print(f"⚡ Starting OCR inference...")
            result = ocr_engine.perform_ocr_custom_prompt(
                image_path=image_path,
                custom_prompt=prompt_text,
                max_tokens=max_tokens
            )
            
            processing_time = time.time() - start_time
            
            # Enhanced result formatting
            enhanced_result = {
                'success': True,
                'image_path': image_path,
                'image_name': image_name,
                'prompt_key': prompt_key,
                'prompt_validation': validation,
                'processing_time': processing_time,
                'ocr_results': result,
                'text_extracted': result.get('extracted_text', ''),
                'text_regions': result.get('text_regions', 0),
                'bounding_boxes': result.get('bounding_boxes', []),
                'confidence_scores': result.get('confidence_scores', []),
                'quantization_used': result.get('quantization_info', 'unknown')
            }
            
            print(f"✅ OCR completed in {processing_time:.2f}s")
            print(f"📊 Text regions found: {enhanced_result['text_regions']}")
            
            return enhanced_result
            
        except Exception as e:
            error_result = {
                'success': False,
                'image_path': image_path,
                'image_name': image_name,
                'prompt_key': prompt_key,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
            print(f"❌ OCR failed: {e}")
            return error_result
    
    def process_image_with_custom_markdown(self, 
                                         image_path: str, 
                                         prompt_key: str = 'markdown_comprehensive',
                                         max_tokens: int = 2048,
                                         temperature: float = 0.1) -> Dict[str, Any]:
        """Process image with custom Markdown prompt"""
        
        if prompt_key not in self.prompts:
            raise ValueError(f"Unknown prompt key: {prompt_key}")
        
        prompt_text = self.prompts[prompt_key]
        image_name = Path(image_path).stem
        
        print(f"\\n📝 Markdown Processing: {image_name}")
        print(f"🎯 Using prompt: {prompt_key}")
        print(f"📄 Prompt preview: {prompt_text[:100]}...")
        
        start_time = time.time()
        
        try:
            # Validate prompt
            validation = self.validate_prompt(prompt_text, prompt_key)
            
            # Get Markdown engine
            md_engine = self._get_md_engine()
            
            # Load and validate image
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Process with custom prompt
            print(f"⚡ Starting Markdown generation...")
            result = md_engine.generate_markdown_custom_prompt(
                image_path=image_path,
                custom_prompt=prompt_text,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            processing_time = time.time() - start_time
            
            # Analyze generated markdown
            markdown_text = result.get('generated_markdown', '')
            
            # Enhanced result formatting
            enhanced_result = {
                'success': True,
                'image_path': image_path,
                'image_name': image_name,
                'prompt_key': prompt_key,
                'prompt_validation': validation,
                'processing_time': processing_time,
                'markdown_results': result,
                'generated_markdown': markdown_text,
                'word_count': len(markdown_text.split()),
                'char_count': len(markdown_text),
                'headers': markdown_text.count('#'),
                'lists': markdown_text.count('- ') + markdown_text.count('* '),
                'tables': markdown_text.count('|'),
                'code_blocks': markdown_text.count('```'),
                'quantization_used': result.get('quantization_info', 'unknown')
            }
            
            print(f"✅ Markdown completed in {processing_time:.2f}s")
            print(f"📊 Generated {enhanced_result['word_count']} words, {enhanced_result['headers']} headers")
            
            return enhanced_result
            
        except Exception as e:
            error_result = {
                'success': False,
                'image_path': image_path,
                'image_name': image_name,
                'prompt_key': prompt_key,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
            print(f"❌ Markdown generation failed: {e}")
            return error_result
    
    def process_image_dual_mode(self, 
                              image_path: str,
                              ocr_prompt_key: str = 'ocr_detailed',
                              md_prompt_key: str = 'markdown_comprehensive',
                              ocr_max_tokens: int = 1024,
                              md_max_tokens: int = 2048,
                              temperature: float = 0.1) -> Dict[str, Any]:
        """Process image with both OCR and Markdown using custom prompts"""
        
        image_name = Path(image_path).stem
        
        print(f"\\n🎯 DUAL MODE PROCESSING: {image_name}")
        print("="*60)
        
        start_time = time.time()
        
        # Process OCR
        print(f"\\n1️⃣ OCR PHASE:")
        ocr_result = self.process_image_with_custom_ocr(
            image_path=image_path,
            prompt_key=ocr_prompt_key,
            max_tokens=ocr_max_tokens
        )
        
        # Process Markdown
        print(f"\\n2️⃣ MARKDOWN PHASE:")
        md_result = self.process_image_with_custom_markdown(
            image_path=image_path,
            prompt_key=md_prompt_key,
            max_tokens=md_max_tokens,
            temperature=temperature
        )
        
        total_time = time.time() - start_time
        
        # Combined results
        combined_result = {
            'image_path': image_path,
            'image_name': image_name,
            'total_processing_time': total_time,
            'ocr_result': ocr_result,
            'markdown_result': md_result,
            'both_successful': ocr_result.get('success', False) and md_result.get('success', False),
            'prompts_used': {
                'ocr_prompt': ocr_prompt_key,
                'markdown_prompt': md_prompt_key
            }
        }
        
        print(f"\\n🎊 DUAL MODE SUMMARY:")
        print(f"   OCR Success: {'✅' if ocr_result.get('success') else '❌'}")
        print(f"   Markdown Success: {'✅' if md_result.get('success') else '❌'}")
        print(f"   Total Time: {total_time:.2f}s")
        
        return combined_result
    
    def save_results(self, results: Dict[str, Any], output_dir: str):
        """Save processing results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_name = results['image_name']
        
        # Save combined results
        results_file = output_path / f"{image_name}_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # Save OCR text if successful
        if results['ocr_result'].get('success'):
            ocr_text = results['ocr_result'].get('text_extracted', '')
            if ocr_text:
                ocr_file = output_path / f"{image_name}_ocr.txt"
                with open(ocr_file, 'w', encoding='utf-8') as f:
                    f.write(ocr_text)
        
        # Save Markdown if successful
        if results['markdown_result'].get('success'):
            markdown_text = results['markdown_result'].get('generated_markdown', '')
            if markdown_text:
                md_file = output_path / f"{image_name}_markdown.md"
                with open(md_file, 'w', encoding='utf-8') as f:
                    f.write(markdown_text)
        
        print(f"\\n💾 Results saved to: {output_path}")
        return str(output_path)
    
    def process_batch(self, input_path: str, output_dir: str, 
                     ocr_prompt_key: str = 'ocr_structured',
                     md_prompt_key: str = 'markdown_comprehensive',
                     ocr_max_tokens: int = 2048,
                     md_max_tokens: int = 2048,
                     temperature: float = 0.1,
                     image_extensions: List[str] = None) -> Dict[str, Any]:
        """Process a single image or batch of images with comprehensive reporting"""
        
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        # Determine if input is file or directory
        input_path_obj = Path(input_path)
        
        if input_path_obj.is_file():
            # Single image processing
            if input_path_obj.suffix.lower() in image_extensions:
                images_to_process = [str(input_path_obj)]
            else:
                raise ValueError(f"File {input_path} is not a supported image format")
        elif input_path_obj.is_dir():
            # Directory processing - find all images
            images_to_process = []
            for ext in image_extensions:
                for case_ext in [ext.lower(), ext.upper()]:
                    pattern = f"*{case_ext}"
                    found_files = list(input_path_obj.glob(pattern))
                    images_to_process.extend([str(f) for f in found_files])
            
            # Remove duplicates and sort
            images_to_process = list(set(images_to_process))
            images_to_process.sort()
            
            if not images_to_process:
                raise ValueError(f"No supported images found in directory {input_path}")
        else:
            raise ValueError(f"Input path {input_path} does not exist")
        
        logger.info(f"🔍 Found {len(images_to_process)} image(s) to process")
        print(f"📊 Processing {len(images_to_process)} image(s)")
        
        # Initialize batch results
        batch_results = {
            'total_images': len(images_to_process),
            'successful_images': 0,
            'failed_images': 0,
            'individual_results': [],
            'batch_start_time': time.time(),
            'configuration': {
                'ocr_prompt_key': ocr_prompt_key,
                'md_prompt_key': md_prompt_key,
                'ocr_max_tokens': ocr_max_tokens,
                'md_max_tokens': md_max_tokens,
                'temperature': temperature,
                'model_checkpoint': self.model_checkpoint,
                'use_8bit': self.use_8bit,
                'mixed_precision': self.mixed_precision
            }
        }
        
        # Process each image
        for i, image_path in enumerate(images_to_process, 1):
            print(f"\\n{'=' * 60}")
            print(f"📷 Processing image {i}/{len(images_to_process)}: {Path(image_path).name}")
            print(f"{'=' * 60}")
            
            try:
                result = self.process_image_dual_mode(
                    image_path=image_path,
                    ocr_prompt_key=ocr_prompt_key,
                    md_prompt_key=md_prompt_key,
                    ocr_max_tokens=ocr_max_tokens,
                    md_max_tokens=md_max_tokens,
                    temperature=temperature,
                    output_dir=output_dir
                )
                
                batch_results['individual_results'].append(result)
                
                if result['both_successful']:
                    batch_results['successful_images'] += 1
                    logger.info(f"✅ Successfully processed {Path(image_path).name}")
                else:
                    batch_results['failed_images'] += 1
                    logger.warning(f"❌ Failed to process {Path(image_path).name}")
                    
            except Exception as e:
                batch_results['failed_images'] += 1
                error_result = {
                    'image_path': image_path,
                    'image_name': Path(image_path).name,
                    'success': False,
                    'error': str(e),
                    'both_successful': False
                }
                batch_results['individual_results'].append(error_result)
                logger.error(f"💥 Error processing {Path(image_path).name}: {e}")
        
        # Calculate batch statistics
        batch_results['batch_end_time'] = time.time()
        batch_results['total_batch_time'] = batch_results['batch_end_time'] - batch_results['batch_start_time']
        batch_results['success_rate'] = (batch_results['successful_images'] / batch_results['total_images']) * 100
        batch_results['average_time_per_image'] = batch_results['total_batch_time'] / batch_results['total_images']
        
        # Save batch report
        batch_report_file = Path(output_dir) / "batch_processing_report.json"
        try:
            with open(batch_report_file, 'w', encoding='utf-8') as f:
                json.dump(batch_results, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"📄 Batch report saved to: {batch_report_file}")
        except Exception as e:
            logger.error(f"Failed to save batch report: {e}")
        
        # Print batch summary
        self._print_batch_summary(batch_results)
        
        return batch_results
    
    def _print_batch_summary(self, batch_results: Dict[str, Any]):
        """Print comprehensive batch processing summary"""
        print(f"\\n{'=' * 80}")
        print("📊 BATCH PROCESSING SUMMARY")
        print(f"{'=' * 80}")
        
        print(f"\\n📈 STATISTICS:")
        print(f"  Total Images: {batch_results['total_images']}")
        print(f"  Successful: {batch_results['successful_images']} ✅")
        print(f"  Failed: {batch_results['failed_images']} ❌")
        print(f"  Success Rate: {batch_results['success_rate']:.1f}%")
        
        print(f"\\n⏱️  TIMING:")
        print(f"  Total Time: {batch_results['total_batch_time']:.2f}s")
        print(f"  Average per Image: {batch_results['average_time_per_image']:.2f}s")
        print(f"  Images per Minute: {(batch_results['total_images'] / batch_results['total_batch_time']) * 60:.1f}")
        
        config = batch_results['configuration']
        print(f"\\n⚙️  CONFIGURATION:")
        print(f"  OCR Prompt: {config['ocr_prompt_key']}")
        print(f"  Markdown Prompt: {config['md_prompt_key']}")
        print(f"  OCR Tokens: {config['ocr_max_tokens']}")
        print(f"  MD Tokens: {config['md_max_tokens']}")
        print(f"  Temperature: {config['temperature']}")
        print(f"  8-bit Quantization: {'Enabled' if config['use_8bit'] else 'Disabled'}")
        print(f"  Mixed Precision: {'Enabled' if config['mixed_precision'] else 'Disabled'}")
        
        print(f"\\n{'=' * 80}")
        
        if batch_results['failed_images'] > 0:
            print("⚠️  Some images failed to process. Check individual results for details.")
        else:
            print("🎉 All images processed successfully!")


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Enhanced Custom Batch Processor for Kosmos-2.5 with dedicated prompts',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output arguments (required)
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Path to input image file or folder containing images')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Path to output folder for results')
    
    # Model configuration
    parser.add_argument('--model_checkpoint', '-m', type=str, 
                       default="fp8/models/kosmos-fp8-mixed",
                       help='Path to model checkpoint')
    parser.add_argument('--device', '-d', type=str, default=None,
                       help='Device to use (auto-detected if not specified)')
    
    # Processing configuration
    parser.add_argument('--no_8bit', action='store_true',
                       help='Disable 8-bit quantization')
    parser.add_argument('--no_mixed_precision', action='store_true',
                       help='Disable mixed precision')
    
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
        print("🔍 Debug mode enabled")
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
        print("📝 Verbose mode enabled")
    
    # Validate arguments
    if args.temperature < 0 or args.temperature > 1.0:
        print(f"❌ Error: Temperature must be between 0.0 and 1.0, got {args.temperature}")
        sys.exit(1)
    
    print("🚀 Enhanced Custom Batch Processor for Kosmos-2.5")
    print("="*60)
    print(f"📥 Input: {args.input}")
    print(f"📤 Output: {args.output}")
    print(f"🤖 Model: {args.model_checkpoint}")
    print(f"🔍 OCR Prompt: {args.ocr_prompt}")
    print(f"📝 Markdown Prompt: {args.md_prompt}")
    print(f"⚡ 8-bit Quantization: {'Disabled' if args.no_8bit else 'Enabled'}")
    print(f"🔥 Mixed Precision: {'Disabled' if args.no_mixed_precision else 'Enabled'}")
    print(f"🌡️  Temperature: {args.temperature}")
    print("="*60)
    
    try:
        # Initialize processor
        processor = CustomBatchProcessor(
            model_checkpoint=args.model_checkpoint,
            device=args.device,
            use_8bit=not args.no_8bit,
            mixed_precision=not args.no_mixed_precision
        )
        
        # Display custom prompts if requested
        if args.show_prompts:
            processor.display_prompts()
        
        # Validate input path
        if not os.path.exists(args.input):
            print(f"❌ Error: Input path not found: {args.input}")
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
        
        # Exit with appropriate code
        if batch_results['failed_images'] == 0:
            print("\\n🎉 All images processed successfully!")
            sys.exit(0)
        else:
            print(f"\\n⚠️  {batch_results['failed_images']} image(s) failed to process")
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\\n🛑 Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        logger.error(f"Fatal error in main: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()