"""
Simple Custom Batch Processor for KOSMOS-2.5 with dedicated prompts
Based on fp8 mixed precision implementation without emoji encoding issues
"""

import os
import sys
import time
import logging
import traceback
import argparse
from pathlib import Path
from PIL import Image

# Configure simple logging without emojis
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('batch_custom.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import engines
try:
    from fp8.ocr_fp8_mixed import EightBitOCRInference
    from fp8.md_fp8_mixed import EightBitMarkdownInference
    print("Successfully imported FP8 mixed precision modules")
except ImportError as e:
    print(f"Error importing 8-bit mixed precision modules: {e}")
    sys.exit(1)

class SimpleCustomBatchProcessor:
    """Simple custom batch processor with dedicated prompts"""
    
    def __init__(self, model_checkpoint="fp8/models/kosmos-fp8-mixed", use_8bit=True, mixed_precision=True):
        """Initialize with custom prompts"""
        
        self.model_checkpoint = model_checkpoint
        self.use_8bit = use_8bit
        self.mixed_precision = mixed_precision
        
        # Define custom prompts without format string issues
        self.prompts = {
            'ocr_detailed': "<grounding>Perform comprehensive OCR analysis on this image. Extract all visible text with precise bounding box coordinates. Include: 1. Main headings and titles 2. Body text and paragraphs 3. Tables, lists, and structured data 4. Form fields and labels 5. Small text and footnotes 6. Any numerical data or codes. Provide exact text extraction with spatial relationships.",
            
            'ocr_structured': "<grounding>Extract all text from this document image in a structured format. Identify and categorize: Headers and section titles, Paragraph content, Bullet points and numbered lists, Table data with row/column structure, Captions and labels, Page numbers and metadata. Maintain the document's hierarchical structure and reading order.",
            
            'markdown_comprehensive': "<md>Convert this document image to clean, well-structured Markdown format. Generate comprehensive markdown that includes proper heading hierarchy, paragraph formatting with appropriate spacing, lists with correct indentation, tables with proper alignment, code blocks for technical content. Maintain original document flow and organization, preserve technical terminology and formatting, include all visible text content. Use semantic markdown elements, ensure proper nesting and hierarchy, create readable professional output following markdown best practices. Generate complete publication-ready markdown.",
            
            'markdown_technical': "<md>Analyze this technical document and convert it to specialized Markdown optimized for technical content. Include code snippets in appropriate language blocks, mathematical expressions and formulas, technical diagrams descriptions, API documentation formatting, configuration examples, command-line instructions. Use clear section hierarchy with descriptive headers, technical glossaries and definitions, step-by-step procedures as ordered lists, important notes and warnings as blockquotes. Apply consistent code fence languages, proper table formatting for specifications, emphasis for key terms and concepts, inline code for technical terms. Create comprehensive technical documentation in markdown format.",
            
            'markdown_business': "<md>Transform this business document into professional Markdown suitable for corporate documentation. Structure with executive summary sections, key findings and recommendations, financial data in well-formatted tables, process flows and procedures, organizational charts descriptions, project timelines and milestones. Apply clear heading hierarchy for navigation, bullet points for key insights, numbered lists for procedures, tables for data presentation, blockquotes for important statements, professional language and tone. Maintain consistent formatting throughout, appropriate emphasis and highlighting, clean readable layout, corporate documentation style. Generate polished business-ready markdown documentation."
        }
        
        # Initialize engines
        self._ocr_engine = None
        self._md_engine = None
        
        logger.info("Simple Custom Batch Processor initialized")
        logger.info(f"Model: {self.model_checkpoint}")
        logger.info(f"8-bit: {self.use_8bit}, Mixed Precision: {self.mixed_precision}")
        logger.info(f"Loaded {len(self.prompts)} custom prompts")
    
    def display_prompts(self):
        """Display all custom prompts for validation"""
        print("\n" + "="*80)
        print("CUSTOM PROMPTS VALIDATION")
        print("="*80)
        
        for prompt_name, prompt_text in self.prompts.items():
            print(f"\nPROMPT: {prompt_name.upper().replace('_', ' ')}")
            print("-" * 60)
            print(prompt_text[:200] + "..." if len(prompt_text) > 200 else prompt_text)
            print("-" * 60)
            print(f"Length: {len(prompt_text)} characters")
            print(f"Token estimate: ~{len(prompt_text.split())} words")
    
    def validate_prompt(self, prompt_text, prompt_key):
        """Validate prompt structure and content"""
        validation = {
            'prompt_key': prompt_key,
            'length': len(prompt_text),
            'word_count': len(prompt_text.split()),
            'has_tag': prompt_text.startswith('<grounding>') or prompt_text.startswith('<md>'),
            'tag_type': 'grounding' if prompt_text.startswith('<grounding>') else 'markdown' if prompt_text.startswith('<md>') else 'unknown'
        }
        return validation
    
    def _get_ocr_engine(self):
        """Get or create OCR engine"""
        if self._ocr_engine is None:
            self._ocr_engine = EightBitOCRInference(
                model_checkpoint=self.model_checkpoint,
                use_8bit=self.use_8bit,
                mixed_precision=self.mixed_precision
            )
        return self._ocr_engine
    
    def _get_md_engine(self):
        """Get or create Markdown engine"""
        if self._md_engine is None:
            self._md_engine = EightBitMarkdownInference(
                model_checkpoint=self.model_checkpoint,
                use_8bit=self.use_8bit,
                mixed_precision=self.mixed_precision
            )
        return self._md_engine
    
    def process_ocr_custom(self, image_path, prompt_key='ocr_structured', max_tokens=2048):
        """Process image with custom OCR prompt"""
        start_time = time.time()
        
        try:
            # Get prompt
            if prompt_key not in self.prompts:
                raise ValueError(f"Unknown prompt key: {prompt_key}")
            
            prompt_text = self.prompts[prompt_key]
            image_name = os.path.basename(image_path)
            
            print(f"\nOCR Processing: {image_name}")
            print(f"Using prompt: {prompt_key}")
            print(f"Prompt preview: {prompt_text[:80]}...")
            
            # Validate prompt
            validation = self.validate_prompt(prompt_text, prompt_key)
            
            # Get OCR engine
            ocr_engine = self._get_ocr_engine()
            
            # Load and validate image
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Process with custom prompt
            print("Starting OCR inference...")
            result = ocr_engine.perform_ocr_custom_prompt(
                image_path=image_path,
                custom_prompt=prompt_text,
                max_tokens=max_tokens
            )
            
            processing_time = time.time() - start_time
            
            # Enhanced result
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
                'inference_time': result.get('inference_time', 0),
            }
            
            print(f"OCR completed in {processing_time:.2f}s")
            return enhanced_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"OCR failed: {str(e)}")
            logger.error(f"OCR processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': processing_time,
                'image_path': image_path
            }
    
    def process_markdown_custom(self, image_path, prompt_key='markdown_comprehensive', max_tokens=2048, temperature=0.1):
        """Process image with custom Markdown prompt"""
        start_time = time.time()
        
        try:
            # Get prompt
            if prompt_key not in self.prompts:
                raise ValueError(f"Unknown prompt key: {prompt_key}")
            
            prompt_text = self.prompts[prompt_key]
            image_name = os.path.basename(image_path)
            
            print(f"\nMarkdown Processing: {image_name}")
            print(f"Using prompt: {prompt_key}")
            print(f"Prompt preview: {prompt_text[:80]}...")
            
            # Validate prompt
            validation = self.validate_prompt(prompt_text, prompt_key)
            
            # Get Markdown engine
            md_engine = self._get_md_engine()
            
            # Load and validate image
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Process with custom prompt
            print("Starting Markdown generation...")
            result = md_engine.generate_markdown_custom_prompt(
                image_path=image_path,
                custom_prompt=prompt_text,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            processing_time = time.time() - start_time
            
            # Analyze generated markdown
            markdown_text = result.get('generated_markdown', '')
            
            # Enhanced result
            enhanced_result = {
                'success': True,
                'image_path': image_path,
                'image_name': image_name,
                'prompt_key': prompt_key,
                'prompt_validation': validation,
                'processing_time': processing_time,
                'markdown_results': result,
                'generated_markdown': markdown_text,
                'markdown_analysis': {
                    'length': len(markdown_text),
                    'lines': markdown_text.count('\n'),
                    'headers': markdown_text.count('#'),
                    'lists': markdown_text.count('- ') + markdown_text.count('* '),
                    'tables': markdown_text.count('|'),
                    'code_blocks': markdown_text.count('```')
                },
                'inference_time': result.get('inference_time', 0),
            }
            
            print(f"Markdown completed in {processing_time:.2f}s")
            return enhanced_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"Markdown generation failed: {str(e)}")
            logger.error(f"Markdown processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': processing_time,
                'image_path': image_path
            }
    
    def process_image_dual_mode(self, image_path, ocr_prompt='ocr_structured', md_prompt='markdown_comprehensive', output_dir='custom_processing_results'):
        """Process image with both OCR and Markdown using custom prompts"""
        total_start = time.time()
        
        print(f"\nProcessing image: {os.path.basename(image_path)}")
        print(f"DUAL MODE PROCESSING: {os.path.splitext(os.path.basename(image_path))[0]}")
        print("="*60)
        
        # Phase 1: OCR Processing
        print("\n1. OCR PHASE:")
        ocr_result = self.process_ocr_custom(image_path, ocr_prompt)
        ocr_success = ocr_result.get('success', False)
        
        # Phase 2: Markdown Processing
        print("\n2. MARKDOWN PHASE:")
        md_result = self.process_markdown_custom(image_path, md_prompt)
        md_success = md_result.get('success', False)
        
        total_time = time.time() - total_start
        
        # Summary
        print(f"\nDUAL MODE SUMMARY:")
        print(f"   OCR Success: {'YES' if ocr_success else 'NO'}")
        print(f"   Markdown Success: {'YES' if md_success else 'NO'}")
        print(f"   Total Time: {total_time:.2f}s")
        
        # Save results
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Save OCR results
        if ocr_success and ocr_result.get('text_extracted'):
            ocr_file = os.path.join(output_dir, f"{base_name}_ocr_{ocr_prompt}.txt")
            with open(ocr_file, 'w', encoding='utf-8') as f:
                f.write(ocr_result['text_extracted'])
            print(f"   OCR saved: {ocr_file}")
        
        # Save Markdown results
        if md_success and md_result.get('generated_markdown'):
            md_file = os.path.join(output_dir, f"{base_name}_markdown_{md_prompt}.md")
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(md_result['generated_markdown'])
            print(f"   Markdown saved: {md_file}")
        
        print(f"\nResults saved to: {output_dir}")
        
        return {
            'ocr_result': ocr_result,
            'markdown_result': md_result,
            'total_processing_time': total_time,
            'both_successful': ocr_success and md_success
        }

    def process_batch(self, input_path, output_dir, ocr_prompt='ocr_structured', md_prompt='markdown_comprehensive', 
                     image_extensions=['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']):
        """Process a single image or batch of images"""
        
        # Determine if input is file or directory
        input_path_obj = Path(input_path)
        
        if input_path_obj.is_file():
            # Single image processing
            if input_path_obj.suffix.lower() in image_extensions:
                images_to_process = [str(input_path_obj)]
            else:
                print(f"Error: File {input_path} is not a supported image format")
                return False
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
                print(f"Error: No supported images found in directory {input_path}")
                return False
        else:
            print(f"Error: Input path {input_path} does not exist")
            return False
        
        print(f"Found {len(images_to_process)} image(s) to process")
        
        # Process each image
        total_successful = 0
        for i, image_path in enumerate(images_to_process, 1):
            print(f"\n{'='*60}")
            print(f"Processing image {i}/{len(images_to_process)}: {Path(image_path).name}")
            print(f"{'='*60}")
            
            try:
                result = self.process_image_dual_mode(
                    image_path=image_path,
                    ocr_prompt=ocr_prompt,
                    md_prompt=md_prompt,
                    output_dir=output_dir
                )
                
                if result['both_successful']:
                    total_successful += 1
                    
            except Exception as e:
                print(f"Error processing {Path(image_path).name}: {e}")
                logger.error(f"Error processing {image_path}: {e}")
        
        print(f"\n{'='*80}")
        print("BATCH PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"Total images processed: {len(images_to_process)}")
        print(f"Successful: {total_successful}")
        print(f"Failed: {len(images_to_process) - total_successful}")
        print(f"Success rate: {(total_successful/len(images_to_process))*100:.1f}%")
        print(f"Output directory: {output_dir}")
        
        return total_successful == len(images_to_process)

def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Simple Custom Batch Processor for Kosmos-2.5 with dedicated prompts',
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
    
    # Token limits
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
    
    return parser.parse_args()

def main():
    """Main execution function with command line argument support"""
    args = get_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        print("Verbose mode enabled")
    
    # Validate arguments
    if args.temperature < 0 or args.temperature > 1.0:
        print(f"Error: Temperature must be between 0.0 and 1.0, got {args.temperature}")
        sys.exit(1)
    
    print("Simple Custom Batch Processor for Kosmos-2.5")
    print("="*50)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Model: {args.model_checkpoint}")
    print(f"OCR Prompt: {args.ocr_prompt}")
    print(f"Markdown Prompt: {args.md_prompt}")
    print(f"8-bit Quantization: {'Disabled' if args.no_8bit else 'Enabled'}")
    print(f"Mixed Precision: {'Disabled' if args.no_mixed_precision else 'Enabled'}")
    
    try:
        # Initialize processor
        processor = SimpleCustomBatchProcessor(
            model_checkpoint=args.model_checkpoint,
            use_8bit=not args.no_8bit,
            mixed_precision=not args.no_mixed_precision
        )
        
        # Display custom prompts if requested
        if args.show_prompts:
            processor.display_prompts()
        
        # Validate input path
        if not os.path.exists(args.input):
            print(f"Error: Input path not found: {args.input}")
            sys.exit(1)
        
        # Process images
        success = processor.process_batch(
            input_path=args.input,
            output_dir=args.output,
            ocr_prompt=args.ocr_prompt,
            md_prompt=args.md_prompt,
            image_extensions=args.image_extensions
        )
        
        if success:
            print("\n✓ All images processed successfully!")
            sys.exit(0)
        else:
            print("\n⚠ Some images failed to process. Check logs for details.")
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        logger.error(f"Fatal error in main: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()