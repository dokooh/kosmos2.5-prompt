# KOSMOS-2.5 Custom Batch Processor

A powerful batch processing framework for [Microsoft's KOSMOS-2.5](https://huggingface.co/microsoft/kosmos-2.5) multimodal model with dedicated custom prompts for OCR and Markdown extraction tasks.

## Features

### 🎯 Custom Prompts
- **5 Dedicated Prompts** optimized for different document types and extraction tasks
- **OCR Prompts**: Detailed text extraction and structured document parsing
- **Markdown Prompts**: Comprehensive, technical, and business-focused formatting

### ⚡ Precision Options
- **FP16 Version** (`batch_processor_prompt_fp16.py`): Standard precision with BFloat16/Float32
- **FP8 Version** (`batch_processor_custom.py`): 8-bit mixed precision for memory efficiency

### 🔄 Processing Modes
- Single image processing
- Batch directory processing
- Dual-mode (OCR + Markdown in one pass)
- Thread-safe multi-processing support

### 📊 Output Formats
- Plain text OCR results with bounding boxes
- Clean Markdown documents
- Comprehensive JSON results with metadata
- Batch processing summaries

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- 16GB+ RAM (32GB+ recommended for large batches)

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd kosmos2.5-prompt
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Verify installation**
```bash
python batch_processor_prompt_fp16.py --show_prompts --input dummy --output dummy
```

## Usage

### Quick Start - FP16 Version (Recommended)

**Display all custom prompts:**
```bash
python batch_processor_prompt_fp16.py --input test.png --output results --show_prompts
```

**Process a single image:**
```bash
python batch_processor_prompt_fp16.py \
  --input page_042.png \
  --output results \
  --ocr_prompt ocr_detailed \
  --md_prompt markdown_comprehensive
```

**Process a batch of images:**
```bash
python batch_processor_prompt_fp16.py \
  --input ./images_folder \
  --output ./batch_results \
  --ocr_prompt ocr_structured \
  --md_prompt markdown_technical \
  --ocr_tokens 2048 \
  --md_tokens 2048
```

### Advanced Usage

**Technical documentation with custom settings:**
```bash
python batch_processor_prompt_fp16.py \
  --input technical_docs/ \
  --output tech_results/ \
  --ocr_prompt ocr_structured \
  --md_prompt markdown_technical \
  --temperature 0.1 \
  --ocr_tokens 3072 \
  --md_tokens 4096 \
  --verbose
```

**Business document processing:**
```bash
python batch_processor_prompt_fp16.py \
  --input business_docs/ \
  --output business_results/ \
  --ocr_prompt ocr_detailed \
  --md_prompt markdown_business \
  --temperature 0.2
```

**Specify custom device:**
```bash
python batch_processor_prompt_fp16.py \
  --input images/ \
  --output results/ \
  --device cuda:1
```

## Custom Prompts

### OCR Prompts

#### 1. `ocr_detailed`
Comprehensive OCR analysis with precise bounding box coordinates. Extracts:
- Main headings and titles
- Body text and paragraphs
- Tables, lists, and structured data
- Form fields and labels
- Small text and footnotes
- Numerical data and codes

**Use case**: Maximum text extraction with spatial relationships

#### 2. `ocr_structured`
Structured text extraction with hierarchical organization. Identifies:
- Headers and section titles
- Paragraph content
- Bullet points and numbered lists
- Table data with row/column structure
- Captions and labels
- Page numbers and metadata

**Use case**: Document structure preservation and categorized extraction

### Markdown Prompts

#### 3. `markdown_comprehensive`
Complete document conversion to clean, well-structured Markdown:
- Proper heading hierarchy
- Paragraph formatting with appropriate spacing
- Lists (ordered and unordered) with correct indentation
- Tables with proper alignment
- Code blocks for technical content
- Links and references

**Use case**: General-purpose, publication-ready markdown

#### 4. `markdown_technical`
Specialized Markdown optimized for technical content:
- Code snippets in appropriate language blocks
- Mathematical expressions and formulas
- Technical diagrams descriptions
- API documentation formatting
- Configuration examples
- Command-line instructions

**Use case**: Technical documentation, API docs, code repositories

#### 5. `markdown_business`
Professional Markdown for corporate documentation:
- Executive summary sections
- Key findings and recommendations
- Financial data in well-formatted tables
- Process flows and procedures
- Organizational charts descriptions
- Project timelines and milestones

**Use case**: Business reports, corporate documentation, presentations

## Command Line Arguments

### Required Arguments
- `--input, -i`: Path to input image file or folder
- `--output, -o`: Path to output folder for results

### Model Configuration
- `--model_checkpoint, -m`: Model path or repository (default: `microsoft/kosmos-2.5`)
- `--device, -d`: Device to use (default: auto-detect)

### Prompt Selection
- `--ocr_prompt`: OCR prompt to use
  - Choices: `ocr_detailed`, `ocr_structured`
  - Default: `ocr_structured`
- `--md_prompt`: Markdown prompt to use
  - Choices: `markdown_comprehensive`, `markdown_technical`, `markdown_business`
  - Default: `markdown_comprehensive`

### Generation Parameters
- `--ocr_tokens`: Maximum tokens for OCR (default: 2048)
- `--md_tokens`: Maximum tokens for Markdown (default: 2048)
- `--temperature, -t`: Temperature for generation (default: 0.1, range: 0.0-1.0)

### File Handling
- `--image_extensions`: Image file extensions to process
  - Default: `.jpg .jpeg .png .bmp .tiff .webp`

### Display Options
- `--show_prompts`: Display all custom prompts and exit
- `--verbose, -v`: Enable verbose output
- `--debug`: Enable debug output

## Output Structure

When processing images, the script generates:

```
output_folder/
├── image_name_ocr.txt              # Plain text OCR results
├── image_name_markdown.md          # Generated markdown
├── image_name_results.json         # Complete results with metadata
└── batch_summary.json              # Batch processing statistics
```

### Example Output Files

**OCR Text (`*_ocr.txt`):**
```
Section 1: Introduction
This document provides an overview...
Table 1: Results
Column A    Column B    Column C
```

**Markdown (`*_markdown.md`):**
```markdown
# Section 1: Introduction

This document provides an overview...

## Table 1: Results

| Column A | Column B | Column C |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |
```

**JSON Results (`*_results.json`):**
```json
{
  "image_path": "page_042.png",
  "ocr_result": {
    "success": true,
    "text_extracted": "...",
    "processing_time": 2.34
  },
  "markdown_result": {
    "success": true,
    "generated_markdown": "...",
    "content_analysis": {
      "word_count": 450,
      "line_count": 78
    }
  }
}
```

## Performance Tips

### Memory Management
- **FP16 Version**: ~8-12GB GPU memory per model instance
- **FP8 Version**: ~4-6GB GPU memory per model instance (if implemented)
- Use `--device cpu` for CPU-only processing (slower)

### Speed Optimization
- Process images in batches for better throughput
- Use CUDA if available (10-50x faster than CPU)
- Lower token limits reduce processing time
- Temperature 0.0-0.1 for deterministic, faster results

### Quality vs Speed
- Higher `--ocr_tokens` and `--md_tokens` for complete extraction
- Lower temperature (0.0-0.1) for consistent results
- Higher temperature (0.2-0.5) for more creative formatting

## Project Structure

```
kosmos2.5-prompt/
├── batch_processor_prompt_fp16.py      # FP16 version (recommended)
├── batch_processor_custom.py           # FP8 version with emojis
├── batch_processor_custom_simple.py    # FP8 version without emojis
├── requirements.txt                    # Python dependencies
├── README.md                          # This file
├── CUSTOM_BATCH_PROCESSOR_SUMMARY.md  # Implementation summary
├── fp16/                              # FP16 inference modules
│   ├── ocr.py                         # OCR script
│   ├── md.py                          # Markdown script
│   └── batch_processor.py             # Original batch processor
└── fp8/                               # FP8 inference modules
    ├── ocr_fp8_mixed.py               # FP8 OCR engine
    ├── md_fp8_mixed.py                # FP8 Markdown engine
    └── batch_processor_fp8_mixed.py   # FP8 batch processor
```

## Architecture

### FP16 Version Architecture

```
CustomBatchProcessorFP16
├── FP16OCRInference
│   ├── Model Loading (BFloat16/Float32)
│   ├── Image Processing
│   ├── Custom Prompt Handling
│   └── OCR Post-processing
└── FP16MarkdownInference
    ├── Model Loading (BFloat16/Float32)
    ├── Image Processing
    ├── Custom Prompt Handling
    └── Markdown Generation
```

### Processing Pipeline

1. **Initialization**: Load KOSMOS-2.5 model with appropriate precision
2. **Image Loading**: Support for local files and URLs
3. **Prompt Selection**: Choose from 5 dedicated prompts
4. **Inference**: Generate OCR/Markdown with custom prompts
5. **Post-processing**: Extract and format results
6. **Output**: Save text, markdown, and JSON results

## Troubleshooting

### Common Issues

**CUDA Out of Memory:**
```bash
# Solution 1: Use CPU
python batch_processor_prompt_fp16.py --device cpu --input images/ --output results/

# Solution 2: Process images one at a time (already default)
# Solution 3: Reduce token limits
python batch_processor_prompt_fp16.py --ocr_tokens 1024 --md_tokens 1024 --input images/ --output results/
```

**Model Download Fails:**
```bash
# Pre-download the model
python -c "from transformers import AutoProcessor, Kosmos2_5ForConditionalGeneration; \
Kosmos2_5ForConditionalGeneration.from_pretrained('microsoft/kosmos-2.5'); \
AutoProcessor.from_pretrained('microsoft/kosmos-2.5')"
```

**Import Errors:**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

**Slow Processing:**
- Verify CUDA is being used: Check for "Using device: cuda:0" in output
- Reduce image resolution if very large (>4K)
- Use lower token limits

## Examples

### Example 1: Research Paper Processing
```bash
python batch_processor_prompt_fp16.py \
  --input research_papers/ \
  --output paper_results/ \
  --ocr_prompt ocr_structured \
  --md_prompt markdown_technical \
  --ocr_tokens 4096 \
  --md_tokens 4096 \
  --temperature 0.05
```

### Example 2: Business Reports
```bash
python batch_processor_prompt_fp16.py \
  --input quarterly_reports/ \
  --output report_results/ \
  --ocr_prompt ocr_detailed \
  --md_prompt markdown_business \
  --temperature 0.1
```

### Example 3: Mixed Documents
```bash
python batch_processor_prompt_fp16.py \
  --input mixed_docs/ \
  --output mixed_results/ \
  --ocr_prompt ocr_structured \
  --md_prompt markdown_comprehensive \
  --verbose
```

## Contributing

Contributions are welcome! Areas for improvement:
- Additional custom prompts for specific domains
- Performance optimizations
- Support for additional output formats
- Enhanced error handling
- Multi-GPU support

## License

This project uses Microsoft's KOSMOS-2.5 model. Please refer to the [model card](https://huggingface.co/microsoft/kosmos-2.5) for licensing information.

## Citation

If you use this tool in your research, please cite the original KOSMOS-2.5 paper:

```bibtex
@article{lv2023kosmos,
  title={Kosmos-2.5: A Multimodal Literate Model},
  author={Lv, Tengchao and others},
  journal={arXiv preprint arXiv:2309.11419},
  year={2023}
}
```

## Acknowledgments

- Microsoft Research for the KOSMOS-2.5 model
- Hugging Face for the transformers library
- The open-source community

## Support

For issues, questions, or contributions:
1. Check existing issues in the repository
2. Create a new issue with detailed description
3. Include error messages and system information

---

**Version**: 1.0.0  
**Last Updated**: October 2025  
**Python**: 3.8+  
**Model**: microsoft/kosmos-2.5
