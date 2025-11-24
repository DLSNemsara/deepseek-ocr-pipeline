# High-Volume Legal Document OCR Pipeline

An enterprise-grade Optical Character Recognition (OCR) pipeline designed for processing high volumes of complex legal documents (multi-column forms, handwriting, tables). This project leverages **DeepSeek-OCR (VLM)** for state-of-the-art layout preservation and text extraction.

---

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![DeepSeek](https://img.shields.io/badge/Model-DeepSeek--OCR-orange.svg)
![Hardware](https://img.shields.io/badge/Hardware-NVIDIA%20T4%20%7C%20A100-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

</div>

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Features](#features)
- [Performance & Hardware Analysis](#performance--hardware-analysis)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Example Output](#example-output)
- [Technical Highlights](#technical-highlights)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Contact](#contact)
- [License](#license)

---

## Architecture Overview

This repository hosts the **Inference Engine (Layer 3)**. In the full production environment, this engine acts as the processing core within a microservices architecture:

1. **Ingestion Layer (External System):**
   - Connects to legacy **IBM AS400 (DB2)** databases via JDBC to query document metadata.
   - Manages batch queues based on priority and filing dates.
2. **Retrieval Layer (External System):**
   - Interacts with an internal **Spring Boot Image Server**.
   - Securely fetches raw byte streams (TIFF/PDF) via REST API.
   - Handles on-the-fly image conversion and sanitization.
3. **Inference Layer (This Repository):**
   - **Model:** `deepseek-ai/DeepSeek-OCR` (Simulated here on NVIDIA T4).
   - **Optimization:** Implements layout-aware prompting to output structured Markdown.

> **Note:** The DB2 extraction and Spring Boot integration scripts are proprietary components and are excluded from this public repository. This repo focuses on the ML inference and OCR optimization logic.

---

## Features

- **Layout Awareness:** Correctly parses complex legal forms, checkboxes, and dual-column layouts.
- **Format Agnostic:** Handles PDF, TIFF (multi-page), JPG, and PNG.
- **Privacy-First:** Reporting module automatically masks filenames to protect sensitive court data.
- **Performance Reporting:** Automatically generates Markdown reports calculating latency (sec/page) and throughput.

---

## Performance & Hardware Analysis

**Test Environment:** Google Colab T4 GPU (16GB VRAM)

| Metric | Value | Notes |
| :--- | :--- | :--- |
| **Model Size** | 7B Parameters | Running in bfloat16 precision |
| **Observed Latency** | ~40s / page | On T4 GPU (Free Tier) |
| **Success Rate** | 100% | Tested on mixed legal forms |

> **⚠️ Production Scaling Note:**  
> Benchmarks in this repository represent a baseline using older hardware (T4).  
> For a production environment processing **200k+ pages/day**, the following upgrades are recommended:
>
> - **Hardware:** Switch to NVIDIA A100 (40GB/80GB) or H100.  
> - **Optimization:** Implement 4-bit quantization (AWQ) to reduce latency to <5s/page.  
> - **Scaling:** Use multiple GPU nodes behind a queue (e.g., RabbitMQ).  

---

## System Requirements

### Hardware Requirements

- **GPU:** NVIDIA GPU with at least 16GB VRAM
  - Recommended: T4 (16GB), A100 (40GB/80GB), or H100
  - Minimum CUDA Compute Capability: 7.0+
- **RAM:** 16GB+ system memory
- **Storage:** 20GB+ free disk space for model and dependencies

### Software Requirements

- **Python:** 3.10 or higher
- **CUDA:** 11.8+ or 12.1+ (for GPU acceleration)
- **Operating System:** Linux (Ubuntu 20.04+), Windows 10/11, or macOS
- **Package Manager:** pip or conda

### Cloud Platform Support

- Google Colab (Free/Pro tier with T4 GPU)
- AWS EC2 (g4dn.xlarge or higher)
- Azure NC-series VMs
- Google Cloud Platform (n1-standard with T4/V100)

---

## Installation

### Option 1: Google Colab (Recommended for Quick Start)

1. Open the notebook directly in Google Colab:
   - Click the "Open in Colab" badge in `DeepSeek_OCR_Pipeline.ipynb`
   - Or visit: [Open in Colab](https://colab.research.google.com/github/DLSNemsara/deepseek-ocr-pipeline/blob/main/DeepSeek_OCR_Pipeline.ipynb)

2. Ensure GPU is enabled:
   - Go to `Runtime` → `Change runtime type`
   - Select `T4 GPU` (or `A100` if using Colab Pro)

3. Run all cells sequentially - dependencies will be installed automatically

### Option 2: Local Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/DLSNemsara/deepseek-ocr-pipeline.git
   cd deepseek-ocr-pipeline
   ```

2. **Create Virtual Environment** (Recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Install System Dependencies** (Linux/macOS)

   ```bash
   # For PDF processing
   sudo apt-get install poppler-utils  # Ubuntu/Debian
   # or
   brew install poppler  # macOS
   ```

5. **Verify Installation**

   ```bash
   python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
   ```

---

## Configuration

The pipeline supports various configuration options in the notebook:

```python
# Model loading configuration
MODEL_PATH = "deepseek-ai/DeepSeek-OCR"
torch_dtype = torch.bfloat16        # Use bfloat16 for stability and performance
attn_implementation = "eager"       # Attention implementation (eager/flash)
device_map = "auto"                 # Automatic GPU placement
use_safetensors = True              # Use safetensors format for security

# Inference configuration
base_size = 1024                    # Base resolution for model internals
image_size = 640                    # Processing resolution (resize target)
crop_mode = True                    # Enable cropping for small text regions
test_compress = False               # Disable compression for maximum accuracy
max_new_tokens = 4096               # Maximum output length
do_sample = False                   # Deterministic generation
```

### Advanced Configuration Options

You can further customize the pipeline by adjusting:

- **Memory Optimization:**
  - `offload_folder`: Directory for CPU offloading to reduce VRAM usage
  - Adjust `device_map` for multi-GPU setups

- **Image Processing:**
  - `dpi`: Resolution for PDF conversion (default: 300)
  - Image preprocessing steps (resize, normalize)

- **Generation Parameters:**
  - `temperature`: Sampling temperature (if `do_sample=True`)
  - `top_p`: Nucleus sampling parameter
  - `repetition_penalty`: Control repetitive output

---

## Usage

### Quick Start with Jupyter Notebook

1. **Open the Notebook**
   
   Launch `DeepSeek_OCR_Pipeline.ipynb` in Google Colab or Jupyter:
   ```bash
   jupyter notebook DeepSeek_OCR_Pipeline.ipynb
   ```

2. **Install Dependencies** (Cell 1)
   
   Run the first cell to install all required packages:
   ```python
   # Note: Specific versions are installed automatically
   # See requirements.txt for exact version specifications
   !pip install transformers tokenizers einops pillow pdf2image PyPDF2
   !apt-get install poppler-utils
   ```

3. **Verify GPU** (Cell 2)
   
   Check that GPU is available and properly configured:
   - Will display GPU name and memory
   - Raises error if GPU not found

4. **Upload Documents** (Cell 3)
   
   Upload your documents using the file picker:
   - Supported formats: **PDF, TIFF (multi-page), JPG, PNG**
   - Can upload multiple files at once
   - Files are stored in `/content/uploads/`

5. **Load Model** (Cell 4)
   
   Load the DeepSeek-OCR model:
   - Takes 2-3 minutes on first run
   - Model weights (~14GB) are cached automatically
   - Uses bfloat16 precision for optimal performance

6. **Convert Documents** (Cell 5)
   
   Convert PDFs and TIFFs to images:
   - Multi-page PDFs → Individual PNG files
   - Multi-page TIFFs → Individual PNG files
   - Images copied to `/content/processed/`

7. **Process Documents** (Cell 6)
   
   Run OCR on all prepared documents:
   - Processes each page sequentially
   - Extracts text in structured Markdown format
   - Shows real-time progress with timing info

8. **Generate Report** (Cell 7)
   
   Create performance analysis report:
   - Calculates latency and throughput metrics
   - Generates success rate statistics
   - Creates Markdown report with masked filenames (privacy)

9. **View Results** (Cell 8-9)
   
   Display and review OCR outputs:
   - Shows sample outputs with bounding boxes
   - Preview extracted text
   - View performance summary

10. **Download Results** (Cell 10)
    
    Package and download all outputs:
    - Creates timestamped ZIP archive
    - Includes OCR results, metadata, and report
    - Downloads automatically in Colab

### Supported Document Formats

- **PDF:** Single and multi-page documents
- **TIFF:** Single and multi-page TIFF files
- **JPG/JPEG:** Standard image format
- **PNG:** Lossless image format

### Output Structure

```
results/
├── DeepSeek_OCR_Performance_Report.md  # Main performance report
├── processing_metadata.json            # Processing statistics
└── doc_001/                            # Per-document results
    ├── result.mmd                      # Extracted text (Markdown)
    ├── result_with_boxes.jpg           # Bounding box visualization
    └── [other processing artifacts]
```

---

## Example Output

### Sample OCR Result

**Input:** Multi-column legal document (court filing)

**Output:** Structured Markdown preserving layout

```markdown
# COURT ORDER

**Case No:** 2024-CV-12345  
**Date Filed:** January 15, 2024

## Parties

**Plaintiff:** John Doe  
**Defendant:** Jane Smith

## Order Details

The court hereby orders that...

| Item | Description | Amount |
|------|-------------|--------|
| Filing Fee | Court costs | $350.00 |
| Service Fee | Process service | $75.00 |
```

### Performance Metrics Example

For a batch of 10 legal documents (mixed formats):

- **Success Rate:** 100%
- **Average Latency:** 38.5 seconds/page
- **Total Characters:** 125,000+
- **Throughput:** ~94 documents/hour (on T4 GPU)

---

## Technical Highlights

This project demonstrates:

- **ML Engineering:** Model loading, inference optimization, VRAM management, bfloat16 precision
- **Pipeline Design:** Multi-format document handling (TIFF/PDF → image conversion)
- **Data Engineering:** Integration patterns for legacy databases (AS400/DB2) and microservices
- **Analytics:** Automated performance metrics, reporting, and benchmarking
- **Privacy:** Filename masking in reports to protect sensitive information

### Key Technical Decisions

1. **bfloat16 Precision:** Balance between accuracy and memory efficiency
2. **No Compression:** `test_compress=False` prioritizes accuracy over speed
3. **Eager Attention:** Compatible with wider range of hardware
4. **Deterministic Generation:** `do_sample=False` ensures consistent outputs
5. **Batch-free Processing:** Sequential processing for stability on free-tier GPUs

### Planned Enhancements

- GPU batching and asynchronous queue ingestion
- Quantized model variants (4-bit AWQ) for faster inference
- Structured JSON schema output for downstream NLP tasks
- Integration tests simulating Spring Boot image retrieval
- Docker containerization for deployment
- REST API wrapper for microservices integration

---

## Troubleshooting

### Common Issues

#### 1. GPU Not Detected

**Error:** `RuntimeError: GPU not available`

**Solutions:**
- **Google Colab:** Go to `Runtime` → `Change runtime type` → Select `T4 GPU`
- **Local:** Verify CUDA installation: `nvidia-smi` and check PyTorch: `torch.cuda.is_available()`
- Ensure CUDA-compatible PyTorch is installed: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

#### 2. Out of Memory (OOM) Error

**Error:** `CUDA out of memory`

**Solutions:**
- Reduce batch size (if processing multiple images)
- Use CPU offloading: set `offload_folder="/tmp/offload"`
- Process documents sequentially instead of in batch
- Use Google Colab Pro for A100 GPU (80GB VRAM)
- Lower `base_size` or `image_size` parameters

#### 3. PDF Conversion Fails

**Error:** `Unable to convert PDF to images`

**Solutions:**
- Install poppler-utils: `sudo apt-get install poppler-utils`
- Check PDF is not encrypted or password-protected
- Verify PDF file is not corrupted: try opening in a PDF viewer
- For large PDFs, process page-by-page manually

#### 4. Model Loading Timeout

**Error:** Model download is slow or times out

**Solutions:**
- Use Hugging Face CLI to pre-download: `huggingface-cli download deepseek-ai/DeepSeek-OCR`
- Check internet connection and firewall settings
- Try using a mirror or proxy for Hugging Face
- In Colab, restart runtime and try again (cached downloads persist)

#### 5. Poor OCR Quality

**Issue:** Text extraction is inaccurate or incomplete

**Solutions:**
- Ensure input images are high quality (300 DPI minimum for scanned docs)
- Set `test_compress=False` for maximum accuracy
- Check image preprocessing (rotation, contrast, brightness)
- For handwriting, results may vary - consider post-processing
- Verify document is in a supported language

#### 6. Dependencies Installation Fails

**Error:** pip install errors or version conflicts

**Solutions:**
- Use Python 3.10 or 3.11 (not 3.12+ due to compatibility)
- Create fresh virtual environment: `python -m venv venv`
- Update pip: `pip install --upgrade pip`
- Install dependencies from requirements.txt: `pip install -r requirements.txt`
- Check system architecture (ARM vs x86_64)

### Getting Help

If you encounter issues not listed here:

1. Check the [Issues](https://github.com/DLSNemsara/deepseek-ocr-pipeline/issues) page
2. Review DeepSeek-OCR [official documentation](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
3. Create a new issue with:
   - Python version and OS
   - GPU model and VRAM
   - Full error traceback
   - Sample document (if not confidential)

---

## Contributing

Contributions are welcome! This project serves as a demonstration and learning resource.

### How to Contribute

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/my-improvement`
3. **Make** your changes with clear commit messages
4. **Test** your changes thoroughly
5. **Submit** a Pull Request with description of changes

### Areas for Contribution

- Performance optimizations and benchmarking
- Support for additional document formats
- Quantization and model optimization
- Deployment guides (Docker, Kubernetes)
- Integration examples with other systems
- Documentation improvements
- Test coverage expansion

### Code Style

- Follow PEP 8 for Python code
- Add docstrings for functions and classes
- Include type hints where appropriate
- Keep functions focused and modular

---

## Contact

**Sinel Nemsara**

- GitHub: [@DLSNemsara](https://github.com/DLSNemsara)  
- LinkedIn: [sinel-nemsara](https://www.linkedin.com/in/sinel-nemsara)

Feel free to reach out for questions, collaboration opportunities, or feedback!

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for full details.

### Summary

You are free to:
- ✅ Use commercially
- ✅ Modify and distribute
- ✅ Use privately
- ✅ Sublicense

Under the conditions:
- 📄 Include original license and copyright notice
- ⚖️ No liability or warranty provided

**Note:** This repository serves as a technical demonstration of an enterprise OCR architecture. The DeepSeek-OCR model has its own license terms from deepseek-ai.

---

**⭐ If you find this project useful, please consider giving it a star on GitHub!**
