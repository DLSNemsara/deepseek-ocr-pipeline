# High-Volume Legal Document OCR Pipeline

An enterprise-grade Optical Character Recognition (OCR) pipeline designed for processing high volumes of complex legal documents (multi-column forms, handwriting, tables). This project leverages **DeepSeek-OCR (VLM)** for state-of-the-art layout preservation and text extraction.

---

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![DeepSeek](https://img.shields.io/badge/Model-DeepSeek--OCR-orange.svg)
![Hardware](https://img.shields.io/badge/Hardware-NVIDIA%20T4%20%7C%20A100-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

</div>

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Features](#features)
- [Performance & Hardware Analysis](#performance--hardware-analysis)
- [Configuration](#configuration)
- [Usage](#usage)
- [Technical Highlights](#technical-highlights)
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

## Configuration

The pipeline supports various configuration options in the notebook:

```python
# Model loading options
torch_dtype = torch.bfloat16   # Use bfloat16 for stability
attn_implementation = "eager"  # Attention implementation choice
device_map = "auto"            # Automatic GPU placement

# Inference options
base_size = 1024               # Base resolution for model internals
image_size = 640               # Processing resolution (resize target)
crop_mode = True               # Enable cropping for small text regions
test_compress = False          # Disable compression for maximum accuracy
```

You can further tune:
- `batch_size` (if batching images)
- `max_new_tokens` for output length control
- `temperature / top_p` (if the model exposes generative decoding parameters)

---

## Usage

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   sudo apt-get install poppler-utils
   ```

2. **Run the Pipeline**  
   Open `DeepSeek_OCR_Pipeline.ipynb` in Google Colab or a local GPU environment.

3. **Upload Documents**  
   Supports batch uploads of PDF, TIFF (multi-page), JPG, PNG, and mixed types.

4. **Generate Output**  
   The notebook will produce:
   - Raw extracted text
   - Structured Markdown (layout-preserving)
   - Performance summary (latency, throughput)

5. **Reporting**  
   A Markdown performance report can be exported for audit or benchmarking.

---

## Technical Highlights

This project demonstrates:

- **ML Engineering:** Model loading, inference optimization, VRAM management  
- **Pipeline Design:** TIFF/PDF → image conversion prior to inference  
- **Data Engineering:** Integration logic for legacy DBs (AS400) and microservices (conceptual)  
- **Analytics:** Automated performance metrics and report generation  

Planned enhancements (roadmap ideas):
- GPU batching and asynchronous queue ingestion
- Quantized model variants (4-bit AWQ)
- Structured JSON schema output for downstream NLP
- Integration tests simulating Spring Boot image retrieval

---

## Contact

**Sinel Nemsara**

- GitHub: [@DLSNemsara](https://github.com/DLSNemsara)  
- LinkedIn: [sinel-nemsara](https://www.linkedin.com/in/sinel-nemsara)

---

## License

This repository serves as a technical demonstration of an enterprise OCR architecture.  

---
