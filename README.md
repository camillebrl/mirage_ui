# MIRAGE Citation & Interpretability Tool

A web interface for the [MIRAGE](https://aclanthology.org/2024.emnlp-main.347/) interpretability tool, providing interactive visualization for attribution-based citation analysis in language models.

## Overview

MIRAGE (Model Interpretability through Retrieval-Augmented Generation Explanations) is a method for understanding how language models use context when generating responses. This web interface allows you to:

- Upload documents and ask questions
- Generate responses with automatic citations
- Visualize token-level attributions
- Examine document highlights showing which parts influenced the model's answer

## How MIRAGE Works

### 1. CTI (Context–Token Importance)
For each generated token *y<sub>i</sub>*, MIRAGE compares the model's predictive distributions with and without context:
- *P<sup>ctx</sup><sub>i</sub>*: distribution with context
- *P<sup>no-ctx</sup><sub>i</sub>*: distribution without context

It calculates the KL divergence: **m<sub>i</sub> = KL(P<sup>ctx</sup><sub>i</sub> || P<sup>no-ctx</sup><sub>i</sub>)**

Tokens with *m<sub>i</sub> ≥ m\** are considered context-sensitive.

### 2. CCI (Context–Context Importance)
For context-sensitive tokens, MIRAGE measures the importance of context tokens using gradient-based attribution via the [Inseq](https://github.com/inseq-team/inseq) library.

### 3. Citation Generation
The system aggregates attribution scores by document and generates citations when document influence exceeds a threshold.

## Installation

### Prerequisites
- Python 3.10 or higher
- Poetry package manager
- CUDA-compatible GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd mirage-ui
```

2. Install dependencies with Poetry:
```bash
poetry install
```

3. Activate the virtual environment:
```bash
poetry shell
```

4. Run the application:
```bash
python src/mirage_ui/mirage_app.py
```

The web interface will be available at `http://localhost:8000`

## Usage

### Web Interface

1. **System Instruction**: Customize how the model should behave (default: helpful assistant with citations)

2. **Question**: Enter your question about the provided documents

3. **Documents**: Add documents manually or upload a JSON file with the format:
```json
{
  "question": "Your question here",
  "docs": [
    {"title": "Document 1", "text": "Document content..."},
    {"title": "Document 2", "text": "More content..."}
  ],
  "output": "Optional pre-generated output"
}
```

4. **Pre-generated Output** (Optional): If you already have model output, enable this toggle to skip generation and go directly to attribution analysis

5. **Advanced Parameters**:
   - **Model**: Choose from Qwen, Llama, or other supported models
   - **CTI Threshold**: Controls context sensitivity detection
   - **CCI Threshold**: Controls context attribution filtering
   - **Temperature**: Sampling temperature for generation
   - **Max Tokens**: Maximum length of generated response

### Results

The tool provides three main visualizations:

1. **Generated Answer with Citations**: The model's response with automatic citations [1], [2], etc.

2. **Token-level Highlights**: Interactive visualization showing which documents influenced each token, with color coding and intensity based on attribution strength

3. **Document Highlights**: Shows specific text spans in each document that contributed to the answer

## API Usage

The application also provides a REST API at `/process`:

```python
import requests

response = requests.post("http://localhost:8000/process", json={
    "instruction": "You are a helpful assistant...",
    "question": "What is machine learning?",
    "documents": [
        {"title": "ML Guide", "text": "Machine learning is..."}
    ],
    "model": "Qwen/Qwen3-0.6B",
    "cti_threshold": 1,
    "cci_threshold": -5
})

result = response.json()
print(result["output"])  # Generated text with citations
```

## Supported Models

- **Qwen**: Qwen3-0.6B, Qwen2.5-1.5B
- **Llama**: Llama-3.2-1B
- **Custom models**: Any HuggingFace transformers model (may require configuration adjustments)

## Configuration

### Memory Management
The application automatically manages GPU memory and supports:
- Automatic device mapping
- CPU offloading for large models
- Memory cleanup between operations

### Citation Parameters
- **CTI Threshold**: Higher values = more selective context sensitivity
- **CCI Threshold**: Negative values use percentage-based filtering
- **Max Citations**: Limit citations per sentence (default: 3)

## Troubleshooting

### Memory Issues
If you encounter out-of-memory errors:
1. Try smaller models (Qwen3-0.6B instead of larger variants)
2. Reduce `max_new_tokens`
3. Process fewer documents at once

### No Citations Generated
If no citations appear:
1. Lower the CTI threshold
2. Adjust the CCI threshold (try values closer to 0)
3. Check that documents are relevant to the question

### Model Loading Errors
If models fail to load:
1. Ensure you have sufficient disk space
2. Check internet connection for model downloads
3. Verify CUDA installation for GPU support

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [MIRAGE](https://aclanthology.org/2024.emnlp-main.347/) original research
- [Inseq](https://github.com/inseq-team/inseq) library for attribution methods
- [Transformers](https://huggingface.co/transformers/) library for model support