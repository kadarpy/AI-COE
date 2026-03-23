# RAG Evaluation Proof of Concept

A hands-on learning project for evaluating Retrieval-Augmented Generation (RAG) systems using structured LLM evaluation techniques. This POC demonstrates how to systematically measure quality issues like hallucination, unfaithfulness, and answer irrelevance in GenAI applications.

##  Project Objective

**Key Question:** Does structured LLM evaluation actually catch problems that manual testing doesn't?

This project answers that question by:
1. Building a minimal RAG-based Q&A bot
2. Creating 20 test questions (answerable, partially answerable, and unanswerable)
3. Running structured evaluations using **DeepEval** metrics
4. Analyzing results to quantify which issues evaluation catches vs. manual testing misses
5. Documenting findings with specific examples

**This is not about building a reusable tool—it's about learning how LLMs fail and how to measure that failure.**

---

##  Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Test Cases](#test-cases)
- [Results & Analysis](#results--analysis)
- [Architecture](#architecture)
- [Supported LLM Providers](#supported-llm-providers)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [References](#references)

---

##  Features

### Core Functionality
- **RAG Pipeline**: Documents → Chunking → Vector Embeddings → ChromaDB → Retrieval + LLM Generation
- **Multi-Provider LLM Support**: Groq (free), OpenAI, and Anthropic
- **DeepEval Metrics**: Hallucination, Faithfulness, Answer Relevancy, Contextual Recall
- **Test Case Management**: YAML-based test definition with expected answers and source context
- **Evaluation Dashboard**: Streamlit UI for interactive evaluation and visualization

### User Interfaces
- **Streamlit Web UI**: Interactive Q&A interface with real-time evaluation
- **FastAPI REST API**: RESTful endpoints for programmatic access
- **Command-Line Tools**: Direct Python scripts for evaluation runs

---

##  Quick Start

### Prerequisites
- Python 3.8+
- A free Groq API key (recommended) OR OpenAI API key

### 1. Clone and Setup

```bash
git clone <repository-url>
cd rag_eval_poc
python -m venv venv
venv\Scripts\activate  # Windows
# or: source venv/bin/activate  # macOS/Linux
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Create a `.env` file in the `config/` folder:

```env
# Choose your LLM provider (groq, openai)
LLM_PROVIDER=groq
API_KEY=your_API_KEY_here
LLM_MODEL=llama-3.3-70b-versatile

# OR for OpenAI:
# OPENAI_API_KEY=your_openai_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com)

### 4. Run the Application

```bash
# Streamlit UI (recommended for exploration)
python run_app.py

# FastAPI Server
python run_api.py

# Run evaluation tests
python tests/evaluation/run_eval.py
```

---

##  Installation

### Full Setup with Virtual Environment

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# Verify installation
python -c "import deepeval, langchain, chromadb; print('✓ All dependencies installed')"
```

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `langchain` | ≥0.2.0 | LLM framework and RAG chains |
| `chromadb` | Latest | Vector database for embeddings |
| `deepeval` | Latest | LLM evaluation metrics framework |
| `sentence-transformers` | ≥2.0.0 | Embedding generation |
| `streamlit` | ≥1.28.0 | Web UI framework |
| `fastapi` | ≥0.104.0 | REST API framework |
| `groq` | ≥0.9.0 | Groq LLM API client |

---

## ⚙️ Configuration

### Environment Variables

Set these in `config/.env`:

```env
# LLM Provider
LLM_PROVIDER=groq                           # groq, openai, anthropic
API_KEY=your_key_here                  # For Groq (free)
LLM_MODEL=llama-3.3-70b-versatile          # Groq model selection

# OpenAI (alternative)
OPENAI_API_KEY=your_key_here                # For OpenAI API
OPENAI_API_BASE=https://api.openai.com/v1   # Optional custom endpoint

# Optional
DOCUMENT_PATH=/custom/path/to/documents     # Custom document directory
```

### Application Configuration

Edit [src/config.py](src/config.py) for system settings:

```python
# Document Processing
PDF_CHUNK_SIZE = 800          # Characters per chunk
PDF_CHUNK_OVERLAP = 100       # Character overlap between chunks
RETRIEVER_K = 1               # Number of documents to retrieve

# Vector Store
VECTOR_STORE_TYPE = "chroma"  # Embedding storage backend
```

---

##  Project Structure

```
rag_eval_poc/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── run_app.py                         # Entry point: Streamlit UI
├── run_api.py                         # Entry point: FastAPI server
├── TO_DO.md                           # Project scope & roadmap
│
├── config/
│   ├── .env                           # Environment variables (DO NOT COMMIT)
│   └── .streamlit/
│       └── config.toml                # Streamlit settings
│
├── src/
│   ├── app.py                         # Streamlit web interface
│   ├── api.py                         # FastAPI REST server
│   ├── config.py                      # Configuration management
│   ├── demo.py                        # RAG bot demonstration
│   ├── evaluation.py                  # Evaluation orchestration
│   ├── validators.py                  # Input/output validation
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── loader.py                  # Document loading utilities
│   │   ├── vector_store.py            # ChromaDB integration
│   │   └── rag_chain.py               # LangChain RAG pipeline
│   │
│   ├── data/
│   │   └── documents/
│   │       └── document.txt           # Sample document
│   │
│   └── chroma_db/                     # Vector database storage
│       ├── chroma.sqlite3
│       └── [embedding collections]/
│
├── tests/
│   ├── evaluation/
│   │   ├── test_cases.yaml            # 20 test questions (YAML format)
│   │   ├── run_eval.py                # Evaluation runner script
│   │   └── LLM_MODEL.py              # Groq-specific evaluation
│   │
│   └── results/
│       ├── evaluation_report_1.md     # Markdown evaluation reports
│       ├── evaluation_report_2.md
│       ├── evaluation_results_1.json  # JSON results for analysis
│       └── evaluation_results_2.json
```

---

##  Usage

### Interactive Web UI (Recommended)

```bash
python run_app.py
```

Visit `http://localhost:8501` to:
- Ask questions to the RAG bot
- Run single-question evaluations
- View evaluation metrics in real-time
- Analyze historical results with visualizations

### REST API Server

```bash
python run_api.py
```

Start at `http://localhost:8000` with:
- `/docs` - Interactive API documentation (Swagger)
- `/health` - Health check endpoint
- `/ask` - Query the RAG bot
- `/evaluate` - Evaluate a query

### Example API Request

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?"}'
```

### Command-Line Evaluation

```bash
# Run full evaluation suite
python tests/evaluation/run_eval.py

# Run with Groq backend
python tests/evaluation/LLM_MODEL.py
```

---

##  Evaluation Metrics

This project uses **4 core DeepEval metrics**:

### 1. **Hallucination Score** (0-1, higher is better)
- **Measures**: Does the bot invent information not in the source documents?
- **Use Case**: Catches confident false statements
- **Example**: Q: "Does the company have an office in Mars?" → Bot: "Yes, in downtown Mars" → Hallucination detected

### 2. **Faithfulness Score** (0-1, higher is better)
- **Measures**: Does the answer stick to source material or add external knowledge?
- **Use Case**: Ensures bot uses provided documents, not training data
- **Example**: Q: "What happened in 1969?" → Bot uses Wikipedia instead of documents → Unfaithful

### 3. **Answer Relevancy Score** (0-1, higher is better)
- **Measures**: Does the answer actually address the question asked?
- **Use Case**: Catches off-topic or rambling responses
- **Example**: Q: "What's the price?" → Bot: "Companies exist" → Irrelevant

### 4. **Contextual Recall Score** (0-1, higher is better)
- **Measures**: What percentage of necessary context was retrieved?
- **Use Case**: Identifies retrieval failures
- **Example**: Q requires 3 documents but only 1 retrieved → Low recall

---

##  Test Cases

### Test Case Format (YAML)

Test cases are defined in [tests/evaluation/test_cases.yaml](tests/evaluation/test_cases.yaml):

```yaml
- id: "Q1"
  category: "straightforward"
  question: "What is the main purpose of this document?"
  expected_answer: "The document describes..."
  source_context: "Section 1, Paragraph 2"
  
- id: "Q2"
  category: "tricky"
  question: "What happens when you combine information from page 3 and page 5?"
  expected_answer: "..."
  source_context: "Page 3 + Page 5"
  
- id: "Q3"
  category: "unanswerable"
  question: "What color is the author's car?"
  expected_answer: "This information is not in the provided documents."
  source_context: "Not included"
```

### Test Distribution

- **10 Straightforward**: Answers clearly in documents
- **5 Tricky**: Requires combining multiple chunks or contextual reasoning
- **5 Unanswerable**: Correct answer is "I don't know" or "not in documents"

---

##  Results & Analysis

### Result Files

Evaluation results are stored in `tests/results/`:

- `evaluation_report_X.md` - Human-readable markdown reports
- `evaluation_results_X.json` - Machine-readable result data

### Example Analysis

```python
import json

with open('tests/results/evaluation_results_1.json') as f:
    results = json.load(f)

# Analyze metric scores
for test in results['tests']:
    print(f"Q{test['id']}: Hallucination={test['hallucination_score']:.2f}, "
          f"Faithfulness={test['faithfulness_score']:.2f}")

# Identify patterns
failed_tests = [t for t in results['tests'] if t['overall_score'] < 0.7]
print(f"\nFailed Tests: {len(failed_tests)}/{len(results['tests'])}")
```

### Key Insights to Look For

1. **Hallucination Patterns**: Which question types trigger false information?
2. **Retrieval Issues**: Do tricky questions fail due to incomplete context?
3. **Metric Correlation**: Do all metrics agree, or do they catch different issues?
4. **False Positives**: Which failures are metric errors vs. actual bot errors?
5. **Coverage**: How many real failures does structured evaluation catch?

---

##  Architecture

### RAG Pipeline Flow

```
Documents
    ↓
[PDF Loader] → Text extraction & cleaning
    ↓
[Text Splitter] → 800-char chunks with 100-char overlap
    ↓
[Embeddings] → sentence-transformers model
    ↓
[ChromaDB] → Vector storage & indexing
    ↓
User Question
    ↓
[Retriever] → Fetch top-K relevant documents
    ↓
[LLM Chain] → (Question + Context) → Answer (Groq/OpenAI)
    ↓
[Evaluator] → DeepEval metrics (Hallucination, Faithfulness, etc.)
    ↓
Results & Visualization
```

### Component Responsibilities

| Module | Purpose |
|--------|---------|
| `rag/loader.py` | PDF/TXT document loading |
| `rag/vector_store.py` | ChromaDB initialization & retrieval |
| `rag/rag_chain.py` | LangChain retrieval chain |
| `demo.py` | RAG bot orchestration |
| `evaluation.py` | DeepEval integration & scoring |
| `validators.py` | Input/output validation |
| `app.py` | Streamlit UI |
| `api.py` | FastAPI REST server |

---

##  Supported LLM Providers

### Groq (Recommended - Free)

- **Cost**: Free tier includes generous limits
- **Models**: `llama-3.3-70b-versatile` (recommended), `llama-3.1-405b-reasoning`
- **Speed**: ~150-300 tokens/second
- **Signup**: [console.groq.com](https://console.groq.com)

```env
LLM_PROVIDER=groq
API_KEY=your_key
LLM_MODEL=llama-3.3-70b-versatile
```

### OpenAI

- **Cost**: ~$0.03 per 1K input tokens (gpt-4o)
- **Models**: `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo`
- **Quality**: Highest quality (best for evaluation judge)

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key
```

### Switching Providers

Change `LLM_PROVIDER` in `.env` and restart the application. All components auto-detect and use the configured provider.

---

##  Troubleshooting

### Common Issues

#### Q: "API key not found"
```
ERROR: API_KEY not set in environment
```

**Solution**: Create `config/.env` with your API key:
```env
API_KEY=your_actual_key_here
```

#### Q: "ChromaDB connection error"
**Solution**:
```bash
# Remove corrupted database
rm -rf src/chroma_db

# Restart application to rebuild
python run_app.py
```

#### Q: "No documents found"
**Solution**: Add documents to `src/data/documents/`:
```bash
cp your_documents.pdf src/data/documents/
# Application auto-reloads on next query
```

#### Q: Slow response times
**Solution**: Reduce chunk retrieval:
```python
# In src/config.py
RETRIEVER_K = 1  # Retrieve fewer chunks
```

#### Q: "Module not found" errors
```bash
# Ensure you're in the project root and venv is activated
cd rag_eval_poc
venv\Scripts\activate
pip install -r requirements.txt
```

---

##  Learning Resources

### Background Reading
Before diving in, understand these concepts:

1. **RAG (Retrieval-Augmented Generation)**
   - Augmenting LLMs with external documents to reduce hallucination
   
2. **Vector Embeddings & Semantic Search**
   - How documents are converted to vectors and matched to queries
   
3. **LLM Hallucination**
   - When models confidently produce false information
   
4. **DeepEval Metrics**
   - Structured techniques for measuring LLM output quality

See **GenAI Concepts Glossary (AICOE-11)** for beginner-friendly definitions.

### Official Documentation
- [DeepEval Docs](https://docs.confident-ai.com/)
- [LangChain Docs](https://docs.langchain.com/)
- [ChromaDB Docs](https://docs.trychroma.com/)
- [Streamlit Docs](https://docs.streamlit.io/)

---

##  Project Workflow

### Step-by-Step Guide

1. **Setup** (30 mins)
   - Install dependencies
   - Configure API key
   - Add sample documents

2. **Build RAG Bot** (1-2 hours)
   - Load documents into ChromaDB
   - Test retrieval manually
   - Verify LLM chain works

3. **Create Test Cases** (1-2 hours) ← Most Important
   - Write 20 questions in YAML
   - Mix straightforward, tricky, and unanswerable
   - Document expected answers

4. **Run Evaluation** (30 mins)
   - Execute test suite: `python tests/evaluation/run_eval.py`
   - Collect results and metrics

5. **Analyze Results** (2-3 hours)
   - Which metrics caught which issues?
   - Which were false positives/negatives?
   - What does "good enough" look like?

6. **Document Findings**
   - Write evaluation report
   - Compare manual vs. automated testing
   - Provide recommendations

---

##  Contributing

This is a POC for learning purposes. Contributions welcome!

### Areas for Enhancement
- [ ] Additional evaluation metrics (ROUGE, BLEU, BERTScore)
- [ ] Batch evaluation for large test suites
- [ ] Result visualization dashboard
- [ ] Multi-document Q&A support
- [ ] Custom prompt templates
- [ ] Performance benchmarking

### Development Setup

```bash
# Create development branch
git checkout -b feature/your-feature

# Make changes, test locally
python -m pytest tests/  # When tests are added

# Submit pull request
```

---

##  License

[Specify your license here]

---

##  Support & Questions

- **Issues**: Create an issue on GitHub
- **Questions**: Contact the AICOE team
- **Learning**: See the Background Reading section above

---

##  Key Takeaways

By completing this POC, you'll understand:

 How RAG systems work and when they hallucinate  
 Why structured evaluation catches problems manual testing misses  
 How to measure LLM output quality with metrics  
 What "good enough" really means for GenAI apps  
 Practical tools (DeepEval, LangChain, ChromaDB) for evaluation  

**The key insight**: Evaluation metrics aren't perfect, but they're far more scalable and consistent than manual spot-checking. The combination of structured evaluation + human analysis is how you build reliable GenAI systems.

---

**Last Updated**: March 2026  
**Status**: Active POC  
**Maintained By**: AICOE Team
