# RAG Evaluation POC - Complete Project Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Objective & Scope](#project-objective--scope)
3. [Technology Stack](#technology-stack)
4. [Project Architecture](#project-architecture)
5. [Directory Structure](#directory-structure)
6. [Core Components & Modules](#core-components--modules)
7. [Configuration System](#configuration-system)
8. [Data Pipeline](#data-pipeline)
9. [User Interfaces](#user-interfaces)
10. [Evaluation System](#evaluation-system)
11. [Key Concepts Explained](#key-concepts-explained)
12. [How Everything Works Together](#how-everything-works-together)

---

## Project Overview

### What is This Project?

This is a **Retrieval-Augmented Generation (RAG) Evaluation Proof of Concept** — a learning project designed to understand how LLM evaluation works and why it matters.

### Core Mission

**Key Question:** *Does structured LLM evaluation actually catch problems that manual testing doesn't?*

The project answers this by:
1. Building a minimal RAG-based Q&A bot
2. Creating 20 test questions (mix of answerable, partially answerable, and unanswerable)
3. Running structured evaluations using DeepEval metrics
4. Analyzing results to quantify which issues evaluation catches
5. Documenting findings with specific examples

### Why RAG?

Traditional LLMs have a "knowledge cutoff" — they only know information from their training data. **Retrieval-Augmented Generation (RAG)** fixes this by:
- Retrieving relevant documents first
- Using those documents as context
- Generating answers based on retrieved content

This makes answers more accurate, verifiable, and reducible to source documents.

---

## Project Objective & Scope

### What This Project IS
-  A learning sandbox for understanding LLM evaluation
-  A demonstration of how to measure quality issues (hallucination, unfaithfulness, irrelevance)
-  A proof-of-concept with minimal complexity
-  A test bed for comparing manual vs. automated evaluation

### What This Project IS NOT
-  A production-ready RAG system
-  A reusable evaluation framework
-  A complete CI/CD pipeline
-  An optimization benchmark for multiple models

---

## Technology Stack

### Core Dependencies & Why They're Used

| Package | Version | Purpose |
|---------|---------|---------|
| **langchain** | ≥0.2.0 | LLM orchestration framework (chains, prompts, retrievers) |
| **langchain-groq** | ≥0.1.0 | Integration with Groq LLM API |
| **langchain-huggingface** | ≥0.0.1 | Free embeddings using HuggingFace models |
| **langchain-text-splitters** | ≥0.1.0 | Smart document chunking for RAG |
| **chromadb** | Latest | Vector database for storing & retrieving embeddings |
| **deepeval** | Latest | Structured LLM evaluation metrics |
| **sentence-transformers** | ≥2.0.0 | Generate embeddings from text |
| **streamlit** | ≥1.28.0 | Web UI for interactive demo |
| **fastapi** | ≥0.104.0 | REST API endpoints (optional) |
| **uvicorn** | ≥0.24.0 | ASGI server for FastAPI |
| **pydantic** | ≥2.0.0 | Data validation & type checking |
| **pypdf** | Latest | Extract text from PDF documents |
| **python-dotenv** | Latest | Load environment variables from .env files |
| **pyyaml** | Latest | Parse YAML test case files |
| **pandas** | ≥2.0.0 | Data analysis & result aggregation |
| **plotly** | ≥5.0.0 | Interactive data visualization |
| **groq** | ≥0.9.0 | Direct Groq API client |
| **openai** | ≥1.0.0 | OpenAI API integration (optional) |
| **reportlab** | Latest | Generate evaluation reports (PDF) |

### Why These Specific Tools?

**LangChain** → Simplifies building LLM applications by handling chains, memory, and retrieval logic

**Groq API** → Free, fast LLM provider (alternative to expensive OpenAI)

**ChromaDB** → Lightweight, serverless vector database perfect for POC

**DeepEval** → Pre-built metrics for evaluating hallucination, faithfulness, relevancy (no need to write scoring logic)

**Streamlit** → Rapid UI development for demos (no frontend coding needed)

**HuggingFace Embeddings** → Free, high-quality embeddings (no API key required)

---

## Project Architecture

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        RAG BOT ARCHITECTURE                             │
└─────────────────────────────────────────────────────────────────────────┘

INPUT LAYER:
  User Question (or Test Case)
         │
         ▼
┌─────────────────────────────────────────┐
│ 1. INPUT VALIDATION                    │ (validators.py)
│    - Check question length              │
│    - Check valid characters            │
│    - Sanitize input                    │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ 2. RETRIEVAL LAYER (TRUE RAG)           │ (rag_chain.py)
│    - Query vector store (ChromaDB)     │
│    - Retrieve K similar documents      │
│    - Score relevance                   │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ 3. CONTEXT BUILDING                     │ (rag_chain.py)
│    - Combine retrieved documents       │
│    - Format context with sources       │
│    - Preserve metadata                 │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ 4. PROMPT ENGINEERING                   │ (rag_chain.py)
│    - Build system prompt               │
│    - Inject retrieved context          │
│    - Include question                  │
│    - Add balancing instructions        │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ 5. LLM GENERATION                       │ (rag_chain.py)
│    - Call LLM (Groq/OpenAI)           │
│    - Generate answer based on context  │
│    - Return with source metadata       │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ 6. OUTPUT VALIDATION                    │ (validators.py)
│    - Check answer quality              │
│    - Verify source documents exist     │
│    - Format for UI                     │
└─────────────────────────────────────────┘
         │
         ▼
OUTPUT LAYER:
  Answer + Source Documents + Metadata
```

---

## Directory Structure

```
rag_eval_poc/
│
├── README.md                          # Project overview & quick start guide
├── TO_DO.md                           # Project roadmap & learning objectives
├── requirements.txt                   # Python dependencies (pip install -r)
├── PROJECT_DOCUMENTATION.md           # THIS FILE - Complete explanation
├── run_app.py                         # Entry point to launch Streamlit UI
│
├── config/                            # Configuration directory
│   ├── .env                           # Environment variables (API keys, model names)
│   └── .streamlit/
│       └── config.toml                # Streamlit UI theme & behavior config
│
├── src/                               # Main source code directory
│   ├── app.py                         # Streamlit web UI (interactive demo)
│   ├── config.py                      # Configuration loader & validator
│   ├── demo.py                        # RAGBotDemo class for manual testing
│   ├── evaluation.py                  # Evaluation logic for Streamlit UI
│   ├── validators.py                  # Input/output validation
│   │
│   ├── data/                          # Data storage directory
│   │   └── documents/                 # Where user documents go
│   │       └── document.txt           # Sample document for testing
│   │
│   ├── chroma_db/                     # Vector database storage
│   │   ├── chroma.sqlite3             # ChromaDB database file
│   │   └── [collection folders]/      # Embedded documents & vectors
│   │
│   └── rag/                           # RAG pipeline modules
│       ├── __init__.py                # Package initialization
│       ├── loader.py                  # Document loading (PDF/TXT parsing)
│       ├── vector_store.py            # Vector database management
│       └── rag_chain.py               # RAG chain implementation
│
└── tests/                             # Testing & evaluation directory
    └── evaluation/                    # Structured evaluation tests
        ├── test_cases.yaml            # Test questions & expected answers
        ├── run_eval.py                # Script to run DeepEval evaluations
        ├── LLM_MODEL.py              # Groq LLM wrapper for DeepEval
        └── results/                   # Evaluation results output
            ├── evaluation_report_1.md # Markdown report of results
            ├── evaluation_results_1.json # JSON results (machine-readable)
            └── [... more reports ...]
```

---

## Core Components & Modules

### 1. **Configuration System** (`src/config.py`)

#### What It Does
Centralizes all configuration values in one place and validates them at startup.

#### Key Components
```python
class Config:
    # Path Configuration
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    DOCUMENTS_DIR = DATA_DIR / "documents"
    CHROMA_DB_DIR = BASE_DIR / "chroma_db"
    
    # LLM Provider
    LLM_PROVIDER = "groq"
    API_KEY = os.getenv("API_KEY")
    LLM_MODEL = "llama-3.3-70b-versatile"
    
    # Document Processing
    PDF_CHUNK_SIZE = 800              # Characters per chunk
    PDF_CHUNK_OVERLAP = 100           # Overlap between chunks
    
    # Retriever
    RETRIEVER_K = 1                   # Number of documents to retrieve
    
    # Validation Rules
    MIN_QUESTION_LENGTH = 3
    MAX_QUESTION_LENGTH = 500
```

#### Why This Approach?
- **Single Source of Truth**: All settings in one place
- **Environment Separation**: Different settings for dev/prod
- **Type Safety**: Pydantic validates all values
- **Easy to Modify**: Change behavior without touching code

#### Environment Variables (.env)
Located in `config/.env`:
```
LLM_PROVIDER=groq
API_KEY=your_key_here
LLM_MODEL=llama-3.3-70b-versatile
```

---

### 2. **Input/Output Validation** (`src/validators.py`)

#### What It Does
Validates user inputs before processing and ensures outputs meet quality standards.

#### Key Validators

**InputValidator.validate_question()**
- Checks question is not empty
- Enforces min/max length (3-500 chars)
- Requires at least one alphanumeric character
- Returns (is_valid, error_message)

**OutputValidator.validate_answer()**
- Ensures answer meets minimum length
- Checks source documents are cited
- Validates metadata completeness

**DocumentValidator.validate_chunks()**
- Ensures chunks are not empty
- Validates chunk boundaries
- Checks for acceptable character sets

#### Why Validate?
- **Fail Fast**: Catch issues before expensive LLM calls
- **Better UX**: Clear error messages for users
- **Prevent Hallucinations**: Stop bad inputs that confuse LLMs
- **Cost Control**: No wasted API calls on invalid inputs

---

### 3. **Document Loader** (`src/rag/loader.py`)

#### What It Does
Loads, parses, and chunks documents into pieces suitable for embeddings.

#### Process
```
Input Document (PDF/TXT)
         │
         ▼
1. LOAD: PyPDFLoader or TextLoader
   - Extract text from file
   - Handle different encodings
   │
         ▼
2. SPLIT: RecursiveCharacterTextSplitter
   - Break into 800-char chunks
   - Keep 100-char overlap for context
   - Use smart separators (\n\n, \n, space, char)
   │
         ▼
3. VALIDATE: DocumentValidator
   - Check chunks not empty
   - Verify character encoding
   │
         ▼
Output: List of document chunks
```

#### Why Chunking?
- **Embedding Size Limits**: Large documents exceed embedding limits
- **Relevance**: Smaller chunks allow more precise retrieval
- **Context Window**: LLMs have limited input size
- **Overlap**: Preserves context between chunk boundaries

#### Code Example
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,          # ~150-200 words
    chunk_overlap=100,       # 1-2 sentences overlap
    separators=["\n\n", "\n", " ", ""]  # Smart boundaries
)
chunks = splitter.split_documents(docs)
```

---

### 4. **Vector Store Management** (`src/rag/vector_store.py`)

#### What It Does
Manages embeddings and vector database (ChromaDB) for semantic search.

#### Two Embedding Options

**Option 1: HuggingFace Embeddings (Recommended)**
```python
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",     # 384-dimensional vectors
    model_kwargs={"device": "cpu"}     # Run on CPU (no GPU needed)
)
```
-  Free (no API key)
-  Fast (384-dim vectors)
-  Quality (trained on 215M+ sentence pairs)
-  No external calls

**Option 2: TF-IDF Fallback**
-  Works without HuggingFace
-  Less accurate than transformers
-  No semantic understanding

#### ChromaDB: Vector Database

What it does:
- Stores text chunks + embeddings
- Similarity search (find similar documents)
- Persistent storage (SQLite)
- Simple API (no server needed)

How it works:
```python
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="src/chroma_db"  # Save to disk
)
```

#### Why Vector Databases?

| Traditional Search | Vector Search |
|-------------------|---------------|
| Keyword matching | Semantic matching |
| "cat" ≠ "kitten" | "cat" ≈ "kitten" |
| Boolean logic | Similarity scores |
| Brittle | Robust |

---

### 5. **RAG Chain** (`src/rag/rag_chain.py`)

#### What It Does
Orchestrates the entire RAG pipeline: retrieval → context → prompt → generation.

#### Step-by-Step Execution

**Step 1: Get LLM**
```python
def get_llm():
    if provider == "groq":
        return ChatGroq(model=config.LLM_MODEL, api_key=config.API_KEY)
```
- Loads LLM based on configured provider
- Lazy-loads to avoid unnecessary API calls
- Supports multiple providers (Groq, OpenAI, Ollama)

**Step 2: Create Retriever**
```python
self.retriever = vectordb.as_retriever(search_kwargs={"k": config.RETRIEVER_K})
```
- Wraps vector database as retriever
- K=1 means retrieve top-1 most similar document
- Can increase K for more context

**Step 3: Invoke (Main RAG Logic)**
```python
docs = self.retriever.invoke(query)  # RETRIEVAL FIRST
```
1. Query vector store with user question
2. Retrieve similar documents
3. Build context from retrieved docs
4. Create prompt with context
5. Call LLM to generate answer
6. Return answer + source documents

#### The Crucial Prompt
```
System Prompt:
"You are a helpful assistant. Answer questions based on the provided documents.

RULES:
1. Use the documents as your PRIMARY source
2. If answer is clearly in documents, provide it
3. If you need to connect multiple parts, explain reasoning
4. Only say 'not in documents' if genuinely unsearchable
5. If you can reasonably infer something, do so
6. Prioritize document information over general knowledge
7. Cite document sections when relevant

[RETRIEVED DOCUMENTS HERE]

User Question: {question}

Answer (based on the documents):"
```

Why these rules?
- Balances inference with faithfulness
- Prevents hallucination
- Encourages source citation
- Tells LLM when to say "I don't know"

#### Output Structure
```python
{
    "result": "The answer text...",               # Generated answer
    "source_documents": [doc1, doc2],             # Retrieved docs
    "retrieval_count": 1,                         # How many retrieved
    "is_rag": True                                # Proof of RAG
}
```

---

### 6. **Streamlit Web UI** (`src/app.py`)

#### What It Does
Provides an interactive web interface for asking questions and seeing evaluation results.

#### Key Features

**Professional Styling**
```css
.hero → Blue gradient header (AI brand colors)
.chat-window → Clean white cards
.user-msg → Blue-bordered user messages
.ai-msg → Green-bordered AI responses
.source-card → Document source citations
```

**Session State Management**
```python
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
```
- Preserves chat history across page refreshes
- Maintains UI state during Streamlit reruns
- Allows multi-turn conversations

**Components**

1. **Input Section**
   - Text input for questions
   - Validation before submission
   - Real-time character counter

2. **Chat Display**
   - Shows conversation history
   - User messages (blue)
   - AI responses (green)
   - Timestamps

3. **Source Documents**
   - Shows which documents were retrieved
   - Displays relevance scores
   - Links to source content

4. **Evaluation Dashboard**
   - Hallucination score
   - Faithfulness score
   - Answer relevancy
   - Contextual recall
   - Visual charts & metrics

---

### 7. **Evaluation System** (`src/evaluation.py` + `tests/evaluation/run_eval.py`)

#### What It Does
Uses DeepEval library to systematically evaluate RAG bot quality.

#### Four Core Metrics

**1. Hallucination Metric**
- **What**: Did the bot make up facts?
- **How**: Judge compares answer against retrieved documents
- **Trigger**: Answer contains info NOT in documents
- **Example**: 
  - Doc says: "Python released 2020"
  - Bot says: "Python released 1992"
  - Metric: FAIL (hallucination)

**2. Faithfulness Metric**
- **What**: Did the bot stick to source documents?
- **How**: Checks if answer only uses document content
- **Trigger**: Answer strays from provided context
- **Example**:
  - Doc talks about ML training
  - Bot answers about ML deployment (not in doc)
  - Metric: FAIL (unfaithful)

**3. Answer Relevancy Metric**
- **What**: Does the answer actually respond to the question?
- **How**: Judge checks question-answer alignment
- **Trigger**: Answer doesn't address the question
- **Example**:
  - Q: "What is AI?"
  - A: "Computers are useful devices"
  - Metric: FAIL (irrelevant)

**4. Contextual Recall Metric**
- **What**: Did the bot retrieve all necessary context?
- **How**: Checks if retrieved docs contain answer
- **Trigger**: Retrieved docs don't have answer info
- **Example**:
  - Q: "What is RAG?"
  - Retrieved: Document about LLMs (no RAG content)
  - Metric: FAIL (missing context)

#### How Evaluation Works

```python
# 1. Create test case
test_case = LLMTestCase(
    input="What is AI?",
    expected_output="Expected answer here",
    actual_output="Bot's actual answer",
    retrieval_context=["Retrieved doc 1", "Retrieved doc 2"]
)

# 2. Create metrics
hallucination_metric = HallucinationMetric()
faithfulness_metric = FaithfulnessMetric()

# 3. Measure
hallucination_metric.measure(test_case)  # 0.0-1.0 score
print(f"Score: {hallucination_metric.score}")
print(f"Reason: {hallucination_metric.reason}")
```

#### Test Cases (test_cases.yaml)
```yaml
test_cases:
  - id: 1
    category: straightforward
    question: "What is AI?"
    expected_answer: "AI is computer systems..."
    source_context: "AI refers to systems capable of..."
    difficulty: easy
```

Three categories:
- **Straightforward (10)**: Direct factual questions
- **Tricky (5)**: Require reasoning/inference
- **Unanswerable (5)**: Not in documents (test "I don't know")

#### Groq as Judge LLM

Why use Groq instead of OpenAI?
-  Free (unlimited requests)
-  Fast (optimized for reasoning)
-  Works offline-like (no rate limits)
-  Less guaranteed accuracy than GPT-4

```python
class GroqModel(DeepEvalBaseLLM):
    def __init__(self, api_key, model_name):
        self.groq_client = ChatGroq(api_key=api_key, model=model_name)
    
    def generate(self, prompt):
        response = self.groq_client.invoke(prompt)
        return response.content
```

---

### 8. **Demo Class** (`src/demo.py`)

#### What It Does
Programmatic way to test RAG bot without UI (for scripts/notebooks).

#### Usage
```python
bot = RAGBotDemo("path/to/document.txt")
bot.setup()
result = bot.ask("What is AI?")
print(result["answer"])
print(result["sources"])
```

#### Methods
- `setup()`: Initialize RAG pipeline
- `ask(question)`: Get answer to question
- `reload_documents()`: Reload documents from disk
- `get_vector_store_stats()`: Get DB statistics

---

## Configuration System

### config/.env File

Purpose: Store secrets and configuration outside code (security best practice)

```env
# ======================================
# LLM PROVIDER SELECTION
# ======================================
LLM_PROVIDER=groq

# ======================================
# GROQ CONFIGURATION (FREE!)
# ======================================
API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
LLM_MODEL=llama-3.3-70b-versatile
```

### Why .env?
- 🔒 **Security**: API keys never in version control
-  **Flexibility**: Easy to switch between environments
-  **Deployment**: Different keys for dev/test/prod
-  **Best Practice**: Industry standard

### config/.streamlit/config.toml

Purpose: Configure Streamlit UI appearance and behavior

```toml
[theme]
primaryColor = "#2563EB"              # Blue for AI/tech
backgroundColor = "#F8FAFC"           # Light background
textColor = "#0F172A"                 # Dark text
font = "sans serif"                   # Modern font

[client]
showErrorDetails = false              # Hide stack traces
toolbarMode = "minimal"               # Professional look
caching = true                        # Faster rendering
```

---

## Data Pipeline

### End-to-End Data Flow

```
1. USER UPLOADS DOCUMENT
   document.txt (1000+ characters)
         │
         ▼
2. LOAD & PARSE
   loader.py → PyPDFLoader/TextLoader
   Output: Raw text from file
         │
         ▼
3. CHUNK INTO SEGMENTS
   loader.py → RecursiveCharacterTextSplitter
   Input: Full document text
   Output: ~12 chunks of 800 chars each (with overlap)
   
   Example:
   Chunk 1: "AI refers to computer systems... [continues]"
   Chunk 2: "[overlap] ...system learning... [new content]"
   Chunk 3: "[overlap] ...pattern recognition... [new]"
         │
         ▼
4. GENERATE EMBEDDINGS
   vector_store.py → HuggingFaceEmbeddings
   Input: Each chunk (string)
   Process: Model converts to 384-dim vector
   Output: 12 vectors × 384 dimensions
   
   Conceptually:
   "AI is machine learning" → [0.23, 0.45, ..., 0.78]
         │
         ▼
5. STORE IN VECTOR DATABASE
   vector_store.py → ChromaDB
   Stores: {
     chunk_text: "AI refers to...",
     embedding: [0.23, 0.45, ...],
     metadata: {source: "document.txt", page: 0}
   }
   ChromaDB creates: src/chroma_db/chroma.sqlite3
         │
         ▼
6. USER ASKS QUESTION
   "What is AI?"
   app.py → question validation
         │
         ▼
7. RETRIEVE SIMILAR DOCUMENTS
   rag_chain.py → retriever.invoke(question)
   Input: "What is AI?" → Convert to embedding
   Process: Vector similarity search (cosine distance)
   Output: Top-1 most similar chunk
   
   Similarity calculation:
   cos_similarity("What is AI?", Chunk1) = 0.89
   cos_similarity("What is AI?", Chunk2) = 0.45
   → Retrieve Chunk1 (highest similarity)
         │
         ▼
8. BUILD CONTEXT PROMPT
   rag_chain.py → format context + question
   
   Context = """
   [Document 1: document.txt]
   AI refers to computer systems capable of...
   """
   
   Full Prompt = """
   [System rules...]
   [DOCUMENTS...]
   [Question...]
   """
         │
         ▼
9. GENERATE ANSWER
   rag_chain.py → LLM.invoke(prompt)
   LLM Provider: Groq (llama-3.3-70b-versatile)
   Output: "AI refers to computer systems..."
         │
         ▼
10. VALIDATE OUTPUT
   validators.py → OutputValidator
   - Check answer not empty
   - Verify sources cited
   - Check metadata complete
         │
         ▼
11. DISPLAY TO USER
   app.py → Streamlit UI
   Shows:
   ├─ User Question
   ├─ AI Answer
   ├─ Source Documents
   ├─ Evaluation Metrics
   └─ Confidence Scores
         │
         ▼
12. (OPTIONAL) EVALUATE
   evaluation.py → DeepEval metrics
   For each metric (Hallucination, Faithfulness, etc.):
   - Judge LLM analyzes answer
   - Returns score (0.0-1.0)
   - Provides reason for score
```

---

## User Interfaces

### 1. **Streamlit Web UI** (Primary)

**Launch Command**
```bash
python run_app.py
```

**Features**
- Interactive chat interface
- Real-time Q&A
- Source document display
- Evaluation metrics dashboard
- Chat history management

**Layout**
```
┌─────────────────────────────────────────┐
│  RAG Intelligence Platform              │ (Hero)
└─────────────────────────────────────────┘
│                  Sidebar                 │
│  • Upload Document                      │
│  • Clear History                        │
│  • Run Evaluation                       │
├─────────────────────────────────────────┤
│  Chat History                           │
│  ┌─────────────────────────────────────┐│
│  │ You: What is AI?                   ││
│  │ Bot: AI refers to...               ││
│  │ Sources: [Document 1]              ││
│  └─────────────────────────────────────┘│
├─────────────────────────────────────────┤
│  Metrics Display                        │
│  Hallucination: 0.95 ░░░░░            │
│  Faithfulness: 0.88 ░░░░               │
│  Answer Relevancy: 0.92 ░░░░░          │
│  Contextual Recall: 0.85 ░░░░          │
└─────────────────────────────────────────┘
```

### 2. **Command Line / Python API**

**Via Demo Class**
```python
from src.demo import RAGBotDemo

bot = RAGBotDemo()
bot.setup()
result = bot.ask("What is AI?")
print(result["answer"])
```

### 3. **Evaluation Script**

**Launch**
```bash
python tests/evaluation/run_eval.py
```

**Output**
- Generates evaluation_report_N.md (readable results)
- Generates evaluation_results_N.json (machine-readable)
- Shows pass/fail for each metric
- Provides detailed reasons

---

## Evaluation System

### How Structured Evaluation Works

#### The Problem It Solves

Manual testing:
-  Doesn't scale (20 questions manually evaluated = 30 min)
-  Subjective ("looks good to me" vs. "missed this detail")
-  Misses subtle issues (hallucinations that sound plausible)
-  Can't measure improvement over time

Automated evaluation:
-  Scalable (1000 questions in seconds)
-  Consistent metrics (same standard each time)
-  Catches subtle issues (LLM as judge)
-  Quantified progress (scores improve over time)

#### How DeepEval Works

```
Test Case / Actual Output
         │
         ▼
    Metric (e.g., Hallucination)
         │
         ▼
    Send to Judge LLM with Prompt:
    "Did this model hallucinate? [answer] [documents]"
         │
         ▼
    Judge LLM Returns:
    Score: 0.95 (confidence)
    Reason: "The answer accurately reflected..."
         │
         ▼
    Result Captured
```

#### Key Metrics Explained

**Hallucination Score**
- Range: 0.0-1.0 (higher = less hallucination)
- 0.0 = Complete hallucination (all made up)
- 1.0 = No hallucination (fact-checked)
- Typical: 0.5-0.9

Example:
```
Document: "Python was released in 1991"
Answer: "Python was released in 1990"
Judge: "This is factually incorrect - hallucination"
Score: 0.0 (failed)
Reason: "The answer states 1990 but documents say 1991"
```

**Faithfulness Score**
- Range: 0.0-1.0 (higher = more faithful)
- 0.0 = Answer ignores documents
- 1.0 = Answer strictly from documents
- Typical: 0.6-0.95

Example:
```
Document: "RAG combines retrieval with generation"
Answer: "RAG is great" (too vague, doesn't match doc)
Judge: "Answer doesn't use provided context"
Score: 0.2 (low faithfulness)
Reason: "No specific information from documents"
```

**Answer Relevancy Score**
- Range: 0.0-1.0 (higher = more relevant)
- 0.0 = Complete non-sequitur
- 1.0 = Perfect answer to question
- Typical: 0.5-0.95

Example:
```
Question: "What is AI?"
Answer: "Computers need electricity"
Judge: "Answer doesn't address the question"
Score: 0.1 (irrelevant)
Reason: "Question asks for AI definition, answer about power"
```

**Contextual Recall Score**
- Range: 0.0-1.0 (higher = better recall)
- 0.0 = Retrieved docs had no relevant info
- 1.0 = Retrieved docs had all necessary info
- Typical: 0.3-0.9

Example:
```
Question: "What is RAG?"
Retrieved Doc: "About neural networks"
Answer: "RAG combines retrieval with generation"
Judge: "RAG info wasn't in retrieved documents"
Score: 0.0 (retrieval failed)
Reason: "Needed RAG-specific doc, got neural nets instead"
```

### Test Cases Structure

File: `tests/evaluation/test_cases.yaml`

```yaml
test_cases:
  # Easy Questions (directly answerable)
  - id: 1
    category: straightforward
    question: "What is AI?"
    expected_answer: "AI refers to computer systems..."
    source_context: "AI systems capable of learning..."
    difficulty: easy
    notes: "Direct definition"
  
  # Tricky Questions (require reasoning)
  - id: 11
    category: tricky
    question: "Why is data quality important?"
    expected_answer: "Poor data leads to incorrect predictions"
    source_context: "High-quality data must be accurate"
    difficulty: medium
    notes: "Cause-effect reasoning"
  
  # Unanswerable Questions (test "I don't know")
  - id: 16
    category: unanswerable
    question: "Will aliens visit Earth tomorrow?"
    expected_answer: "Not in documents"
    source_context: "Not available"
    difficulty: hard
    notes: "Should return 'I don't know'"
```

### Results Interpretation

Example output:
```json
{
  "test_id": 1,
  "question": "What is AI?",
  "hallucination": {
    "score": 0.95,
    "passed": true,
    "reason": "No hallucinations detected"
  },
  "faithfulness": {
    "score": 0.88,
    "passed": true,
    "reason": "Answer closely follows source documents"
  },
  "answer_relevancy": {
    "score": 0.92,
    "passed": true,
    "reason": "Answer directly addresses question"
  },
  "contextual_recall": {
    "score": 0.85,
    "passed": true,
    "reason": "Retrieved documents contained necessary info"
  },
  "overall_score": 0.90,
  "all_passed": true
}
```

**Interpreting Results**
-  green (0.7-1.0): Good performance
- 🟡 yellow (0.5-0.7): Acceptable but risky
-  red (0.0-0.5): Clear failure

---

## Key Concepts Explained

### Retrieval-Augmented Generation (RAG)

**What**: Combining document retrieval with LLM generation

**Why**: 
- LLMs have knowledge cutoff (don't know recent info)
- LLMs hallucinate (make up facts)
- Need to verify answers against sources

**How**:
```
Question → Retrieve Relevant Docs → Add Context to Prompt → LLM Generates Answer
```

**Example**:
```
Without RAG:
Q: "What new feature was added to Python 3.12?"
A: "I don't know, my training data ends at April 2024"

With RAG:
Q: "What new feature was added to Python 3.12?"
[Retrieve: Python 3.12 release notes]
A: "Python 3.12 added Per-Interpreter GIL for better concurrency"
[With source cited]
```

### Embeddings

**What**: Converting text to numbers that capture meaning

**Why**: 
- Computers work with numbers, not text
- Similar meanings should be close together
- Enables similarity search

**How**:
```
Text: "The cat sat on the mat"
   ↓ HuggingFace Embeddings Model
Vector: [0.23, 0.45, 0.67, ..., 0.89] (384 numbers)
```

**Key Property - Semantic Similarity**:
```
"The cat sat on the mat" ≈ [0.23, 0.45, ...]
"A kitty sat on a rug"   ≈ [0.24, 0.46, ...] (similar!)

Cosine Similarity = 0.92 (very similar)
```

### Vector Similarity Search

**What**: Finding similar vectors quickly

**How** (Cosine Similarity):
```
Query: "What is AI?"
   ↓ Convert to embedding
Embedding: [0.1, 0.5, 0.3, ...]

Compare to all document embeddings:
Chunk1: [0.11, 0.51, 0.31, ...] → Similarity = 0.98  TOP MATCH
Chunk2: [0.3, 0.2, 0.9, ...]   → Similarity = 0.42 
Chunk3: [0.01, 0.02, 0.03, ...] → Similarity = 0.15 
```

### Prompt Engineering

**What**: Carefully crafting instructions for LLMs

**Why**: Small wording changes can dramatically change outputs

**Examples**:

```
Bad Prompt:
"Answer the question using the documents"

Problem: Might ignore documents if confident about topic

Better Prompt:
"You MUST base your answer on the provided documents.
If the answer is not in documents, say 'I don't know'"

Best Prompt:
"You are a helpful assistant. Answer questions based on provided documents.

RULES:
1. Use documents as PRIMARY source
2. If answer in documents, provide it
3. If not in documents, say 'I don't know'
4. Cite which document section you used
5. Do not use general knowledge about this topic"
```

### Hallucination

**What**: LLM confidently stating false information

**Why It Happens**:
- LLM trained to be helpful → fills gaps with plausible-sounding stuff
- No built-in fact-checking
- Doesn't know what it doesn't know

**Examples**:
```
Q: "When was Python 4.0 released?"
Bad Answer: "Python 4.0 was released in 2025" (HALLUCINATION - doesn't exist!)
Good Answer: "Python 4.0 hasn't been released yet"

Q: "What color is an apple?"
Retrieved Doc: "An apple is a fruit"
Bad Answer: "An apple is red and round" (HALLUCINATION - doc doesn't say color!)
Good Answer: "Apples are fruits"
```

### Faithfulness

**What**: Staying true to source documents

**Different from Hallucination**:
- Hallucination = Making things up (false positive)
- Unfaithful = Straying from documents (missing context)

**Example**:
```
Doc: "Python is a programming language. It's fun to use."

Unfaithful (strays from doc):
"Python is beautiful" (not in doc)
"Python is hard" (contradicts doc)

Faithful:
"Python is a programming language"
"Python is a programming language and fun to use"
```

### Relevancy

**What**: Answer directly addresses the question

**Example**:
```
Q: "What is AI?"
Bad: "Computers are useful" (relevant to nothing)
Good: "AI is machine learning and automation"

Q: "How many legs does a spider have?"
Bad: "A spider eats insects" (related but not relevant)
Good: "A spider has eight legs"
```

### Contextual Recall

**What**: Retrieving all necessary information for answering

**Example**:
```
Multi-part question: "What is ML and why is it important?"

Retrieved: "ML is learning from data"
Problem: Got half the answer (no "why important" info)
Score: 0.5 (partial recall)

Retrieved: "ML enables systems to learn, improving automation"
Better: Full context available
Score: 0.9 (good recall)
```

---

## How Everything Works Together

### Complete Use Case: User Asks a Question

#### Step 1: User Submits Question (1 second)
```
User Types: "What is Artificial Intelligence?"
                    ↓
         Input validation
         - Check length (3-500 chars) ✓
         - Check alphanumeric ✓
         - Store in session state
```

#### Step 2: Retrieve Relevant Documents (0.5 seconds)
```
Query Embedding (HuggingFace):
  "What is Artificial Intelligence?"
           ↓
  [0.23, 0.45, 0.67, ..., 0.89] (384-dim)
           ↓
Vector Similarity Search (ChromaDB):
  Compare to all chunks in database
  Chunk1: "AI is computer systems..." → Sim=0.94 
  Chunk2: "ML is subset of AI..." → Sim=0.82
  Chunk3: "NLP processes text..." → Sim=0.45
           ↓
Retrieve Top-1:
  "AI refers to computer systems capable of learning..."
           ↓
Metadata:
  Source: "document.txt"
  Page: 1
```

#### Step 3: Build Prompt with Context (0.1 seconds)
```
System: "You are helpful assistant..."
Context: "[Document 1: document.txt]\n AI refers to..."
Question: "What is Artificial Intelligence?"

Full Prompt:
[System rules about using documents]
[Retrieved document text]
[Question asked by user]
"Answer based on the documents above:"
```

#### Step 4: Generate Answer with LLM (1-3 seconds)
```
Call Groq API:
  model: "llama-3.3-70b-versatile"
  temperature: 0.0 (deterministic)
  
LLM Processes:
  - Reads prompt
  - Sees retrieved context
  - Generates answer based on docs
  - Stays grounded in context
  
Response:
  "Artificial Intelligence refers to computer systems
   capable of learning, reasoning, and problem-solving
   without explicit human programming. AI systems can
   process large volumes of data and recognize patterns."
```

#### Step 5: Validate Output (0.5 seconds)
```
Output Validator checks:
  ✓ Answer length > 10 chars
  ✓ Source documents provided
  ✓ Metadata complete
  ✓ No obvious errors
```

#### Step 6: Display to User (0.1 seconds)
```
Streamlit UI shows:
┌─────────────────────────────────┐
│ Question: What is AI?           │
│ Answer: AI refers to...         │
│ Source: document.txt (Page 1)   │
│ Relevance Score: 0.94           │
└─────────────────────────────────┘
```

#### Step 7 (Optional): Run Evaluation (5-10 seconds)
```
For each metric:
  
  Hallucination Metric:
    Input: {answer, documents}
    Judge LLM: "Did this hallucinate?"
    Output: Score 0.95 (PASS)
    
  Faithfulness Metric:
    Input: {answer, documents}
    Judge LLM: "Stayed faithful to docs?"
    Output: Score 0.88 (PASS)
    
  Answer Relevancy Metric:
    Input: {question, answer}
    Judge LLM: "Relevant to question?"
    Output: Score 0.92 (PASS)
    
  Contextual Recall Metric:
    Input: {question, documents}
    Judge LLM: "Had all needed info?"
    Output: Score 0.85 (PASS)

Final Report:
  Overall Score: 0.90
  All Tests: PASS ✓
```

---

## Performance Optimization

### Key Settings & Why

| Setting | Value | Reason |
|---------|-------|--------|
| Chunk Size | 800 chars | Sweet spot: granular but contextual |
| Chunk Overlap | 100 chars | Prevents context loss at boundaries |
| Retriever K | 1 | Fast & focused; increase if needed |
| Temperature | 0.0 | Deterministic; prevents randomness |
| Embedding Model | all-MiniLM-L6-v2 | Fast (384-dim), good quality, free |

### Scaling Strategies

**If Too Slow:**
- Increase Chunk Size (fewer vectors to compare)
- Reduce Chunk Overlap (fewer chunks total)
- Reduce Retriever K (fewer documents processed)
- Switch to Ollama (local LLM, no API latency)

**If Quality Drops:**
- Increase Chunk Size (more context per chunk)
- Increase Retriever K (more documents to choose from)
- Adjust prompt instructions (more specific guidance)
- Switch to better LLM (GPT-4 vs. Groq)

---

## Troubleshooting Guide

### Common Issues

**Issue: "API_KEY not found"**
```
Cause: API key not in .env or environment
Fix:
1. Create config/.env
2. Add: API_KEY=your_key_here
3. Get free key at: console.groq.com
```

**Issue: "No documents found"**
```
Cause: Document files missing from src/data/documents/
Fix:
1. Place PDF/TXT files in: src/data/documents/
2. Restart app
3. Select document in UI
```

**Issue: "ChromaDB error"**
```
Cause: Database corrupted
Fix:
1. Delete: src/chroma_db/
2. Restart app
3. Reload documents
```

**Issue: "Slow response time"**
```
Cause: Large documents or many chunks
Fix:
1. Increase PDF_CHUNK_SIZE in config.py
2. Reduce RETRIEVER_K
3. Switch to Groq (faster than OpenAI)
```

---

## Summary: The Complete Picture

Your RAG bot project demonstrates:

1. **Document Understanding**: How to load, parse, and chunk documents
2. **Semantic Search**: How embeddings enable meaning-based retrieval
3. **Context-Aware Generation**: How to use retrieved context to improve LLM outputs
4. **Structured Evaluation**: How to measure hallucination, faithfulness, relevancy, recall
5. **Professional UX**: How to build interactive AI applications with Streamlit

**The learning goal**: Understanding that **evaluation matters** — automated metrics catch issues manual testing misses, enabling systematic improvement of AI systems.

---

## Next Steps & Enhancements

Potential improvements (out of future scope):
- [ ] Multi-document RAG (combine info from multiple docs)
- [ ] Re-ranking (semantic re-ranking of retrieved docs)
- [ ] Fine-tuning embeddings on domain-specific data
- [ ] Caching for repeated queries
- [ ] API response streaming (long answers)
- [ ] Prompt optimization (A/B test different prompts)
- [ ] Multiple LLM providers (compare Groq vs OpenAI)

---

**Document Version**: 1.0  
**Last Updated**: March 2026  
**Project Status**: POC (Proof of Concept) - Learning Project
