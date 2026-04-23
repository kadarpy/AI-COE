Evaluate a GenAI Application Using LLM Evaluation Techniques — POC

## Overview

This POC is about learning how to evaluate GenAI applications properly. You'll build a small RAG-based Q&A bot (or use any simple LLM-powered app), then systematically evaluate its quality using LLM evaluation techniques — measuring hallucination, faithfulness, answer relevancy, and more. The goal is to prove that structured evaluation catches quality issues that manual "looks good to me" testing misses.

This is not about building a reusable tool. It's about getting hands-on experience with how LLMs fail and how to measure that. By the end, you'll understand why evaluation matters, which metrics catch which problems, and what "good enough" looks like for a GenAI app.

## Objective

1. Build a small RAG-based Q&A bot using a sample document set (5-10 pages)

2. Create 20 test questions — a mix of answerable, partially answerable, and unanswerable queries

3. Run structured evaluations using DeepEval and document the results

4. Identify real quality issues the bot has (hallucinations, wrong answers, missed context) and show that evaluation metrics caught them

5. Write up findings: what worked, what failed, what surprised you

The key question to answer: **Does structured LLM evaluation actually catch problems that manual testing doesn't?**

## Background & Context

When you ask an LLM a question and it gives a decent-looking answer, how do you know if it's actually correct? Manual spot-checking doesn't scale and misses subtle issues — a model might sound confident while completely making up facts (hallucination), or it might answer from its training data instead of the documents you gave it (unfaithfulness).

LLM evaluation solves this by using automated metrics. Some metrics compare the answer against source documents (faithfulness). Others check if the model invented information (hallucination). Others measure whether the answer actually addresses the question asked (relevancy). The interesting part: some of these metrics use another LLM as a judge — essentially asking a second AI "did the first AI do a good job?"

DeepEval is an open-source Python library that makes this practical. It works like Pytest — you write test cases with inputs and expected behavior, run them, and get scored results. It has 50+ built-in metrics so you don't need to implement scoring logic yourself.

New to GenAI concepts? See the **GenAI Concepts Glossary (AICOE-11)** for beginner-friendly definitions of RAG, hallucination, embeddings, and other terms you'll encounter.

## Scope

### In Scope

* Build a minimal RAG Q&A bot (simple is fine — this isn't the point of the POC)

* Write 20 test questions with expected answers and source context

* Run evaluations using 4 DeepEval metrics: Hallucination, Faithfulness, Answer Relevancy, Contextual Recall

* Analyze results: which questions failed? Why? Did the metrics catch real issues?

* Document findings with specific examples of caught vs. missed problems

### Out of Scope

* Building a production-quality RAG pipeline (keep it minimal)

* Creating a reusable evaluation framework or dashboard (that's the Asset phase)

* Custom metrics or advanced evaluation techniques

* CI/CD integration or automation

* Evaluating multiple models or prompt variations (nice-to-have, not required)

## Step-by-Step Build Guide

### 1. Set Up the Environment

Create a Python project. Install: `deepeval`, `openai` (or `anthropic`), `langchain` (or `llama-index`), `chromadb`, `pypdf`. Set up your OpenAI API key as an environment variable. DeepEval uses OpenAI's GPT-4o as a judge model by default.

### 2. Build a Minimal RAG Bot

Pick a sample document set — something you can read yourself so you know the "right" answers. Good options: a company's FAQ page, a product manual, or a Wikipedia article on a specific topic (5-10 pages max). Load the documents, chunk them, embed them into ChromaDB, and wire up a simple retrieval + LLM generation chain. Use LangChain's `RetrievalQA` or equivalent. Don't over-engineer this — the bot is the thing being tested, not the deliverable.

### 3. Write 20 Test Questions

This is the most important step. Create a spreadsheet or YAML file with:

* **10 straightforward questions** where the answer is clearly in the documents

* **5 tricky questions** where the answer requires combining info from multiple chunks or where the document only partially answers

* **5 unanswerable questions** where the correct behavior is "I don't know" or "this isn't in the documents"

For each question, write: the question, the expected answer, and which part of the source document contains the answer (or "not in documents").

### 4. Run Your First Evaluation

Use DeepEval to create test cases from your 20 questions. Run all 4 metrics against each question. Don't worry about tuning thresholds yet — just run with defaults and look at what comes back. See DeepEval's Getting Started guide for the exact code pattern.

### 5. Analyze the Results

This is where the real learning happens. For each failing test case:

* Read the actual output from the bot

* Read the metric's reason for failure

* Check: was the metric right? Did the bot actually hallucinate / miss context / go off-topic?

* Note any false positives (metric flagged something that was actually fine)

* Note any false negatives (metric passed something that was actually wrong)

Create a simple table: Question | Metric | Score | Was the metric right? | Notes

### 6. Find the Interesting Failures

Look for patterns:

* Do unanswerable questions cause the most hallucination? (They usually do)

* Are multi-chunk questions less faithful? (Often yes — the model fills gaps)

* Does the bot give relevant-sounding but wrong answers? (Relevancy score high, faithfulness score low = dangerous)

Pick 3-5 of the most interesting findings to highlight in your write-up.

### 7. (Optional) Try Improving the Bot

If time allows, take 2-3 of the worst-performing questions and try to fix them — maybe adjust the prompt, change the chunking strategy, or add a "refuse to answer if unsure" instruction. Re-run the evaluation and see if scores improve. This demonstrates the eval-improve-eval loop that production GenAI teams use.

### 8. Write Up Findings

Document everything using the AICOE POC Output Template. Focus on:

* What did the evaluation catch that manual testing would have missed?

* Which metrics were most useful? Which were noisy?

* What's your confidence that a "passing" bot is actually good enough?

* Recommendation: should AICOE build a reusable evaluation harness (Asset)?

## Tech Stack

| Component        | Tool                    | Purpose                                     |
| ---------------- | ----------------------- | ------------------------------------------- |
| Language         | Python 3.11+            | Primary language                            |
| RAG Framework    | LangChain or LlamaIndex | Build the minimal Q&A bot                   |
| Vector Store     | ChromaDB                | Store document embeddings (local, no setup) |
| Eval Framework   | DeepEval                | Run evaluation metrics                      |
| LLM              | OpenAI GPT-4o           | Both the bot's brain and DeepEval's judge   |
| Document Loading | PyPDF or Unstructured   | Load sample documents                       |

## Getting Started Resources

### DeepEval (Your Main Tool)

* [DeepEval Getting Started](https://deepeval.com/docs/getting-started) — Install, write first test case, run your first eval

* [DeepEval Metrics Introduction](https://deepeval.com/docs/metrics-introduction) — All 50+ metrics explained with when to use each

* [DeepEval Tutorials](https://deepeval.com/tutorials/tutorial-introduction) — Step-by-step for RAG evaluation specifically

* [confident-ai/deepeval on GitHub](https://github.com/confident-ai/deepeval) — Source code, examples, community issues

### Understanding LLM Evaluation (Read Before You Start)

* [Using DeepEval for LLM Evaluation (Codecademy)](https://www.codecademy.com/article/using-deepeval-for-llm-evaluation-python) — Beginner-friendly walkthrough with code

* [Evaluate LLMs Effectively (DataCamp)](https://www.datacamp.com/tutorial/deepeval) — Practical guide with examples

* [LLM Evaluation Metrics: The Ultimate Guide (Confident AI)](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation) — What each metric measures and why

### RAG Setup (Quick Reference)

* [LangChain RetrievalQA Quickstart](https://python.langchain.com/docs/tutorials/rag/) — Build a basic RAG chain

* [ChromaDB Getting Started](https://docs.trychroma.com/getting-started) — Local vector store setup

### Evaluation Concepts

* [LLM Evaluation Frameworks Compared (Comet)](https://www.comet.com/site/blog/llm-evaluation-frameworks/) — DeepEval vs RAGAS vs others — helpful context

* [RAG Evaluation Metrics Best Practices (Patronus AI)](https://www.patronus.ai/llm-testing/rag-evaluation-metrics) — Faithfulness, precision, relevancy explained

## Deliverables

* Working RAG bot (minimal — just needs to run and answer questions)

* 20 test questions in YAML/spreadsheet with expected answers and source context

* Evaluation results: per-question metric scores

* Analysis write-up: findings, interesting failures, metric accuracy assessment

* POC output document following the **AICOE POC Output Template**

* Recommendation: should AICOE build a reusable eval Asset?

## Demo Criteria

10-minute walkthrough showing:

* Ask the RAG bot a question → show the answer

* Show the DeepEval results for that question (metric scores)

* Highlight 2-3 cases where metrics caught a real problem (e.g., the bot hallucinated and the hallucination metric flagged it)

* Show 1 case where the bot "looked fine" manually but the metric revealed an issue (this is the money shot — proves evaluation's value)

## Before You Start

* **Accounts & Access:** OpenAI API key with GPT-4o access (for both the RAG bot and DeepEval's judge). Estimated cost: ~$2-5 total for running 20 test cases through evaluation.

* **Time Estimate:** Day 1-2: Set up environment + build minimal RAG bot. Day 3-4: Write test questions + run first evaluation. Day 5-7: Analyze results, find interesting failures, optional improvements. Day 8-10: Write up findings + prepare demo.

* **Common Pitfalls:**

  * Don't spend too long perfecting the RAG bot — it's the evaluation target, not the deliverable. A mediocre bot is actually better for this POC because it gives you more interesting evaluation results.

  * Write test questions BEFORE running the bot. If you write questions after seeing the bot's answers, you'll unconsciously bias toward questions it handles well.

  * DeepEval's judge model costs API credits per evaluation. Run one question first to check costs before running all 20.

* **References:** GenAI Concepts Glossary (AICOE-11) for terminology. AICOE POC Output Template for final deliverable format.

## Acceptance Criteria

| Criteria                       | Threshold                                                    |
| ------------------------------ | ------------------------------------------------------------ |
| Working RAG bot                | Answers questions from loaded documents                      |
| Test questions                 | 20 questions: 10 straightforward + 5 tricky + 5 unanswerable |
| Metrics evaluated              | 4 DeepEval metrics run on all test cases                     |
| Analysis depth                 | 3+ specific examples of evaluation catching real issues      |
| False positive/negative review | Documented assessment of metric accuracy                     |
| Findings write-up              | Clear conclusions on evaluation value                        |
| Demo ready                     | Can show passing, failing, and "sneaky" cases live           |
| POC output document            | Completed using AICOE POC Output Template                    |
