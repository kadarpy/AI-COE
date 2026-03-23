# RAG Bot Evaluation Report

**Generated**: 2026-03-22 20:13:54

## Summary Statistics

- **Total Tests**: 6
- **Passed**: 0
- **Failed**: 6
- **Pass Rate**: 0.0%
- **Profile Used**: {'faithfulness': 0.7, 'relevancy': 0.75, 'recall': 0.7, 'hallucination': 0.25}

## Failed Test Cases

16, 17, 18, 19, 20, 21

## Metric Performance

### Hallucination
- Average Score: 0.00
- Min: 0.00, Max: 0.00
- Passed: 6/6

### Faithfulness
- Average Score: 0.78
- Min: 0.00, Max: 1.00
- Passed: 4/6

### AnswerRelevancy
- Average Score: 0.50
- Min: 0.33, Max: 0.67
- Passed: 0/6

### ContextualRecall
- Average Score: 0.00
- Min: 0.00, Max: 0.00
- Passed: 0/6

## Results by Category

- **unanswerable**: 0/6 passed

## Detailed Results

### Test #16
**Category**: unanswerable
**Question**: What is the accuracy percentage of AI systems in healthcare?

**Expected**: This information is not available in the provided document

**Actual**: Not in documents. The provided documents do not mention the accuracy percentage of AI systems in healthcare. They only discuss the key characteristics of AI systems and the integration of AI into various industries, including healthcare, but do not provide specific information on accuracy percentages. (Refer to Document 1)

**Metrics**:
- Hallucination: 0.00 PASS
  - Reason: The score is 0.00 because the actual output does not contradict the provided context and only mentions the absence of specific information, which aligns with the given guidelines.
- Faithfulness: 0.67 FAIL
  - Reason: The score is 0.67 because the actual output incorrectly implies that the retrieval context specifically mentions the integration of AI into the healthcare industry, when in fact it only discusses AI integration across industries in general.
- AnswerRelevancy: 0.50 FAIL
  - Reason: The score is 0.50 because the actual output acknowledges the lack of information regarding the accuracy percentage of AI systems in healthcare, but fails to provide a relevant answer, resulting in a mediocre score due to the presence of irrelevant statements that do not address the input question.
- ContextualRecall: 0.00 FAIL
  - Reason: The score is 0.00 because the sentence in the expected output cannot be attributed to any node in the retrieval context, specifically the 1st node, as it discusses unrelated topics like 'Organizations across industries...'.

---

### Test #17
**Category**: unanswerable
**Question**: Which vector database is the most widely used in production?

**Expected**: The document lists databases but does not rank them

**Actual**: Not in documents. 

The documents provided list popular vector databases, including ChromaDB, Pinecone, Weaviate, Milvus, and FAISS (Section 6: Vector Databases), but they do not specify which one is the most widely used in production.

**Metrics**:
- Hallucination: 0.00 PASS
  - Reason: The score is 0.00 because the actual output does not contradict the provided context and only states a fact that is not specified in the documents, without introducing any unfounded or false information.
- Faithfulness: 1.00 PASS
  - Reason: The score is 1.00 because there are no contradictions found, indicating a perfect alignment between the actual output and the retrieval context.
- AnswerRelevancy: 0.67 FAIL
  - Reason: The score is 0.67 because the actual output partially addresses the question but lacks specific information about vector databases used in production, as it does not provide any details about their usage, resulting in a moderate score.
- ContextualRecall: 0.00 FAIL
  - Reason: The score is 0.00 because the sentence in the expected output cannot be attributed to any node in the retrieval context, specifically nodes 1 and 2, which discuss vector databases but do not mention ranking information, making it impossible to establish a connection.

---

### Test #18
**Category**: unanswerable
**Question**: What is the training dataset size required for large language models?

**Expected**: The document does not specify dataset sizes

**Actual**: Not in documents. The provided documents do not mention the training dataset size required for large language models. They only describe the workflow of a RAG system and its components, but do not provide information on training dataset sizes.

**Metrics**:
- Hallucination: 0.00 PASS
  - Reason: The score is 0.00 because the actual output does not contradict the provided context and only states information that is not mentioned in the context, without introducing any unfounded or false information.
- Faithfulness: 1.00 PASS
  - Reason: The score is 1.00 because there are no contradictions found, indicating a perfect alignment between the actual output and the retrieval context.
- AnswerRelevancy: 0.50 FAIL
  - Reason: The score is 0.50 because the actual output acknowledges the lack of information in the provided documents regarding the training dataset size required for large language models, but fails to provide a relevant answer or alternative solution, resulting in a mediocre score.
- ContextualRecall: 0.00 FAIL
  - Reason: The score is 0.00 because the sentence in the expected output cannot be attributed to any node in the retrieval context, as none of the nodes mention 'dataset sizes' or related information.

---

### Test #19
**Category**: unanswerable
**Question**: Which company developed the first AI system?

**Expected**: This historical information is not included in the document

**Actual**: Not in documents. 

The provided document does not mention the company that developed the first AI system. It only provides a general introduction to Artificial Intelligence and its applications in enterprises. (Document 1)

**Metrics**:
- Hallucination: 0.00 PASS
  - Reason: The score is 0.00 because the actual output does not contradict the provided context and accurately states that the information about the company that developed the first AI system is not in the document.
- Faithfulness: 1.00 PASS
  - Reason: The score is 1.00 because there are no contradictions found, indicating a perfect alignment between the actual output and the retrieval context.
- AnswerRelevancy: 0.33 FAIL
  - Reason: The score is 0.33 because the actual output acknowledges the lack of information about the company that developed the first AI system, but fails to provide a relevant answer, resulting in a low score due to the presence of irrelevant statements.
- ContextualRecall: 0.00 FAIL
  - Reason: The score is 0.00 because the sentence in the expected output cannot be attributed to any node in the retrieval context, specifically the 1st node, as it does not mention 'historical information' or its exclusion from a document.

---

### Test #20
**Category**: unanswerable
**Question**: What is the cost of implementing an enterprise AI system?

**Expected**: Cost details are not provided in the document

**Actual**: Not in documents. 

The provided document does not mention the cost of implementing an enterprise AI system. It only provides a general introduction to Artificial Intelligence and its applications in enterprises, but does not discuss implementation costs. (Refer to Document 1)

**Metrics**:
- Hallucination: 0.00 PASS
  - Reason: The score is 0.00 because the actual output does not contradict the provided context and correctly states that certain information is not present in the document.
- Faithfulness: 1.00 PASS
  - Reason: The score is 1.00 because there are no contradictions found, indicating a perfect alignment between the actual output and the retrieval context, which is absolutely fantastic!
- AnswerRelevancy: 0.50 FAIL
  - Reason: The score is 0.50 because the actual output acknowledges the lack of relevant information regarding the cost of implementing an enterprise AI system, but fails to provide a direct answer or useful alternative, resulting in a mediocre score due to the presence of irrelevant statements that do not address the input question.
- ContextualRecall: 0.00 FAIL
  - Reason: The score is 0.00 because the sentence in the expected output cannot be attributed to any node in the retrieval context, specifically the 1st node, as it does not mention 'cost details' or related information.

---

### Test #21
**Category**: unanswerable
**Question**: what year was python released?

**Expected**: 1991

**Actual**: According to Document 1, Python was released in 2005. 

(Note: However, it's worth noting that this information might be incorrect, as Python was actually first released in 1991. But based on the provided document, the answer is 2005.)

**Metrics**:
- Hallucination: 0.00 PASS
  - Reason: The score is 0.00 because the actual output aligns with the provided context and does not introduce any non-factual information.
- Faithfulness: 0.00 FAIL
  - Reason: The score is 0.00 because the actual output incorrectly stated that Python was released in 2005, which directly contradicts historical records that confirm Python's first release was in 1991.
- AnswerRelevancy: 0.50 FAIL
  - Reason: The score is 0.50 because the actual output contains incorrect information about the release year of Python, which is directly relevant to the input question, but the fact that it attempted to provide a specific year, albeit incorrect, is why it's not a complete failure, hence the moderate score.
- ContextualRecall: 0.00 FAIL
  - Reason: The score is 0.00 because the sentence '1991' cannot be attributed to any part of the nodes in retrieval context, specifically the 1st node, which discusses unrelated topics.

---

