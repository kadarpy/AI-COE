# RAG Bot Evaluation Report

**Generated**: 2026-03-18 18:50:46

 Summary Statistics

- **Total Tests**: 21
- **Passed**: 3
- **Failed**: 1
- **Pass Rate**: 14.3%

 Metric Performance

# Hallucination
- Average Score: 0.00
- Min: 0.00, Max: 0.00
- Passed: 4/21

# Faithfulness
- Average Score: 1.00
- Min: 1.00, Max: 1.00
- Passed: 3/21

# AnswerRelevancy
- Average Score: 0.83
- Min: 0.50, Max: 1.00
- Passed: 2/21

# ContextualRecall
- Average Score: 1.00
- Min: 1.00, Max: 1.00
- Passed: 3/21

 Results by Category

- **straightforward**: 3/4 passed

 Detailed Results

# Test #1
**Category**: straightforward
**Question**: What is Artificial Intelligence?

**Expected**: Artificial Intelligence refers to computer systems capable of performing tasks that require human intelligence

**Actual**: According to Document 1, section 1: Introduction to Artificial Intelligence, Artificial Intelligence (AI) refers to "the development of computer systems capable of performing tasks that typically require human intelligence." These tasks include learning, reasoning, problem-solving, understanding language, recognizing patterns, and making decisions.

**Metrics**:
- Hallucination: 0.00 PASS
  - Reason: The score is 0.00 because the actual output perfectly aligns with the provided context about Artificial Intelligence (AI) without introducing any unfounded or contradictory information.
- Faithfulness: 1.00 PASS
  - Reason: The score is 1.00 because there are no contradictions found, indicating a perfect alignment between the actual output and the retrieval context.
- AnswerRelevancy: 1.00 PASS
  - Reason: The score is 1.00 because the output perfectly addresses the question about Artificial Intelligence without any irrelevant statements, demonstrating a strong understanding of the topic.
- ContextualRecall: 1.00 PASS
  - Reason: The score is 1.00 because the expected output perfectly matches the information in the node(s) in retrieval context, specifically node 1, which confirms that Artificial Intelligence refers to computer systems capable of performing tasks that require human intelligence.

---

# Test #2
**Category**: straightforward
**Question**: Name two key characteristics of AI systems

**Expected**: AI systems can process large data and recognize patterns

**Actual**: Based on the provided document (Document 1), two key characteristics of AI systems are:

1. Ability to process large volumes of data
2. Pattern recognition and prediction capabilities

These characteristics are listed in the section "Key characteristics of AI systems include:" in Document 1.

**Metrics**:
- Hallucination: 0.00 PASS
  - Reason: The score is 0.00 because the actual output accurately aligns with the provided context, listing key characteristics of AI systems without introducing any unfounded or contradictory information.
- Faithfulness: 1.00 PASS
  - Reason: The score is 1.00 because there are no contradictions found, indicating a perfect alignment between the actual output and the retrieval context.
- AnswerRelevancy: 0.50 FAIL
  - Reason: The score is 0.50 because the actual output partially addresses the input by mentioning the location of characteristics in a document, but fails to provide the characteristics themselves, resulting in a mediocre score.
- ContextualRecall: 1.00 PASS
  - Reason: The score is 1.00 because the sentence in the expected output can be perfectly attributed to the 1st node in the retrieval context, which mentions relevant capabilities, making it a great match!

---

# Test #3
**Category**: straightforward
**Question**: What is Machine Learning?

**Expected**: Machine Learning is a subset of AI that enables systems to learn from data without explicit programming

**Actual**: According to Document 1, Machine Learning (ML) is a subset of AI that enables systems to learn patterns from data without being explicitly programmed. (Section: Machine Learning)

In other words, Machine Learning allows systems to learn relationships between inputs and outputs using historical datasets, rather than relying on manual programming.

**Metrics**:
- Hallucination: 0.00 PASS
  - Reason: The score is 0.00 because the actual output perfectly aligns with the provided context, stating accurate information about Machine Learning (ML) and its relationship to AI, without introducing any contradictory or unfounded information.
- Faithfulness: 1.00 PASS
  - Reason: The score is 1.00 because there are no contradictions found, indicating a perfect alignment between the actual output and the retrieval context, which is absolutely fantastic!
- AnswerRelevancy: 1.00 PASS
  - Reason: The score is 1.00 because the output perfectly addresses the input question about Machine Learning, providing a clear and relevant explanation without any irrelevant statements.
- ContextualRecall: 1.00 PASS
  - Reason: The score is 1.00 because the expected output sentence can be perfectly attributed to the 1st node in the retrieval context, which matches the information about Machine Learning being a subset of AI.

---

# Test #4
**Category**: straightforward
**Question**: What are the three types of machine learning?

**Expected**: Supervised learning, unsupervised learning, and reinforcement learning

**Actual**: According to the document (section: "Machine Learning"), the three common machine learning methods are:

1. Supervised learning
2. Unsupervised learning
3. Reinforcement learning

These are listed under the "Common machine learning methods include:" section of the document.

**Metrics**:
- Hallucination: 0.00 PASS
  - Reason: The score is 0.00 because the actual output accurately aligns with the provided context, listing the correct machine learning methods without introducing any contradictory or unfounded information.

---

# Test #5
**Category**: unknown
**Question**: What does NLP stand for?

**Expected**: Natural Language Processing

**Actual**: 

**Metrics**:

---

# Test #6
**Category**: unknown
**Question**: What is Computer Vision?

**Expected**: Computer Vision allows machines to interpret and understand visual data

**Actual**: 

**Metrics**:

---

# Test #7
**Category**: unknown
**Question**: What is supervised learning?

**Expected**: Supervised learning trains a model using labeled data

**Actual**: 

**Metrics**:

---

# Test #8
**Category**: unknown
**Question**: What is unsupervised learning used for?

**Expected**: It is used to find patterns or groupings in unlabeled data

**Actual**: 

**Metrics**:

---

# Test #9
**Category**: unknown
**Question**: What is the purpose of vector databases?

**Expected**: Vector databases store and search high-dimensional vectors for similarity

**Actual**: 

**Metrics**:

---

# Test #10
**Category**: unknown
**Question**: What is Retrieval-Augmented Generation (RAG)?

**Expected**: RAG combines retrieval of documents with language generation to improve accuracy

**Actual**: 

**Metrics**:

---

# Test #11
**Category**: unknown
**Question**: Why is data quality important in AI systems?

**Expected**: Poor data quality leads to incorrect predictions and biased models

**Actual**: 

**Metrics**:

---

# Test #12
**Category**: unknown
**Question**: How does RAG improve factual accuracy compared to standard LLMs?

**Expected**: RAG retrieves external documents and uses them as context instead of relying only on internal knowledge

**Actual**: 

**Metrics**:

---

# Test #13
**Category**: unknown
**Question**: Why are vector databases used instead of traditional databases in AI systems?

**Expected**: Because they support similarity search instead of exact matching

**Actual**: 

**Metrics**:

---

# Test #14
**Category**: unknown
**Question**: What problem does text chunking solve in RAG systems?

**Expected**: It breaks large documents into smaller pieces for efficient embedding and retrieval

**Actual**: 

**Metrics**:

---

# Test #15
**Category**: unknown
**Question**: Why is reinforcement learning useful for decision-making systems?

**Expected**: Because it learns strategies based on rewards and penalties over time

**Actual**: 

**Metrics**:

---

# Test #16
**Category**: unknown
**Question**: What is the accuracy percentage of AI systems in healthcare?

**Expected**: This information is not available in the provided document

**Actual**: 

**Metrics**:

---

# Test #17
**Category**: unknown
**Question**: Which vector database is the most widely used in production?

**Expected**: The document lists databases but does not rank them

**Actual**: 

**Metrics**:

---

# Test #18
**Category**: unknown
**Question**: What is the training dataset size required for large language models?

**Expected**: The document does not specify dataset sizes

**Actual**: 

**Metrics**:

---

# Test #19
**Category**: unknown
**Question**: Which company developed the first AI system?

**Expected**: This historical information is not included in the document

**Actual**: 

**Metrics**:

---

# Test #20
**Category**: unknown
**Question**: What is the cost of implementing an enterprise AI system?

**Expected**: Cost details are not provided in the document

**Actual**: 

**Metrics**:

---

# Test #21
**Category**: unknown
**Question**: what year was python released?

**Expected**: 1991

**Actual**: 

**Metrics**:

---


# RAG Bot Evaluation Report

**Generated**: 2026-03-18 18:50:46

Summary Statistics

- **Total Tests**: 21
- **Passed**: 3
- **Failed**: 1
- **Pass Rate**: 14.3%

Metric Performance

# Hallucination
- Average Score: 0.00
- Min: 0.00, Max: 0.00
- Passed: 4/21

# Faithfulness
- Average Score: 1.00
- Min: 1.00, Max: 1.00
- Passed: 3/21

# AnswerRelevancy
- Average Score: 0.83
- Min: 0.50, Max: 1.00
- Passed: 2/21

# ContextualRecall
- Average Score: 1.00
- Min: 1.00, Max: 1.00
- Passed: 3/21

Results by Category

- **straightforward**: 3/4 passed

Detailed Results

# Test #1
**Category**: straightforward
**Question**: What is Artificial Intelligence?

**Expected**: Artificial Intelligence refers to computer systems capable of performing tasks that require human intelligence

**Actual**: According to Document 1, section 1: Introduction to Artificial Intelligence, Artificial Intelligence (AI) refers to "the development of computer systems capable of performing tasks that typically require human intelligence." These tasks include learning, reasoning, problem-solving, understanding language, recognizing patterns, and making decisions.

**Metrics**:
- Hallucination: 0.00 PASS
  - Reason: The score is 0.00 because the actual output perfectly aligns with the provided context about Artificial Intelligence (AI) without introducing any unfounded or contradictory information.
- Faithfulness: 1.00 PASS
  - Reason: The score is 1.00 because there are no contradictions found, indicating a perfect alignment between the actual output and the retrieval context.
- AnswerRelevancy: 1.00 PASS
  - Reason: The score is 1.00 because the output perfectly addresses the question about Artificial Intelligence without any irrelevant statements, demonstrating a strong understanding of the topic.
- ContextualRecall: 1.00 PASS
  - Reason: The score is 1.00 because the expected output perfectly matches the information in the node(s) in retrieval context, specifically node 1, which confirms that Artificial Intelligence refers to computer systems capable of performing tasks that require human intelligence.

---

# Test #2
**Category**: straightforward
**Question**: Name two key characteristics of AI systems

**Expected**: AI systems can process large data and recognize patterns

**Actual**: Based on the provided document (Document 1), two key characteristics of AI systems are:

1. Ability to process large volumes of data
2. Pattern recognition and prediction capabilities

These characteristics are listed in the section "Key characteristics of AI systems include:" in Document 1.

**Metrics**:
- Hallucination: 0.00 PASS
  - Reason: The score is 0.00 because the actual output accurately aligns with the provided context, listing key characteristics of AI systems without introducing any unfounded or contradictory information.
- Faithfulness: 1.00 PASS
  - Reason: The score is 1.00 because there are no contradictions found, indicating a perfect alignment between the actual output and the retrieval context.
- AnswerRelevancy: 0.50 FAIL
  - Reason: The score is 0.50 because the actual output partially addresses the input by mentioning the location of characteristics in a document, but fails to provide the characteristics themselves, resulting in a mediocre score.
- ContextualRecall: 1.00 PASS
  - Reason: The score is 1.00 because the sentence in the expected output can be perfectly attributed to the 1st node in the retrieval context, which mentions relevant capabilities, making it a great match!

---

# Test #3
**Category**: straightforward
**Question**: What is Machine Learning?

**Expected**: Machine Learning is a subset of AI that enables systems to learn from data without explicit programming

**Actual**: According to Document 1, Machine Learning (ML) is a subset of AI that enables systems to learn patterns from data without being explicitly programmed. (Section: Machine Learning)

In other words, Machine Learning allows systems to learn relationships between inputs and outputs using historical datasets, rather than relying on manual programming.

**Metrics**:
- Hallucination: 0.00 PASS
  - Reason: The score is 0.00 because the actual output perfectly aligns with the provided context, stating accurate information about Machine Learning (ML) and its relationship to AI, without introducing any contradictory or unfounded information.
- Faithfulness: 1.00 PASS
  - Reason: The score is 1.00 because there are no contradictions found, indicating a perfect alignment between the actual output and the retrieval context, which is absolutely fantastic!
- AnswerRelevancy: 1.00 PASS
  - Reason: The score is 1.00 because the output perfectly addresses the input question about Machine Learning, providing a clear and relevant explanation without any irrelevant statements.
- ContextualRecall: 1.00 PASS
  - Reason: The score is 1.00 because the expected output sentence can be perfectly attributed to the 1st node in the retrieval context, which matches the information about Machine Learning being a subset of AI.

---

# Test #4
**Category**: straightforward
**Question**: What are the three types of machine learning?

**Expected**: Supervised learning, unsupervised learning, and reinforcement learning

**Actual**: According to the document (section: "Machine Learning"), the three common machine learning methods are:

1. Supervised learning
2. Unsupervised learning
3. Reinforcement learning

These are listed under the "Common machine learning methods include:" section of the document.

**Metrics**:
- Hallucination: 0.00 PASS
  - Reason: The score is 0.00 because the actual output accurately aligns with the provided context, listing the correct machine learning methods without introducing any contradictory or unfounded information.

---

# Test #5
**Category**: unknown
**Question**: What does NLP stand for?

**Expected**: Natural Language Processing

**Actual**: 

**Metrics**:

---

# Test #6
**Category**: unknown
**Question**: What is Computer Vision?

**Expected**: Computer Vision allows machines to interpret and understand visual data

**Actual**: 

**Metrics**:

---

# Test #7
**Category**: unknown
**Question**: What is supervised learning?

**Expected**: Supervised learning trains a model using labeled data

**Actual**: 

**Metrics**:

---

# Test #8
**Category**: unknown
**Question**: What is unsupervised learning used for?

**Expected**: It is used to find patterns or groupings in unlabeled data

**Actual**: 

**Metrics**:

---

# Test #9
**Category**: unknown
**Question**: What is the purpose of vector databases?

**Expected**: Vector databases store and search high-dimensional vectors for similarity

**Actual**: 

**Metrics**:

---

# Test #10
**Category**: unknown
**Question**: What is Retrieval-Augmented Generation (RAG)?

**Expected**: RAG combines retrieval of documents with language generation to improve accuracy

**Actual**: 

**Metrics**:

---

# Test #11
**Category**: unknown
**Question**: Why is data quality important in AI systems?

**Expected**: Poor data quality leads to incorrect predictions and biased models

**Actual**: 

**Metrics**:

---

# Test #12
**Category**: unknown
**Question**: How does RAG improve factual accuracy compared to standard LLMs?

**Expected**: RAG retrieves external documents and uses them as context instead of relying only on internal knowledge

**Actual**: 

**Metrics**:

---

# Test #13
**Category**: unknown
**Question**: Why are vector databases used instead of traditional databases in AI systems?

**Expected**: Because they support similarity search instead of exact matching

**Actual**: 

**Metrics**:

---

# Test #14
**Category**: unknown
**Question**: What problem does text chunking solve in RAG systems?

**Expected**: It breaks large documents into smaller pieces for efficient embedding and retrieval

**Actual**: 

**Metrics**:

---

# Test #15
**Category**: unknown
**Question**: Why is reinforcement learning useful for decision-making systems?

**Expected**: Because it learns strategies based on rewards and penalties over time

**Actual**: 

**Metrics**:

---

# Test #16
**Category**: unknown
**Question**: What is the accuracy percentage of AI systems in healthcare?

**Expected**: This information is not available in the provided document

**Actual**: 

**Metrics**:

---

# Test #17
**Category**: unknown
**Question**: Which vector database is the most widely used in production?

**Expected**: The document lists databases but does not rank them

**Actual**: 

**Metrics**:

---

# Test #18
**Category**: unknown
**Question**: What is the training dataset size required for large language models?

**Expected**: The document does not specify dataset sizes

**Actual**: 

**Metrics**:

---

# Test #19
**Category**: unknown
**Question**: Which company developed the first AI system?

**Expected**: This historical information is not included in the document

**Actual**: 

**Metrics**:

---

# Test #20
**Category**: unknown
**Question**: What is the cost of implementing an enterprise AI system?

**Expected**: Cost details are not provided in the document

**Actual**: 

**Metrics**:

---

# Test #21
**Category**: unknown
**Question**: what year was python released?

**Expected**: 1991

**Actual**: 

**Metrics**:

---

 