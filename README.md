# ScholarLLM

## Overview
ScholarLLM is a project aimed at developing a domain-specific Large Language Model (LLM) for the computer science (CS) research community. Our goal is to assist researchers in efficiently processing, understanding, and utilizing scientific knowledge.

## Objectives
- Generate a CS-focused LLM to aid researchers in scientific tasks.
- Explore and compare different approaches for improving model performance.
- Provide tools and models that can enhance the efficiency of research-related activities.

## Methods
We explored three different strategies to achieve our objectives:

1. **Multi-task Fine-tuning**
   - Fine-tune a single model to handle multiple tasks simultaneously.

2. **Instruction Tuning**
   - Perform instruction-based fine-tuning for individual tasks.

3. **Retrieval-Augmented Generation (RAG)**
   - Implement RAG to augment responses with relevant retrieved information from external knowledge sources.

## Directory Structure
- **code_multitask_model**: Code and resources related to multi-task fine-tuning.
- **instruction_tunning**: Code and resources for instruction-based tuning.
- **rag**: Code and resources for the implementation of RAG.

## Usage
The models and scripts provided can be utilized to perform tasks such as document summarization, question answering, and simplification in the CS domain.


## Acknowledgment
Insturction fine-tunning relies on [Unsloth](https://github.com/unslothai/unsloth)
