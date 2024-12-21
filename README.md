# VisualWebBench Project - Fine-Tuning Multimodal Large Language Models (MLLMs)

This repository contains the implementation, evaluation, and fine-tuning pipelines for improving Multimodal Large Language Models (MLLMs) on the [VisualWebBench](https://visualwebbench.github.io/) benchmark using the [MultiUI](https://neulab.github.io/MultiUI/) dataset. Our work was conducted as part of the coursework at the Carnegie Mellon University Language Technologies Institute (LTI) within the School of Computer Science ([Course Website](https://cmu-mmml.github.io/)).

---

## Project Overview

This project focuses on advancing **open-source multimodal models with compact architectures (models with up to 7 billion parameters)** for web-based tasks. Our motivation stems from the need to bridge performance gaps in Multimodal Machine Learning (MMML) by exploring innovative techniques and model architectures. While prior works have often showcased new MMML techniques by benchmarking results on VisualWebBench, none have demonstrated training models on a similar WebUI dataset and subsequently improving their performance on VisualWebBench tasks.

Key Highlights:
1. **Fine-Tuning Techniques**: Leveraged the MultiUI dataset to fine-tune LLaVA-v1.5-7B, improving its ability to reason about web UI elements and perform tasks such as OCR, grounding, and captioning.
2. **Prompt Optimization and Visual Attention Analysis**: Conducted in-depth qualitative analyses of visual attention mechanisms to enhance alignment between textual and visual modalities. This led to task-specific prompt designs and preprocessing strategies that significantly boosted performance.
3. **State-of-the-Art Results**: Achieved competitive results, including state-of-the-art performance in Action Prediction (77.94%) and second-best open-source results for Heading OCR (54.82%).
4. **Comprehensive Research**: Conducted a detailed evaluation of baseline models and systematically demonstrated improvements through model fine-tuning and prompt engineering.

Our findings not only validate the utility of training open-source MMML models on WebUI datasets but also highlight the potential of task-specific fine-tuning, prompt optimization, and alignment improvements in achieving robust multimodal performance.

---

## Repository Structure

- **`R2/`**: Contains baseline models and inference code used to evaluate existing methods on VisualWebBench.
- **`R4/`**: Includes everything related to fine-tuning LLaVA on the MultiUI dataset, including data preprocessing, fine-tuning scripts, and result evaluation.

> **Note**: Due to the large size of the model weights and related files, they are not included in this repository. To use this code, replace the model paths with your own.

---

## Datasets

### [MultiUI Dataset](https://neulab.github.io/MultiUI/)
A diverse collection of multimodal instructions synthesized from over 1 million websites, enabling robust training for tasks like visual reasoning, OCR, and grounding.

### [VisualWebBench](https://visualwebbench.github.io/)
A benchmark that evaluates multimodal models across seven tasks:
- Webpage QA
- Action Prediction
- Action Grounding
- Element Grounding
- Element OCR
- Heading OCR
- Captioning

For detailed use cases, refer to the original [paper](https://arxiv.org/pdf/2401.13649.pdf).

---

## Key Contributions

1. Fine-tuned LLaVA-v1.5-7B on a subset of MultiUI for improved multimodal reasoning and OCR capabilities.
2. Developed structured task-specific prompts and pre-processing strategies, supported by visual attention analysis, to enhance grounding and action prediction tasks.
3. Conducted a detailed analysis of baseline models to guide improvements.

---

## Results

Our fine-tuned LLaVA model achieved:
- Significant improvements in OCR (54.82%) and Action Prediction (77.94%) tasks.
- Insights into failure modes and strategies for further enhancements.

Refer to the `results/` directory for detailed metrics and visualizations.

---

## Team

This project was conducted by AI Engineering students at Carnegie Mellon University:
- **Akshay Badagabettu**
- **Nikolaj Hindsbo**
- **Aayush Shah**
- **Sai Yarlagadda**

Contact: 
`{nikolajhindsbo} at gmail`
`{abadagab, aayushsh, saisravy}@andrew.cmu.edu`

---
