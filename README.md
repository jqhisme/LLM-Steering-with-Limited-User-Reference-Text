# LLM-Steering-with-Limited-User-Reference-Text
Develop a user friendly pipelines that control LLM output by training a model to steer LLM using limited input text reference 
<img width="1513" height="904" alt="Screenshot 2025-05-25 130113" src="https://github.com/user-attachments/assets/698da4b6-1c81-4831-92a0-e64c855e7c90" />

## Dependencies
SAELens\
TransformerLens\
Transformers\
Sentence-Transfromers

## Training
Train with ranked loss during the steering process 
```
python train_ranksteer.py
```

Train with triplet loss 
```
pythonn train_tripletLoss.py
```


## Project Motivation
**Simplify LLM Steering**: Large language models contain thousands of features. Manually navigating these features from sparse autoencoders (SAEs) is overwhelming and inefficient. Can we create a smarter way to align small user inputs with these features for precise control?

**Leverage Small Data**:Using small, user-provided reference texts can streamline feature alignment and make LLM control more accessible.

## Project Setup
Data: Reddit tldr-17 dataset

Assume social media style text input, 3.8 million user post, no longer than 512 tokens, relatively distinct themes and speaking styles;

LLM: GPT-2,Short Inference time and more well studied for mechanistic interpretability

SAE: gpt2-small · 8-res_fs768-jb

Relatively small latent dimension, and sufficient level of abstraction (determined by the layer) and variance explained.

Explained Variance: 92.57%

Alive Feature: 100%

## Training Approach
**Approach 1:**
<img width="1799" height="1014" alt="Screenshot 2025-08-09 125639" src="https://github.com/user-attachments/assets/e4fe7444-4630-47ff-8858-65b826b83acc" />

**Approach 2: end -to-end**
<img width="1921" height="1139" alt="Screenshot 2025-08-09 125842" src="https://github.com/user-attachments/assets/22a6590e-6c1b-474d-a584-bd96764c1c5f" />


