
# Translation Efficiency Prediction Tutorial Series 🧬

## Overview

This tutorial series will guide you through using OmniGenBench to complete a **Translation Efficiency (TE) prediction** task, which is an important problem in genomics. We will use rice mRNA sequence data to train a genomic foundation model to predict sequence translation efficiency (high/low).

## Learning Paths

### 🎯 Choose Your Learning Path

#### Complete Learners (New to Machine Learning or Genomics)
- ⏱️ **Estimated Time**: 4-6 hours
- 📋 **Recommended Path**: Complete all sub-tutorials in sequence
- 🎓 **Learning Goal**: Comprehensive understanding of the complete data-to-deployment pipeline

#### Practitioners (ML Background, New to Genomics)
- ⏱️ **Estimated Time**: 2-3 hours  
- 📋 **Recommended Path**: Quick overview of 01 → Focus on 02,03 → Practice 04
- 🎓 **Learning Goal**: Master genomics-specific models and data processing

#### Genomics Experts (Biology Background, New to Deep Learning)
- ⏱️ **Estimated Time**: 2-3 hours
- 📋 **Recommended Path**: Skip biology background → Focus on 02,03 → Deep dive into 04
- 🎓 **Learning Goal**: Understand deep learning applications in genomics

## Tutorial Structure

### 📊 [01_Data_Preparation.ipynb](tutorials/01_Data_Preparation.ipynb)
**Data Preparation: From Biological Sequences to Machine Learning Data**
- 🧬 Introduction to genomic data types (DNA vs RNA)
- 🔢 Machine learning task classification (Classification vs Regression vs Sequence Labeling)
- 🗂️ Data processing best practices
- ➡️ **Transitions to**: Model selection and initialization

### 🤖 [02_Model_Initialization.ipynb](tutorials/02_Model_Initialization.ipynb)  
**Model Initialization: Understanding and Selecting Genomic Foundation Models**
- 🏗️ Foundation model concepts and advantages
- 🔧 Model-task matching principles
- 📥 Model and tokenizer loading
- ➡️ **Transitions to**: Model training process

### 🏋️ [03_Model_Training.ipynb](tutorials/03_Model_Training.ipynb)
**Model Training: Fine-tuning Your Genomic Model**
- 📚 Supervised learning and fine-tuning principles
- ⚖️ Comparison of three trainer types and selection
- 🔄 Complete training workflow
- ➡️ **Transitions to**: Model deployment and inference

### 🚀 [04_Inference_and_Deployment.ipynb](tutorials/04_Inference_and_Deployment.ipynb)
**Inference and Deployment: Making Your Model Serve Real Applications**
- 🔮 Model inference pipeline
- 🌐 Deployment solutions (FastAPI, etc.)
- 📈 Performance evaluation and optimization
- 🔄 **Returns to**: Project summary

## Quick Start

If you want to quickly experience the complete workflow, you can run:

```bash
# Install dependencies
pip install omnigenbench -U

# Enter tutorial directory
cd tutorials/

# Run tutorials in sequence
jupyter notebook 01_Data_Preparation.ipynb
```

## Technical Requirements

- Python 3.8+
- PyTorch 2.0+
- 8GB+ RAM
- (Optional) GPU acceleration

## Support and Feedback

If you encounter problems during your learning process, please:
1. Check the FAQ sections in each tutorial
2. Consult the [OmniGenBench documentation](https://omnigenbenchdoc.readthedocs.io/)
3. Submit an issue on GitHub

---

**Start your genomic machine learning journey!** 🧬✨