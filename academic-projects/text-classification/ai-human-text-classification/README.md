# AI vs Human Text Classification

A comprehensive machine learning project comparing multiple approaches for distinguishing between human-written and AI-generated text across different domains.

## Project Overview

This project implements and compares various machine learning models to classify text as either human-written or AI-generated. The study focuses on domain adaptation and cross-domain generalization capabilities of different approaches.

## Key Features

- **Multi-Model Comparison**: LightGBM, CNN, MLP, RNN, and SVM implementations
- **Domain Adaptation**: Training strategies for cross-domain generalization
- **Data Balancing**: Techniques to handle imbalanced datasets
- **Performance Analysis**: Comprehensive evaluation across different domains

## Project Structure

### EDA (`EDA/`)
- **EDA.ipynb**: Exploratory data analysis including:
  - Word distribution analysis between domains
  - Text length statistics for human vs machine-generated content
  - Distinctive word count analysis across domains

### Models (`Models/`)
- **lgb_final.ipynb**: Final LightGBM implementation with data balancing
- **lgb_approach_two.ipynb**: Two-step training approach (domain 1 → domain 2 fine-tuning)
- **lgb_fine_tuning.ipynb**: Hyperparameter optimization for LightGBM
- **MLP+RNN.ipynb**: Neural network implementations (MLP and RNN)
- **svm.ipynb**: Support Vector Machine with PCA and TF-IDF features
- **CNN.ipynb**: Convolutional Neural Network for text classification
- **logistic_regression_baseline.ipynb**: Baseline logistic regression model
- **Balanced.ipynb**: Data balancing techniques and implementations

## Performance

- **Best Model**: LightGBM achieved 85% accuracy on test set
- **Domain Adaptation**: Successful cross-domain generalization
- **Ensemble Approach**: Improved performance through model combination

## Technical Stack

- **Machine Learning**: LightGBM, Scikit-learn, PyTorch
- **NLP**: NLTK, TF-IDF, Word embeddings
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn 


