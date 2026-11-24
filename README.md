# BERT-Based AG News Classification

## 📌 Project Overview

This project implements a **News Text Classification System using BERT (Bidirectional Encoder Representations from Transformers)** to automatically categorize news articles into four different categories: World, Sports, Business, and Sci/Tech.

The project uses the **AG News dataset** and fine-tunes the pre-trained `bert-base-uncased` model for multi-class text classification.

It leverages the Hugging Face Transformers library for model training and evaluation, along with PyTorch and Scikit-learn for deep learning and performance measurement.

## 🎯 Project Objectives

- Implement a BERT-based text classification model.
- Perform text tokenization and preprocessing using BERT Tokenizer.
- Fine-tune a pre-trained BERT model on the AG News dataset.
- Evaluate model performance using standard classification metrics.
- Predict the category of previously unseen news articles.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Deep Learning Framework:** PyTorch
- **NLP Model:** BERT (bert-base-uncased)
- **Libraries:** Hugging Face Transformers, Datasets, NumPy, Scikit-learn
- **Dataset:** AG News Dataset

## 📂 Project Structure

```text
BERT-AG-News-Classification/
│
├── bert.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 📊 Dataset

This project uses the **AG News Dataset**, a widely used benchmark dataset for news topic classification.

The dataset contains four news categories:

| Label | Category |
|---|---|
| 0 | World |
| 1 | Sports |
| 2 | Business |
| 3 | Sci/Tech |

The dataset is loaded directly using the Hugging Face Datasets library.

```python
dataset = load_dataset("ag_news")
```

For faster experimentation and CPU-based training, a subset of the dataset is selected:

- Training Samples: 2,000
- Testing Samples: 500

## ⚙️ Implementation Workflow

### 1. Data Loading

Load the AG News dataset using the Hugging Face Datasets library.

### 2. Text Preprocessing and Tokenization

Use the pre-trained BERT tokenizer to convert news articles into numerical input representations.

- Padding: `max_length`
- Truncation: Enabled
- Maximum Sequence Length: 64

### 3. Model Initialization

Load the pre-trained `bert-base-uncased` model and configure it for four-class sequence classification.

### 4. Model Training

Fine-tune the BERT model using the following training configuration:

| Parameter | Value |
|---|---|
| Pre-trained Model | bert-base-uncased |
| Number of Classes | 4 |
| Learning Rate | 2e-5 |
| Training Batch Size | 8 |
| Evaluation Batch Size | 8 |
| Epochs | 1 |
| Weight Decay | 0.01 |
| Optimizer | Hugging Face Trainer default |

### 5. Model Evaluation

Evaluate the trained model using the test dataset.

The following performance metrics are calculated:

- **Accuracy:** Measures the proportion of correctly classified news articles.
- **Precision:** Measures the correctness of positive predictions.
- **Recall:** Measures how effectively the model identifies relevant samples.
- **F1-Score:** Measures the balance between precision and recall.

Weighted averages are used for precision, recall, and F1-score.

### 6. News Category Prediction

The trained model predicts the category of a new news article.

**Sample Input:**

```text
Apple launches new AI chip for advanced smartphones.
```

**Expected Output Format:**

```text
Predicted Category: Sci/Tech
```

*Note: The actual prediction depends on the trained model.*

## 🚀 Installation and Setup

### Prerequisites

- Python 3.9 or later
- pip package manager
- Internet connection for downloading the dataset and pre-trained BERT model

### Step 1: Clone the Repository

```bash
git clone https://github.com/Lakshanika1/BERT-AG-News-Classification.git
```

### Step 2: Navigate to the Project Directory

```bash
cd BERT-AG-News-Classification
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Project

```bash
python bert.py
```

The program downloads the required dataset and pre-trained BERT model, performs fine-tuning, evaluates the model, and displays the predicted category for the sample news article.

## 📈 Output

The program displays:

1. Training progress.
2. Evaluation metrics:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
3. Predicted category for the sample news article.

Example output format:

```text
{'eval_loss': ..., 'eval_accuracy': ..., 'eval_f1': ..., 'eval_precision': ..., 'eval_recall': ...}

Predicted Category: Sci/Tech
```

*Evaluation values may vary depending on the training environment and model execution.*

## 💡 Key Features

- BERT-based contextual text understanding.
- Automated news article classification.
- Four-category multi-class classification.
- Transfer learning using a pre-trained Transformer model.
- Performance evaluation using multiple classification metrics.
- Reduced dataset size and sequence length for faster CPU experimentation.

## 🔮 Future Enhancements

- Train the model using the complete AG News dataset.
- Increase the number of training epochs for improved performance.
- Experiment with different Transformer architectures such as DistilBERT and RoBERTa.
- Implement a web-based interface using Streamlit or Flask.
- Save and load the fine-tuned model for independent predictions.
- Optimize model performance using GPU acceleration.

## 📝 Conclusion

This project demonstrates the application of **BERT and Transformer-based deep learning techniques for automated news classification**.

By fine-tuning a pre-trained BERT model on the AG News dataset, the system learns contextual representations of news articles and classifies them into four major categories.

The project provides practical experience in Natural Language Processing (NLP), transfer learning, text preprocessing, Transformer architectures, and model evaluation.

It also establishes a foundation for developing more advanced NLP applications using pre-trained language models.

## 👩‍💻 Author

**Lakshanika A P**

B.E. Computer Science and Engineering

---

⭐ If you find this project useful, consider giving the repository a star!
