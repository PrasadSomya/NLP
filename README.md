# Stock News Sentiment Analysis and Weekly Market Intelligence

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-green)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Logistic%20Regression-purple)
![LLM](https://img.shields.io/badge/LLM-Llama--2--7B-red)

An end-to-end Natural Language Processing project that analyzes stock-related news, classifies each article as **negative**, **neutral**, or **positive**, studies its relationship with market activity, and generates weekly summaries of potentially market-moving events using a Large Language Model.

## Project Overview

Financial markets react quickly to company announcements, analyst opinions, economic developments, and media reports. Manually reviewing a large volume of financial news is time-consuming and may lead to delayed or inconsistent decisions.

This project builds an AI-driven workflow to:

- explore historical stock-news and market data;
- identify relationships between sentiment, prices, returns, volatility, and volume;
- compare multiple text-embedding approaches for sentiment classification;
- classify news into negative, neutral, and positive sentiment;
- aggregate daily news at a weekly level; and
- use an LLM to extract the most important positive and negative weekly events.

> **Disclaimer:** This project is intended for educational and analytical purposes only. It does not provide financial or investment advice.

## Business Objective

The objective is to help financial analysts process stock-related news more efficiently and convert unstructured text into actionable market intelligence.

The solution supports two primary use cases:

1. **News sentiment classification** — predict whether a news article has a negative, neutral, or positive market impact.
2. **Weekly event summarization** — consolidate multiple news articles and identify the major positive and negative events that may affect the stock price.

## Dataset

The notebook uses `stock_news.csv`, containing historical news and stock-market information for a NASDAQ-listed company.

| Column | Description |
|---|---|
| `Date` | Date on which the news was released |
| `News` | Stock-related news content |
| `Open` | Opening stock price |
| `High` | Highest stock price during the day |
| `Low` | Lowest stock price during the day |
| `Close` | Adjusted closing stock price |
| `Volume` | Number of shares traded during the day |
| `Label` | News sentiment: `-1` negative, `0` neutral, `1` positive |

### Dataset Summary

- **Rows:** 349
- **Columns:** 8
- **Missing values:** None
- **Duplicate rows:** None
- **Sentiment distribution:**
  - Neutral: 170 records — 48.71%
  - Negative: 99 records — 28.37%
  - Positive: 80 records — 22.92%

The target distribution is moderately imbalanced, with neutral news forming the largest class.

## Project Workflow

```text
Stock News and Price Data
          │
          ▼
Data Inspection and Cleaning
          │
          ▼
Exploratory Data Analysis
          │
          ▼
Chronological Train/Validation/Test Split
          │
          ▼
Text Embedding Generation
 ┌────────┼─────────────┐
 │        │             │
Word2Vec  GloVe         SBERT
 │        │             │
 └────────┴─────────────┘
          │
          ▼
Logistic Regression + GridSearchCV
          │
          ▼
Sentiment Classification
          │
          ▼
Weekly News Aggregation
          │
          ▼
Llama-2 Weekly Event Summarization
```

## Exploratory Data Analysis

The notebook performs univariate and bivariate analysis across news, sentiment, and stock-market variables.

### Engineered Features

The following additional variables are created:

- `News_Length` — length of each news article;
- `Price_Change` — difference between closing and opening price;
- `Daily_Return_Pct` — daily percentage return;
- `Volatility` — difference between daily high and low price; and
- `Moving_Avg_7` — seven-period moving average of closing price.

### Key EDA Findings

- Neutral sentiment is the dominant target class.
- Opening, high, low, and closing prices are strongly correlated, as expected.
- Trading volume has a weak linear correlation with sentiment and daily price changes.
- Sentiment does not have a strong simple linear relationship with price, although positive news tends to align with stronger returns and negative news may coincide with higher volatility.
- The class imbalance can reduce the model's ability to identify positive and negative news consistently.

## Data Preprocessing

The data is sorted chronologically before splitting to reduce the risk of future information leaking into model training.

| Dataset | Records | Share |
|---|---:|---:|
| Training | 244 | 70% |
| Validation | 52 | 15% |
| Test | 53 | 15% |

The split uses `shuffle=False` so that temporal order is preserved.

For Word2Vec and GloVe, the text is:

- converted to lowercase;
- cleaned using regular expressions;
- tokenized with NLTK; and
- filtered to remove English stop words.

SBERT is applied to the original news text so that contextual information is retained.

## Text Embedding Approaches

Three embedding strategies are compared.

### 1. Word2Vec

- Trained from scratch using the project training data
- Vector size: 100
- Document representation: average of word vectors

### 2. GloVe

- Pretrained model: `glove-wiki-gigaword-100`
- Vector size: 100
- Document representation: average of available word vectors

### 3. Sentence Transformer

- Model: `all-MiniLM-L6-v2`
- Embedding size: 384
- Captures contextual and semantic relationships in complete sentences

## Sentiment Classification

A Logistic Regression classifier is trained separately on each embedding representation.

### Model Configuration

- Algorithm: Logistic Regression
- Class balancing: `class_weight="balanced"`
- Maximum iterations: 1,000
- Hyperparameter tuning: `GridSearchCV`
- Cross-validation: 3 folds
- Candidate `C` values: `0.01`, `0.1`, `1`, `10`, `100`
- Primary evaluation metric: Macro F1-score

Macro F1 is selected because it gives equal importance to negative, neutral, and positive classes despite their unequal frequencies.

## Model Results

### Validation Performance

| Embedding | Best C | Validation Accuracy | Validation Macro F1 |
|---|---:|---:|---:|
| Word2Vec | 100 | 0.3269 | 0.2915 |
| GloVe | 1 | 0.4038 | 0.4024 |
| Sentence Transformer | 0.1 | **0.5192** | **0.5290** |

The Sentence Transformer representation provides the strongest validation performance and is selected for final evaluation.

### Final Test Performance

| Metric | Score |
|---|---:|
| Accuracy | 0.4151 |
| Macro F1-score | 0.3988 |

### Class-wise Test Results

| Sentiment | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Negative | 0.40 | 0.38 | 0.39 | 16 |
| Neutral | 0.44 | 0.48 | 0.46 | 25 |
| Positive | 0.36 | 0.33 | 0.35 | 12 |

The decrease from validation to test performance highlights the difficulty of generalizing from a small, imbalanced financial-news dataset.

## Weekly News Summarization with Llama-2

The project also implements an LLM-based summarization workflow.

### LLM Configuration

- Model repository: `TheBloke/Llama-2-7B-Chat-GGUF`
- Model file: `llama-2-7b-chat.Q4_K_M.gguf`
- Runtime library: `llama-cpp-python`
- Context window: 4,000 tokens
- GPU configuration: all supported layers offloaded to the GPU

### Summarization Process

1. Convert the date column to datetime format.
2. Group the data by week.
3. Combine all news articles from each week.
4. Prompt Llama-2 to identify the top positive and negative events.
5. Request structured JSON output.
6. Parse and format the model output into a presentation-ready DataFrame.

The notebook demonstrates the workflow on the first five aggregated weeks.

### LLM Limitation

Llama-2 can identify meaningful weekly events, but it does not always return perfectly valid JSON. The project therefore includes post-processing and fallback logic for extracting and formatting the generated events.

## Technologies Used

- Python
- Jupyter Notebook / Google Colab
- Pandas and NumPy
- Matplotlib and Seaborn
- NLTK
- Gensim
- Sentence Transformers
- Scikit-learn
- Hugging Face Hub
- Llama CPP Python
- tqdm

## Repository Structure

```text
NLP/
├── NLP_Project_Full_Code_1.ipynb   # Complete analysis and implementation
├── NLP_Stocks.html                 # Executed HTML version of the notebook
└── README.md                       # Project documentation
```

The dataset is not included in the repository and must be added separately.

## Installation

Clone the repository:

```bash
git clone https://github.com/PrasadSomya/NLP.git
cd NLP
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the core dependencies:

```bash
pip install pandas numpy matplotlib seaborn nltk gensim \
    sentence-transformers transformers scikit-learn \
    huggingface-hub tqdm jupyter
```

Install `llama-cpp-python` according to your CPU, operating system, and CUDA version. The notebook contains a Google Colab installation command for a CUDA 12.2 environment.

## How to Run

### Google Colab

1. Open `NLP_Project_Full_Code_1.ipynb` in Google Colab.
2. Upload `stock_news.csv` to Google Drive.
3. Update the dataset path in the notebook:

```python
df = pd.read_csv('/content/drive/MyDrive/path/to/stock_news.csv')
```

4. Select a GPU runtime for faster embedding generation and LLM inference:

```text
Runtime → Change runtime type → T4 GPU
```

5. Run all cells sequentially.

### Local Jupyter Environment

Place `stock_news.csv` in the repository and change the loading code to:

```python
df = pd.read_csv('stock_news.csv')
```

Then launch Jupyter:

```bash
jupyter notebook
```

Open the notebook and execute the cells in order.

## Main Conclusions

- Contextual SBERT embeddings outperform Word2Vec and GloVe for this financial-news classification problem.
- A small and imbalanced dataset limits out-of-sample model performance.
- Financial sentiment and stock-price movements have complex relationships that are not fully captured by simple correlations.
- LLMs can convert large weekly news collections into concise event summaries, but structured-output validation is essential.

## Recommended Improvements

- Fine-tune a financial-domain model such as FinBERT.
- Collect a larger and more recent labeled dataset.
- Apply time-series cross-validation instead of relying on a single chronological split.
- Experiment with stronger imbalance-handling techniques.
- Add TF-IDF and n-gram baselines for comparison.
- Tune decision thresholds using business costs rather than accuracy alone.
- Use constrained generation or schema-validation libraries to guarantee valid LLM output.
- Evaluate generated summaries using human review and factual-consistency checks.
- Backtest whether sentiment features improve return, direction, or volatility forecasting.
- Package the pipeline as an API or interactive dashboard for analyst use.

## Author

**Somya Prasad**

GitHub: [PrasadSomya](https://github.com/PrasadSomya)
