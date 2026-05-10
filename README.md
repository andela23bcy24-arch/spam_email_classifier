# Spam Email Classifier

A simple, effective machine learning application that classifies emails as spam or not spam using the Naive Bayes algorithm.

## Features

- Text cleaning & preprocessing
- TF-IDF vectorization
- Naive Bayes model for spam detection
- Visual evaluation (confusion matrix)
- Human-readable, clean code



## Getting Started

**1. Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/spam-email-classifier.git
cd spam-email-classifier
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run the classifier:**
```bash
python src/train.py
```

**4. Customize dataset:**  
Replace or expand `data/sample_emails.csv` with your own dataset in the format:  

| text | label |
|------|-------|
| "Email message here." | spam/ham |

---

## Project Structure

- `src/preprocess.py` — Data loading and cleaning functions
- `src/train.py` — Main training & evaluation script
- `data/sample_emails.csv` — Example dataset
- `requirements.txt` — Dependency list

## License

MIT License
