# Phishing Website Detection using Machine Learning

A machine learning project that classifies websites as **legitimate** or **phishing** from 30 URL, domain and page-based features. It trains three classifiers, compares them using standard metrics, and saves the best model for reuse.

## Why this project

Phishing is one of the most common ways attackers steal credentials and spread malware. Automated detection helps security teams flag malicious sites before users interact with them. This project explores how machine learning can support that task.

## Dataset

- **Source:** UCI Machine Learning Repository, *Phishing Websites* dataset (also hosted on OpenML, id 4534)
- **Size:** 11,055 websites, 30 features
- **Labels:** phishing or legitimate
- **Example features:** presence of an IP address in the URL, URL length, use of URL shortening services, `@` symbol, number of subdomains, SSL certificate state, domain registration length, redirects, and others

## Approach

1. Load the dataset and convert labels to `0 = legitimate`, `1 = phishing`
2. Split into 80% training and 20% testing (stratified, `random_state=42`)
3. Train and compare three models:
   - Logistic Regression (baseline)
   - Decision Tree
   - Random Forest
4. Evaluate using accuracy, precision, recall and F1-score
5. Select the best model by F1-score, plot its confusion matrix and the top 10 most important features
6. Save the best model with `joblib`

## How to run

```bash
git clone https://github.com/Samee2406/phishing-detector.git
cd phishing-detector
pip install -r requirements.txt
python phishing_detector.py
```

The script downloads the dataset automatically from OpenML. To use a local copy instead:

```bash
python phishing_detector.py --csv path/to/phishing.csv --target Result
```

## Output files

| File | Description |
|------|-------------|
| `results.csv` | Metrics for all three models |
| `confusion_matrix.png` | Confusion matrix of the best model |
| `feature_importance.png` | Top 10 features used by the best model |
| `phishing_model.joblib` | Saved best model |

## Results

> Run the script and paste your own numbers here.

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----|
| Logistic Regression | | | | |
| Decision Tree | | | | |
| Random Forest | | | | |

**Observations:** *(Write 2-3 lines in your own words: which model performed best, which features mattered most, and where it made mistakes. In security, a false negative, meaning a phishing site marked as legitimate, is usually more costly than a false positive.)*

## Limitations

- The dataset is a fixed snapshot, so newer phishing techniques may not be represented.
- Several features (for example web traffic and Google index status) depend on external lookups and cannot be computed from a URL alone.
- Results on a single train/test split can vary; cross-validation would give a more reliable estimate.

## Future work

- Add cross-validation and hyperparameter tuning
- Build a URL feature extractor so the model can score a raw URL
- Try gradient boosting (XGBoost / LightGBM)
- Wrap the model in a small Flask or Streamlit app

## Tech stack

Python, scikit-learn, pandas, matplotlib, joblib
