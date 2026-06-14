# 📊 Instagram Trend Analyzer using NLP & Machine Learning

## Overview

Instagram Trend Analyzer is an end-to-end Data Science and Machine Learning project that analyzes Instagram engagement patterns, predicts post virality, and recommends relevant hashtags using Natural Language Processing (NLP).

The project combines Exploratory Data Analysis (EDA), Feature Engineering, Machine Learning, and Recommendation Systems to extract actionable insights from Instagram content performance data.

---

## Problem Statement

Content creators and businesses often struggle to determine:

* Which content is likely to go viral
* Which hashtags should be used
* What posting times maximize engagement
* Which topics perform best

This project aims to solve these problems using data-driven techniques.

---

## Dataset

**Source:** Social Media Performance Dataset

### Dataset Statistics

* Total Social Media Posts: 10,000
* Instagram Posts Analyzed: 2,500
* Multiple engagement and metadata features

### Available Features

* Platform
* Content Type
* Topic
* Language
* Region
* Post Date & Time
* Hashtags
* Sentiment Score
* Views
* Likes
* Comments
* Shares
* Engagement Rate
* Viral Label

---

## Project Architecture

```text
Dataset
   │
   ▼
Data Cleaning
   │
   ▼
Exploratory Data Analysis
   │
   ▼
Feature Engineering
   │
   ├── Posting Hour
   ├── Day of Week
   ├── Hashtag Count
   └── Engagement Score
   │
   ▼
Machine Learning
   │
   ├── Random Forest
   └── XGBoost
   │
   ▼
Virality Prediction
   │
   ▼
NLP Recommendation System
   │
   ├── TF-IDF
   ├── Cosine Similarity
   └── Hashtag Recommendation
```

---

## Exploratory Data Analysis

Performed comprehensive analysis on Instagram posts including:

* Viral vs Non-Viral content distribution
* Topic-wise engagement comparison
* Best posting day analysis
* Best posting hour analysis
* Feature importance analysis
* Engagement trend visualization

### Key Insights

#### Best Posting Day

Thursday showed the highest average engagement among all days.

#### Best Posting Hour

4 PM demonstrated the highest average engagement.

#### Most Important Features

The machine learning model identified:

1. Comments
2. Likes
3. Shares
4. Engagement Score

as the strongest indicators of virality.

---

## Feature Engineering

Created additional features to improve predictive performance:

| Feature          | Description                        |
| ---------------- | ---------------------------------- |
| posting_hour     | Hour extracted from post timestamp |
| day_of_week      | Day extracted from post timestamp  |
| hashtag_count    | Number of hashtags used            |
| engagement_score | Likes + Comments + Shares          |

---

## Machine Learning Models

### Random Forest Classifier

Used for viral post classification.

**Performance**

* Accuracy: 77.2%
* Viral Post F1 Score: 0.83

### XGBoost Classifier

Used as an alternative ensemble model.

**Performance**

* Accuracy: 75.6%

### Model Comparison

| Model         | Accuracy |
| ------------- | -------- |
| Random Forest | 77.2%    |
| XGBoost       | 75.6%    |

**Best Performing Model:** Random Forest

---

## NLP-Based Hashtag Recommendation Engine

Implemented a recommendation system using:

* TF-IDF Vectorization
* Cosine Similarity

### Workflow

```text
User Query
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Similarity Search
      │
      ▼
Topic Detection
      │
      ▼
Hashtag Recommendation
```

### Example

Input:

```text
AI machine learning startup
```

Output:

```text
#AI
#Innovation
#TechTrends
#CodingLife
#Programming
```

---

## Technologies Used

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib

### Machine Learning

* Scikit-Learn
* Random Forest
* XGBoost

### NLP

* TF-IDF
* Cosine Similarity

### Model Persistence

* Joblib

---

## Repository Structure

```text
instagram-trend-analyzer/

├── data/
│   └── social_media_performance.csv

├── models/
│   └── viral_post_predictor.pkl

├── notebooks/
│   └── Instagram_Trend_Analyzer.ipynb

├── README.md

└── requirements.txt
```

---

## Future Improvements

* Streamlit Dashboard
* FastAPI Backend
* Real-Time Instagram Data Collection
* Transformer-Based Recommendations
* Trend Forecasting Models
* Deployment on Cloud Platforms

---

## Learning Outcomes

Through this project, I learned:

* Data Cleaning and Preprocessing
* Exploratory Data Analysis
* Feature Engineering
* Classification Models
* Model Evaluation
* NLP Fundamentals
* Recommendation Systems
* GitHub Project Organization

---

## Author

Divyanshu

B.Tech, IIITDM Kurnool

Interested in Machine Learning, Data Science, NLP, and Software Development.
