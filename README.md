# Movie Recommendation System By ScaleCraft - Data Engineering at Scale

A production-ready movie recommendation system built with **PySpark** and **Streamlit**, processing the **MovieLens 25M dataset** (25 million ratings, 62,000 movies) to deliver personalized recommendations using hybrid collaborative filtering and content-based approaches.

---

## Table of Contents

- [Movie Recommendation System By ScaleCraft - Data Engineering at Scale](#movie-recommendation-system-by-scalecraft---data-engineering-at-scale)
  - [Table of Contents](#table-of-contents)
  - [What is This Project?](#what-is-this-project)
    - [Key Features](#key-features)
  - [Why This Project?](#why-this-project)
    - [Business Problem](#business-problem)
    - [Technical Challenges](#technical-challenges)
    - [Solution Approach](#solution-approach)
  - [How Does It Work?](#how-does-it-work)
    - [End-to-End Workflow](#end-to-end-workflow)
    - [Recommendation Algorithm (Cold-Start)](#recommendation-algorithm-cold-start)
  - [Project Architecture](#project-architecture)
    - [Data Flow](#data-flow)
    - [Technology Stack](#technology-stack)
  - [Quick Start](#quick-start)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
      - [**Step 1: Install Java 17**](#step-1-install-java-17)
      - [**Step 2: Clone the Repository**](#step-2-clone-the-repository)
      - [**Step 3: Download MovieLens 25M Dataset**](#step-3-download-movielens-25m-dataset)
      - [**Step 4: Generate Processed Data \& Models**](#step-4-generate-processed-data--models)
      - [**Step 5: Setup App Environment**](#step-5-setup-app-environment)
      - [**Step 6: Run the Streamlit App**](#step-6-run-the-streamlit-app)
      - [**Step 7: Use the App**](#step-7-use-the-app)
    - [Data Management](#data-management)
  - [Dataset Information](#dataset-information)
    - [MovieLens 25M Dataset](#movielens-25m-dataset)
    - [Processed Data](#processed-data)
    - [Trained Models](#trained-models)
  - [Technologies Used](#technologies-used)
  - [Project Structure](#project-structure)
  - [Quick Navigation Guide](#quick-navigation-guide)
    - ["I want to..."](#i-want-to)

---

## What is This Project?

This is a **scalable movie recommendation system** that demonstrates data engineering best practices for building production-ready ML applications. The system:

- **Processes 25 million ratings** using distributed computing (PySpark)
- **Trains hybrid recommendation models** combining collaborative filtering (ALS) and content-based filtering (LSH)
- **Handles cold-start problems** for new users with no rating history
- **Delivers real-time recommendations** through an interactive web interface
- **Uses pre-trained models** for fast inference (2-3 seconds vs 10-15 seconds)

### Key Features

**Hybrid Recommendation Engine**
- Collaborative Filtering (70% weight) - ALS matrix factorization
- Content-Based Filtering (30% weight) - Genome vector similarity

**Cold-Start Solution**
- LSH (Locality Sensitive Hashing) for new users
- Genome vectors (1,128 dimensions) for content similarity

**Production-Ready Architecture**
- Pre-trained model persistence
- Distributed data processing with PySpark
- Interactive web UI with Streamlit

**Comprehensive Evaluation**
- RMSE, MAE, Precision@K, Recall@K, NDCG@K
- Hyperparameter tuning with cross-validation

---

## Why This Project?

### Business Problem

Movie streaming platforms face two critical challenges:

1. **Recommendation Quality**: Users expect personalized suggestions that match their taste
2. **Cold-Start Problem**: New users have no rating history, making personalization difficult

### Technical Challenges

1. **Scale**: Processing 25M ratings requires distributed computing
2. **Latency**: Users expect recommendations in < 3 seconds
3. **Accuracy**: Balancing collaborative and content-based signals
4. **Cold-Start**: Providing quality recommendations without user history

### Solution Approach

This project demonstrates how to:
- **Scale data processing** using PySpark for 25M+ records
- **Optimize inference** using pre-trained models and caching
- **Solve cold-start** using content-based LSH with genome vectors
- **Build hybrid models** combining collaborative and content signals
- **Deploy interactively** with a user-friendly Streamlit interface

---

## How Does It Work?

### End-to-End Workflow

```
┌─────────────────┐
│  Raw CSV Data   │  MovieLens 25M Dataset
│  (25M ratings)  │  - movies.csv (62K movies)
└────────┬────────┘  - ratings.csv (25M ratings)
         │           - genome-scores.csv (15M tag relevances)
         ▼
┌────────────────────────────────────────────────────────┐
│  STEP 1: Data Preparation (01-data-preparation.ipynb)  │
├────────────────────────────────────────────────────────┤
│  • Load CSV files into PySpark DataFrames              │
│  • Clean and validate data (remove nulls, outliers)    │
│  • Create genome vectors (1,128 dimensions per movie)  │
│  • Join movies + ratings + genome data                 │
│  • Save processed data as Parquet files                │
└────────┬───────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│  STEP 2: Model Training (02-model_training.ipynb)      │
├────────────────────────────────────────────────────────┤
│  A. Collaborative Filtering (ALS)                      │
│     • Train ALS model on user-movie ratings            │
│     • Hyperparameter tuning (rank, regParam, alpha)    │
│     • Cross-validation with RMSE evaluation            │
│     • Save best model → models/als_best_model/         │
│                                                        │
│  B. Content-Based Filtering (LSH)                      │
│     • Create item features (genome + ML features)      │
│     • Train LSH model for similarity search            │
│     • Save LSH model → models/lsh_model/               │
│                                                        │
│  C. Hybrid Recommendations                             │
│     • Combine ALS (70%) + Content (30%)                │
│     • Generate recommendations for all users           │
│     • Save → data/hybrid_recommendations.parquet       │
│                                                        │
│  D. Evaluation                                         │
│     • RMSE, MAE (rating prediction accuracy)           │
│     • Precision@K, Recall@K, NDCG@K (ranking quality)  │
└────────┬───────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│  STEP 3: Web Application (app/movie_recommender_app.py)│
├────────────────────────────────────────────────────────┤
│  • Load pre-trained LSH model from disk                │
│  • User selects and rates 5 movies                     │
│  • Create user profile from genome vectors             │
│  • Find similar movies using LSH                       │
│  • Return top 5 recommendations with match scores      │
└────────────────────────────────────────────────────────┘
```

### Recommendation Algorithm (Cold-Start)

For new users with no history:

1. **User Input**: User rates 5 movies (scale 0.5-5.0)
2. **Profile Creation**: Weighted average of genome vectors
   ```
   user_profile = Σ(rating_i × genome_vector_i) / Σ(rating_i)
   ```
3. **Similarity Search**: LSH finds approximate nearest neighbors
4. **Ranking**: Cosine similarity scores
5. **Output**: Top 5 recommendations with match percentages

---

## Project Architecture

### Data Flow

```
Raw Data (CSV) → PySpark Processing → Parquet Storage → Model Training → Pre-trained Models → Streamlit App
```

### Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Data Processing** | PySpark 4.0.1 | Distributed processing of 25M records |
| **Storage** | Parquet | Columnar storage for efficient I/O |
| **ML Framework** | PySpark MLlib | ALS, LSH, model persistence |
| **Web Framework** | Streamlit 1.50.0 | Interactive UI |
| **Runtime** | Java 17 | Required for PySpark 4.x |
| **Language** | Python 3.9+ | Core development |

---

## Quick Start

### Prerequisites

- **Python 3.9+**
- **Java 17** (required for PySpark 4.x)
- **8GB RAM** minimum (16GB recommended)
- **~1.5GB disk space** for data and models

### Installation

#### **Step 1: Install Java 17**

```bash
# macOS (Homebrew)
brew install openjdk@17
export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
export PATH="$JAVA_HOME/bin:$PATH"

# Verify installation
java -version  # Should show 17.x.x
```

#### **Step 2: Clone the Repository**

```bash
git clone https://github.com/thangarajn01/Movie-Recommender-By-ScaleCraft.git
cd Movie-Recommender-By-ScaleCraft
```

#### **Step 3: Download MovieLens 25M Dataset**

**Important**: Large data files (>25MB) are **not included in Git** to keep the repository lightweight.

**Option A - Automated Setup (Recommended):**

```bash
cd code
./setup_data.sh
cd ..
```

This script will:
- Download MovieLens 25M dataset (~250MB compressed)
- Extract all CSV files to `data/` directory
- Set up directory structure
- Show next steps

**Option B - Manual Download:**

1. Download: https://files.grouplens.org/datasets/movielens/ml-25m.zip
2. Extract the zip file
3. Copy all `.csv` files to the `data/` directory

**What gets downloaded:**
- `movies.csv` (3MB) - Movie metadata
- `ratings.csv` (647MB) - User ratings
- `genome-scores.csv` (415MB) - Tag relevance scores
- `genome-tags.csv` (20KB) - Tag descriptions
- `tags.csv` (37MB) - User-generated tags
- `links.csv` (1.3MB) - IMDB/TMDB links

#### **Step 4: Generate Processed Data & Models**

Run the Jupyter notebooks to create processed data and train models:

```bash
cd code

# Install Jupyter if needed
pip install jupyter

# Run data preparation (creates parquet files)
jupyter notebook 01-data-preparation.ipynb  # Run all cells

# Run model training (creates models)
jupyter notebook 02-model_training.ipynb    # Run all cells

cd ..
```

**What gets generated:**
- `data/movie_ratings.parquet/` (332MB) - Cleaned ratings
- `data/genome_vector.parquet/` (23MB) - Genome vectors
- `data/items_features.parquet/` (45MB) - LSH features
- `data/movie_avg.parquet/` (492KB) - Average ratings
- `data/hybrid_recommendations.parquet/` (35MB) - Pre-computed recommendations
- `models/als_best_model/` (26MB) - Trained ALS model
- `models/lsh_model/` (52KB) - Trained LSH model

**⏱️ Processing time**: ~100-180 minutes (depending on hardware)

#### **Step 5: Setup App Environment**

```bash
cd app
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

#### **Step 6: Run the Streamlit App**

```bash
./run_app.sh
```

Or manually:
```bash
export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
streamlit run movie_recommender_app.py
```

#### **Step 7: Use the App**

1. Open browser at `http://localhost:8501`
2. Rate 5 movies using the sliders
3. Click "Get Recommendations"
4. View your personalized top 5 movie recommendations!

---

### Data Management

**Why aren't data files in Git?**

GitHub has file size limits (100MB max). Our dataset includes:
- `ratings.csv` (647MB) ❌ Too large
- `genome-scores.csv` (415MB) ❌ Too large
- Processed parquet files (332MB+) ❌ Too large

**Solution**: Use `setup_data.sh` to download the publicly available MovieLens dataset, then run notebooks to generate processed data.

**See also**:
- [`data/README.md`](data/README.md) - Data directory documentation
- [`models/README.md`](models/README.md) - Models directory documentation

---

## Dataset Information

### MovieLens 25M Dataset

**Source**: [GroupLens Research](https://grouplens.org/datasets/movielens/25m/)

**Statistics**:
- 25,000,095 ratings
- 62,423 movies
- 162,541 users
- 1,128 genome tags
- Ratings from 1995 to 2019

**Files** (downloaded via `setup_data.sh`):

| File | Size | Records | Description |
|------|------|---------|-------------|
| `movies.csv` | 3MB | 62,423 | Movie metadata (ID, title, genres) |
| `ratings.csv` | 647MB | 25,000,095 | User ratings (userId, movieId, rating, timestamp) |
| `genome-scores.csv` | 415MB | 15,584,448 | Tag relevance scores (movieId, tagId, relevance) |
| `genome-tags.csv` | 20KB | 1,128 | Tag descriptions |
| `tags.csv` | 37MB | 1,093,360 | User-generated tags |
| `links.csv` | 1.3MB | 62,423 | IMDB/TMDB links |

### Processed Data

**Generated by running Jupyter notebooks** (`01-data-preparation.ipynb` and `02-model_training.ipynb`):

| File | Size | Format | Description |
|------|------|--------|-------------|
| `movie_ratings.parquet/` | 332MB | Parquet | Joined movies + ratings |
| `genome_vector.parquet/` | 23MB | Parquet | 1,128-dim vectors per movie |
| `items_features.parquet/` | 45MB | Parquet | Genome + ML features for LSH |
| `movie_avg.parquet/` | 492KB | Parquet | Average ratings per movie |
| `hybrid_recommendations.parquet/` | 35MB | Parquet | Pre-computed recommendations |

### Trained Models

**Generated by running `02-model_training.ipynb`**:

| Model | Size | Algorithm | Purpose |
|-------|------|-----------|---------|
| `als_best_model/` | 26MB | ALS (Alternating Least Squares) | Collaborative filtering |
| `lsh_model/` | 52KB | LSH (Locality Sensitive Hashing) | Cold-start recommendations |

---

## Technologies Used

- **PySpark 4.0.1** - Distributed data processing and ML
- **Streamlit 1.50.0** - Interactive web interface
- **Pandas 2.3.3** - Data manipulation
- **NumPy 2.0.2** - Numerical computing
- **PyArrow 21.0.0** - Parquet I/O
- **Java 17** - PySpark runtime

---

## Project Structure

```
DataEnginneringatScale/
├── README.md                              # This file - Main project documentation
│
├── code/                                  # Jupyter notebooks and data setup
│   ├── setup_data.sh                      # Data download script (run this first!)
│   ├── 01-data-preparation.ipynb          # Data cleaning & genome vectors
│   └── 02-model_training.ipynb            # ALS, LSH, hybrid model training
│
├── data/                                  # Raw and processed data (see data/README.md)
│   ├── README.md                          # Data directory documentation
│   │
│   ├── movies.csv                         # Download via setup_data.sh
│   ├── genome-tags.csv                    # Download via setup_data.sh
│   ├── links.csv                          # Download via setup_data.sh
│   ├── tags.csv                           # Download via setup_data.sh
│   ├── ratings.csv                        # Download via setup_data.sh
│   ├── genome-scores.csv                  # Download via setup_data.sh
│   │
│   ├── movie_ratings.parquet/             # Generate via notebook
│   ├── genome_vector.parquet/             # Generate via notebook
│   ├── items_features.parquet/            # Generate via notebook
│   ├── movie_avg.parquet/                 # Generate via notebook
│   └── hybrid_recommendations.parquet/    # Generate via notebook
│
├── models/                                # Trained ML models (see models/README.md)
│   ├── README.md                          # Models directory documentation
│   ├── als_best_model/                    # Generate via notebook
│   └── lsh_model/                         # Generate via notebook
│
├── app/                                   # Streamlit web application
│   ├── README.md                          # App-specific documentation
│   ├── movie_recommender_app.py           # Main Streamlit app
│   ├── requirements.txt                   # Python dependencies
│   ├── run_app.sh                         # Launch script
```
---

## Quick Navigation Guide

### "I want to..."

**...get started quickly**
→ Follow: [Quick Start](#quick-start) section above

**...download the dataset**
→ Run: `cd code && ./setup_data.sh`

**...understand what this project is about**
→ Read: [What is This Project?](#what-is-this-project)

**...run the Streamlit app**
→ Read: [`app/README.md`](app/README.md) or follow [Step 6](#step-6-run-the-streamlit-app)

**...understand the recommendation algorithm**
→ Read: [Recommendation Algorithm](#recommendation-algorithm-cold-start) or [`app/README.md`](app/README.md)

**...train models from scratch**
→ Run: `code/01-data-preparation.ipynb` then `code/02-model_training.ipynb`

**...understand the data pipeline**
→ Read: [End-to-End Workflow](#end-to-end-workflow)

**...know what data files are needed**
→ Read: [`data/README.md`](data/README.md)

**...understand pre-trained models**
→ Read: [`models/README.md`](models/README.md)

**...troubleshoot Java issues**
→ Read: [`app/README.md`](app/README.md) (Troubleshooting section)

**...understand why data isn't in Git**
→ Read: [Data Management](#-data-management) section

**...see the project structure**
→ Read: [Project Structure](#project-structure)

**...understand the tech stack**
→ Read: [Technologies Used](#technologies-used)

---

**ScaleCraft for Data Engineering at Scale Project**

