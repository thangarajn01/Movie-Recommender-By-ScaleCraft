# Movie Recommender - Streamlit Application by ScaleCraft

An interactive web application that provides **personalized movie recommendations** for new users using **content-based filtering** with **LSH (Locality Sensitive Hashing)** and **genome vectors**.

---

## Table of Contents

- [What is This App?](#what-is-this-app)
- [Why This Approach?](#why-this-approach)
- [How Does It Work?](#how-does-it-work)
- [How to Use It?](#how-to-use-it)
- [Installation](#installation)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)

---

## What is This App?

This is a **cold-start movie recommendation engine** that solves the challenge of recommending movies to **new users with no rating history**.

### Key Features

- **No History Required** - Works for brand new users  
- **Fast Recommendations** - 2-3 seconds with pre-trained models  
- **High Quality** - Uses 1,128-dimensional genome vectors  
- **Interactive UI** - Beautiful, Netflix-inspired interface  
- **Scalable** - Built on PySpark for production deployment  

### What You Can Do

1. **Select 5 movies** from a list of popular movies
2. **Rate them** on a scale of 0.5 to 5.0 stars
3. **Get 5 personalized recommendations** with match scores
4. **See detailed info** - genres, average ratings, popularity

---

## Why This Approach?

### The Cold-Start Problem

**Challenge**: How do you recommend movies to a user who has never rated anything?

**Traditional Collaborative Filtering** (e.g., "Users who liked X also liked Y") **fails** because:
- No user history exists
- No similar users to compare with
- Matrix factorization requires existing ratings

### Our Solution: Content-Based Filtering with LSH

**Why Content-Based?**
- Uses **movie features** (genome vectors) instead of user history
- Works immediately for new users
- Provides explainable recommendations

**Why LSH (Locality Sensitive Hashing)?**
- **Fast**: O(log n) similarity search vs O(n) brute force
- **Scalable**: Handles 62,000 movies efficiently
- **Accurate**: Approximate nearest neighbors with high precision

**Why Genome Vectors?**
- **Rich features**: 1,128 dimensions capturing movie characteristics
- **Semantic**: Tags like "action", "romance", "twist ending"
- **Proven**: MovieLens genome data is research-grade

---

## How Does It Work?

### Step-by-Step Process

```
┌────────────────────────────────────────────────────────────┐
│  STEP 1: User Input                                        │
├────────────────────────────────────────────────────────────┤
│  User selects and rates 5 movies:                          │
│  • Toy Story (1995) → 5.0 stars                            │
│  • Jurassic Park (1993) → 4.5 stars                        │
│  • The Matrix (1999) → 5.0 stars                           │
│  • Forrest Gump (1994) → 4.0 stars                         │
│  • Inception (2010) → 5.0 stars                            │
└────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: Create User Profile                                │
├─────────────────────────────────────────────────────────────┤
│  Weighted average of genome vectors:                        │
│                                                             │
│  user_profile = Σ(rating_i × genome_vector_i) / Σ(rating_i) │
│                                                             │
│  Result: 1,128-dimensional vector representing user taste   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────┐
│  STEP 3: LSH Similarity Search                             │
├────────────────────────────────────────────────────────────┤
│  • Load pre-trained LSH model from disk                    │
│  • Find approximate nearest neighbors                      │
│  • Use BucketedRandomProjectionLSH (PySpark MLlib)         │
│  • Search space: 62,000 movies                             │
│  • Time complexity: O(log n) instead of O(n)               │
└────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: Rank and Filter                                    │
├─────────────────────────────────────────────────────────────┤
│  • Calculate cosine similarity scores                       │
│  • Remove already-rated movies                              │
│  • Sort by similarity (highest first)                       │
│  • Select top 5 recommendations                             │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 5: Display Results                                    │
├─────────────────────────────────────────────────────────────┤
│  Show for each recommendation:                              │
│  • Movie title and year                                     │
│  • Genres                                                   │
│  • Match score (e.g., 94% match)                            │
│  • Average rating (e.g., 4.2/5.0)                           │
│  • Popularity (number of ratings)                           │
└─────────────────────────────────────────────────────────────┘
```

### Algorithm Details

**Cosine Similarity Formula**:
```
similarity(A, B) = (A · B) / (||A|| × ||B||)
```

**Match Score Calculation**:
```
match_percentage = (1 - distance) × 100%
```

Where `distance` is the Euclidean distance in LSH space.

---

## How to Use It?

### Quick Start (2 Steps)

1. **Launch the app**:
   ```bash
   cd app
   ./run_app.sh
   ```

2. **Open browser**: Navigate to `http://localhost:8501`

### Using the Interface

**Step 1: Select Movies**
- Browse the list of popular movies (sorted by number of ratings)
- Use the search box to find specific titles
- Select exactly 5 movies

**Step 2: Rate Your Selections**
- Rate each movie from 0.5 to 5.0 stars
- Higher ratings = stronger preference signal
- Be honest - this affects recommendation quality!

**Step 3: Get Recommendations**
- Click "Get Recommendations"
- Wait 2-3 seconds for processing
- View your personalized top 5 picks

**Step 4: Explore Results**
- See match scores (e.g., "94% match")
- Check genres to understand why it was recommended
- View average ratings and popularity metrics

---

## Installation

### Prerequisites

- **Python 3.9+**
- **Java 17** (required for PySpark 4.x)
- **8GB RAM** minimum

### Step-by-Step Setup

**1. Install Java 17**

```bash
# macOS (Homebrew)
brew install openjdk@17
export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
export PATH="$JAVA_HOME/bin:$PATH"

# Verify installation
java -version  # Should show: openjdk version "17.x.x"
```

**2. Create Virtual Environment**

```bash
cd app
python3 -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

**3. Install Dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the App**

```bash
# Option A: Using the launch script (recommended)
./run_app.sh

# Option B: Manual launch
export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
streamlit run movie_recommender_app.py
```

**5. Access the App**

Open your browser and navigate to: `http://localhost:8501`

---

## Technical Details

### Architecture

**Frontend**: Streamlit (Python web framework)  
**Backend**: PySpark 4.0.1 (distributed ML)  
**ML Algorithm**: BucketedRandomProjectionLSH  
**Data Format**: Parquet (columnar storage)  
**Caching**: Streamlit @st.cache_resource and @st.cache_data  

### Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **Startup Time** | 5-8 seconds | First load (initializes Spark) |
| **Recommendation Time** | 2-3 seconds | With pre-trained model |
| **Recommendation Time** | 10-15 seconds | Without pre-trained model (fallback) |
| **Memory Usage** | ~2GB | Spark driver memory |
| **Dataset Size** | 62,423 movies | Full MovieLens 25M catalog |

### Pre-trained Models

The app uses pre-trained models for fast inference:

**Models Loaded**:
- `models/lsh_model/` - LSH model for similarity search
- `data/items_features.parquet` - Movie features (genome + ML)
- `data/movie_avg.parquet` - Average ratings per movie
- `data/genome_vector.parquet` - 1,128-dim genome vectors

**Benefits**:
- 5x faster loading (2-3s vs 10-15s)
- Consistent recommendations across sessions
- Production-ready approach

**Fallback**: If models not found, trains on-the-fly (slower but functional)

### Dependencies

```
streamlit>=1.28.0    # Web framework
pandas>=2.0.0        # Data manipulation
numpy>=1.24.0        # Numerical computing
pyarrow>=13.0.0      # Parquet I/O
pyspark>=3.4.0       # Distributed ML
```

---

## Troubleshooting

### Common Issues

**1. "Java gateway process exited before sending its port number"**

**Cause**: Java not installed or wrong version

**Solution**:
```bash
# Install Java 17
brew install openjdk@17

# Set JAVA_HOME
export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home

# Verify
java -version  # Must show 17.x.x
```

**2. "Pre-trained model not found"**

**Cause**: Models not generated from Jupyter notebook

**Solution**: Run `code/02-model_training.ipynb` to generate models, or use fallback (slower)

**3. "Port 8501 already in use"**

**Solution**:
```bash
# Kill existing Streamlit process
pkill -f streamlit

# Or use different port
streamlit run movie_recommender_app.py --server.port 8502
```

**4. Slow recommendations (>10 seconds)**

**Cause**: Using fallback training instead of pre-trained models

**Solution**: Generate pre-trained models by running the Jupyter notebook

---

## Learn More

- **Project README**: `../README.md` - Full project documentation
- **Data Preparation**: `../code/01-data-preparation.ipynb`
- **Model Training**: `../code/02-model_training.ipynb`
- **Models Info**: `../models/README.md`

---

**Enjoy discovering your next favorite movie!**

