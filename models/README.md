# 🤖 Models Directory

This directory contains trained machine learning models for the movie recommender system.

---

## 🚨 Important: Models Not Included in Git

**Trained models (>25MB) are not included in this repository** to keep it lightweight.

---

## 📥 How to Generate Models

Run the model training notebook:

```bash
cd code
jupyter notebook 02-model_training.ipynb
```

This will create:
- `models/lsh_model/` (~52KB)
- `models/als_best_model/` (~26MB)

---

## 📁 Model Files

| Model | Size | Purpose | Used By |
|-------|------|---------|---------|
| **`lsh_model/`** | ~52KB | LSH for cold-start recommendations | Streamlit app |
| **`als_best_model/`** | ~26MB | Collaborative filtering (ALS) | Evaluation, future use |

### **LSH Model (Locality Sensitive Hashing)**

- **Purpose**: Fast similarity search for cold-start users
- **Algorithm**: BucketedRandomProjectionLSH
- **Input**: Genome vectors (1,128 dimensions)
- **Output**: Similar movies based on content features
- **Used in**: Streamlit app for new user recommendations

### **ALS Model (Alternating Least Squares)**

- **Purpose**: Collaborative filtering recommendations
- **Algorithm**: PySpark ALS
- **Features**: User-item matrix factorization
- **Hyperparameters**: Tuned via cross-validation
- **Used in**: Hybrid recommendations (notebook)

---

## 🚀 Usage

### **In Streamlit App**

The app automatically loads pre-trained models:

```python
# Load LSH model
lsh_model = BucketedRandomProjectionLSHModel.load("../models/lsh_model")

# Load item features
items_features = spark.read.parquet("../data/items_features.parquet")
```

### **Benefits of Pre-trained Models**

- ✅ **5x faster loading** (2-3 seconds vs 10-15 seconds)
- ✅ **Consistent recommendations** across sessions
- ✅ **Production-ready** approach
- ✅ **No training overhead** on app startup

---

## 🔄 Fallback Behavior

If pre-trained models are not found, the Streamlit app will:

1. ⚠️ Show a warning message
2. 🔧 Train the LSH model on-the-fly (~10 seconds)
3. ✅ Use the freshly trained model for the session
4. 💡 Recommend running the notebook to save the model

---

## 📊 Model Details

### **LSH Model Structure**

```
lsh_model/
├── metadata/
│   ├── _SUCCESS
│   └── part-00000
└── data/
    ├── _SUCCESS
    └── part-00000-*.parquet
```

### **ALS Model Structure**

```
als_best_model/
├── metadata/
│   ├── _SUCCESS
│   └── part-00000
├── itemFactors/
│   ├── _SUCCESS
│   └── part-*.parquet
└── userFactors/
    ├── _SUCCESS
    └── part-*.parquet
```

---

## 🗂️ Directory Structure

```
models/
├── README.md              # This file
├── lsh_model/             # ❌ Generate via notebook (~52KB)
│   ├── metadata/
│   └── data/
└── als_best_model/        # ❌ Generate via notebook (~26MB)
    ├── metadata/
    ├── itemFactors/
    └── userFactors/
```

---

## ⚠️ Troubleshooting

### **Missing models error**

If you see "Model not found" errors:

```bash
cd code
jupyter notebook 02-model_training.ipynb
```

Run all cells to generate and save the models.

### **Model loading errors**

Ensure you have:
- ✅ PySpark installed
- ✅ Java 17 configured
- ✅ Correct file paths (relative to app directory)

---

**Need help?** See the main [README.md](../README.md) for more information.

