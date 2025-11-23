# How to Run the Notebooks

### **Prerequisites**

1. **Python 3.9+**
2. **Java 17** (for PySpark)
3. **Jupyter Notebook or JupyterLab**

### **Setup**

```bash
# 1. Install Jupyter
pip install jupyter notebook

# 2. Install dependencies
pip install pyspark pandas numpy

# 3. Set Java environment (if not already set)
export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
export PATH="$JAVA_HOME/bin:$PATH"
```

### **Running the Notebooks**

```bash
# Navigate to the code directory
cd code

# Start Jupyter Notebook
jupyter notebook

# Or use JupyterLab
jupyter lab
```

Then open and run:
1. `01-data-preparation.ipynb` (first time only, or when data changes)
2. `02-model_training.ipynb` (to train models)

---