#!/bin/bash

# Movie Recommender App Launcher
echo "🎬 Starting Movie Recommendation Engine..."
echo ""

# Set Java 17 for PySpark 4.x
export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
export PATH="$JAVA_HOME/bin:$PATH"

# Activate virtual environment
source myenv/bin/activate

echo "Java version: $(java -version 2>&1 | head -n 1)"
echo "Python: $(which python)"
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null
then
    echo "❌ Streamlit is not installed."
    echo "Installing dependencies..."
    pip3 install -r requirements.txt
fi

# Check if data files exist
if [ ! -f "../data/movies.csv" ]; then
    echo "❌ Error: movies.csv not found in ../data/"
    exit 1
fi

if [ ! -f "../data/ratings.csv" ]; then
    echo "❌ Error: ratings.csv not found in ../data/"
    exit 1
fi

if [ ! -d "../data/genome_vector.parquet" ]; then
    echo "⚠️  Warning: genome_vector.parquet not found. App will use fallback recommendations."
fi

echo "✅ All checks passed!"
echo ""
echo "🚀 Launching Streamlit app..."
echo "📱 Open your browser at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the app
streamlit run movie_recommender_app.py

