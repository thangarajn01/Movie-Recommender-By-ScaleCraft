#!/usr/bin/env python3
"""
Test script to verify pre-trained model loading

This script tests:
1. Whether pre-trained models exist
2. Whether they can be loaded successfully
3. Performance comparison between pre-trained and on-the-fly training
"""

import os
import sys
import time

def check_file_exists(path, description):
    """Check if a file or directory exists"""
    exists = os.path.exists(path)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {path}")
    return exists

def test_model_files():
    """Test if all required model files exist"""
    print("\n" + "="*60)
    print("TESTING MODEL FILES")
    print("="*60)
    
    files_to_check = [
        ("../models/lsh_model", "LSH Model Directory"),
        ("../data/items_features.parquet", "Items Features"),
        ("../data/movie_avg.parquet", "Movie Averages"),
        ("../data/genome_vector.parquet", "Genome Vectors"),
        ("../data/movie_ratings.parquet", "Movie Ratings"),
    ]
    
    all_exist = True
    for path, desc in files_to_check:
        if not check_file_exists(path, desc):
            all_exist = False
    
    return all_exist

def test_pyspark_import():
    """Test if PySpark is installed and can be imported"""
    print("\n" + "="*60)
    print("TESTING PYSPARK")
    print("="*60)
    
    try:
        from pyspark.sql import SparkSession
        from pyspark.ml.feature import BucketedRandomProjectionLSHModel
        print("✅ PySpark imported successfully")
        return True
    except ImportError as e:
        print(f"❌ PySpark import failed: {e}")
        print("💡 Install with: pip install pyspark>=3.4.0")
        return False

def test_load_pretrained_model():
    """Test loading pre-trained LSH model"""
    print("\n" + "="*60)
    print("TESTING PRE-TRAINED MODEL LOADING")
    print("="*60)
    
    try:
        from pyspark.sql import SparkSession
        from pyspark.ml.feature import BucketedRandomProjectionLSHModel
        
        # Initialize Spark
        print("Initializing Spark session...")
        spark = SparkSession.builder \
            .appName("ModelLoadingTest") \
            .config("spark.driver.memory", "2g") \
            .getOrCreate()
        
        spark.sparkContext.setLogLevel("ERROR")
        
        # Test loading pre-trained model
        print("\nAttempting to load pre-trained LSH model...")
        start_time = time.time()
        
        lsh_model = BucketedRandomProjectionLSHModel.load("../models/lsh_model")
        items_features = spark.read.parquet("../data/items_features.parquet")
        movie_avg_df = spark.read.parquet("../data/movie_avg.parquet")
        
        load_time = time.time() - start_time
        
        print(f"✅ Pre-trained model loaded successfully!")
        print(f"⏱️  Load time: {load_time:.2f} seconds")
        print(f"📊 Items features count: {items_features.count():,}")
        print(f"📊 Movie averages count: {movie_avg_df.count():,}")
        
        spark.stop()
        return True, load_time
        
    except Exception as e:
        print(f"❌ Failed to load pre-trained model: {e}")
        return False, None

def test_train_on_the_fly():
    """Test training LSH model on-the-fly (fallback behavior)"""
    print("\n" + "="*60)
    print("TESTING ON-THE-FLY TRAINING (FALLBACK)")
    print("="*60)
    
    try:
        from pyspark.sql import SparkSession
        from pyspark.ml.feature import BucketedRandomProjectionLSH
        from pyspark.ml.linalg import Vectors, VectorUDT
        from pyspark.sql.functions import udf
        from pyspark.sql import functions as F
        
        # Initialize Spark
        print("Initializing Spark session...")
        spark = SparkSession.builder \
            .appName("OnTheFlyTrainingTest") \
            .config("spark.driver.memory", "2g") \
            .getOrCreate()
        
        spark.sparkContext.setLogLevel("ERROR")
        
        # Test training on-the-fly
        print("\nTraining LSH model on-the-fly...")
        start_time = time.time()
        
        # Load genome vectors
        genome_df = spark.read.parquet("../data/genome_vector.parquet")
        
        # Convert to ML vectors
        to_vec = udf(lambda xs: Vectors.dense(xs) if xs else Vectors.dense([0.0]*1128), VectorUDT())
        items_features = genome_df.withColumn("features", to_vec("genome_vector"))
        
        # Train LSH model
        brp = BucketedRandomProjectionLSH(
            inputCol="features",
            outputCol="hashes",
            bucketLength=2.0,
            numHashTables=3
        )
        lsh_model = brp.fit(items_features)
        
        # Load ratings and compute averages
        ratings_df = spark.read.parquet("../data/movie_ratings.parquet")
        movie_avg_df = ratings_df.groupBy("movieId") \
            .agg(F.avg("rating").alias("movie_avg"), 
                 F.count("rating").alias("num_ratings"))
        
        train_time = time.time() - start_time
        
        print(f"✅ On-the-fly training completed!")
        print(f"⏱️  Training time: {train_time:.2f} seconds")
        print(f"📊 Items features count: {items_features.count():,}")
        print(f"📊 Movie averages count: {movie_avg_df.count():,}")
        
        spark.stop()
        return True, train_time
        
    except Exception as e:
        print(f"❌ Failed to train on-the-fly: {e}")
        return False, None

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("PRE-TRAINED MODEL LOADING TEST")
    print("="*60)
    
    # Test 1: Check files
    files_exist = test_model_files()
    
    # Test 2: Check PySpark
    pyspark_ok = test_pyspark_import()
    
    if not pyspark_ok:
        print("\n❌ Cannot proceed without PySpark")
        sys.exit(1)
    
    # Test 3: Load pre-trained model
    pretrained_ok, pretrained_time = test_load_pretrained_model()
    
    # Test 4: Train on-the-fly (for comparison)
    onthefly_ok, onthefly_time = test_train_on_the_fly()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if pretrained_ok and onthefly_ok:
        speedup = onthefly_time / pretrained_time
        print(f"\n✅ Both methods work!")
        print(f"\n📊 Performance Comparison:")
        print(f"   Pre-trained model: {pretrained_time:.2f} seconds")
        print(f"   On-the-fly training: {onthefly_time:.2f} seconds")
        print(f"   Speedup: {speedup:.1f}x faster with pre-trained model! 🚀")
        
        if speedup >= 3:
            print(f"\n🎉 Excellent! Pre-trained model is {speedup:.1f}x faster!")
        elif speedup >= 2:
            print(f"\n✅ Good! Pre-trained model is {speedup:.1f}x faster!")
        else:
            print(f"\n⚠️  Pre-trained model is only {speedup:.1f}x faster (expected 3-5x)")
    
    elif pretrained_ok:
        print("\n✅ Pre-trained model works!")
        print("⚠️  On-the-fly training failed (but not critical)")
    
    elif onthefly_ok:
        print("\n⚠️  Pre-trained model not available")
        print("✅ On-the-fly training works (fallback is functional)")
        print("\n💡 Recommendation: Run the Jupyter notebook to save the model!")
    
    else:
        print("\n❌ Both methods failed!")
        print("🔧 Please check your setup and data files")
        sys.exit(1)

if __name__ == "__main__":
    main()

