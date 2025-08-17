#!/usr/bin/env python3
"""
Simple test script for the fraud detection model
Run this to verify everything is working correctly
"""

def test_imports():
    """Test that all required packages can be imported."""
    print("🧪 Testing package imports...")
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
        
        import pandas as pd
        print("✅ Pandas imported successfully")
        
        import matplotlib.pyplot as plt
        print("✅ Matplotlib imported successfully")
        
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
        print("✅ Scikit-learn imported successfully")
        
        from pyod.models.auto_encoder import AutoEncoder
        print("✅ PyOD AutoEncoder imported successfully")
        
        import joblib
        print("✅ Joblib imported successfully")
        
        print("\n🎉 All packages imported successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please run: pip install -r requirements_simple.txt")
        return False

def test_data_file():
    """Test if the credit card data file exists."""
    print("\n📁 Testing data file...")
    
    import os
    
    if os.path.exists('creditcard.csv'):
        print("✅ creditcard.csv found")
        
        # Try to read the file
        try:
            import pandas as pd
            data = pd.read_csv('creditcard.csv')
            print(f"✅ File read successfully: {data.shape}")
            print(f"   Features: {data.shape[1]}")
            print(f"   Transactions: {data.shape[0]:,}")
            
            # Check for required columns
            if 'Class' in data.columns:
                fraud_count = data['Class'].sum()
                print(f"   Fraudulent transactions: {fraud_count}")
                print(f"   Fraud rate: {fraud_count/len(data)*100:.3f}%")
                return True
            else:
                print("❌ 'Class' column not found in data")
                return False
                
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return False
    else:
        print("❌ creditcard.csv not found")
        print("Please download it from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        return False

def test_autoencoder_creation():
    """Test if we can create an AutoEncoder instance."""
    print("\n🧠 Testing AutoEncoder creation...")
    
    try:
        from pyod.models.auto_encoder import AutoEncoder
        
        # Create a simple AutoEncoder
        autoencoder = AutoEncoder(random_state=42)
        print("✅ AutoEncoder created successfully")
        
        # Check if it has basic methods
        if hasattr(autoencoder, 'fit'):
            print("✅ AutoEncoder has 'fit' method")
        else:
            print("❌ AutoEncoder missing 'fit' method")
            return False
            
        if hasattr(autoencoder, 'decision_function'):
            print("✅ AutoEncoder has 'decision_function' method")
        else:
            print("❌ AutoEncoder missing 'decision_function' method")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error creating AutoEncoder: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 SIMPLE FRAUD DETECTION - SYSTEM TEST")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Package imports
    if test_imports():
        tests_passed += 1
        print("✅ Import test PASSED")
    else:
        print("❌ Import test FAILED")
    
    # Test 2: Data file
    if test_data_file():
        tests_passed += 1
        print("✅ Data file test PASSED")
    else:
        print("❌ Data file test FAILED")
    
    # Test 3: AutoEncoder creation
    if test_autoencoder_creation():
        tests_passed += 1
        print("✅ AutoEncoder test PASSED")
    else:
        print("❌ AutoEncoder test FAILED")
    
    # Summary
    print("\n" + "=" * 50)
    print("TESTING SUMMARY")
    print("=" * 50)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("\n🎉 ALL TESTS PASSED!")
        print("Your system is ready to run the fraud detection model!")
        print("\nNext step: python simple_fraud_detection.py")
    else:
        print("\n⚠️ Some tests failed. Please fix the issues above.")
        print("\nCommon solutions:")
        print("1. Install packages: pip install -r requirements_simple.txt")
        print("2. Download data: Get creditcard.csv from Kaggle")
        print("3. Check Python version: Use Python 3.7+")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main() 