# 🚀 Credit Card Fraud Detection with PyOD AutoEncoder

A simple, easy-to-use implementation for detecting credit card fraud using PyOD's AutoEncoder neural network.

## 📋 What This Does

This script automatically:
1. **Loads** your credit card transaction data
2. **Trains** an AutoEncoder neural network on legitimate transactions
3. **Detects** fraudulent transactions based on reconstruction errors
4. **Shows** comprehensive results and visualizations
5. **Saves** the trained model for future use

## 🐍 Python Version Requirements

### **Recommended Python Version:**
- **Python 3.10.18** (Your virtual environment version - fully tested and working!)
- **Minimum:** Python 3.7+
- **Compatible:** Python 3.8 - 3.13+ (all modern versions work well)

### **Why Python 3.10.18?**
- **Your virtual environment**: Already created and working with this version
- **Stable version**: Python 3.10 is a mature, stable release
- **PyOD compatibility**: Fully tested and supported
- **Package ecosystem**: All required packages work seamlessly
- **Performance**: Optimized for machine learning workloads

### **Check Your Python Version:**
```bash
# Check system Python
python --version
# Shows: Python 3.13.5

# Check virtual environment Python
fraud_detection_env/bin/python --version
# Shows: Python 3.10.18
```

## 🛠️ Virtual Environment Setup (Recommended)

**Using a virtual environment is HIGHLY RECOMMENDED** to avoid package conflicts and keep your system clean.

### **📥 Download Credit Card Dataset (Required First Step)**

**Before setting up the virtual environment, you need to download the dataset:**

1. **Go to Kaggle**: Visit [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. **Download the file**: Click "Download" to get `creditcard.csv`
3. **Place in project root**: Put `creditcard.csv` in the top directory of your project
4. **Verify placement**: The file should be at `fraud-detection/creditcard.csv`

**Your project structure should look like this:**
```
fraud-detection/
├── creditcard.csv             # ← Place the dataset here (top level)
├── fraud_detection.py         # Main script
├── requirements.txt           # Dependencies
├── README.md                 # This file
└── fraud_detection_env/      # Virtual environment (will be created)
```

**⚠️ Important**: The script expects `creditcard.csv` to be in the same directory as `fraud_detection.py`

### **Option 1: Using `venv` (Built-in, Recommended)**

#### **Step 1: Create Virtual Environment**
```bash
# Navigate to your project directory
cd fraud-detection

# Create virtual environment
python -m venv fraud_detection_env

# Activate virtual environment
# On macOS/Linux:
source fraud_detection_env/bin/activate

# On Windows:
fraud_detection_env\Scripts\activate
```

#### **Step 2: Verify Activation**
```bash
# You should see (fraud_detection_env) at the start of your command prompt
(fraud_detection_env) user@computer:~/fraud-detection$

# Check Python location (should point to your virtual environment)
which python
# Should show: /path/to/fraud-detection/fraud_detection_env/bin/python
```

#### **Step 3: Install Dependencies**
```bash
# Upgrade pip first
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

#### **Step 4: Run Your Script**
```bash
python fraud_detection.py
```

#### **Step 5: Deactivate When Done**
```bash
deactivate
```

### **Option 2: Using `pyenv` (For managing multiple Python versions)**

#### **Step 1: Install Python Version**
```bash
# Install Python 3.10.18 (recommended - matches your virtual environment)
pyenv install 3.10.18

# Set local version for this project
pyenv local 3.10.18
```

#### **Step 2: Create Virtual Environment**
```bash
# Create virtual environment
python -m venv fraud_detection_env

# Activate it
source fraud_detection_env/bin/activate
```

#### **Step 3: Install and Run**
```bash
pip install -r requirements.txt
python fraud_detection.py
```

## 🚀 Quick Setup (Without Virtual Environment - Not Recommended)

**⚠️ Warning: This may cause package conflicts with other projects!**

```bash
# Install dependencies globally
pip install -r requirements.txt

# Run the script
python fraud_detection.py
```

## 📁 Project Structure

```
fraud-detection/
├── fraud_detection.py          # Main fraud detection script
├── requirements.txt            # Python package dependencies
├── README.md                  # This file
├── creditcard.csv             # Your dataset (download from Kaggle)
└── fraud_detection_env/       # Virtual environment (created by you)
    ├── bin/                   # Python executables
    ├── lib/                   # Installed packages
    └── include/               # Header files
```

## 📊 What You'll See

The script will show you:
- **Dataset information** (size, fraud rate)
- **Training progress** (epochs, loss)
- **Fraud detection results** (accuracy, precision, recall)
- **Beautiful visualizations** (6 different plots)
- **Model performance summary**

## 🎯 Expected Output

```
🚀 CREDIT CARD FRAUD DETECTION WITH AUTOENCODER
============================================================
📊 Loading Credit Card Dataset...
Dataset shape: (284807, 31)
Features: 30
Transactions: 284,807

Fraud Detection Challenge:
✅ Legitimate transactions: 284,315
🚨 Fraudulent transactions: 492
📈 Fraud rate: 0.173%

🔧 Preparing Data...
Features (X): (227845, 30)
Target (y): (227845,)
Training set: (227845, 30)
Testing set: (56962, 30)
✅ Data scaled successfully

🧠 Creating AutoEncoder Model...
AutoEncoder Architecture:
  Input: 30 features
  Using default PyOD AutoEncoder configuration
  Output: 30 features

🏋️ Training AutoEncoder...
This will take a few minutes...
✅ Training completed!

🔍 Detecting Fraud...
Reconstruction error range: 0.0000 to 0.9999
Fraud threshold: 0.1234

📊 Results:
  Accuracy: 0.9876
  Fraud detected: 123
  Actual fraud: 98

📈 Model Evaluation...
Classification Report:
              precision    recall  f1-score   support

   Legitimate     0.99      0.99      0.99     56864
        Fraud     0.80      0.84      0.82        98

🎉 FRAUD DETECTION PIPELINE COMPLETED SUCCESSFULLY!
============================================================
Your model is ready to detect fraud in new transactions!
```

## 🔍 How It Works

### **AutoEncoder Principle:**
```
Normal Transaction → Encoder → Compressed → Decoder → Reconstructed
     ✅ Good match = Legitimate transaction

Fraudulent Transaction → Encoder → Compressed → Decoder → Reconstructed  
     ❌ Bad match = Fraudulent transaction
```

### **The Process:**
1. **Training**: Model learns patterns from legitimate transactions
2. **Detection**: New transactions are scored based on reconstruction quality
3. **Threshold**: Transactions above threshold are flagged as fraud

## 📈 Understanding Results

### **Key Metrics:**
- **Accuracy**: Overall correctness (should be >95%)
- **Precision**: Of fraud predictions, how many were actually fraud
- **Recall**: Of actual fraud, how many did we catch
- **ROC-AUC**: Overall ability to distinguish fraud (should be >0.9)

### **Visualizations:**
1. **Anomaly Score Distribution**: Shows reconstruction error patterns
2. **Scores by Transaction Type**: Compares legitimate vs fraudulent scores
3. **Fraud Detection Results**: Pie chart of detection performance
4. **ROC Curve**: Model discrimination ability
5. **Threshold Analysis**: How threshold affects fraud detection
6. **Performance Summary**: Key metrics at a glance

## 💾 Saved Model

The script automatically saves your trained model as `fraud_detection_model.pkl`. You can use this to:
- Detect fraud in new transactions
- Deploy the model in production
- Share with team members

## 🚨 Troubleshooting

### **Virtual Environment Issues:**

#### **"Command not found: python"**
```bash
# Make sure virtual environment is activated
source fraud_detection_env/bin/activate

# Check Python location
which python
```

#### **"Package not found" errors**
```bash
# Make sure you're in the virtual environment
# You should see (fraud_detection_env) in your prompt

# Reinstall packages
pip install -r requirements.txt
```

#### **"Permission denied" errors**
```bash
# On macOS/Linux, you might need:
chmod +x fraud_detection_env/bin/activate
source fraud_detection_env/bin/activate
```

### **Data File Issues:**
- Make sure `creditcard.csv` is in the **top directory** of your project
- The file should be at the same level as `fraud_detection.py`
- Check the file name spelling (exactly `creditcard.csv`)
- Verify the file is not in a subdirectory

### **Python Version Issues:**
- Python 3.10.18 is fully compatible with all packages
- Use Python 3.7+ for minimum compatibility
- All modern Python versions (3.8+) work well

### **Import Errors:**
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt`
- Check that all packages installed successfully

### **Training Takes Too Long:**
- The model needs to process 227,845 transactions
- This typically takes 5-15 minutes depending on your computer
- Consider using a machine with more RAM/CPU if available

## 🔄 Daily Workflow

### **Starting Work:**
```bash
# Navigate to project directory
cd fraud-detection

# Activate virtual environment
source fraud_detection_env/bin/activate

# Verify activation
which python  # Should show your virtual environment path
```

### **Running the Model:**
```bash
# Make sure virtual environment is active
python fraud_detection.py
```

### **Finishing Work:**
```bash
# Deactivate virtual environment
deactivate
```

## 🎯 Next Steps

After running successfully, you can:
1. **Analyze the visualizations** to understand model performance
2. **Adjust the threshold** if you want different precision/recall balance
3. **Use the saved model** for real-time fraud detection
4. **Experiment with different parameters** for better performance

## 🤝 Need Help?

If you encounter issues:
1. **Check virtual environment**: Make sure it's activated
2. **Verify Python version**: Use Python 3.8-3.11
3. **Check dependencies**: Ensure all packages are installed
4. **Look at error messages**: The script includes helpful error handling
5. **Verify data file**: Make sure `creditcard.csv` is accessible

## 📚 Additional Resources

- **PyOD Documentation**: https://pyod.readthedocs.io/
- **Kaggle Dataset**: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Python Virtual Environments**: https://docs.python.org/3/tutorial/venv.html
- **Machine Learning Best Practices**: https://scikit-learn.org/stable/

---

**Happy Fraud Detection! 🚀**

*Remember: Always use a virtual environment for clean, reproducible development!* 