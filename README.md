# DeepLOB-lite

An implementation of the **DeepLOB** architecture (CNN + Inception + LSTM) for predicting short-term price movements from **limit order book (LOB) data**.

This project reproduces and simplifies ideas from *DeepLOB: Deep Convolutional Neural Networks for Limit Order Books* (Zhang et al., 2018), applied to the **FI-2010 benchmark dataset**.

---

## 🔑 Features
- **Data preprocessing**: sliding windows of 100 LOB snapshots × 40 features.
- **Model architecture**:
  - Convolutional blocks to capture spatial structure of LOB levels.
  - Inception module for multi-scale feature extraction.
  - LSTM to model temporal dependencies.
- **Training**:
  - Early stopping, learning rate scheduling, and class weighting.
  - Achieved ~73% test accuracy and balanced F1-scores across classes (Down/Stationary/Up).
- **Evaluation**: classification report, confusion matrix, and performance plots.
- **Trading backtest**: simple simulation (buy on Up, sell on Down, hold otherwise) showing improved stability compared to naïve buy & hold.

---

## 📊 Results
- **Test accuracy**: ~73%
- **F1-scores**: ~0.72–0.74 across all three classes.
- **Trading simulation**: strategy outperformed market buy & hold baseline.

---

## 📂 Repository Structure
```
DeepLOB-lite/
│
├── notebooks/                # Jupyter notebooks
│   ├── deeplob-lite.ipynb     # main cleaned implementation
│   └── DeepLOB.ipynb          # reference notebook
│
├── data/                     # datasets (add to .gitignore if too large)
│   ├── FI2010_train.csv
│   └── FI2010_test.csv
│
├── models/                   # trained models
│   └── deeplob_fi2010.h5
│
├── results/                  # outputs, figures, reports
│   ├── confusion_matrix.png
│   ├── trading_simulation.png
│   └── classification_report.txt
│
├── src/                      # Python scripts (optional modularization)
│   ├── preprocess.py
│   ├── model.py
│   └── backtest.py
│
├── README.md                 # project description
├── requirements.txt          # dependencies
└── .gitignore                # ignore large data/models
```

---

## 🚀 Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/DeepLOB-lite.git
   cd DeepLOB-lite
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download FI-2010 dataset:
   - Place `FI2010_train.csv` and `FI2010_test.csv` in the `data/` directory.
   - Dataset source: [Ntakaris et al., FI-2010 Benchmark](https://arxiv.org/abs/1705.03233)

4. Run the notebook:
   - Open `notebooks/deeplob-lite.ipynb`
   - Train the model, evaluate results, and run the trading simulation.

---

## 📈 Example Outputs

**Confusion Matrix**
![confusion_matrix](results/confusion_matrix.png)

**Trading Simulation**
![trading_simulation](results/trading_simulation.png)

---

## ⚙️ Requirements
- Python 3.8+
- TensorFlow 2.x
- NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn

Install via:
```bash
pip install -r requirements.txt
```

---

## 📖 Reference
- Z. Zhang, S. Zohren, and S. Roberts, *DeepLOB: Deep Convolutional Neural Networks for Limit Order Books*, arXiv:1808.03668, 2018.

---

## 📜 License
This project is released under the MIT License.
