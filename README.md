# 📌 AG News Classification with Deep Learning

## 📖 Overview
This project implements **text classification** on the **AG News dataset** using two deep learning architectures:
- 🧠 **CNN (Convolutional Neural Network)** for efficient feature extraction.
- ⚡ **HAN (Hierarchical Attention Network)** for improved context understanding.

The models are trained and evaluated based on **accuracy** and **inference time**, and deployed using **Streamlit** for real-time classification.

---

## 📊 Dataset Description
The **AG News Dataset** consists of news articles categorized into four classes:
| Category | Description |
|----------|-------------|
| 🌍 **World** | International News |
| ⚽ **Sports** | Sports News & Updates |
| 💰 **Business** | Financial & Economic News |
| 🔬 **Science/Technology** | Tech & Science News |

---

## 📂 Project Structure
```
📦 ag-news-classification
├── 📁 models
│   ├── 📝 cnn_ag_news.h5
│   ├── 📝 han_ag_news.h5
├── 📝 tokenizer.pkl
├── 📜 app.py  # Streamlit Application
├── 📜 train.py  # Model Training Script
├── 📄 requirements.txt
├── 📖 README.md
```

---

## ⚙️ Installation & Setup
### 1️⃣ Clone the repository:
```bash
git clone https://github.com/yourusername/ag-news-classification.git
cd ag-news-classification
```

### 2️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit app:
```bash
streamlit run app.py
```

---

## 🎯 Model Training
To train the **CNN** and **HAN** models, run:
```bash
python train.py
```
📌 The trained models will be saved in the `models/` directory.

---

## 📈 Performance Comparison
| 🏆 Model | 🎯 Accuracy | ⏱ Inference Time (sec) |
|---------|----------|------------------|
| 🚀 **CNN** | ~85% | ~0.02 sec |
| 🏅 **HAN** | ~88% | ~0.08 sec |

📌 **CNN** is faster for real-time applications, while **HAN** provides higher accuracy.

---

## 🌍 Deployment
The **Streamlit app** allows real-time classification of news articles. After running `streamlit run app.py`, open your browser at:
```
🔗 http://localhost:8501/
```

---

## 🚀 Future Improvements
✅ Implement **Transformer-based models** (e.g., BERT) for better accuracy.
✅ Optimize **inference speed** for real-time applications.
✅ Extend the project to handle **multi-label classification**.

---

## ✨ Author
👤 **Your Name** - [GitHub Profile](https://github.com/sharukg)

