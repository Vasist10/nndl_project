# 🧠 Hate Speech Detection System

<a href="https://www.python.org/" target="_blank">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python" />
</a>
<a href="https://flask.palletsprojects.com/" target="_blank">
  <img src="https://img.shields.io/badge/Backend-Flask-orange?style=for-the-badge&logo=flask" />
</a>
<img src="https://img.shields.io/badge/ML-BiLSTM-purple?style=for-the-badge" />

---

## 🔍 Overview

The **Hate Speech Detection System** is an AI-powered tool that identifies toxic and hate speech in text in real time. It uses a fine-tuned BERT model served through a Flask backend with a clean web UI.

- 🌐 **Web App** — enter any text and get an instant toxicity prediction with confidence score
- 🧩 **Chrome Extension** — automatically blurs toxic comments on YouTube & Instagram
- 🤖 **BiLSTM model** — trained on the HateXplain dataset for multi-class hate speech detection

> **Note:** The app is not yet deployed. Follow the local setup instructions below to run it.

---

## 🛠️ Local Setup

### Prerequisites
- Python 3.10+
- The model weights (`hate_model/model.pt` or `hate_model/model.safetensors`) — not included in this repo due to file size (~418 MB). Download them separately and place them in the `hate_model/` folder.

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/Vasist10/nndl_project.git
cd nndl_project

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Flask app
python app.py
```

Then open `http://localhost:5000` in your browser.

---

## 🧩 Chrome Extension Setup

1. Download and unzip the extension file.
2. Open Chrome and go to `chrome://extensions/`.
3. Enable **Developer Mode** (top-right toggle).
4. Click **Load Unpacked** and select the unzipped folder.
5. Visit YouTube or Instagram — toxic comments will be blurred automatically.
6. Click **Show** on any blurred comment to reveal it.

---

## 🖼️ Screenshots

### 🌐 Web App
![Web App Screenshot](images/web-ui.jpg)

### 🧩 Chrome Extension in Action
![Extension Screenshot](images/extension-demo.jpg)

---

## 🛠️ Tech Stack

| Layer            | Technology                               |
|------------------|------------------------------------------|
| **Frontend**     | HTML, CSS, JavaScript                    |
| **Backend**      | Flask (Python)                           |
| **ML Model**     | BiLSTM (`PyTorch`)                       |
| **Browser Ext.** | JavaScript, Manifest v3                  |
| **Dataset**      | HateXplain                               |

---

## 📖 Usage

### Web App
1. Enter any text or comment in the input box.
2. Click **"Check Toxicity"**.
3. See the prediction (Toxic / Not Toxic) and confidence score instantly.

### Extension
- Visit YouTube or Instagram.
- Toxic comments are blurred automatically.
- Click **Show** on any comment to unhide it.

---

## 📚 Acknowledgments
- 🤗 **Hugging Face** for the tokenizer and `transformers` library
- 🐍 **Flask** for serving the ML model as an API
- 📊 **HateXplain** dataset for model training

---

### ⭐ Star this repo if you find it useful!
