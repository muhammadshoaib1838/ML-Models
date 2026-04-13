# 🐾 Animal Classifier AI

A production-ready **Streamlit** web application that classifies animal images into **90 categories** using a hybrid architecture:

- **Feature extraction**: Pretrained **MobileNetV2** (ImageNet weights, Global Average Pooling → 1 280-dim vector)
- **Classifier**: Classical ML model trained with `scikit-learn` (e.g. SVM / Random Forest / Gradient Boosting)

---

## 📁 Project Structure

```
animal_classifier/
├── app.py                  # Main Streamlit application
├── best_model.pkl          # Trained ML classifier  ← YOU provide this
├── label_encoder.pkl       # LabelEncoder           ← YOU provide this
├── requirements.txt        # Python dependencies
├── packages.txt            # System-level packages (Streamlit Cloud)
├── .streamlit/
│   └── config.toml         # Theme & server settings
└── README.md               # This file
```

> **Note:** `best_model.pkl` and `label_encoder.pkl` are **not** included in this repo.  
> Copy them from your training environment before running or deploying.

---

## ⚡ Quick Start (Local)

### 1 – Clone the repo

```bash
git clone https://github.com/<your-username>/animal-classifier.git
cd animal-classifier
```

### 2 – Copy your model files

```bash
cp /path/to/best_model.pkl   .
cp /path/to/label_encoder.pkl .
```

### 3 – Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 4 – Install dependencies

```bash
pip install -r requirements.txt
```

### 5 – Run the app

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 🧠 How It Works

```
User uploads image
       │
       ▼
PIL Image  →  RGB  →  resize 224×224
       │
       ▼
MobileNetV2 (frozen, ImageNet weights)
  → Global Average Pooling
  → Feature vector [1 × 1 280]
       │
       ▼
Classical ML model  (best_model.pkl)
  → predict()          → class index
  → predict_proba()    → top-5 probabilities (if supported)
       │
       ▼
LabelEncoder.inverse_transform()
  → Human-readable class name
```

---

## 🚀 Deploy on Streamlit Cloud

### Step 1 – Push to GitHub

```bash
# Initialise (if not already a git repo)
git init
git add .
git commit -m "Initial commit: Animal Classifier AI"

# Create a new repo on GitHub (via web UI or gh CLI)
git remote add origin https://github.com/<your-username>/animal-classifier.git
git branch -M main
git push -u origin main
```

> ⚠️ **Large files warning**: `best_model.pkl` and `label_encoder.pkl` may exceed GitHub's 100 MB limit.  
> If they do, use **Git LFS**:
> ```bash
> git lfs install
> git lfs track "*.pkl"
> git add .gitattributes
> git add *.pkl
> git commit -m "Add model files via LFS"
> git push
> ```

---

### Step 2 – Deploy on Streamlit Cloud

1. Go to **https://share.streamlit.io** and sign in with your GitHub account.
2. Click **"New app"**.
3. Fill in the form:

   | Field | Value |
   |-------|-------|
   | Repository | `<your-username>/animal-classifier` |
   | Branch | `main` |
   | Main file path | `app.py` |

4. Click **"Deploy!"**

Streamlit Cloud will automatically:
- Install packages from `packages.txt` (system) and `requirements.txt` (Python)
- Cache the MobileNetV2 model weights on first boot
- Serve the app on a public URL like `https://<your-username>-animal-classifier-app-xxxx.streamlit.app`

---

## 🔑 Secrets / Environment Variables (optional)

If your app needs API keys or other secrets, add them via the Streamlit Cloud dashboard:

**App → Settings → Secrets**

```toml
# .streamlit/secrets.toml  (local only, never commit this file!)
SOME_API_KEY = "your-secret-key"
```

Access in Python:

```python
import streamlit as st
key = st.secrets["SOME_API_KEY"]
```

---

## 🗂️ 90 Animal Classes

The model was trained on the **Animals-90** dataset which includes:

`antelope`, `badger`, `bat`, `bear`, `bee`, `beetle`, `bison`, `boar`, `butterfly`,
`cat`, `caterpillar`, `chimpanzee`, `cockroach`, `cow`, `coyote`, `crab`, `crow`,
`deer`, `dog`, `dolphin`, `donkey`, `dragonfly`, `duck`, `eagle`, `elephant`,
`flamingo`, `fly`, `fox`, `goat`, `goldfish`, `goose`, `gorilla`, `grasshopper`,
`hamster`, `hare`, `hedgehog`, `hippopotamus`, `hornbill`, `horse`, `hummingbird`,
`hyena`, `jellyfish`, `kangaroo`, `koala`, `ladybugs`, `leopard`, `lion`,
`lizard`, `lobster`, `lynx`, `magpie`, `monkey`, `moose`, `moth`, `mouse`,
`octopus`, `okapi`, `orangutan`, `otter`, `owl`, `ox`, `oyster`, `panda`,
`parrot`, `pelecaniformes`, `penguin`, `pig`, `pigeon`, `porcupine`, `possum`,
`raccoon`, `rat`, `reindeer`, `rhinoceros`, `sandpiper`, `seahorse`, `seal`,
`shark`, `sheep`, `snake`, `sparrow`, `squid`, `squirrel`, `starfish`,
`swan`, `tiger`, `turkey`, `turtle`, `whale`, `wolf`, `wombat`, `woodpecker`, `zebra`

---

## 🛠️ Troubleshooting

| Issue | Fix |
|-------|-----|
| `FileNotFoundError: best_model.pkl` | Copy `best_model.pkl` to the same folder as `app.py` |
| `FileNotFoundError: label_encoder.pkl` | Same as above for `label_encoder.pkl` |
| TensorFlow not installing | Use Python 3.10 or 3.11; avoid 3.12 |
| Apple Silicon errors | Replace `tensorflow` with `tensorflow-macos` + `tensorflow-metal` in `requirements.txt` |
| Streamlit Cloud build fails | Check `packages.txt` contains `libgl1` |
| Model predicts wrong class | Ensure the same preprocessing pipeline was used during training |
| Git push rejected (file too large) | Use **Git LFS** for `.pkl` files > 100 MB |

---

## 📄 License

MIT License – feel free to use, modify, and distribute.

---

*Built with ❤️ using Streamlit · TensorFlow · scikit-learn*
