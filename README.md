# 🏨 MASC-Hotel: Multimodal Aspect Sentiment Classification

This repository implements a **MASC** (Multimodal Aspect Sentiment Classification) model using **BART-base** enhanced with **Visible Matrix (Graph Matrix)** and **Visual Fusion**. 

The goal is to classify the sentiment (Positive, Neutral, Negative) of a *pre-defined* aspect within a hotel review, leveraging both textual context and image-based knowledge triples.

## 🛠️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/quanglap100905/SeqCSG-MASC.git
cd SeqCSG-MASC

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download SpaCy language model
python -m spacy download en_core_web_sm

# 4. Run
python prepare_data.py
python train.py
