
# Smart Resume Parser

Smart Resume Parser is a web-based application that extracts structured information such as Name, Email, Skills, Education, and more from resumes. It supports both rule-based and machine learning-based Named Entity Recognition (NER) using spaCy and a fine-tuned BERT model.

## 🔍 Features
- Upload resume PDF or paste plain text.
- Extracts entities like Name, Email, Skills, Degree, and more.
- Choose between spaCy (rule-based) or BERT (ML-based) NER.
- Download results in structured JSON format.

## ⚙️ Technologies Used
- Python
- Huggingface Transformers (BERT)
- spaCy NLP (EntityRuler)
- Streamlit (Web App)
- PyPDF2 (PDF Reading)
- scikit-learn (Evaluation Metrics)

## 🛠 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/smart-resume-parser.git
cd smart-resume-parser
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```

## 🧪 Example Output (JSON)
```json
[
  {"label": "Name", "text": "Rounak Laddha"},
  {"label": "Email", "text": "rladdha@hawk.iit.edu"},
  {"label": "Skills", "text": "Python, SQL, Machine Learning"}
]
```

## 📈 Accuracy Metrics
- **BERT F1 Score**: ~0.66 (test set)
- **spaCy**: Works reliably for predefined patterns

## 📂 Folder Structure
```
smart-resume-parser/
├── app.py                      # Streamlit app
├── resume_parser.ipynb        # Training notebook
├── requirements.txt           # Python packages
├── README.md                  # Project overview
├── Smart_Resume_Parser_Project_Summary.docx
├── ner_resume_model_final/    # Trained model files
└── label_map.json             # BIO label mapping
```

## 📌 Resume STAR Entry
- Built a resume entity extractor using BERT and spaCy NER models.
- Deployed via Streamlit with PDF upload and JSON download.
- Compared models via precision/recall/F1 and switched to spaCy for reliability.
- Achieved 0.66 F1 score and fully working front-end interface.

## 📤 Deployment
You can host this project locally or on platforms like Streamlit Cloud.

##Author
**Rounak Laddha** – Master's in Data Science
