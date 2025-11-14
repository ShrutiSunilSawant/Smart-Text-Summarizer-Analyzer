#  Smart Text Summarizer & Analyzer

An **AI-powered web app** built with **Streamlit** that can:

- Summarize long text using a transformer model  
- Analyze sentiment and basic readability  
- Extract named entities and key phrases  
- Translate the summary into multiple languages  
- Work with **raw text**, **PDF**, and **Word (.docx)** files  

This project uses modern NLP libraries such as **Hugging Face Transformers**, **spaCy**, **TextBlob**, and **deep-translator** to deliver a complete text-processing solution.

All through a clean, browser-based UI.

---

##  Features

### 🔹 1. Intelligent Text Summarization
- Uses Hugging Face `transformers` **BART (facebook/bart-large-cnn)** model.
- Splits long text into chunks and generates a combined summary.
- Configurable **maximum** and **minimum** summary length from the sidebar.

### 🔹 2. Text Analysis
- Uses **spaCy** (`en_core_web_sm`) for:
  - Named Entity Recognition (NER) – people, places, orgs, etc.
  - Noun chunks to extract **key phrases**.
- Uses **TextBlob** for:
  - **Sentiment analysis** (polarity score between -1 and 1).
- Basic readability stats:
  - Total words  
  - Total sentences  
  - Average sentence length  

### 🔹 3. Sentiment Visualization
- Displays a **gauge chart** using Plotly:
  - Red zone → Negative sentiment  
  - Gray zone → Neutral  
  - Green zone → Positive  

### 🔹 4. Multilingual Translation
- Uses `deep-translator`’s `GoogleTranslator` to translate the **generated summary**.
- Supported languages (as of now):
  - English (`en`)
  - Spanish (`es`)
  - French (`fr`)
  - German (`de`)
  - Italian (`it`)
  - Portuguese (`pt`)

### 🔹 5. File Upload Support
You can upload:
- `.txt` files  
- `.pdf` files (text is extracted with **PyPDF2**)  
- `.docx` files (text is extracted with **python-docx**)  

Or just paste raw text in the text area.

---

##  Tech Stack

- **Frontend / Web App**: [Streamlit](https://streamlit.io/)
- **Summarization**: `transformers`, `facebook/bart-large-cnn`
- **NLP / NER / Key Phrases**: `spaCy` (`en_core_web_sm`)
- **Sentiment Analysis**: `TextBlob`
- **Translation**: `deep-translator` (`GoogleTranslator`)
- **Visualization**: `plotly`
- **PDF Parsing**: `PyPDF2`
- **DOCX Parsing**: `python-docx`
- **Environment**: Python 3.x, `venv`

---

##  Project Structure

```text
SMART TEXT SUMMARIZER/
│
├── venv/                 # Python virtual environment (ignored in Git)
├── app.py                # Main Streamlit application
└── requirements.txt      # Python dependencies
```
---

## Installation & Setup

### Clone the Repository

```bash
git clone https://github.com/your-username/smart-text-summarizer.git
cd smart-text-summarizer
```
### Create Virtual Environment

```bash
python -m venv venv
```
Activate it:
Windows (PowerShell / CMD)
```bash
venv\Scripts\activate
```
Git Bash
```bash
source venv/Scripts/activate
```
### Install Dependencies

```bash
pip install -r requirements.txt
```
### Download spaCy English Model

```bash
python -m spacy download en_core_web_sm
```
## Run the Application

With the virtual environment activated:
```bash
streamlit run app.py
```
---

## How to Use

1. Paste your text or upload a file (.txt, .pdf, .docx)
2. Choose:
   - Output language
   - Summary length
3. Click Analyze and Summarize
4. View:
   - Summary
   - Named entities
   - Sentiment chart
   - Key phrases
   - Text statistics

---

## Future Enhancements

- Add extractive summarization mode
- Support more languages
- PDF image-to-text OCR
- Export analysis to PDF / DOCX
- Keyword scoring and topic modeling
- Save history of analyses

---

## Contributing

Pull requests are welcome!
For major changes, open an issue first to discuss what you want to add.

---

## Support

If you like this project, please give it a star ⭐ on GitHub!
