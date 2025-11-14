import io

import nltk
import plotly.graph_objects as go
import PyPDF2
import spacy
import streamlit as st
import torch
from deep_translator import GoogleTranslator
from docx import Document
from PIL import Image
from textblob import TextBlob
from transformers import pipeline


class AdvancedTextSummarizer:
    def __init__(self):
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.nlp = spacy.load("en_core_web_sm")
        self.supported_languages = {
            'en': 'English', 'es': 'Spanish', 'fr': 'French',
            'de': 'German', 'it': 'Italian', 'pt': 'Portuguese'
        }
        
    def summarize_text(self, text, max_length=150, min_length=50):
        chunks = self._split_into_chunks(text)
        summaries = []
        for chunk in chunks:
            summary = self.summarizer(
                chunk,
                max_length=max_length,
                min_length=min_length
            )[0]['summary_text']
            summaries.append(summary)
        return " ".join(summaries)
    
    def analyze_text(self, text):
        doc = self.nlp(text)
        
        # Extract entities
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Get key phrases
        key_phrases = []
        for chunk in doc.noun_chunks:
            key_phrases.append(chunk.text)
            
        # Sentiment analysis
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        
        # Readability
        words = len(text.split())
        sentences = len(list(doc.sents))
        avg_sentence_length = words / sentences if sentences > 0 else 0
        
        return {
            'entities': entities,
            'key_phrases': key_phrases[:10],
            'sentiment': sentiment,
            'stats': {
                'words': words,
                'sentences': sentences,
                'avg_sentence_length': avg_sentence_length
            }
        }
    
    def translate_text(self, text, target_lang='en'):
        translator = GoogleTranslator(source='auto', target=target_lang)
        return translator.translate(text)
    
    def _split_into_chunks(self, text, max_chunk_size=1000):
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_size += len(word) + 1
            if current_size > max_chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_size = len(word) + 1
            else:
                current_chunk.append(word)
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks


def create_analysis_visualizations(analysis_results):
    # Sentiment visualization
    sentiment_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=analysis_results['sentiment'],
        title={'text': "Sentiment Score"},
        gauge={
            'axis': {'range': [-1, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-1, -0.3], 'color': "red"},
                {'range': [-0.3, 0.3], 'color': "gray"},
                {'range': [0.3, 1], 'color': "green"}
            ]
        }
    ))
    
    return sentiment_fig


def main():
    st.title("Advanced Text Analyzer & Summarizer")
    
    # Initialize summarizer
    summarizer = AdvancedTextSummarizer()
    
    # Sidebar options
    st.sidebar.header("Settings")
    target_language = st.sidebar.selectbox(
        "Select Output Language",
        options=list(summarizer.supported_languages.keys()),
        format_func=lambda x: summarizer.supported_languages[x]
    )
    
    max_length = st.sidebar.slider("Maximum Summary Length", 50, 500, 150)
    min_length = st.sidebar.slider("Minimum Summary Length", 30, 100, 50)
    
    # Main content
    text_input = st.text_area("Enter your text here:", height=200)
    uploaded_file = st.file_uploader("Or upload a document", type=['txt', 'pdf', 'docx'])
    
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            reader = PyPDF2.PdfReader(uploaded_file)
            text_input = ""
            for page in reader.pages:
                text_input += page.extract_text()
        elif uploaded_file.type == "text/plain":
            text_input = uploaded_file.read().decode()
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            # ✅ Use Document from python-docx
            doc = Document(uploaded_file)
            text_input = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    
    if text_input:
        if st.button("Analyze and Summarize"):
            with st.spinner("Processing..."):
                # Generate summary
                summary = summarizer.summarize_text(text_input, max_length, min_length)
                
                # Translate if needed
                if target_language != 'en':
                    summary = summarizer.translate_text(summary, target_language)
                
                # Analyze text
                analysis = summarizer.analyze_text(text_input)
                
                # Display results
                st.subheader("Summary")
                st.write(summary)
                
                # Display analysis
                st.subheader("Text Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Key Statistics:")
                    st.write(f"- Words: {analysis['stats']['words']}")
                    st.write(f"- Sentences: {analysis['stats']['sentences']}")
                    st.write(f"- Avg. Sentence Length: {analysis['stats']['avg_sentence_length']:.1f}")
                
                with col2:
                    st.write("Named Entities:")
                    for entity, label in analysis['entities'][:5]:
                        st.write(f"- {entity} ({label})")
                
                # Visualizations
                st.subheader("Sentiment Analysis")
                sentiment_fig = create_analysis_visualizations(analysis)
                st.plotly_chart(sentiment_fig)
                
                # Key phrases
                st.subheader("Key Phrases")
                st.write(", ".join(analysis['key_phrases']))


if __name__ == "__main__":
    main()
