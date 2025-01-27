import os
import json
import torch
import gradio as gr
import numpy as np
import nltk
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any
import plotly.graph_objs as go
import torch.nn.functional as F
from pathlib import Path
import torch.nn as nn
import re

# Initialize FastAPI with root_path for Spaces
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for analyzing sentiment in text",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up NLTK data path
NLTK_DATA = Path("nltk_data")
NLTK_DATA.mkdir(exist_ok=True)
nltk.data.path.append(str(NLTK_DATA))

# Download NLTK data to local directory
for package in ['punkt', 'stopwords', 'wordnet']:
    try:
        nltk.download(package, download_dir=str(NLTK_DATA))
    except Exception as e:
        print(f"Error downloading {package}: {e}")

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize NLTK components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# [Keep your model architecture classes here - MultiHeadAttention, PositionalEncoding, 
# TransformerBlock, TwitterSentimentTransformer - unchanged]
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)
        return output, attention_weights
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=512):
        super().__init__()
        
        # Create a matrix of positional encodings
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Then the transformer block
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Multi-head attention layer
        self.attention = MultiHeadAttention(d_model, num_heads)
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward neural network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Apply attention and add residual connection
        attention_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))
        
        # Apply feed-forward and add residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

# Finally, the main transformer model
class TwitterSentimentTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=8, num_layers=4, d_ff=1024, max_seq_length=128, dropout=0.1):
        super().__init__()
        
        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Positional encoding layer
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Stack of transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Classification layers
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, 3)  # 3 classes: negative, neutral, positive
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Convert input tokens to embeddings
        x = self.embedding(x)
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# Input validation model
class SentimentRequest(BaseModel):
    text: str

# Response model
class SentimentResponse(BaseModel):
    sentiment: str
    probabilities: Dict[str, float]
    graph: Dict[str, Any]

# Test endpoint
@app.get("/")
async def root():
    return {"message": "Sentiment Analysis API is running"}

# Test endpoint with echo
@app.post("/echo")
async def echo(request: SentimentRequest):
    return {"received_text": request.text}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )

def preprocess_text(text: str) -> str:
    """Enhanced text preprocessing with better error handling"""
    try:
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'@\w+', '@user', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        return ' '.join(tokens)
    except Exception as e:
        raise ValueError(f"Text preprocessing failed: {str(e)}")

def load_model_and_vocab():
    """Load model and vocabulary with enhanced error handling"""
    try:
        # Load vocabulary
        vocab_path = os.path.join(os.path.dirname(__file__), 'vocab.json')
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}")
            
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
        
        # Initialize model
        device = torch.device('cpu')  # Force CPU usage for Spaces
        model = TwitterSentimentTransformer(
            vocab_size=len(vocab),
            d_model=256,
            num_heads=8,
            num_layers=4
        ).to(device)
        
        # Load model weights
        model_path = os.path.join(os.path.dirname(__file__), 'model.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        return model, vocab, device
    except Exception as e:
        print(f"Error loading model or vocabulary: {str(e)}")
        raise

# Initialize model and vocabulary
try:
    model, vocab, device = load_model_and_vocab()
    print("Model and vocabulary loaded successfully")
except Exception as e:
    print(f"Failed to load model or vocabulary: {str(e)}")
    # In production, you might want to exit here
    model, vocab, device = None, None, None

def analyze_sentiment(text: str) -> Dict[str, Any]:
    """Analyze sentiment with comprehensive error handling"""
    if model is None or vocab is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
        
    try:
        # Preprocess text
        preprocessed_text = preprocess_text(text)
        if not preprocessed_text:
            raise ValueError("Preprocessing resulted in empty text")

        # Tokenize
        tokens = [vocab.get(word, vocab.get('<UNK>', 0)) for word in preprocessed_text.split()]
        if not tokens:
            raise ValueError("No valid tokens found")

        # Pad or truncate sequence
        tokens = tokens[:128] if len(tokens) > 128 else tokens + [vocab.get('<PAD>', 0)] * (128 - len(tokens))

        # Model inference
        with torch.no_grad():
            input_tensor = torch.tensor([tokens]).to(device)
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)[0].cpu().numpy()

        # Create visualization
        labels = ['Negative', 'Neutral', 'Positive']
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        
        fig = {
            'data': [{
                'type': 'bar',
                'x': labels,
                'y': probabilities.tolist(),
                'marker': {'color': colors}
            }],
            'layout': {
                'title': 'Sentiment Analysis Results',
                'yaxis': {'title': 'Probability', 'range': [0, 1]},
                'xaxis': {'title': 'Sentiment'}
            }
        }

        return {
            'sentiment': labels[np.argmax(probabilities)],
            'probabilities': {label: float(prob) for label, prob in zip(labels, probabilities)},
            'graph': fig
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# FastAPI endpoint
@app.post("/api/analyze", response_model=SentimentResponse)
async def analyze_text(request: SentimentRequest):
    """API endpoint for sentiment analysis"""
    try:
        result = analyze_sentiment(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Gradio interface
def gradio_interface(text):
    """Gradio interface function"""
    try:
        result = analyze_sentiment(text)
        return (
            result['sentiment'],
            result['probabilities'],
            result['graph']
        )
    except Exception as e:
        raise gr.Error(str(e))

# Create Gradio interface
iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(label="Enter text to analyze"),
    outputs=[
        gr.Textbox(label="Sentiment"),
        gr.JSON(label="Probabilities"),
        gr.Plot(label="Sentiment Distribution")
    ],
    title="Sentiment Analysis",
    description="Analyze the sentiment of text using a transformer-based model."
)

# Mount Gradio app to FastAPI
app = gr.mount_gradio_app(app, iface, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)