# AI Agent GUI - Local Setup Guide

## Prerequisites

### 1. Python Installation
Make sure you have Python 3.8+ installed on your system.

### 2. Required Dependencies

Create a `requirements.txt` file with the following content:

```txt
# Core AI/ML libraries
torch>=1.9.0
transformers>=4.20.0
sentence-transformers>=2.2.0
scikit-learn>=1.0.0
numpy>=1.21.0
pandas>=1.3.0

# GUI libraries
tkinter  # Usually comes with Python
matplotlib>=3.5.0
Pillow>=8.3.0

# Database
faiss-cpu>=1.7.2  # or faiss-gpu if you have CUDA
sqlite3  # Usually comes with Python

# Utilities
logging
json
threading
time
os
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: `tkinter` usually comes pre-installed with Python. If not:
- **Ubuntu/Debian**: `sudo apt-get install python3-tk`
- **macOS**: Should be included with Python from python.org
- **Windows**: Usually included with Python installer

## File Structure

Create the following directory structure:

```
ai-agent-gui/
├── main.py                    # Main application file
├── improved_ai_agent.py       # Core AI agent classes
├── config.json               # Configuration file
├── requirements.txt          # Dependencies
├── logs/                     # Log files directory
│   └── (auto-created)
├── data/                     # Data storage
│   └── (auto-created)
└── models/                   # Model cache
    └── (auto-created)
```

## Configuration

### 1. Create `config.json`

```json
{
  "model": {
    "embedding_model": "all-MiniLM-L6-v2",
    "generation_model": "gpt2",
    "use_gpu": false,
    "model_cache_dir": "./models"
  },
  "database": {
    "path": "./data/vector_db.db",
    "vector_dimension": 384
  },
  "learning": {
    "feedback_threshold": 0.7,
    "learning_rate": 0.001
  },
  "security": {
    "safe_mode": true,
    "execution_timeout": 30,
    "allowed_operations": ["read", "analyze", "generate"]
  },
  "logging": {
    "level": "INFO",
    "file": "./logs/ai_agent.log"
  }
}
```

### 2. Create Core AI Agent Classes

You'll need to create the `improved_ai_agent.py` file with the core classes that the GUI imports. Here's a basic implementation:

```python
import os
import json
import logging
import sqlite3
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def setup_logger(name: str, log_file: str) -> logging.Logger:
    """Set up a logger with file and console handlers."""
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

class ConfigManager:
    """Manages configuration settings."""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Return default configuration
            return {
                "model": {
                    "embedding_model": "all-MiniLM-L6-v2",
                    "generation_model": "gpt2",
                    "use_gpu": False,
                    "model_cache_dir": "./models"
                },
                "database": {
                    "path": "./data/vector_db.db",
                    "vector_dimension": 384
                },
                "learning": {
                    "feedback_threshold": 0.7,
                    "learning_rate": 0.001
                },
                "security": {
                    "safe_mode": True,
                    "execution_timeout": 30
                }
            }
    
    def save_config(self) -> bool:
        """Save configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def get(self, *keys) -> Any:
        """Get configuration value using dot notation."""
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value
    
    def set(self, *args) -> None:
        """Set configuration value using dot notation."""
        if len(args) < 2:
            return
        
        keys = args[:-1]
        value = args[-1]
        
        config_ref = self.config
        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]
        
        config_ref[keys[-1]] = value

class ModelManager:
    """Manages AI models."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.embedding_model = None
        self.generation_model = None
        self.tokenizer = None
        self.logger = setup_logger("ModelManager", "logs/model_manager.log")
    
    def load_embedding_model(self) -> bool:
        """Load the embedding model."""
        try:
            model_name = self.config.get("model", "embedding_model")
            cache_dir = self.config.get("model", "model_cache_dir")
            
            os.makedirs(cache_dir, exist_ok=True)
            
            self.embedding_model = SentenceTransformer(
                model_name,
                cache_folder=cache_dir
            )
            
            self.logger.info(f"Loaded embedding model: {model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading embedding model: {e}")
            return False
    
    def load_generation_model(self) -> Tuple[Optional[Any], Optional[Any]]:
        """Load the generation model."""
        try:
            model_name = self.config.get("model", "generation_model")
            cache_dir = self.config.get("model", "model_cache_dir")
            
            os.makedirs(cache_dir, exist_ok=True)
            
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
            
            self.generation_model = GPT2LMHeadModel.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
            
            # Add padding token
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.logger.info(f"Loaded generation model: {model_name}")
            return self.generation_model, self.tokenizer
            
        except Exception as e:
            self.logger.error(f"Error loading generation model: {e}")
            return None, None
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a list of texts."""
        if self.embedding_model is None:
            raise ValueError("Embedding model not loaded")
        
        return self.embedding_model.encode(texts)
    
    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 0.7) -> str:
        """Generate text using the generation model."""
        if self.generation_model is None or self.tokenizer is None:
            return "Generation model not available. Please load the model first."
        
        try:
            # Encode the prompt
            inputs = self.tokenizer.encode(prompt, return_tensors='pt')
            
            # Generate text
            with torch.no_grad():
                outputs = self.generation_model.generate(
                    inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the prompt from the generated text
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text if generated_text else "I'm thinking about that..."
            
        except Exception as e:
            self.logger.error(f"Error generating text: {e}")
            return f"Sorry, I encountered an error while generating a response: {str(e)}"

class VectorDatabase:
    """Simple vector database using SQLite and FAISS."""
    
    def __init__(self, db_path: str, vector_dim: int):
        self.db_path = db_path
        self.vector_dim = vector_dim
        self.logger = setup_logger("VectorDatabase", "logs/vector_db.log")
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    source TEXT,
                    tags TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
    
    def add_embedding(self, text: str, embedding: np.ndarray, source: str = "", tags: List[str] = None):
        """Add an embedding to the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Convert embedding to bytes
            embedding_bytes = embedding.tobytes()
            
            # Convert tags to JSON string
            tags_json = json.dumps(tags) if tags else "[]"
            
            cursor.execute('''
                INSERT INTO embeddings (text, embedding, source, tags)
                VALUES (?, ?, ?, ?)
            ''', (text, embedding_bytes, source, tags_json))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Added embedding for text: {text[:50]}...")
            
        except Exception as e:
            self.logger.error(f"Error adding embedding: {e}")
    
    def search_similar(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar embeddings."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT id, text, embedding, source, tags FROM embeddings')
            results = cursor.fetchall()
            
            if not results:
                return []
            
            # Calculate similarities
            similarities = []
            for row in results:
                stored_embedding = np.frombuffer(row[2], dtype=np.float32)
                
                # Reshape if necessary
                if len(stored_embedding) != len(query_embedding):
                    continue
                
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, stored_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
                )
                
                similarities.append({
                    'id': row[0],
                    'text': row[1],
                    'source': row[3],
                    'tags': json.loads(row[4]),
                    'similarity': similarity
                })
            
            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            conn.close()
            return similarities[:k]
            
        except Exception as e:
            self.logger.error(f"Error searching similar embeddings: {e}")
            return []
```

## Running the Application

1. **Save the main GUI code** as `main.py` (the code from your document)

2. **Create the directory structure** and files as described above

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python main.py
   ```

## First Run Instructions

1. **Initial Setup**: On first run, the application will:
   - Create necessary directories (`logs/`, `data/`, `models/`)
   - Download the embedding model (this may take a few minutes)
   - Create the vector database

2. **Load Models**: Click the "Load Models" button in the GUI to initialize the AI models

3. **Start Chatting**: Once models are loaded, you can start conversing with the AI agent

## Troubleshooting

### Common Issues

1. **tkinter not found**:
   - Install tkinter for your system (see prerequisites)

2. **Models downloading slowly**:
   - The first run will download models (~500MB), be patient
   - Models are cached locally for future runs

3. **GPU not detected**:
   - Install `torch` with CUDA support if you have a compatible GPU
   - Or set `use_gpu: false` in config.json

4. **Permission errors**:
   - Make sure you have write permissions in the application directory

### Performance Tips

1. **First run**: Models will download and cache locally
2. **Subsequent runs**: Much faster startup time
3. **GPU acceleration**: Enable in settings if you have a compatible GPU
4. **Memory usage**: The application uses ~2-4GB RAM with default models

## Features

- **Interactive Chat**: Real-time conversation with the AI agent
- **Visualizations**: Multiple visualization types (conversation flow, embeddings, sentiment, topics)
- **Configuration**: Adjustable model and learning parameters
- **Data Export**: Export chat history and visualizations
- **Analysis Tools**: Conversation analysis with statistics and insights

## Next Steps

Once running, you can:
1. Customize the configuration through the GUI
2. Add your own models by modifying the config
3. Extend the visualization capabilities
4. Integrate with external APIs or databases
