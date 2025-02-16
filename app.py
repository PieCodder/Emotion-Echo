from flask import Flask, request, jsonify
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import sqlite3
from datetime import datetime
from flask_cors import CORS
from emotion_analyzer import get_dominant_emotion
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize models
emotion_classifier = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base')
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Initialize the sentence transformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to SQLite database
conn = sqlite3.connect('emotions.db', check_same_thread=False)
cursor = conn.cursor()

# Create a table to store messages, emotions, and timestamps
cursor.execute('''
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message TEXT NOT NULL,
    emotion TEXT NOT NULL,
    response TEXT,
    context TEXT,
    timestamp TEXT NOT NULL
)
''')
conn.commit()

def migrate_database():
    try:
        cursor.execute('''
        ALTER TABLE messages 
        ADD COLUMN response TEXT;
        ''')
        cursor.execute('''
        ALTER TABLE messages 
        ADD COLUMN context TEXT;
        ''')
        conn.commit()
    except sqlite3.OperationalError:
        # Columns might already exist
        pass

migrate_database()

def get_dominant_emotion(text):
    result = emotion_classifier(text, top_k=1)
    return result[0]['label']

def save_message_emotion(message, emotion):
    timestamp = datetime.now().isoformat()
    cursor.execute('''
    INSERT INTO messages (message, emotion, timestamp) VALUES (?, ?, ?)
    ''', (message, emotion, timestamp))
    conn.commit()

def get_last_emotion():
    cursor.execute('''
    SELECT emotion FROM messages ORDER BY timestamp DESC LIMIT 1
    ''')
    result = cursor.fetchone()
    return result[0] if result else None

def find_similar_experiences(message, emotion):
    """Find similar past conversations with the same emotion"""
    cursor.execute('''
    SELECT message, response, context
    FROM messages 
    WHERE emotion = ? 
    ORDER BY timestamp DESC 
    LIMIT 3
    ''', (emotion,))
    return cursor.fetchall()

def save_conversation(message, emotion, response, context=None):
    """Save the full conversation context"""
    timestamp = datetime.now().isoformat()
    cursor.execute('''
    INSERT INTO messages (message, emotion, response, context, timestamp) 
    VALUES (?, ?, ?, ?, ?)
    ''', (message, emotion, response, context, timestamp))
    conn.commit()

def generate_response(prompt, emotion, context=None):
    """Generate response considering emotion and past experiences"""
    if context:
        input_text = f"Context: {context}\nEmotion: {emotion}\nUser: {prompt}"
    else:
        input_text = f"Emotion: {emotion}\nUser: {prompt}"
        
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    response_ids = model.generate(
        input_ids, 
        max_length=100, 
        num_return_sequences=1,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(response_ids[0], skip_special_tokens=True)

def find_most_similar_experience(new_message, top_k=3):
    """
    Find the most similar past experiences using semantic similarity.
    
    Args:
        new_message (str): The current user message
        top_k (int): Number of similar experiences to return
        
    Returns:
        list: Top k similar experiences with their emotions and similarity scores
    """
    # Get all past messages
    cursor.execute('''
    SELECT message, emotion, response, context
    FROM messages
    ORDER BY timestamp DESC
    ''')
    past_experiences = cursor.fetchall()
    
    if not past_experiences:
        return []
        
    # Get embeddings for all messages
    past_messages = [exp[0] for exp in past_experiences]
    new_embedding = sentence_model.encode([new_message])[0]
    past_embeddings = sentence_model.encode(past_messages)
    
    # Calculate similarities
    similarities = cosine_similarity([new_embedding], past_embeddings)[0]
    
    # Get indices of top k similar experiences
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Return top k similar experiences with similarity scores
    similar_experiences = []
    for idx in top_indices:
        if similarities[idx] > 0.3:  # Similarity threshold
            similar_experiences.append({
                'message': past_experiences[idx][0],
                'emotion': past_experiences[idx][1],
                'response': past_experiences[idx][2],
                'context': past_experiences[idx][3],
                'similarity': float(similarities[idx])
            })
    
    return similar_experiences

@app.route('/')
def home():
    return "Welcome to the Emotion Analyzer API!"

@app.route('/process', methods=['POST'])
def process_message():
    try:
        data = request.get_json()
        message = data.get('message')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400

        # Get emotion from the message
        emotion = get_dominant_emotion(message)
        
        # Find similar past experiences based on semantic similarity
        similar_experiences = find_most_similar_experience(message)
        
        # Create context from similar experiences
        context = None
        if similar_experiences:
            context = "Similar past experiences:\n" + "\n".join([
                f"Message: {exp['message']}\n"
                f"Emotion: {exp['emotion']}\n"
                f"Response: {exp['response']}\n"
                f"Similarity: {exp['similarity']:.2f}"
                for exp in similar_experiences
            ])

        # Generate response based on emotion and context
        response = generate_response(message, emotion, context)
        
        # Save the full conversation
        save_conversation(message, emotion, response, context)

        return jsonify({
            'emotion': emotion,
            'response': response,
            'similar_experiences': similar_experiences
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/emotions', methods=['GET'])
def get_emotions():
    cursor.execute('''
    SELECT emotion, timestamp FROM messages ORDER BY timestamp
    ''')
    emotions = cursor.fetchall()
    return jsonify(emotions)

@app.route('/experiences', methods=['GET'])
def get_experiences():
    """Endpoint to retrieve past experiences for a specific emotion"""
    emotion = request.args.get('emotion')
    if not emotion:
        return jsonify({'error': 'No emotion specified'}), 400
        
    cursor.execute('''
    SELECT message, response, context, timestamp 
    FROM messages 
    WHERE emotion = ? 
    ORDER BY timestamp DESC
    ''', (emotion,))
    
    experiences = cursor.fetchall()
    return jsonify([{
        'message': exp[0],
        'response': exp[1],
        'context': exp[2],
        'timestamp': exp[3]
    } for exp in experiences])

if __name__ == '__main__':
    app.run(debug=True) 