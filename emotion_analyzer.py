from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import sqlite3
from datetime import datetime, timedelta
import dateutil.parser

# Initialize the emotion classifier
emotion_classifier = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base')

# Initialize the BlenderBot model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-3B")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-3B")

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('emotions.db')
cursor = conn.cursor()

# Create a table to store messages, emotions, and timestamps
cursor.execute('''
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message TEXT NOT NULL,
    emotion TEXT NOT NULL,
    timestamp TEXT NOT NULL
)
''')
conn.commit()

def get_dominant_emotion(text):
    """
    Analyzes the emotional content of text and returns the highest-confidence emotion.
    
    Args:
        text (str): The text to analyze
        
    Returns:
        str: The dominant emotion detected in the text
    """
    # Analyze the text to detect emotions
    result = emotion_classifier(text, top_k=1)  # Get the top emotion
    dominant_emotion = result[0]['label']
    
    return dominant_emotion

def save_message_emotion(message, emotion):
    timestamp = datetime.now().isoformat()
    cursor.execute('''
    INSERT INTO messages (message, emotion, timestamp) VALUES (?, ?, ?)
    ''', (message, emotion, timestamp))
    conn.commit()

def get_past_emotions():
    cursor.execute('''
    SELECT message, emotion, timestamp FROM messages ORDER BY timestamp
    ''')
    return cursor.fetchall()

def get_last_emotion():
    cursor.execute('''
    SELECT emotion FROM messages ORDER BY timestamp DESC LIMIT 1
    ''')
    result = cursor.fetchone()
    return result[0] if result else None

def get_emotion_by_date(target_date=None):
    """
    Retrieve emotions from a specific date.
    If no date is provided, returns the most recent emotion.
    target_date should be in ISO format (YYYY-MM-DD)
    """
    if target_date:
        # Convert target_date to datetime
        try:
            date = dateutil.parser.parse(target_date)
            next_day = date + timedelta(days=1)
            
            cursor.execute('''
            SELECT message, emotion, timestamp 
            FROM messages 
            WHERE timestamp >= ? AND timestamp < ?
            ORDER BY timestamp DESC
            ''', (date.isoformat(), next_day.isoformat()))
            
        except (ValueError, TypeError):
            return None, "Invalid date format. Please use YYYY-MM-DD"
    else:
        # Get most recent emotion
        cursor.execute('''
        SELECT message, emotion, timestamp 
        FROM messages 
        ORDER BY timestamp DESC 
        LIMIT 1
        ''')
    
    result = cursor.fetchone()
    if result:
        return {
            'message': result[0],
            'emotion': result[1],
            'timestamp': result[2]
        }, None
    return None, "No emotions found for this date"

def get_emotions_in_range(start_date, end_date):
    """
    Retrieve emotions between two dates.
    Dates should be in ISO format (YYYY-MM-DD)
    """
    try:
        start = dateutil.parser.parse(start_date)
        end = dateutil.parser.parse(end_date) + timedelta(days=1)  # Include end date
        
        cursor.execute('''
        SELECT message, emotion, timestamp 
        FROM messages 
        WHERE timestamp >= ? AND timestamp < ?
        ORDER BY timestamp DESC
        ''', (start.isoformat(), end.isoformat()))
        
        results = cursor.fetchall()
        if results:
            return [{
                'message': row[0],
                'emotion': row[1],
                'timestamp': row[2]
            } for row in results], None
        return None, "No emotions found in this date range"
    
    except (ValueError, TypeError):
        return None, "Invalid date format. Please use YYYY-MM-DD"

def generate_emotional_response(prompt, emotion_data):
    """
    Generate a response based on emotion data
    """
    if isinstance(emotion_data, dict):
        emotion = emotion_data['emotion']
        context = f"Previous message: {emotion_data['message']}"
    else:
        emotion = emotion_data
        context = ""

    input_text = f"{context}\nEmotion: {emotion}\nPrompt: {prompt}"
    inputs = tokenizer(input_text, return_tensors='pt')
    response_ids = model.generate(inputs['input_ids'], max_length=100, num_return_sequences=1)
    return tokenizer.decode(response_ids[0], skip_special_tokens=True)

# Example usage
if __name__ == "__main__":
    # Test with current date
    emotion_data, error = get_emotion_by_date()
    if error:
        print(f"Error: {error}")
    else:
        print(f"Most recent emotion: {emotion_data['emotion']}")
        response = generate_emotional_response("Tell me something interesting", emotion_data)
        print(f"Response: {response}")

    # Test with specific date
    specific_date = "2024-01-15"  # Example date
    emotion_data, error = get_emotion_by_date(specific_date)
    if error:
        print(f"Error: {error}")
    else:
        print(f"Emotion from {specific_date}: {emotion_data['emotion']}")
        response = generate_emotional_response("Tell me something interesting", emotion_data)
        print(f"Response: {response}")

    # Test date range
    start_date = "2024-01-01"
    end_date = "2024-01-15"
    emotions, error = get_emotions_in_range(start_date, end_date)
    if error:
        print(f"Error: {error}")
    else:
        print(f"Emotions from {start_date} to {end_date}:")
        for emotion in emotions:
            print(f"Date: {emotion['timestamp']}, Emotion: {emotion['emotion']}")

# Close the connection when done
conn.close() 