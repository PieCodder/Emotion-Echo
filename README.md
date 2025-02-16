# Emotion Echo

Emotion Echo is an AI-powered web application that analyzes text for emotions and generates contextually appropriate responses. The application uses state-of-the-art natural language processing models to detect emotions and maintain emotionally consistent conversations.

## Features

- Real-time emotion detection using Hugging Face's emotion-english-distilroberta-base model
- Contextual response generation that matches the detected emotion
- SQLite database for storing conversation history
- Simple and intuitive web interface
- Date-based emotion retrieval functionality

## Technologies Used

- Python
- Flask
- Hugging Face Transformers
- SQLite
- HTML/JavaScript
- Facebook's BlenderBot for response generation

## Installation

1. Clone the repository: bash
git clone https://github.com/PieCodder/Emotion-Echo.git

2. Install required packages: bash
pip install transformers flask python-dateutil

3. Run the application: bash
python app.py


4. Open your browser and navigate to `http://127.0.0.1:5000`

## Usage

1. Enter your message in the text box
2. Click "Submit" to analyze the emotion
3. View the AI's emotionally-appropriate response
4. Browse past conversations and emotions by date

## Contributing

Feel free to submit issues and enhancement requests!

## License

[MIT License](LICENSE)

## Author

- Joshua Viana (PieCodder)
