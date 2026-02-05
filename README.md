# Generative AI – LSTM Text Generator

## Overview
This project implements a Generative AI text generator using an LSTM (Long Short-Term Memory) neural network.
The model is trained on Shakespeare's complete works and generates new text based on a seed input.

## Dataset
Shakespeare Complete Works  
Source: https://www.gutenberg.org/ebooks/100  

Download the text file and rename it to:
shakespeare.txt

Place the file in the project root directory.

## Technologies Used
- Python 3.11
- TensorFlow / Keras
- NumPy
- Pandas

## Project Steps
1. Loaded and preprocessed text data
2. Converted text to lowercase and cleaned punctuation
3. Character-level tokenization
4. Created input-output sequences
5. Built LSTM-based model
6. Trained the model
7. Generated text using seed input

## How to Run the Project

Install dependencies:
pip install tensorflow numpy pandas

Run the script:
python lstm_text_generator.py

## Sample Output

Seed:
to be, or not to be

Generated Text:
to be, or not to be.
ond one thour wort gethour what the thou krest fraot with daghe prospe

## Project Structure

lstm_text_generator/
├── lstm_text_generator.py
├── shakespeare.txt
└── README.md

## Author
Chandrakanta
