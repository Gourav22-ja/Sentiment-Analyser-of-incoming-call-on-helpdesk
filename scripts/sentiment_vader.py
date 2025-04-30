import sys
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# Get input text from command-line argument
input_text = sys.argv[1] if len(sys.argv) > 1 else ""

if not input_text:
    print(json.dumps({"error": "No text provided"}))
    sys.exit(1)

# Analyze sentiment
scores = analyzer.polarity_scores(input_text)
compound = scores['compound']

# Map compound score to sentiment label
if compound >= 0.05:
    sentiment = "POSITIVE"
elif compound <= -0.05:
    sentiment = "NEGATIVE"
else:
    sentiment = "NEUTRAL"

# Output JSON result
result = {
    "text": input_text,
    "sentiment": sentiment,
    "confidence": abs(compound),  # Simplified confidence metric
    "compound_score": compound
}
print(json.dumps(result))