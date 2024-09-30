import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from flask import Flask, render_template, request, jsonify

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

app = Flask(__name__)

faq_data = {
    "What is your name?": "I am an FAQ bot, here to help you with questions.",
    "Do you restock sold-out sneakers?": "Some popular sneakers are restocked from time to time. You can sign up for restock alerts on the product page or follow us on social media to stay updated on restocks.",
    "How do I find the right size for me?": "We recommend checking our size guide, which is available on each product page. You can also refer to customer reviews for feedback on fit. If you're between sizes, we recommend ordering the next size up.",
    "Are your sneakers authentic?": "Yes, all of our sneakers are 100% authentic and come directly from the brands or authorized retailers.",
    "How do I care for my sneakers?": "To maintain your sneakers, we recommend spot cleaning with a soft cloth and warm water. For specific material care, please refer to the manufacturer's care instructions.",
    "How do I place an order?":"Simply browse our website, select your size, and click 'Add to Cart'. Once you're ready to check out, follow the on-screen instructions to complete your purchase.",
    "Can I modify or cancel my order after placing it?":" You can modify or cancel your order within 1 hour of placing it. Please contact our customer support team immediately if you need to make changes.",
    "How can I track my order?":"Once your order ships, you will receive an email with tracking information. You can also track your order through your account on our website.",
    "Do you offer international shipping?":"Yes, we ship internationally to most countries. Shipping costs and delivery times vary based on the destination.",
    "What payment methods do you accept?":"We accept major credit cards (Visa, MasterCard, American Express), PayPal, Apple Pay, and Google Pay. Some regions may have additional local payment methods.",
    "What is your return policy?":"We accept returns within 30 days of purchase for unworn and unwashed sneakers in their original packaging. Please initiate a return through your account or contact our customer service for help.",
    "How do I return or exchange a pair of sneakers?":"To return or exchange sneakers, log into your account, select the order, and choose the 'Return' or 'Exchange' option. You’ll receive instructions and a return shipping label via email.",
    "Do you sell gift cards?":"Yes, we offer digital gift cards in various amounts. You can purchase them directly from our website, and they will be emailed to the recipient.",
    "Do you offer any discounts or promotional codes?":"Yes, we offer occasional discounts and promotions. Sign up for our newsletter or follow us on social media to stay updated on special offers.",
    "How can I contact customer support?":"You can reach our customer support team via email, phone, or live chat. Our support hours are [9am-5am], and we aim to respond to all inquiries within 24 hours.",

}


# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Preprocess a sentence: tokenize, remove stopwords, and lemmatize
def preprocess(sentence):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(sentence.lower())
    words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(words)

# Preprocess FAQ questions
preprocessed_questions = [preprocess(question) for question in faq_data.keys()]


vectorizer = TfidfVectorizer()

# Fit and transform the FAQ questions
faq_tfidf = vectorizer.fit_transform(preprocessed_questions)


def get_response(user_query):
    # Preprocess the user query
    processed_query = preprocess(user_query)
    
    # Convert the user query into a vector
    query_tfidf = vectorizer.transform([processed_query])
    
    # Compute cosine similarity between user query and all FAQ questions
    similarity_scores = cosine_similarity(query_tfidf, faq_tfidf)
    
    # Find the most similar question
    max_score_index = np.argmax(similarity_scores)
    max_score = similarity_scores[0][max_score_index]
    
    # Define a threshold for similarity
    if max_score > 0.3:
        # Get the corresponding answer from the FAQ data
        question = list(faq_data.keys())[max_score_index]
        return faq_data[question]
    else:
        return "Sorry, I don't have an answer for that. Can you please rephrase your question?"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def chatbot_response():
    user_input = request.form['user_input']
    response = get_response(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)

 
