from flask import Flask, render_template, request
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import nltk
import json
import random

# we define a variable called app
app = Flask(__name__)

lemmatizer = WordNetLemmatizer()

nltk.download('punkt')
nltk.download('wordnet')

intents = json.loads(open('static/model/price_intents.json').read())
lem_words = pickle.load(open('static/model/lem_words.pkl', 'rb'))
classes = pickle.load(open('static/model/classes.pkl', 'rb'))
model = load_model('static/model/price_negotiator_model.h5')

context_state = None  # Check the context of the words


def cleaning(text):
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word.lower()) for word in words]
    return words


def bag_ow(text, words, show_details=True):
    sentence_words = cleaning(text)
    bag_of_words = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag_of_words[i] = 1
    return (np.array(bag_of_words))


def class_prediction(sentence, model):
    p = bag_ow(sentence, lem_words, show_details=False)
    result = model.predict(np.array([p]))[0]
    ER_THRESHOLD = 0.30
    f_results = [[i, r] for i, r in enumerate(result) if r > ER_THRESHOLD]
    f_results.sort(key=lambda x: x[1], reverse=True)
    intent_prob_list = []
    for i in f_results:
        intent_prob_list.append(
            {"intent": classes[i[0]], "probability": str(i[1])})
    return intent_prob_list


def chat(text):
    # Reset the context state since there is no context at the beginning of the conversation
    global context_state
    # This is what the bot will say if it doesn't understand what the user is saying
    default_responses = [
        "Internet connection problem or perhaps you said something that I have not been trained on"]

    text_bag = bag_ow(text, lem_words, show_details=False)
    response = model.predict(np.array([text_bag]))[0]
    response_index = np.argmax(response)

    tag = classes[response_index]

    intents_list = intents['intents']
    result = ''

    if response[response_index] > 0.8:
        for intent in intents_list:
            if (intent['tag'] == tag):
                if 'context_filter' not in intent or 'context_filter' in intent and intent['context_filter'] == context_state:
                    result = random.choice(intent['responses'])
                    # If this intent is associated with a context set, then set the context state
                    if 'context_set' in intent:
                        context_state = intent['context_set']
                    else:
                        context_state = None
                    return result
                else:
                    return random.choice(default_responses)
    else:
        return random.choice(default_responses)


@app.route("/", methods=["POST", "GET"])
def home():
    if request.method == "POST":
        user_text = request.form.get("msg")
        bot_text = chat(user_text)
        return bot_text

    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
