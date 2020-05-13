import flask
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer # Count vector

corpus = pd.read_csv('trainingData_clean.csv')
corpus = corpus[corpus['tweet_text'].isnull() == False]

# Use pickle to load in the pre-trained model
with open(f'model/pickle_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')

# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main.html'))
    
    if flask.request.method == 'POST':
        # Extract the input
        tweet = flask.request.form['tweet']
        print(tweet)

        # Vectorize and predict
        vectorizer = CountVectorizer()
        vectorizer.fit_transform(list(corpus['tweet_text']))
        vector = vectorizer.transform([tweet])
        prediction = model.predict(vector)

        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return flask.render_template('main.html',
                                     original_input={'tweet':tweet,
                                                     },
                                     result=prediction,
                                     )

if __name__ == '__main__':
    app.run()
