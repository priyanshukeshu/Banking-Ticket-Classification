from flask import Flask, render_template, request
import pickle
import nltk, spacy, re
import string
import swifter
import en_core_web_sm

nltk.download('punkt_tab')

# Load the English language model
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    text2=[]
    for i in text:
        if i.isalnum():
           text2.append(i)          #removing all special characters

    text=text2.copy()
    text2.clear()
    for i in text:
        if i not in string.punctuation:
            text2.append(i)

    return ' '.join(text2)

def lemmatize_text(text):
    doc = nlp(text)
    # Lemmatize each token and remove stop words
    lemmatized_text = " ".join([token.lemma_ for token in doc if not token.is_stop])

    return lemmatized_text

def pos_tags(text):
  nn_words = []
  doc = nlp(text)
  for tok in doc:
      if (tok.tag_== 'NN'):
          nn_words.append(tok.lemma_)
  nn_words_str = " ".join(nn_words)
  return nn_words_str

model = pickle.load(open('model.pkl','rb'))
count = pickle.load(open('count.pkl','rb'))
vector = pickle.load(open('tfidf.pkl','rb'))

app = Flask(__name__)

@app.route("/")

def index():
    return render_template('home.html')

@app.route('/classify',methods=['POST'])
def classification():
    t = str(request.form.get('ticket'))
    ticket = pos_tags(lemmatize_text(clean_text(t)))
    # prediction
    if ticket=='':
        result = "Enter a valid query"
    
    else:
        c = count.transform([ticket])
        vec = vector.transform(c)
        result = model.predict(vec)[0]

    return render_template('home.html',result=result)

if __name__ == '__main__':
    app.run(debug=True)