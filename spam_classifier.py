import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

# Convert labels to binary
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2)
#print(df['message'])
# Vectorize text
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predict
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))

#Testing for custom message

while True:
    custom_msg = input("\nEnter a message to check if it's spam (or type 'exit' to quit):\n> ")
    if custom_msg.lower() == 'exit':
        break
    custom_vec = vectorizer.transform([custom_msg])
    result = model.predict(custom_vec)[0]
    print("This message is:", "SPAM" if result else "Not Spam")
