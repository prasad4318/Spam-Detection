import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

# Load the dataset
data = pd.read_csv("D:\spam.csv")

# Display the first few rows of the dataset
print(data.head())

# Print the shape of the dataset
print(data.shape)

# Remove duplicate rows
data.drop_duplicates(inplace=True)

# Print the shape of the dataset after removing duplicates
print(data.shape)

# Check for any missing values
print(data.isnull().sum())

# Replace 'ham' with 'Not Spam' and 'spam' with 'Spam' in the 'Category' column
data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])

# Display the first few rows of the modified dataset
print(data.head())

# Split the data into training and testing sets
msg = data['Message']
cat = data['Category']
(msg_train, msg_test, cat_train, cat_test) = train_test_split(msg, cat, test_size=0.2)

# Initialize CountVectorizer to convert text data into numerical feature vectors
cv = CountVectorizer(stop_words='english')

# Fit and transform the training messages into a sparse matrix of token counts
features = cv.fit_transform(msg_train)

# Initialize the Naive Bayes classifier (Multinomial Naive Bayes)
model = MultinomialNB()

# Train the model using the training features and categories
model.fit(features, cat_train)

# Transform the test messages into a sparse matrix of token counts
feature_test = cv.transform(msg_test)

# Calculate the accuracy of the model on the test data
accuracy = model.score(feature_test, cat_test)
print("Accuracy:", accuracy)

# Define a function to predict whether a message is spam or not
def predict(message):
    predictinput = cv.transform([message]).toarray()  # Transform the input message into a numerical vector
    result = model.predict(predictinput)  # Predict the category of the input message
    return result

# Set the title and icon for the Streamlit web app
st.set_page_config(page_title="Spam Detection", page_icon="📧")

# Display the title and header for the web app
st.title("Spam Detection 📧")
st.header("Enter a message to check if it's spam or not:")

# Input box for the user to type a message
input_msg = st.text_input('Type your message here...')

# Custom CSS styling for the button
st.markdown("<style>.stButton > button {background-color: #007bff; color: white; font-size: 24px; padding: 10px 20px;}</style>", unsafe_allow_html=True)

# Button to trigger the prediction function
if st.button('Validate'):
    output = predict(input_msg)  # Get the prediction result for the input message
    if output[0] == 'Not Spam':
        st.success("This message is *Not Spam*!", icon="✅")  # Display success message if not spam
    else:
        st.error("This message is *Spam*!", icon="❌")  # Display error message if spam