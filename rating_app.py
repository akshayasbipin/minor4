import numpy as np
import pandas as pd
import pickle
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score

# Load the dataset and preprocess
df = pd.read_csv('zomato.csv')

# Clean up and preprocess the dataset
df = df.drop(['url', 'phone', 'menu_item'], axis=1)
df.dropna(inplace=True)
df = df.rename(columns={'approx_cost(for two people)': 'cost', 'listed_in(type)': 'type', 'listed_in(city)': 'city'})

df['cost'] = df['cost'].apply(lambda x: x.replace(',', ''))
df['cost'] = df['cost'].astype(float)

df = df.loc[df.rate != 'NEW']
df['rate'] = df['rate'].apply(lambda x: x.replace('/5', ''))
df['rate'] = df['rate'].apply(lambda x: float(x))

df.online_order[df.online_order == 'Yes'] = 1
df.online_order[df.online_order == 'No'] = 0

# Label encoding for categorical columns
le = LabelEncoder()
df.location = le.fit_transform(df.location)
df.rest_type = le.fit_transform(df.rest_type)
df.cuisines = le.fit_transform(df.cuisines)
df.book_table = le.fit_transform(df.book_table)

# Features and target for model training
X = df[['online_order', 'book_table', 'votes', 'location', 'rest_type', 'cuisines', 'cost']]
y = df['rate']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# Training the model
etr = ExtraTreesRegressor(n_estimators=120)
etr.fit(X_train, y_train)

# Save the model using pickle
pickle.dump(etr, open('model.pkl', 'wb'))

# Streamlit Application for user input
st.title('Zomato Restaurant Rating Prediction')

st.subheader('Enter the details for prediction')

# Collect inputs from the user
online_order = st.selectbox('Online Order', ['Yes', 'No'])
book_table = st.selectbox('Book Table', ['Yes', 'No'])
votes = st.number_input('Votes', min_value=0, max_value=100000, value=100)
location = st.text_input('Location')
restaurant_type = st.text_input('Restaurant Type')
cuisines = st.text_input('Cuisines')

cost = st.number_input('Cost', min_value=0.0, max_value=100000.0, value=2000.0)


# Predict button
if st.button('Predict Rating'):
    # Transform inputs
    online_order = 1 if online_order == 'Yes' else 0
    book_table = 1 if book_table == 'Yes' else 0
    location_encoded = le.transform([location])[0] if location in le.classes_ else 0
    restaurant_type_encoded = le.transform([restaurant_type])[0] if restaurant_type in le.classes_ else 0
    cuisines_encoded = le.transform([cuisines])[0] if cuisines in le.classes_ else 0

    # Prepare the features for prediction
    features = np.array([[online_order, book_table, votes, location_encoded, restaurant_type_encoded, cuisines_encoded, cost]])

    # Load the trained model and make prediction
    model = pickle.load(open('model.pkl', 'rb'))
    prediction = model.predict(features)
    output = round(prediction[0], 1)

    # Display the result
    st.subheader(f"Predicted Rating: {output}")
