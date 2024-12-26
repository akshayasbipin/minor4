import pickle
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load and preprocess data
df = pd.read_csv('zomato.csv')

# Clean and preprocess your dataframe as in your original code

df = df.drop(['url', 'phone', 'menu_item'], axis=1)
df.dropna(inplace=True)
df = df.rename(columns={'approx_cost(for two people)': 'cost', 'listed_in(type)': 'type', 'listed_in(city)': 'city'})

df['cost'] = df['cost'].apply(lambda x: x.replace(',', ''))
df['cost'] = df['cost'].astype(float)

df = df.loc[df.rate != 'NEW']
df['rate'] = df['rate'].apply(lambda x: x.replace('/5', ''))
df['rate'] = df['rate'].apply(lambda x: float(x))

# Label encoding
df.online_order[df.online_order == 'Yes'] = 1
df.online_order[df.online_order == 'No'] = 0

# Apply LabelEncoder to categorical columns
le = LabelEncoder()
df.location = le.fit_transform(df.location)
df.rest_type = le.fit_transform(df.rest_type)
df.cuisines = le.fit_transform(df.cuisines)
df.book_table = le.fit_transform(df.book_table)

# Features and target
X = df[['online_order', 'book_table', 'votes', 'location', 'rest_type', 'cuisines', 'cost']]
y = df['rate']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# Initialize and train the model
etr = ExtraTreesRegressor(n_estimators=120)
etr.fit(X_train, y_train)

# Save the trained model using pickle
with open('model.pkl', 'wb') as model_file:
    pickle.dump(etr, model_file)

# Now the model is saved and you can use the `model.pkl` file in your Streamlit app
