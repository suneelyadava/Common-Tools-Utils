import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the dataset into a pandas DataFrame
df = pd.read_csv('Top 5000 Incidents-dex-2.csv')

# Preprocess the data
df = df.fillna(-1)

# Create a machine learning model
model = RandomForestClassifier()

# Split the data into a training and validation set
train_df = df.sample(frac=0.8, random_state=1)
val_df = df.drop(train_df.index)

# Train the model
X_train = train_df.drop(columns=['Severity'])
y_train = train_df['Severity']
model.fit(X_train, y_train)

# Calculate the accuracy on the validation set
X_val = val_df.drop(columns=['Severity'])
y_val = val_df['Severity']
accuracy = model.score(X_val, y_val)
print(f'Validation accuracy: {accuracy:.2f}')

# Use the model to predict the incident column on the entire dataset
df['prediction'] = model.predict(df.drop(columns=['Severity']))

# Calculate the most occurred incidents
most_occurred = df['prediction'].value_counts().reset_index().rename(columns={'index': 'Severity', 'prediction': 'count'}).sort_values('count', ascending=False)
print(most_occurred)
