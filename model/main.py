import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle as pickle


def create_model(data): 
  X = data.drop(['Potability'], axis=1)
  y = data['Potability']
  
  # scale the data
  scaler = StandardScaler()
  X = scaler.fit_transform(X)
  
  # split the data
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
  )
  
  # train the model
  print(X.shape)
  model = RandomForestClassifier(n_estimators = 100,n_jobs=-1,random_state=42,bootstrap=True,)
  model.fit(X_train, y_train)
  
  # test model
  y_pred = model.predict(X_test)
  #print(model.score(y_pred,y_train))
  print('Accuracy of our model: ', accuracy_score(y_test, y_pred))
  print("Classification report: \n", classification_report(y_test, y_pred))
  
  return model, scaler


def get_clean_data():
  data = pd.read_csv('C:/Users/hp/Desktop/Streamlit water quality/data/data.csv')

  
  data = data.apply(fill_missing_values, axis=1)
  
  return data



potable_ranges = {
    'ph': (6.5, 8.5),
    'Hardness': (60, 120),
    'Solids': (0, 500),
    'Chloramines': (0, 4),
    'Sulfate': (0, 250),
    'Conductivity': (50, 500),
    'Organic_carbon': (0, 4),
    'Trihalomethanes': (0, 80),
    'Turbidity': (0, 1)
}

non_potable_ranges = {
    'ph': (5, 9),
    'Hardness': (0, 500),
    'Solids': (0, 1500),
    'Chloramines': (0, 10),
    'Sulfate': (0, 500),
    'Conductivity': (50, 2000),
    'Organic_carbon': (0, 10),
    'Trihalomethanes': (0, 150),
    'Turbidity': (0, 10)
}


def fill_missing_values(row):
    for feature in potable_ranges.keys():
        if pd.isnull(row[feature]):
            if row['Potability'] == 1:
                row[feature] = np.random.uniform(*potable_ranges[feature])
            else:
                row[feature] = np.random.uniform(*non_potable_ranges[feature])
    return row


def main():
  data = get_clean_data()
  print(data)
  

  model, scaler = create_model(data)

  with open('model/model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
  with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
  

if __name__ == '__main__':
  main()