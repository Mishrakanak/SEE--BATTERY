import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Sample Battery Data - NASA type
# Abhi ke liye dummy data, kal real CSV daalenge
data = {
    'Voltage': [4.2, 4.1, 4.0, 3.9, 3.8],
    'Current': [1.5, 1.6, 1.7, 1.8, 1.9],
    'Temperature': [25, 27, 29, 31, 33],
    'Capacity': [2.0, 1.9, 1.8, 1.7, 1.6]  # Yehi SOH hai
}

df = pd.DataFrame(data)

X = df[['Voltage', 'Current', 'Temperature']]
y = df['Capacity']

model = RandomForestRegressor()
model.fit(X, y)

print("Model Trained Successfully!")
print(f"Accuracy: {model.score(X,y)*100:.2f}%")
print("Ready for 40 LPA - Kanak Mishra @ IITK SEE")
