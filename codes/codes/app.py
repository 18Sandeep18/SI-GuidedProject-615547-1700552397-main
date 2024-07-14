import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from flask import Flask, request


# Load data
car_df = pd.read_csv('C:\\Users\\91950\\Desktop\\vs code\\car_data.csv')

# Split data into train and test sets
X = car_df.iloc[:, [2, 3]]
y = car_df.iloc[:, 4]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# Scale the data
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fit the model
classi = LogisticRegression()
classi.fit(X_train, y_train)

app = Flask(__name__)

home = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Car Purchase Prediction</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f7f7f7;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            position: relative;
        }

        h1 {
            position: absolute;
            top: 120px;
            color: #333;
        }

        form {
            background-color: #fff;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 0 15px rgba(0, 0, 0, 0.3);
            max-width: 400px;
            width: 100%;
        }

        label {
            display: block;
            margin-bottom: 10px;
            font-size: 20px;
            font-weight: 600;
            color: #444;
        }

        input {
            width: 100%;
            padding: 12px;
            margin-bottom: 20px;
            box-sizing: border-box;
            border: 1px solid #ccc;
            border-radius: 4px;
            font-size: 14px;
        }

        button {
            background-color: #4caf50;
            color: #fff;
            padding: 12px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size:21px;
            width: 100%;
        }

        button:hover {
            background-color: #45a049;
        }
    </style>
</head>
<body>
    <h1>Car Purchase Prediction</h1>
    <form action="/predict" method="post">
        <label for="age">Age:</label>
        <input type="text" name="age" required>

        <label for="salary">Estimated Salary:</label>
        <input type="text" name="salary" required>

        <button type="submit">Predict</button>
    </form>
</body>
</html>
'''

result = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Result</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f7f7f7;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }

        .container {
            text-align: center;
            background-color: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 15px rgba(0, 0, 0, 0.3);
            max-width: 400px;
            width: 100%;
        }

        h1 {
            color: #333;
        }

        p {
            font-size: 18px;
            color: #555;
            margin-bottom: 10px;
        }

        a {
            text-decoration: none;
            color: #4caf50;
            font-weight: bold;
        }

        a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Predicted Result</h1>
        <p>answer</p>
        <a href="/">Go back</a>
    </div>
</body>
</html>

'''


@app.route('/')
def index():
    return home


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = float(request.form['age'])
        salary = float(request.form['salary'])

        # Make prediction
        X_new = [[age, salary]]
        X_new_scaled = sc_X.transform(X_new)
        y_new = classi.predict(X_new_scaled)


        # Display prediction
        if y_new == 1:
            pred = "Customer is interested to purchase the car😊"
        else:
            pred = "Customer is not interested to purchase a car😔"

    return result.replace('answer', pred)


if __name__ == '__main__':
    app.run(debug=True)
