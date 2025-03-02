import re
import json
import joblib
import numpy as np
import pandas as pd
import os
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

app = Flask(__name__, template_folder="templates")

def extract_username_from_url(url):
    match = re.search(r"(?:instagram|facebook|twitter)\.com/([a-zA-Z0-9_.]+)", url)
    return match.group(1) if match else None

def fetch_profile_info(form_data):
    username = extract_username_from_url(form_data['url'])
    if not username:
        return None
    
    features = {
        "userFollowerCount": int(form_data['followers']),
        "userFollowingCount": int(form_data['following']),
        "userBiographyLength": int(form_data['bio_length']),
        "userMediaCount": int(form_data['media_count']),
        "userHasProfilPic": int(form_data['profile_pic']),
        "userIsPrivate": int(form_data['is_private']),
        "usernameDigitCount": sum(c.isdigit() for c in username),
        "usernameLength": len(username)
    }
    return np.array([list(features.values())])

def train_fake_account_detector(train_data_path, test_data_path, model_path):
    if not os.path.exists(model_path):
        with open(train_data_path, "r") as train_file:
            train_data = json.load(train_file)
        with open(test_data_path, "r") as test_file:
            test_data = json.load(test_file)
        
        df_train = pd.DataFrame(train_data)
        df_test = pd.DataFrame(test_data)
        
        features = [
            "userFollowerCount",
            "userFollowingCount",
            "userBiographyLength",
            "userMediaCount",
            "userHasProfilPic",
            "userIsPrivate",
            "usernameDigitCount",
            "usernameLength"
        ]
        target = "isFake"
        
        X_train = df_train[features]
        y_train = df_train[target]
        X_test = df_test[features]
        y_test = df_test[target]
        
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, model_path)
        print("Model trained and saved at", model_path)
        print("Test Accuracy:", pipeline.score(X_test, y_test))

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        model_path = "fake_account_detector.joblib"
        if not os.path.exists(model_path):
            train_fake_account_detector("trainData.json", "testData.json", model_path)
        
        model = joblib.load(model_path)
        profile_data = fetch_profile_info(request.form)
        if profile_data is None:
            return render_template("index.html", result="Invalid URL. Unable to extract username.")
        prediction = model.predict(profile_data)
        result = "Fake Account" if prediction[0] == 1 else "Real Account"
        return render_template("index.html", result=result)
    return render_template("index.html", result=None)

if __name__ == '__main__':
    if not os.path.exists("templates"): os.makedirs("templates")
    with open("templates/index.html", "w") as f:
        f.write("""
        <!DOCTYPE html>
<html>
<head>
    <title>Fake Account Detector</title>
    <style>
        body {
            background: linear-gradient(135deg, #6a11cb, #2575fc);
            font-family: Arial, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            color: white;
        }
        .container {
            background: rgba(255, 255, 255, 0.2);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.3);
            width: 50%;
            text-align: center;
        }
        input {
            width: 90%;
            padding: 10px;
            margin: 10px 0;
            border: none;
            border-radius: 5px;
        }
        button {
            background: #ff7eb3;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background: #ff5277;
        }
        .result {
            margin-top: 20px;
            font-size: 20px;
            font-weight: bold;
            color: yellow;
            padding: 10px;
            background: rgba(0, 0, 0, 0.5);
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Fake Account Detector</h1>
        <form method="post">
            <input type="text" name="url" placeholder="Profile URL" required><br>
            <input type="number" name="followers" placeholder="Follower Count" required><br>
            <input type="number" name="following" placeholder="Following Count" required><br>
            <input type="number" name="bio_length" placeholder="Bio Length" required><br>
            <input type="number" name="media_count" placeholder="Media Count" required><br>
            <input type="number" name="profile_pic" placeholder="Has Profile Picture (1 for Yes, 0 for No)" required><br>
            <input type="number" name="is_private" placeholder="Is Private Account (1 for Yes, 0 for No)" required><br>
            <button type="submit">Check Account</button>
        </form>
        {% if result %}
            <div class="result">Prediction: {{ result }}</div>
        {% endif %}
    </div>
</body>
</html>

        """)
    app.run(debug=True)
