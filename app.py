from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load trained ML model
model = joblib.load("food_freshness_model.pkl")
encoder = joblib.load("freshness_encoder.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        # Get values from frontend
        temp = float(data["temp"])
        storage_days = float(data["storage_days"])
        humidity = float(data["humidity"])
        odor = int(data["odor"])
        color_change = int(data["color_change"])
        texture_change = int(data["texture_change"])

        # Create input for ML model
        features = [[
            temp,
            storage_days,
            humidity,
            odor,
            color_change,
            texture_change
        ]]

        # Make prediction
        prediction = model.predict(features)[0]

        # Convert number back to label
        freshness = encoder.inverse_transform([prediction])[0]

        # Get confidence
        probabilities = model.predict_proba(features)[0]
        confidence = round(max(probabilities) * 100, 2)

        # Recommendation
        if freshness == "Fresh":
            recommendation = "Tomato appears fresh. Store it properly for best quality."
        elif freshness == "Moderate":
            recommendation = "Tomato shows some freshness changes. Consider using it soon."
        else:
            recommendation = "Tomato appears not fresh. Do not rely on this prediction alone to determine food safety."

        return jsonify({
            "freshness": freshness,
            "confidence": confidence,
            "recommendation": recommendation
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    print("🍅 Food Freshness AI is starting...")
    print("🌸 Open http://127.0.0.1:5000 in your browser")

    app.run(debug=True)