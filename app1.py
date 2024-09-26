from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler

# Initialize the Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
handler = logging.FileHandler('app.log')
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Load the XGBoost model and StandardScaler from the pickle file
try:
    with open('Wheat_Kernel_Classification_Final.pkl', 'rb') as file:
        classifier, sc = pickle.load(file)
    logger.info('Model and scaler loaded successfully.')
except Exception as e:
    logger.error(f'Error loading model or scaler: {e}')
    raise

# Define the home route to display the input form
@app.route('/')
def home():
    logger.info('Home page accessed.')
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = [float(x) for x in request.form.values()]
        logger.info(f'Received input data: {data}')

        # Convert data into a numpy array and scale it
        final_data = sc.transform(np.array(data).reshape(1, -1))
        logger.debug(f'Scaled data: {final_data}')

        # Predict using the classifier
        prediction = classifier.predict(final_data)
        logger.info(f'Raw prediction: {prediction}')

        # Adjust the prediction to original label by adding 1 (since we subtracted 1 during training)
        output = int(prediction[0]) + 1

        # Map the output to the correct wheat kernel class
        if output == 1:
            result = 'Kama'
        elif output == 2:
            result = 'Rosa'
        else:
            result = 'Canadian'
        
        logger.info(f'Final prediction result: {result}')

        # Render the result
        return render_template('result.html', prediction_text=f'Wheat Kernel Class is: {result}')

    except Exception as e:
        logger.error(f'Error during prediction: {e}')
        return str(e)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
