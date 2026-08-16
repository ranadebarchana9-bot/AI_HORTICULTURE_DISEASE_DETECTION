# How to Run Your Plant Disease Advisory App

## 1. Create a project folder
Make a new folder on your PC, e.g. `plant_disease_app`, and put these
THREE files inside it:
- `app.py`
- `requirements.txt`
- `plant_disease_model_v2_extended.keras`  (the model you downloaded from Kaggle)
- `class_names.json`  (also downloaded from Kaggle)

So the folder should contain 4 files total.

## 2. Install Python (if you don't already have it)
Download from https://www.python.org/downloads/ (any recent version 3.10+).
During installation, tick "Add Python to PATH".

## 3. Open a terminal in that folder
- Windows: open the folder in File Explorer, click the address bar,
  type `cmd`, press Enter.
- This opens a command prompt already inside your project folder.

## 4. Install the required packages
Run this command:
```
pip install -r requirements.txt
```
This may take a few minutes (TensorFlow is a large package).

## 5. Run the app
```
streamlit run app.py
```
This will automatically open the app in your web browser
(usually at http://localhost:8501).

## 6. Using the app
- In the left sidebar, paste your **Gemini API key** and your
  **OpenWeatherMap API key**.
- Upload a leaf image.
- Type your city name.
- Click "Analyze Plant".
- You'll see the predicted disease, confidence, current weather,
  and an AI-generated advisory — all in one screen.

## Notes
- Keep your API keys private — don't share screenshots with them visible.
- The first run may take longer as TensorFlow loads the model.
- If you get a "model not found" error, double check the `.keras`
  and `.json` files are in the exact same folder as `app.py`.
