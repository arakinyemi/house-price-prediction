# House Price Prediction System

This project implements a machine learning model to predict house prices and serves it via a Flask Web GUI.

## Project Structure
- `app.py`: Flask application.
- `model/`: Contains the trained model (`house_price_model.pkl`) and the training notebook (`model_building.ipynb`).
- `static/`: CSS styles.
- `templates/`: HTML templates.
- `requirements.txt`: Project dependencies.

## Local Setup
1. **Create and Activate Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   python app.py
   ```

4. **Access the App**:
   Open your browser and navigate to `http://127.0.0.1:5000`.

## Troubleshooting
- **Model Not Found**: Ensure `house_price_model.pkl` exists in the `model/` directory. If not, run `python model/train_local.py` to generate a temporary one.
- **Port In Use**: If port 5000 is busy, Flask will try another port or fail. Check the terminal output for the correct URL.

## Google Colab Instructions
To train the model on Google Colab:
1. Open `model/model_building.ipynb` in Colab.
2. Upload the `train.csv` dataset.
3. Run all cells to train and save the model.
4. Download `house_price_model.pkl` and place it in the `model/` directory locally.

## Deployment on Render
1. Push this repository to GitHub.
2. Log in to [Render](https://render.com/).
3. Click **New +** -> **Web Service**.
4. Connect your GitHub repository.
5. Configure the service:
   - **Name**: house-price-prediction (or any name)
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
6. Click **Create Web Service**.
