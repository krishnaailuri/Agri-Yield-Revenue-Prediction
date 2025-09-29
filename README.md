
# 🌽 Agri Yield & Revenue Prediction

This project builds a **machine learning pipeline** and a **Streamlit web app** to predict corn yield per acre and estimate revenue, based on USDA crop data and user inputs such as **Year, State, Period, and Data Item**.

---

##  Features
- Cleans and preprocesses raw USDA crop yield data (`corn_yield.csv`).
- Uses **Random Forest Regressor** with categorical encoding (OneHotEncoder).
- Outputs model performance metrics (MSE, MAE, R²).
- Saves the trained pipeline as `crop_price_pipeline.pkl`.
- Streamlit UI for:
  - Entering crop parameters interactively.
  - Predicting bushels per acre, tons per acre, and estimated revenue.
  - Visualizing yield distribution.
- Easy to extend with more crops or different ML models.

---

##  Installation

1. **Clone this repo** and move into the folder:
   ```bash
   git clone https://github.com/yourusername/corn-yield-prediction.git
   cd corn-yield-prediction
````

2. **Create a virtual environment (recommended)**:

   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   Example `requirements.txt`:

   ```txt
   pandas
   numpy
   scikit-learn
   matplotlib
   seaborn
   streamlit
   ```

---

##  Training the Model

1. Place your data file in `./data/corn_yield.csv`.

2. Run training:

   ```bash
   python train_and_app.py
   ```

   This will:

   * Load and clean the dataset.
   * Train a Random Forest Regressor.
   * Print evaluation metrics and feature importances.
   * Save the trained model to `crop_price_pipeline.pkl`.

---

##  Running the Streamlit App

Once you have a trained model (`crop_price_pipeline.pkl`):

```bash
streamlit run train_and_app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

##  Example Output

* **Metrics (training phase):**

  ```
  Mean Squared Error: 1234.56
  Mean Absolute Error: 25.43
  R2 Score: 0.87
  ```

* **Streamlit UI:**

  * Sidebar for input (Year, State, Period, Data Item, Price).
  * Predictions displayed in metric cards.
  * Histogram plot of simulated production distribution.

---

##  Project Structure

```
.
├── data/
│   └── corn_yield.csv              # raw dataset
├── train_and_app.py                # training + streamlit app
├── crop_price_pipeline.pkl         # saved trained model (generated)
├── requirements.txt
└── README.md
```

---

##  Next Steps / Improvements

* Add **GridSearchCV** or **RandomizedSearchCV** for hyperparameter tuning.
* Implement **cross-validation** for better generalization estimates.
* Extend dataset to include weather, soil, or satellite features.
* Add **multi-crop support** (soybean, wheat, etc.).
* Deploy Streamlit app on **Streamlit Cloud, Heroku, or AWS**.

---

##  License

MIT License. Free to use and adapt.

---

👨‍🌾 *Built to support farmers and agri-business with data-driven insights.*
