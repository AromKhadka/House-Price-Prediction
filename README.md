# Nepal House Price Prediction (MIT604 Term Paper)

An end-to-end **Nepal House Price Prediction System** that takes user inputs (**location + area**) and predicts an **estimated price range in NPR**. The project includes a complete machine learning pipeline:

**Data Cleaning → Feature Engineering → Model Training → Model Selection → Prediction → Streamlit UI**

---

## ✨ Features

- Converts Nepali land-unit format (e.g., `1-0-0-0 Aana`) into numeric **square feet**
- Extracts location features such as **city** and **town** from address text
- Parses road information (e.g., `"20 Feet / Blacktopped"`) into numeric **road_width**
- Creates a **road accessibility proxy feature** (`road_distance_score`)
- Trains and compares:
  - **Linear Regression** (baseline model)
  - **Random Forest Regressor** (non-linear regression model)
- Automatically selects the best-performing model using **R² Score**
- Predicts a **price range** instead of a single value
- Displays **Confidence Level** (Low / Medium / High)
- User-friendly **Streamlit Web Interface**

---

## 📁 Project Structure

```text
├── app/
│   └── streamlit_app.py
├── data/
│   └── raw.csv
├── models/
│   ├── best_model.pkl
│   └── metadata.json
├── src/
│   ├── __init__.py
│   ├── feature_engineering.py
│   ├── location_resolver.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── train.py
├── requirements.txt
└── train.py
```

> Note: `.venv/` and `__pycache__/` directories are excluded from version control and should not be uploaded to GitHub.

---

## 🔧 Data Cleaning

### Dataset Challenges

The original dataset contains:

- `price` values requiring numeric conversion
- `address` fields containing location information
- `area` values stored in Nepali land-unit format such as `1-0-0-0 Aana`
- Missing values in some property attributes

### Data Cleaning Process

The preprocessing pipeline:

1. Standardizes column names (lowercase and trimmed spaces)
2. Converts `price` values into numeric format
3. Converts Nepali area units into square feet
4. Extracts `town` information from address strings
5. Extracts `road_width` from road descriptions
6. Removes records with missing essential values (`price` and `area`)
7. Fills missing bedroom and bathroom values using median imputation

### Area Conversion

The project uses:

**1 Aana ≈ 342.25 sq ft**

Example:

```text
1-0-0-0 Aana → 342.25 sq ft
```

---

## 🧠 Feature Engineering

The following features are used for model training:

- `area_sqft`
- `bedrooms`
- `bathrooms`
- `city`
- `town`
- `road_distance_score`

### Road Accessibility Score

Since distance to the main road is not directly available, a proxy feature called `road_distance_score` is created.

Logic:

- Wider road = Better accessibility
- Better accessibility = Lower score

Scale:

| Score | Meaning |
|---------|---------|
| 1 | Excellent road access |
| 2 | Good road access |
| 3 | Average road access |
| 4 | Limited road access |
| 5 | Poor road access |

If road width is unavailable, the score is estimated using location and area information.

---

## 🤖 Model Training and Selection

### Models Evaluated

#### Linear Regression

- Simple and interpretable
- Used as a baseline model

#### Random Forest Regressor

- Handles non-linear relationships
- Typically performs better on real estate data

### Train-Test Split

The dataset is divided into:

- 70% Training Data
- 30% Testing Data

### Model Selection

The performance of each model is evaluated using:

- **R² Score**

The model with the highest R² score is automatically selected and saved as:

```text
models/best_model.pkl
```

Additional model information is stored in:

```text
models/metadata.json
```

---

## 💰 Price Range Prediction

Instead of returning a single predicted value, the system provides a price range.

### Range Calculation

After training:

1. Residual errors are calculated
2. Standard deviation of residuals (`sigma`) is computed

Prediction range:

```text
Minimum Price = Prediction − Sigma
Maximum Price = Prediction + Sigma
```

This provides a more realistic estimate for real-world housing prices.

---

## 📊 Confidence Level

The confidence level is based on prediction uncertainty.

| Confidence | Interpretation |
|------------|---------------|
| High | Low uncertainty |
| Medium | Moderate uncertainty |
| Low | High uncertainty |

Confidence is determined using the ratio between model error (`sigma`) and predicted price.

---

## 🗺 Location Handling

### User Input

Example:

```text
Budhanilkantha
```

The system resolves:

```text
Town → Budhanilkantha
City → Kathmandu
```

### Fallback Logic

If a location is not available in training data:

1. Fall back to city-level information
2. If city is unavailable, use a global `"unknown"` category

This prevents prediction failures and ensures a result is always returned.

---

## 🖥 Streamlit Application

### Inputs

- Location
- Area (Square Feet)

### Outputs

- Estimated Price Range (NPR)
- Confidence Level
- Selected Model
- Prediction Explanation

The explanation considers:

- Location demand
- Property size
- Road accessibility
- Valley vs non-valley influence

---

## ⚙️ Requirements

- Python 3.10+
- Streamlit
- Pandas
- NumPy
- Scikit-Learn
- Joblib

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Step 1: Create Virtual Environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS**

```bash
python -m venv .venv
source .venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Train the Model

```bash
python train.py
```

### Step 4: Launch Streamlit Application

```bash
streamlit run app/streamlit_app.py
```

---

## 🧪 Example

### Input

```text
Location: Budhanilkantha
Area: 1200 sq ft
```

### Output

```text
Estimated Price Range:
NPR 1.8 Crore – NPR 2.3 Crore

Confidence:
Medium

Model:
Random Forest Regressor
```

---

## 📚 Academic Context

This project was developed as a **Term Paper (MIT604)** for the **Master of Information Technology (MIT)** program under **Tribhuvan University (TU)**.

The project demonstrates:

- Data preprocessing
- Feature engineering
- Machine learning model development
- Model evaluation
- Predictive analytics
- Web-based deployment using Streamlit

---

## 👨‍💻 Author

**Arom Khadka**

Master of Information Technology (MIT)  
Tribhuvan University (TU)

---