# 💻 Laptop Price Prediction using Machine Learning

## 📌 Project Overview

Laptop prices depend on many factors such as processor type, RAM size, storage technology, graphics card, display quality, operating system, and brand reputation. Two laptops with similar specifications can often have very different prices due to differences in these components.

The goal of this project is to build an end-to-end Machine Learning model that can accurately predict the price of a laptop based on its specifications.

This project demonstrates the complete machine learning workflow from raw data preprocessing to deployment of the final model using Streamlit.

---

# 🚀 Project Highlights

✅ Data Cleaning and Preprocessing

✅ Feature Engineering

✅ Exploratory Data Analysis (EDA)

✅ Statistical Feature Analysis

✅ Log Transformation of Target Variable

✅ Automated Hyperparameter Tuning

✅ Multiple ML Model Comparison

✅ Automated Best Model Selection

✅ Scikit-Learn Pipelines

✅ Streamlit Deployment

---

# 📂 Dataset

The dataset contains laptop specifications and their corresponding prices.

## Features in Dataset

| Feature | Description |
|----------|-------------|
| Company | Laptop manufacturer |
| TypeName | Laptop category |
| Inches | Screen size in inches |
| ScreenResolution | Display resolution and display type |
| Cpu | Processor details |
| Ram | Installed RAM |
| Memory | Storage configuration |
| Gpu | Graphics card details |
| OpSys | Operating system |
| Weight | Laptop weight |
| Price_euros | Laptop price in Euros |

---

# 🔧 Data Cleaning

Raw datasets are rarely ready for machine learning algorithms. Several preprocessing steps were performed to clean and standardize the data.

---

## 💰 Price Conversion

The original dataset stored prices in Euros.

To make the predictions easier to interpret for Indian users, prices were converted into Indian Rupees.

```python
df['Price_inr'] = df['Price_euros'] * exchange_rate
```

The original `Price_euros` column was then removed.

---

## 🧠 RAM Cleaning

RAM values were originally stored as strings.

Example:

```text
8GB
16GB
32GB
```

Converted into numerical values:

```text
8
16
32
```

This allows machine learning models to treat RAM as a continuous numerical feature.

---

## ⚖️ Weight Cleaning

Weight values contained the `kg` suffix.

Example:

```text
1.37kg
2.10kg
```

Converted into:

```text
1.37
2.10
```

and stored as floating point values.

---

## 💾 Memory Cleaning

The storage column contained highly inconsistent values.

Examples:

```text
128GB SSD
256GB SSD + 1TB HDD
512GB SSD + 2TB HDD
128GB Flash Storage
```

These values were separated into independent storage features:

| SSD | HDD | Hybrid | Flash Storage |
|-----|-----|--------|--------------|
| 128 | 0 | 0 | 0 |
| 256 | 1024 | 0 | 0 |
| 512 | 2048 | 0 | 0 |
| 0 | 0 | 0 | 128 |

This helps the model understand the effect of different storage technologies independently.

---

# 🔨 Feature Engineering

Feature engineering is one of the most important steps in any machine learning project.

Several new features were created from the existing dataset.

---

## 📱 Touchscreen Detection

The `ScreenResolution` column was used to determine whether the laptop supports touchscreen input.

A new binary feature was created:

| Touchscreen | Value |
|------------|-------|
| Yes | 1 |
| No | 0 |

---

## 🎨 IPS Panel Detection

Similarly, another feature was created to identify whether the laptop uses an IPS display panel.

IPS displays usually provide:

- Better color reproduction
- Better viewing angles
- Higher image quality

and therefore often increase laptop prices.

| IPS Panel | Value |
|----------|------|
| Yes | 1 |
| No | 0 |

---

## 🖥️ Resolution Extraction

The original resolution column contained text information mixed with resolution values.

Example:

```text
IPS Panel Full HD 1920x1080
```

Two numerical features were extracted:

```text
X Resolution = 1920
Y Resolution = 1080
```

---

## 🔍 Pixels Per Inch (PPI)

Instead of directly using resolution values, display sharpness was calculated using Pixels Per Inch (PPI).

Formula:

```text
PPI = √(X_resolution² + Y_resolution²) / Screen_Size
```

PPI is a much more informative feature because it combines:

- Screen resolution
- Screen size

For example:

A 1920x1080 display on a 13-inch laptop is much sharper than a 1920x1080 display on a 17-inch laptop.

---

## 🧠 CPU Feature Engineering

The CPU column originally contained hundreds of unique processor names.

Examples:

```text
Intel Core i5 7200U
Intel Core i7 8750H
AMD Ryzen 5 5600H
Intel Celeron
```

These were grouped into simpler categories:

- Intel Core i3
- Intel Core i5
- Intel Core i7
- AMD Processor
- Other Intel Processor

This reduces dimensionality and improves generalization.

---

## 🎮 GPU Feature Engineering

GPU information was simplified into:

- Intel
- Nvidia
- AMD

This captures most of the pricing impact while avoiding unnecessary complexity.

---

## 💻 Operating System Grouping

Operating systems were grouped into:

- Windows
- Mac
- Linux
- Others

---

# 📊 Exploratory Data Analysis (EDA)

EDA was performed to understand patterns in the dataset and identify relationships between features and laptop prices.

---

## Univariate Analysis

Studied the distributions of:

- Price
- RAM
- Weight
- SSD Capacity
- HDD Capacity
- PPI

This helped identify skewed distributions and outliers.

---

## Bivariate Analysis

Studied relationships between features and laptop prices.

Examples include:

- RAM vs Price
- SSD vs Price
- PPI vs Price
- CPU Brand vs Price
- GPU Brand vs Price
- Company vs Price

---

## Multivariate Analysis

Multiple features were analyzed together to understand their combined influence on laptop prices.

Examples:

- CPU + GPU + RAM vs Price
- Company + Type + Price
- Storage + RAM + Price

---

# 📈 Statistical Feature Analysis

For categorical variables, statistical tests were used to measure their importance.

The following features were analyzed using ANOVA F-Test:

- Company
- TypeName
- CPU Brand
- GPU Brand
- Operating System

Features with low statistical significance could potentially be removed to simplify the model.

---

# 📉 Log Transformation of Target Variable

Laptop prices are highly right-skewed.

There are many affordable laptops and relatively few premium laptops costing more than ₹2,00,000.

This skewed distribution can negatively affect regression models.

To solve this problem, a logarithmic transformation was applied:

```python
y = np.log1p(df['Price_inr'])
```

Benefits:

- Reduces skewness
- Stabilizes variance
- Improves model learning
- Reduces influence of outliers

During prediction, the logarithmic transformation is reversed:

```python
predicted_price = np.expm1(prediction)
```

This converts predictions back to Indian Rupees.

---

# ⚙️ Data Preprocessing Pipeline

Scikit-Learn's `ColumnTransformer` was used to automate preprocessing.

---

## Numerical Features

The following preprocessing step was applied:

```text
StandardScaler
```

Applied to:

- RAM
- Weight
- SSD
- HDD
- PPI
- Storage Features

---

## Categorical Features

The following preprocessing step was applied:

```text
OneHotEncoder
```

Applied to:

- Company
- TypeName
- CPU Brand
- GPU Brand
- Operating System

---

# 🤖 Machine Learning Models Used

Multiple regression algorithms were trained and compared.

The following models were used:

- Linear Regression
- ElasticNet Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- AdaBoost Regressor
- Support Vector Regressor (SVR)
- K-Nearest Neighbors Regressor (KNN)
- XGBoost Regressor

---

# 🔍 Hyperparameter Tuning

Hyperparameter tuning was performed using:

```text
GridSearchCV
```

Only the most influential hyperparameters were tuned to reduce computational time while still improving model performance.

---

# 🏆 Automated Model Selection

The training pipeline automatically:

1. Trains all machine learning models.
2. Finds the best hyperparameters for each model.
3. Stores the best version of every model.
4. Evaluates all models on unseen test data.
5. Selects the best performing model.

This creates a completely automated machine learning workflow.

---

# 📏 Model Evaluation Metrics

The models were evaluated using:

## R² Score

Measures how much variance in laptop prices is explained by the model.

Higher values indicate better performance.

---

## Mean Absolute Error (MAE)

Measures the average prediction error in Indian Rupees.

Example:

```text
MAE = ₹18,000
```

means predictions are off by approximately ₹18,000 on average.

---

## Root Mean Squared Error (RMSE)

RMSE penalizes larger prediction errors more heavily.

This metric is useful for identifying large mistakes on premium laptops.

---

# 🌐 Streamlit Deployment

The final model pipeline was saved using:

```python
pickle.dump(best_pipe, open("best_pipe.pkl", "wb"))
```

The Streamlit application performs the following steps automatically:

1. User enters laptop specifications.
2. Saved model pipeline is loaded.
3. Preprocessing is automatically applied.
4. The model predicts the logarithmic price.
5. The prediction is converted back to INR.
6. The estimated laptop price is displayed.

Because preprocessing is stored inside the pipeline, no manual preprocessing is required during deployment.

---

## Example Prediction

### Input

| Feature | Value |
|---------|------|
| Company | Apple |
| TypeName | Ultrabook |
| RAM | 8 GB |
| CPU | Intel Core i5 |
| SSD | 256 GB |
| GPU | Intel |
| Operating System | Mac |
| PPI | 227 |

### Output

```text
Estimated Laptop Price: ₹1,05,000
```

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- XGBoost
- Streamlit
- Pickle

---

# 📁 Project Structure

```text
Laptop-Price-Prediction/
│
├── data/
│   └── laptop.csv
│
├── notebooks/
│   └── laptop_price_prediction.ipynb
│
├── models/
│   └── best_pipe.pkl
│
├── app.py
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

---

# 🚀 Future Improvements

Some possible future improvements include:

- Larger datasets
- Real-time price scraping
- Explainable AI using SHAP values
- Cloud deployment
- Docker containerization

---

# 👨‍💻 Author

**Pritam Halder**

Machine Learning | Data Science | Python | Scikit-Learn

If you found this project useful, feel free to star the repository and connect with me.
