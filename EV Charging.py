import pandas as pd

# Define column headers
columns = [
    "start_date", "station_name", "charging_time_hh_mm_ss", "energy_kwh",
    "address_1", "address_2", "city", "state_province", "zip_postal_code",
    "fee", "model_number", "latitude", "longitude", "geopoint"
]

# Load the original CSV file, treating the first line as data if it contains no headers
input_file_path = 'C:\\Users\\prana\\Downloads\\electric-vehicle-charging-stations 2000.csv'  # Replace with your actual file path
data = pd.read_csv(input_file_path, header=None)

# Remove the first row
data = data.iloc[1:]

if set(data.iloc[0]) == set(columns):
    # Drop the first row if it matches the headers
    data = data.iloc[1:]

# Split the single column into multiple columns based on semicolon delimiter
processed_data = data[0].str.split(';', expand=True)

# Assign the defined column names to the DataFrame
processed_data.columns = columns

# Save the processed DataFrame to a new CSV file
output_file_path = 'C:\\Users\\prana\\Downloads\\Dissertation.csv'
processed_data.to_csv(output_file_path, index=False)

# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import folium
from folium.plugins import HeatMap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'C:\\Users\\prana\\Downloads\\nnn\\NLP\\Dissertation.csv' 
data = pd.read_csv(file_path)

# 1. Basic Data Overview
print("Dataset Shape:", data.shape)  # Rows and Columns
print("Column Info:")
print(data.info())  # Data types and non-null counts
print("Summary Statistics:")
print(data.describe())  # Descriptive statistics for numerical features


# Check for missing values
missing_values = data.isnull().sum()
print("\nMissing Values:\n", missing_values)

# 2. Exploratory Data Analysis (EDA)
# Visualize missing data
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

#data Cleaning
# Drop the specified columns
data = data.drop(['fee', 'state_province', 'city'], axis=1)

# Verify the columns have been removed
print("Remaining Columns:", data.columns)

# Plot distributions for numerical columns
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(data[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()


# Select only numerical columns for correlation matrix
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = data[numerical_columns].corr()

# Plot the correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Visualizing Geographic Distribution of EV Charging Stations (using latitude and longitude)
map = folium.Map(location=[data['latitude'].mean(), data['longitude'].mean()], zoom_start=10)
heat_data = [[row['latitude'], row['longitude']] for index, row in data.iterrows()]
HeatMap(heat_data).add_to(map)
map.save("ev_charger_distribution.html")

from IPython.display import IFrame
IFrame("ev_charger_distribution.html", width=800, height=600)

# Bar plots for categorical features
categorical_columns = data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    plt.figure(figsize=(12, 6))
    sns.countplot(y=col, data=data, palette='viridis')
    plt.title(f"Count of {col}")
    plt.show()

# Handle missing values (example: fillna with median for numerical, mode for categorical)
data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].median())
for col in categorical_columns:
    data[col] = data[col].fillna(data[col].mode()[0])

print(data)

from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

# Separate numeric columns and categorical columns
numeric_cols = data.select_dtypes(include=[np.number]).columns
categorical_cols = data.select_dtypes(exclude=[np.number]).columns

# Handle missing values in numeric columns by filling with the mean
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Handle missing values in categorical columns by filling with the most frequent value
for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

# Check if any missing values remain
print(data.isnull().sum())


# Encode categorical variables
label_enc = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    data[col] = label_enc.fit_transform(data[col])

# Automatically select numeric columns for scaling
numeric_features = data.select_dtypes(include=[np.number]).columns

# Scale numeric features
scaler = StandardScaler()
data[numeric_features] = scaler.fit_transform(data[numeric_features])

# Support Vector Regressor (SVM)
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Define features and target
X = data.drop(['energy_kwh'], axis=1) 
y = data['energy_kwh']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM
svm_model = SVR(kernel='rbf')  # You can try different kernels such as 'linear' or 'poly'
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
print("Support Vector Regressor (SVM) Results:")
print(f"Mean Squared Error: {mean_squared_error(y_test, svm_pred)}")
print(f"R^2 Score: {r2_score(y_test, svm_pred)}")

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print("Random Forest Results:")
print(f"Mean Squared Error: {mean_squared_error(y_test, rf_pred)}")
print(f"R^2 Score: {r2_score(y_test, rf_pred)}")

# Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
print("Gradient Boosting Results:")
print(f"Mean Squared Error: {mean_squared_error(y_test, gb_pred)}")
print(f"R^2 Score: {r2_score(y_test, gb_pred)}")

from sklearn.neighbors import NearestNeighbors
import numpy as np

# Ensure we have a column representing chargers
if "geopoint" not in data.columns:
    raise ValueError("Dataset does not have a column representing charger usage (e.g., 'energy_kwh').")

# Charger locations: where energy_kwh > 0
charger_locations = data[data["geopoint"] > 0][["latitude", "longitude"]]

# Potential locations: all rows (replace if filtering is needed)
potential_locations = data[["latitude", "longitude"]]

# Handle empty charger locations
if charger_locations.empty:
    raise ValueError("No charger locations found in the dataset.")

# Compute distances to the nearest charger using Nearest Neighbors
nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(charger_locations)
distances, indices = nbrs.kneighbors(potential_locations)

# Add the distances to the dataset
data["distance_to_nearest_charger"] = distances

# Simulate columns if missing
if "population_density" not in data.columns:
    print("Simulating 'population_density' column.")
    data["population_density"] = np.random.randint(100, 10000, size=len(data))

if "ev_adoption_rate" not in data.columns:
    print("Simulating 'ev_adoption_rate' column.")
    data["ev_adoption_rate"] = np.random.uniform(0.01, 0.3, size=len(data))


# Define an Underserved Score 
data["underserved_score"] = (
    0.4 * data["population_density"]
    + 0.3 * data["ev_adoption_rate"]
    + 0.3 * data["distance_to_nearest_charger"]
)

# Normalize the underserved score for comparison
data["underserved_score"] = (
    (data["underserved_score"] - data["underserved_score"].min())
    / (data["underserved_score"].max() - data["underserved_score"].min())
)

# Check the results
print(data[["latitude", "longitude", "distance_to_nearest_charger", "underserved_score"]].head())

# Visualize the distribution of the underserved score
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.hist(data["underserved_score"], bins=30, color="blue", alpha=0.7)
plt.title("Distribution of Underserved Scores")
plt.xlabel("Underserved Score")
plt.ylabel("Frequency")
plt.show()

#test
test_EV = np.array([[
    -0.047283,  # start_date
    1.244225,   # station_name
    -1.330790,  # charging_time_hh_mm_ss
    -1.294701,  # address_1
    -0.267709,  # address_2
    -1.271292,  # zip_postal_code
    -0.055951,  # model_number
    -0.339454,  # latitude
    1.039291,   # longitude
    -0.339454,  # geopoint
]])

predicted_energy_kwh = rf_model.predict(test_EV)
print("Predicted Energy (kWh):", predicted_energy_kwh)




*******************************************************************************************************************************************

# Using the test dataset for energy prediction

# Load the test dataset to inspect its structure
import pandas as pd

# File path for the uploaded test dataset
test_file_path = "C:\\Users\\prana\\Downloads\\nnn\\NLP\\Dissertation test dataset.csv"

# Load the test dataset
test_data = pd.read_csv(test_file_path)

# Display the first few rows of the test dataset for inspection
test_data.head(), test_data.info()

from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

#data Cleaning
# Drop the specified columns
test_data = test_data.drop(['fee', 'state_province', 'city'], axis=1)

# Copy the test dataset to avoid altering the original
test_data_processed = test_data.copy()

# Separate numeric and categorical columns
numeric_cols = test_data_processed.select_dtypes(include=[np.number]).columns
categorical_cols = test_data_processed.select_dtypes(exclude=[np.number]).columns

# Handle missing values
# Fill numeric columns with the mean
test_data_processed[numeric_cols] = test_data_processed[numeric_cols].fillna(test_data_processed[numeric_cols].mean())

# Fill categorical columns with the mode
for col in categorical_cols:
    test_data_processed[col] = test_data_processed[col].fillna(test_data_processed[col].mode()[0])

# Encode categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    test_data_processed[col] = le.fit_transform(test_data_processed[col])
    label_encoders[col] = le  # Save the encoder for reference

# Scale numerical columns (excluding the target variable if present)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(test_data_processed[numeric_cols])

# Replace scaled numeric columns in the DataFrame
test_data_processed[numeric_cols] = scaled_features

# Drop the target column 'energy_kwh' for predictions
X_test_dataset = test_data_processed.drop(columns=["energy_kwh"])

# Load rf_model and predict
rf_model_predictions = rf_model.predict(X_test_dataset)

# Add predictions back to the test dataset
test_data["Predicted_energy_kwh"] = rf_model_predictions

# Display the updated test dataset with predictions
test_data.head()
