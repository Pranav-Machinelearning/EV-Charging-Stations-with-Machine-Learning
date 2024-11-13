# Import necessary libraries.
import pandas as pd

# Load the CSV file
path = "C:\\Users\\prana\\Downloads\\nnn\\NLP\\Electric_Vehicle_Charging_Stations.csv"
EV_Data = pd.read_csv(path)

# Show the initial rows and essential information to comprehend the structure of the dataset.
print(EV_Data.head())
print(EV_Data.info())

""" preprcoess the Data - 
1.Drop unnecessary columns.
2. Convert categorical features like City to numeric codes for modeling.
3. Handle any missing values if present.  """

""" Drop columns that may not be useful for the modeling process - Removes columns that don’t provide useful information for the model. 
'Station Name', 'Street Address', and 'EV Other Info' are dropped. """

# code explains
""" axis=1: Specifies that we're dropping columns (instead of rows).
errors='ignore': Ensures that if a column doesn’t exist in the dataset, it won’t raise an error (useful for making the code more adaptable). """

EV_Data = EV_Data.drop(['Station Name', 'Street Address', 'EV Other Info'], axis=1, errors='ignore')  

# Handle categorical EV_Data if needed
# Encodes categorical values in the 'City' column as numeric values, which are easier for machine learning models to interpret.

# code explains
""" .cat.codes: Assigns a unique numeric code to each category (e.g., each city) in the column. This replaces city names with numeric codes (e.g., 0, 1, 2)
astype('category'): Converts the column into a category datatype. """

if 'City' in EV_Data.columns:
    EV_Data['City'] = EV_Data['City'].astype('category').cat.codes

# Check and fill missing values
EV_Data.fillna(0, inplace=True)

# Use Feature engineering.
"""retrieved the "New Georeferenced Column"'s latitude and longitude information and divided it up into new columns. 
 These coordinates will be helpful in identifying underserved areas that may benefit from additional EV chargers."""

 # Extract Latitude and Longitude from 'New Georeferenced Column'
""" The str.extract() function uses a regular expression to capture the values inside the POINT column as separate longitude and latitude components.
The resulting columns are converted to numeric values, which is helpful in case of invalid entries (which would become NaN). """

EV_Data[['Longitude', 'Latitude']] = EV_Data['New Georeferenced Column'].str.extract(r'POINT \(([^ ]+) ([^ ]+)\)')

# Convert the Longitude and Latitude columns to numeric types

EV_Data['Longitude'] = pd.to_numeric(EV_Data['Longitude'], errors='coerce')
EV_Data['Latitude'] = pd.to_numeric(EV_Data['Latitude'], errors='coerce')

# Display the updated dataset with separate Latitude and Longitude columns
print(EV_Data.head())

from sklearn.preprocessing import StandardScaler
import folium

# Set map center to the mean coordinates
center_lat, center_lon = EV_Data['Latitude'].mean(), EV_Data['Longitude'].mean()
ev_map = folium.Map(location=[center_lat, center_lon], zoom_start=10)

# Add each charging station as a marker on the map
for _, row in EV_Data.iterrows():
    popup_text = (f"City: {row['City']}, Level 1: {row['EV Level1 EVSE Num']}, "
                  f"Level 2: {row['EV Level2 EVSE Num']}, DC Fast: {row['EV DC Fast Count']}")
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=popup_text
    ).add_to(ev_map)

# Save the map as HTML
ev_map.save('ev_charging_stations_map.html')

from IPython.display import IFrame

# Display the saved HTML map file in the notebook
IFrame('ev_charging_stations_map.html', width=1000, height=500)

# ************************************************************************************************************************************

from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import folium

# Replace 'NONE' with NaN and convert columns to float
EV_Data[['EV Level1 EVSE Num', 'EV Level2 EVSE Num', 'EV DC Fast Count']] = (
    EV_Data[['EV Level1 EVSE Num', 'EV Level2 EVSE Num', 'EV DC Fast Count']].replace('NONE', np.nan).astype(float)
)

# Impute missing values with the mean of each column
imputer = SimpleImputer(strategy='mean')
EV_Data[['EV Level1 EVSE Num', 'EV Level2 EVSE Num', 'EV DC Fast Count', 'Latitude', 'Longitude']] = imputer.fit_transform(
    EV_Data[['EV Level1 EVSE Num', 'EV Level2 EVSE Num', 'EV DC Fast Count', 'Latitude', 'Longitude']]
)

# Select features and standardize them
features = EV_Data[['Latitude', 'Longitude', 'EV Level1 EVSE Num', 'EV Level2 EVSE Num', 'EV DC Fast Count']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=10, random_state=42)
EV_Data['Cluster'] = kmeans.fit_predict(features_scaled)
# Drop rows with any NaN values in the selected columns
EV_Data.dropna(subset=['Latitude', 'Longitude', 'EV Level1 EVSE Num', 'EV Level2 EVSE Num', 'EV DC Fast Count'], inplace=True)

# Proceed with clustering as before
features = EV_Data[['Latitude', 'Longitude', 'EV Level1 EVSE Num', 'EV Level2 EVSE Num', 'EV DC Fast Count']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=10, random_state=42)
EV_Data['Cluster'] = kmeans.fit_predict(features_scaled)

# Set map center based on mean coordinates
center_lat, center_lon = EV_Data['Latitude'].mean(), EV_Data['Longitude'].mean()
ev_map = folium.Map(location=[center_lat, center_lon], zoom_start=10)

# Add each charging station as a marker on the map
for _, row in EV_Data.iterrows():
    popup_text = (f"City: {row['City']}, Level 1: {row['EV Level1 EVSE Num']}, "
                  f"Level 2: {row['EV Level2 EVSE Num']}, DC Fast: {row['EV DC Fast Count']}, Cluster: {row['Cluster']}")
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=popup_text
    ).add_to(ev_map)

# Add cluster centroids to map
for cluster_id in EV_Data['Cluster'].unique():
    cluster_data = EV_Data[EV_Data['Cluster'] == cluster_id]
    cluster_center = [cluster_data['Latitude'].mean(), cluster_data['Longitude'].mean()]
    folium.Marker(
        location=cluster_center,
        popup=f"Cluster {cluster_id} Center",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(ev_map)

# Save updated map with clusters
ev_map.save('ev_charging_stations_clustered_map.html')

# Display the clustered HTML map file in the notebook
IFrame('ev_charging_stations_clustered_map.html', width=1000, height=500)

# ************************************************************************************************************************

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np


# Create a new 'Future Demand' column as a proxy for demand using available EV_Data
# Here we add the count of chargers as a basic example of demand; adjust this as needed based on actual EV_Data availability

EV_Data['Future Demand'] = (
    0.5 * EV_Data['EV Level1 EVSE Num'] + 
    1.0 * EV_Data['EV Level2 EVSE Num'] + 
    1.5 * EV_Data['EV DC Fast Count']
)

# Fill NaN values in the 'Future Demand' column with the median value as a simple imputation
EV_Data['Future Demand'].fillna(EV_Data['Future Demand'].median(), inplace=True)

# Define features (X) and target (y) for prediction
X = EV_Data[['Latitude', 'Longitude', 'City', 'EV Level1 EVSE Num', 'EV Level2 EVSE Num', 'EV DC Fast Count']]
y = EV_Data['Future Demand']

# Split the EV_Data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Initialize and train a Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=40)
rf_model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Output the trained model’s performance
print("Model Performance:")
print("Mean Squared Error (MSE):", mse)

# *********************************************************************************************************************
# Example new input EV_Data (replace these values with realistic EV_Data as needed)
new_data = pd.DataFrame({
    'Latitude': [34.05],  # Example latitude
    'Longitude': [-118.25],  # Example longitude
    'City': [1],  # Numeric code for the city (assumes pre-encoding)
    'EV Level1 EVSE Num': [10],  # Example number of level 1 chargers
    'EV Level2 EVSE Num': [5],   # Example number of level 2 chargers
    'EV DC Fast Count': [2]      # Example number of DC Fast chargers
})

# Preprocess the new EV_Data in the same way as the training EV_Data (scaling, filling NAs if needed)

# Predict future demand using the trained model
predicted_demand = rf_model.predict(new_data)

print("Predicted Future Demand for the new input:", predicted_demand[0])

