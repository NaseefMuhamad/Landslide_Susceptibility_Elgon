"""
Script 02: Feature Engineering and Sampling
============================================

This script handles:
- Sampling of X (features) and Y (target) data
- Feature engineering for CNN-LSTM model
- Combining spatial and temporal features
- Creating training/validation/test datasets

Outputs:
- final_dataset.csv (combined X and Y data)
"""

# TODO: Implement feature engineering and sampling logic

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.sample import sample_gen
from shapely.geometry import Point
from tqdm import tqdm

# --- 1. CONFIGURATION ---
# Define paths relative to the project root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(ROOT_DIR, '..', 'data', 'raw')
DATA_PROCESSED = os.path.join(ROOT_DIR, '..', 'data', 'processed')

INVENTORY_PATH = os.path.join(DATA_RAW, 'landslides_inventory.csv')
OUTPUT_FILE = os.path.join(ROOT_DIR, '..', 'final_dataset.csv') # Saved in project root

# Processed Static Raster Features from Script 01
RASTER_FEATURES = {
    'elevation': os.path.join(DATA_RAW, 'srtm_30m_elgon.tif'), 
    'slope': os.path.join(DATA_PROCESSED, 'slope.tif'),
    'aspect': os.path.join(DATA_PROCESSED, 'aspect.tif'),
    'twi': os.path.join(DATA_PROCESSED, 't_wi.tif'),
    'soil_class': os.path.join(DATA_PROCESSED, 'soil_class.tif'),
}

# --- 2. CORE FUNCTIONS ---

def create_non_landslide_samples(df_landslide, n_ratio=5):
    """
    Creates 'Non-Landslide' samples (Y=0) by randomly sampling locations 
    within the Mt. Elgon bounding box, ensuring a balanced dataset.
    """
    print(f"⏳ Creating {n_ratio}x negative samples...")
    
    # Use the extent of the DEM to define the sampling area
    try:
        with rasterio.open(RASTER_FEATURES['elevation']) as src:
            bounds = src.bounds
            crs = src.crs
    except Exception:
        # Fallback if DEM isn't found (using approximate Mt. Elgon coordinates)
        print("⚠️ DEM not found. Using hardcoded Mt. Elgon bounds.")
        bounds = (34.0, 0.5, 35.0, 1.5) # [min_lon, min_lat, max_lon, max_lat]
        crs = "EPSG:4326"

    # Define the number of negative samples
    n_landslides = len(df_landslide)
    n_samples = n_landslides * n_ratio
    
    min_lon, min_lat, max_lon, max_lat = bounds
    
    # Randomly generate points within the bounds (in degrees)
    lats = np.random.uniform(min_lat, max_lat, n_samples)
    lons = np.random.uniform(min_lon, max_lon, n_samples)
    
    # Assign a random date from the positive samples to the negative samples 
    # (to enable temporal feature extraction later)
    dates = df_landslide['date_of_event'].sample(n_samples, replace=True).values

    df_negative = pd.DataFrame({
        'latitude': lats,
        'longitude': lons,
        'date_of_event': dates,
        'landslide': 0  # Target variable: No Landslide
    })
    
    # 3. Combine positive and negative samples
    df_landslide['landslide'] = 1 # Target variable: Landslide
    final_df = pd.concat([df_landslide, df_negative], ignore_index=True)
    
    print(f"✅ Created {n_landslides} positive and {n_samples} negative samples.")
    return final_df.reset_index(drop=True)

def sample_static_features(df, features_dict):
    """
    Samples static raster features at the location of every point in the DataFrame.
    """
    df_result = df.copy()
    
    # 1. Convert DataFrame points to a GeoDataFrame (assuming input is in WGS84/EPSG:4326)
    points_gdf = gpd.GeoDataFrame(
        df_result, 
        geometry=gpd.points_from_xy(df_result.longitude, df_result.latitude), 
        crs="EPSG:4326"
    )
    
    # 2. Get the target CRS and transformation from one of the processed rasters
    try:
        with rasterio.open(features_dict['elevation']) as src:
            target_crs = src.crs
            
        points_gdf = points_gdf.to_crs(target_crs)
        coords = [(x, y) for x, y in zip(points_gdf.geometry.x, points_gdf.geometry.y)]
        
    except Exception as e:
        print(f"❌ Error during CRS transformation. Ensure DEM is valid. Error: {e}")
        return df_result # Return without sampling

    # 3. Sample each feature raster
    for feature_name, path in tqdm(features_dict.items(), desc="Sampling Static Features"):
        if not os.path.exists(path):
            print(f"⚠️ Skipping {feature_name}: file not found at {path}")
            continue

        try:
            with rasterio.open(path) as src:
                # Use rasterio's generator for efficient sampling
                sampled_values = [val[0] for val in src.sample(coords)]
                df_result[feature_name] = sampled_values
        except Exception as e:
            print(f"❌ Error sampling {feature_name}: {e}")
            
    # Remove samples that fell on 'No Data' areas after sampling
    initial_count = len(df_result)
    df_result.replace(to_replace=-9999, value=np.nan, inplace=True)
    df_result.dropna(subset=list(features_dict.keys()), inplace=True)
    
    print(f"✅ Spatial sampling complete. Removed {initial_count - len(df_result)} points with NoData.")
    return df_result.drop(columns=['geometry'], errors='ignore')

def extract_temporal_features(df):
    """
    Creates the rainfall time-series sequence for the LSTM input.
    
    NOTE: In a production environment, this function would connect to the 
    CHIRPS asset in GEE, filter by the date_of_event, and extract the 
    precipitation value for the 7 days prior to the event date.
    
    Here, we use a placeholder that generates a random sequence 
    (mimicking 7 days of rainfall in mm) to preserve the required 
    data structure for Script 03.
    """
    print("⏳ Generating temporal rainfall sequence (Placeholder)...")
    
    n_samples = len(df)
    n_timesteps = 7
    
    # Generate a dummy 7-day rainfall sequence (in mm) for each point
    # We use some variation to ensure the model has features to learn from.
    rainfall_data = np.random.uniform(low=0, high=50, size=(n_samples, n_timesteps))
    
    # Add columns to the DataFrame
    for i in range(n_timesteps):
        df[f'rainfall_day_T_minus_{n_timesteps-i}'] = rainfall_data[:, i]

    print(f"✅ Generated {n_timesteps}-day rainfall sequence for {n_samples} points.")
    return df

# --- 3. MAIN EXECUTION ---

def main():
    """Orchestrates the feature engineering and sampling workflow."""
    
    if not os.path.exists(INVENTORY_PATH):
        print(f"❌ Landslide inventory not found at {INVENTORY_PATH}. Please check Step 2.")
        return

    # 1. Load Landslide Inventory
    df_landslide = pd.read_csv(INVENTORY_PATH)
    
    # Ensure date_of_event exists (critical for negative sampling and TS extraction)
    if 'date_of_event' not in df_landslide.columns:
        print("⚠️ 'date_of_event' column missing. Assigning dummy dates for placeholder TS.")
        df_landslide['date_of_event'] = pd.to_datetime('2018-01-01')

    # 2. Create Balanced Dataset (Positive + Negative Samples)
    df_combined = create_non_landslide_samples(df_landslide)

    # 3. Sample Static Raster Features
    df_features = sample_static_features(df_combined, RASTER_FEATURES)
    
    # 4. Extract Temporal Features (Placeholder)
    df_final = extract_temporal_features(df_features)
    
    # 5. Save Final Dataset
    # Select only the relevant features and the target variable
    final_cols = ['latitude', 'longitude', 'landslide'] + list(RASTER_FEATURES.keys()) + [col for col in df_final.columns if 'rainfall' in col]
    df_final = df_final[final_cols]
    
    df_final.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n--- Script 02 Completed Successfully ---")
    print(f"Final training dataset saved to: {OUTPUT_FILE}")
    print(f"Final dataset shape: {df_final.shape}")

if __name__ == '__main__':
    main()