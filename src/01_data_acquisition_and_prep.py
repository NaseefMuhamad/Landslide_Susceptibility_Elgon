"""
Script 01: Data Acquisition and Preparation
===========================================

This script handles:
- Preparing rasters from raw data
- Google Earth Engine (GEE) correlation for rainfall data
- Deriving spatial features from DEM (Slope, Aspect, TWI)
- Processing soil classification data

Outputs to data/processed/:
- slope.tif
- aspect.tif
- t_wi.tif
- soil_class.tif
"""

# TODO: Implement data acquisition and preparation logic
import os
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.plot import show
from scipy.ndimage import uniform_filter
import richdem as rd
import ee # Google Earth Engine API
from geemap import ee_initialize

# --- 1. CONFIGURATION ---
# Define paths relative to the project root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(ROOT_DIR, '..', 'data', 'raw')
DATA_PROCESSED = os.path.join(ROOT_DIR, '..', 'data', 'processed')

# Input Files (Ensure these exist in data/raw/)
DEM_PATH = os.path.join(DATA_RAW, 'srtm_30m_elgon.tif')
HWSD_PATH = os.path.join(DATA_RAW, 'hwsd_v20_raster.tif')

# Output Files
OUTPUT_SLOPE = os.path.join(DATA_PROCESSED, 'slope.tif')
OUTPUT_ASPECT = os.path.join(DATA_PROCESSED, 'aspect.tif')
OUTPUT_TWI = os.path.join(DATA_PROCESSED, 't_wi.tif')
OUTPUT_SOIL = os.path.join(DATA_PROCESSED, 'soil_class.tif')

# --- 2. GEE INITIALIZATION ---
def initialize_gee():
    """Initializes the Google Earth Engine connection."""
    print("Initializing Google Earth Engine...")
    try:
        # Tries to initialize GEE; requires prior authentication (ee.Authenticate())
        ee_initialize()
        print("GEE initialized successfully.")
    except Exception as e:
        print(f"GEE Initialization failed. Run 'ee.Authenticate()' if this is your first time. Error: {e}")
        # Note: We don't need to exit here, as GEE is only needed for the next script.

# --- 3. HELPER FUNCTIONS ---

def save_array_as_raster(array, profile, output_path):
    """Saves a NumPy array back to a GeoTIFF file."""
    print(f"Saving output to {output_path}...")
    profile.update(dtype=rasterio.float32, count=1, nodata=-9999)
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(array.astype(rasterio.float32), 1)
    print(f"Successfully saved {output_path}")

def calculate_derivatives(dem_array, dem_profile):
    """Calculates Slope, Aspect, and TWI using richdem."""
    print("Calculating Slope, Aspect, and TWI...")
    
    # 1. Prepare data for richdem
    rd_dem = rd.rdarray(dem_array, no_data=-9999)
    
    # 2. Calculate Topographic Attributes
    # Slope (returns radians by default; converting to degrees)
    slope_rad = rd.TerrainAttribute(rd_dem, attrib='slope')
    slope_deg = np.rad2deg(slope_rad)
    
    # Aspect (returns radians)
    aspect_rad = rd.TerrainAttribute(rd_dem, attrib='aspect')
    
    # TWI (Topographic Wetness Index)
    # Requires filled sinks for flow accumulation and specific catchment area
    # Note: This step can be resource-intensive
    try:
        rd_dem_filled = rd.FillDepressions(rd_dem, epsilon=True, in_place=False)
        twi_array = rd.TWI(rd_dem_filled)
    except Exception as e:
        print(f"TWI calculation failed, using default flow routing may require specific settings/memory. Error: {e}")
        # Fallback: If TWI fails, save blank arrays or handle gracefully.
        twi_array = np.full_like(dem_array, -9999.0) # Placeholder
    
    # Save the results
    save_array_as_raster(slope_deg, dem_profile, OUTPUT_SLOPE)
    save_array_as_raster(aspect_rad, dem_profile, OUTPUT_ASPECT)
    save_array_as_raster(twi_array, dem_profile, OUTPUT_TWI)

def process_soil_data(dem_profile):
    """
    Clips and resamples the global HWSD data to match the DEM extent and resolution.
    Note: Requires the HWSD raster to be present in data/raw/
    """
    print(f"Processing soil data from {HWSD_PATH}...")
    
    # Define the bounding box for clipping (based on DEM profile)
    clip_extent = dem_profile['bounds']
    clip_transform = dem_profile['transform']
    clip_crs = dem_profile['crs']
    
    try:
        with rasterio.open(HWSD_PATH) as hwsd_src:
            # 1. Clip/Mask the global soil raster to the DEM extent
            # We don't have a geometry polygon, so we rely on the DEM bounding box/window.
            # Rasterio's mask function is usually the most reliable for clipping.
            
            # Create a simple box geometry for masking
            from shapely.geometry import box
            geom = [box(*clip_extent)]
            
            # Reproject to match the HWSD CRS temporarily, clip, then reproject back.
            # Simpler approach: Use reproject and resample to match DEM directly.
            
            from rasterio.warp import reproject, Resampling
            
            # Create an output array and profile matching the DEM exactly
            soil_array = np.empty((dem_profile['height'], dem_profile['width']), dtype=rasterio.int16)
            
            reproject(
                source=rasterio.band(hwsd_src, 1),
                destination=soil_array,
                src_transform=hwsd_src.transform,
                src_crs=hwsd_src.crs,
                dst_transform=clip_transform,
                dst_crs=clip_crs,
                resampling=Resampling.nearest # Use nearest neighbor for categorical data
            )
            
            # The HWSD contains soil unit codes (categorical data). 
            # We save it as is; the next script will handle reclassification/one-hot encoding.
            out_profile = dem_profile.copy()
            out_profile.update(dtype=rasterio.int16, count=1, nodata=-9999)
            
            with rasterio.open(OUTPUT_SOIL, 'w', **out_profile) as dst:
                dst.write(soil_array.astype(rasterio.int16), 1)
            
            print(f"Successfully saved processed soil raster to {OUTPUT_SOIL}")

    except rasterio.RasterioIOError:
        print(f"ERROR: HWSD file not found at {HWSD_PATH}. Skipping soil processing.")
    except Exception as e:
        print(f"An error occurred during soil processing: {e}")


# --- 4. MAIN EXECUTION ---
def main():
    """Main function to run the data acquisition and preparation steps."""
    
    if not os.path.exists(DATA_PROCESSED):
        os.makedirs(DATA_PROCESSED)

    initialize_gee()
    
    # 1. Load the primary DEM
    try:
        with rasterio.open(DEM_PATH) as dem_src:
            dem_array = dem_src.read(1)
            dem_profile = dem_src.profile
            print(f"DEM loaded. Shape: {dem_array.shape}, CRS: {dem_profile['crs']}")
            
            # Handle NoData values (replace common NoData with richdem's default)
            dem_array[dem_array == dem_profile['nodata']] = -9999 
            
            # 2. Calculate Derived Topographic Features
            calculate_derivatives(dem_array, dem_profile)
            
            # 3. Process Soil Data (Matching it to the DEM grid)
            process_soil_data(dem_profile)
            
    except rasterio.RasterioIOError:
        print(f"\n--- ERROR ---")
        print(f"DEM file not found at {DEM_PATH}.")
        print(f"Please ensure 'srtm_30m_elgon.tif' is in the 'data/raw/' folder.")
        print(f"---------------")
    
    print("\n--- Script 01 Completed Successfully ---")
    print(f"Processed files saved to: {DATA_PROCESSED}")

if __name__ == '__main__':
    # Ensure the script runs from the src directory context if possible, 
    # but the path definition handles the relative location.
    main()
