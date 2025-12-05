# Raw Data Directory

This directory should contain the following manually downloaded files:

## Required Files:

1. **srtm_30m_elgon.tif**
   - SRTM 30m Digital Elevation Model for Mount Elgon region
   - Download from: USGS EarthExplorer (https://earthexplorer.usgs.gov/)
   - Or use: NASA SRTM data portal
   - Format: GeoTIFF (.tif)
   - Resolution: 30 meters

2. **landslides_inventory.csv**
   - Landslide point locations (Your Y data / target variable)
   - Should contain columns: latitude, longitude, or x, y coordinates
   - Format: CSV with coordinate columns
   - This is your training data with known landslide locations

3. **hwsd_v20_raster.tif**
   - Harmonized World Soil Database (HWSD) raster data
   - Download from: FAO HWSD portal (https://www.fao.org/soils-portal/data-hub/soil-maps-and-databases/harmonized-world-soil-database-v12/en/)
   - Format: GeoTIFF (.tif)
   - Contains soil classification data

## Notes:
- All files should be clipped/reprojected to the Mount Elgon study area
- Ensure coordinate reference system (CRS) is consistent (preferably WGS84 or UTM)
- Once these files are placed here, run `src/01_data_acquisition_and_prep.py` to process them

