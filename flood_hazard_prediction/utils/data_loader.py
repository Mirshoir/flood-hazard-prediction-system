import geopandas as gpd
import zipfile
import tempfile
import os


def load_shapefile(uploaded_file):
    """
    Load shapefile or geojson from uploaded Streamlit file uploader.
    Supports .shp, .geojson, and zipped shapefile (.zip).
    """
    fname = uploaded_file.name.lower()
    if fname.endswith('.geojson'):
        gdf = gpd.read_file(uploaded_file)
    elif fname.endswith('.zip'):
        # Extract zip to temp dir then read .shp inside
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, 'temp.zip')
            with open(zip_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
            # find .shp file inside extracted files
            shp_files = [f for f in os.listdir(tmpdir) if f.endswith('.shp')]
            if not shp_files:
                raise ValueError("No .shp file found in the uploaded zip.")
            shp_path = os.path.join(tmpdir, shp_files[0])
            gdf = gpd.read_file(shp_path)
    elif fname.endswith('.shp'):
        # This is tricky, shapefile is often multiple files.
        # Streamlit uploads only one file at a time, so best practice
        # is to ask user to zip the shapefile or use geojson.
        raise ValueError("Please upload shapefile as a zipped .zip archive.")
    else:
        raise ValueError("Unsupported file format for spatial data.")

    return gdf
