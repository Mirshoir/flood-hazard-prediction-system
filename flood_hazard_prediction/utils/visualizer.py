import folium
from streamlit_folium import st_folium
import geopandas as gpd
import pandas as pd
import branca.colormap as cm

def display_prediction_map(gdf, preds):
    """
    Merge predictions with spatial data and display on a map.
    Assumes preds dataframe has a 'Predicted' column and the same index order as gdf.
    """

    if gdf.shape[0] != preds.shape[0]:
        # They must have the same number of rows for this simple join
        # Alternatively, could join on IDs if available
        raise ValueError("Spatial data and predictions must have the same number of rows.")

    gdf_copy = gdf.copy()
    gdf_copy['Prediction'] = preds['Predicted']

    # Generate color map based on unique predicted classes
    unique_classes = gdf_copy['Prediction'].unique()
    colormap = cm.StepColormap(
        colors=["green", "yellow", "orange", "red"][:len(unique_classes)],
        index=range(len(unique_classes)+1),
        vmin=0, vmax=len(unique_classes),
        caption="Flood Hazard Prediction"
    )

    def style_function(feature):
        pred_class = feature['properties']['Prediction']
        # Map class to color
        try:
            idx = list(unique_classes).index(pred_class)
            color = colormap.colors[idx]
        except ValueError:
            color = "gray"
        return {
            'fillColor': color,
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.6,
        }

    centroid = gdf_copy.geometry.unary_union.centroid
    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=10)

    folium.GeoJson(
        gdf_copy,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(fields=['Prediction'])
    ).add_to(m)

    colormap.add_to(m)

    st_folium(m, width=700, height=500)
