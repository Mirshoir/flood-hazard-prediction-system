import folium
from streamlit_folium import st_folium


def display_leaflet_map(gdf, zoom_start=10):
    """
    Display a Leaflet map with the given GeoDataFrame overlay.
    """
    if gdf.empty:
        return

    # Center map on the geometry centroid or mean coordinates
    centroid = gdf.geometry.unary_union.centroid
    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=zoom_start)

    # Add GeoDataFrame as a GeoJson overlay
    folium.GeoJson(gdf).add_to(m)

    st_folium(m, width=700, height=500)
