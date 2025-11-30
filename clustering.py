import pandas as pd
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Load data
df = pd.read_parquet("Traffic_Violations.parquet")

# Keep only rows with valid coordinates
df = df[['Latitude', 'Longitude']].dropna()

# Subsample for speed / memory
# df = df.sample(n=10_000, random_state=42)

coords = df[['Latitude', 'Longitude']].values
coords_scaled = StandardScaler().fit_transform(coords)

optics = OPTICS(
    min_samples=5,       # minimum points to form a cluster
    min_cluster_size=0.05 # as fraction of total samples
)
labels = optics.fit_predict(coords_scaled)

# Attach cluster labels
df['cluster'] = labels.astype(str)  # cast to str for discrete colors

# Plot with Plotly (no Mapbox token needed)
fig = px.scatter_map(
    df,
    lat='Latitude',
    lon='Longitude',
    color='cluster',
    hover_data={'cluster': True},
)
fig.write_html("optics_traffic_violations.html")
