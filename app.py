import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
from urllib.request import urlopen, Request

from skyfield.api import load, wgs84
from skyfield.iokit import parse_tle_file
from skyfield.framelib import itrs
from datetime import timedelta

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Satellite Tracker Pro")
st.title("🛰️ Satellite Tracker Pro: Orbit, Visibility & Proximity")

# --- Constants ---
R_EARTH = 6371 # Earth radius in km

# Backup TLE data in case CelesTrak is unreachable
FALLBACK_TLE = """
ISS (ZARYA)
1 25544U 98067A   24017.50000000  .00016717  00000+0  30693-3 0  9993
2 25544  51.6416 280.6231 0005697 325.7688 154.5427 15.49678367491472
HST
1 20580U 90037B   24017.50000000  .00001500  00000+0  10000-3 0  9991
2 20580  28.4699 265.1234 0002500 100.0000 260.0000 15.09000000  1000
GPS BIIA-10 (PRN 32)
1 20959U 90103A   24017.50000000 -.00000050  00000+0  00000+0 0  9995
2 20959  54.8500 100.0000 0150000  45.0000 315.0000  2.00565432 10000
STARLINK-1007
1 44713U 19074A   24017.50000000  .00000678  00000+0  67856-4 0  9995
2 44713  53.0543 178.9876 0001423  87.9752 272.1462 15.06394628230052
"""

# --- Helper Functions ---

def get_category(name):
    """Categorizes satellites based on their name."""
    name = name.upper()
    if 'STARLINK' in name: return 'Starlink'
    elif 'ONEWEB' in name: return 'OneWeb'
    elif 'GPS' in name: return 'GPS'
    elif 'BEIDOU' in name: return 'Beidou'
    elif 'GALILEO' in name: return 'Galileo'
    elif 'GLONASS' in name: return 'GLONASS'
    elif 'IRIDIUM' in name: return 'Iridium'
    elif 'ISS' in name: return 'ISS'
    else: return 'Other'

def get_footprint(lat, lon, alt):
    """Calculate the 3D Cartesian coordinates of the satellite's visibility footprint."""
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    alpha = np.arccos(R_EARTH / (R_EARTH + alt))
    theta = np.linspace(0, 2 * np.pi, 100)

    lat_circle = np.arcsin(
        np.sin(lat_rad) * np.cos(alpha) +
        np.cos(lat_rad) * np.sin(alpha) * np.cos(theta)
    )

    lon_circle = lon_rad + np.arctan2(
        np.sin(theta) * np.sin(alpha) * np.cos(lat_rad),
        np.cos(alpha) - np.sin(lat_rad) * np.sin(lat_circle)
    )

    R_plot = R_EARTH + 10
    x = R_plot * np.cos(lat_circle) * np.cos(lon_circle)
    y = R_plot * np.cos(lat_circle) * np.sin(lon_circle)
    z = R_plot * np.sin(lat_circle)
    return x, y, z

def get_trajectory(sat, ts, t_start, duration_minutes=90):
    """Propagates the orbit forward to generate a trajectory line."""
    start_dt = t_start.utc_datetime()
    time_list = [start_dt + timedelta(minutes=i) for i in range(duration_minutes)]
    times = ts.from_datetimes(time_list)
    
    geocentric = sat.at(times)
    subpoint = wgs84.subpoint(geocentric)
    
    lats = subpoint.latitude.radians
    lons = subpoint.longitude.radians
    alts = subpoint.elevation.km
    
    rs = R_EARTH + alts
    xs = rs * np.cos(lats) * np.cos(lons)
    ys = rs * np.cos(lats) * np.sin(lons)
    zs = rs * np.sin(lats)
    
    return xs, ys, zs

def check_visibility(sat, t, ground_lat, ground_lon):
    """Checks if satellite is visible from a ground station."""
    ground_station = wgs84.latlon(ground_lat, ground_lon)
    difference = sat - ground_station
    topocentric = difference.at(t)
    alt, az, distance = topocentric.altaz()
    return alt.degrees > 0, alt.degrees, distance.km

# --- Caching Functions ---

@st.cache_resource
def load_satellites():
    """Downloads and parses satellite TLE data without local file caching."""
    url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
    lines = []
    
    try:
        # Add headers to mimic a browser
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        
        # Try downloading live data with increased timeout
        with urlopen(req, timeout=30) as response:
            # Read lines directly as bytes
            lines = [line for line in response.readlines()]
            
        if not lines:
            raise ValueError("Empty data received")
            
    except Exception as e:
        st.warning(f"⚠️ Network issue detected ({e}). Switching to OFFLINE mode with sample data.")
        # Fallback to local data
        # Encode strings to bytes to match what Skyfield expects
        lines = [line.encode('ascii') for line in FALLBACK_TLE.strip().splitlines()]

    # ROBUSTNESS CHECK: Ensure all lines are bytes before parsing
    # This handles any edge case where data might be strings
    if lines and isinstance(lines[0], str):
        lines = [l.encode('ascii') for l in lines]
        
    ts = load.timescale(builtin=True)
    satellites = list(parse_tle_file(lines, ts))
    return satellites

@st.cache_data
def get_geometry():
    """Generates Earth mesh and fetches coastline data."""
    phi = np.linspace(0, 2 * np.pi, 100)
    theta = np.linspace(0, np.pi, 100)
    phi, theta = np.meshgrid(phi, theta)
    x_earth = R_EARTH * np.sin(theta) * np.cos(phi)
    y_earth = R_EARTH * np.sin(theta) * np.sin(phi)
    z_earth = R_EARTH * np.cos(theta)

    coast_url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_coastline.geojson"
    xc, yc, zc = [], [], []
    try:
        with urlopen(coast_url, timeout=5) as response:
            coastlines = json.load(response)
        for feature in coastlines['features']:
            coords = feature['geometry']['coordinates']
            if feature['geometry']['type'] == 'LineString': segments = [coords]
            elif feature['geometry']['type'] == 'MultiLineString': segments = coords
            else: continue
            for seg in segments:
                lons, lats = zip(*seg)
                clat, clon = np.radians(np.array(lats)), np.radians(np.array(lons))
                R_coast = R_EARTH + 10
                xc.extend(R_coast * np.cos(clat) * np.cos(clon))
                yc.extend(R_coast * np.cos(clat) * np.sin(clon))
                zc.extend(R_coast * np.sin(clat))
                xc.append(None); yc.append(None); zc.append(None)
    except Exception: pass
    return (x_earth, y_earth, z_earth), (xc, yc, zc)

@st.cache_data
def process_positions(_satellites):
    """Calculates current positions for all satellites."""
    # Use builtin=True to avoid download errors on cloud
    ts = load.timescale(builtin=True)
    t = ts.now()
    data = []
    for i, sat in enumerate(_satellites):
        try:
            geocentric = sat.at(t)
            subpoint = wgs84.subpoint(geocentric)
            lat, lon, alt = subpoint.latitude.degrees, subpoint.longitude.degrees, subpoint.elevation.km
            
            r = R_EARTH + alt
            rad_lat, rad_lon = np.radians(lat), np.radians(lon)
            x = r * np.cos(rad_lat) * np.cos(rad_lon)
            y = r * np.cos(rad_lat) * np.sin(rad_lon)
            z = r * np.sin(rad_lat)
            
            cat = get_category(sat.name)
            
            data.append({'Name': sat.name, 'Category': cat, 'Lat': lat, 'Lon': lon, 'Alt': alt, 'X': x, 'Y': y, 'Z': z, 'Index': i})
        except Exception: continue
    return pd.DataFrame(data), t

# --- Main Execution ---

with st.spinner("Initializing Satellite Data..."):
    satellites = load_satellites()
    
    # Debug Info to verify loading
    st.success(f"DEBUG: Loaded {len(satellites)} satellites.")
    
    if not satellites:
        st.error("⚠️ Failed to load ANY data (online or offline).")
        st.stop()
        
    earth_mesh, coastlines = get_geometry()
    df, t_now = process_positions(satellites)

if df.empty:
    st.warning("No satellites could be processed. Please check data source.")
    st.stop()

# --- Sidebar Controls ---

st.sidebar.header("Target Selection")

# Initialize Session State for Selection
if 'selected_sat_name' not in st.session_state:
    # Default to first Starlink or first item
    sl_match = df[df['Name'].str.contains('STARLINK')]
    st.session_state.selected_sat_name = sl_match.iloc[0]['Name'] if not sl_match.empty else df.iloc[0]['Name']

def update_selection():
    # Callback for selectbox change
    pass # Value is already in key

search = st.sidebar.text_input("Search Satellite", placeholder="ISS").upper()
options = df[df['Name'].str.contains(search, case=False)]['Name'].sort_values() if search else df['Name'].sort_values()

# Handle Search Filter: If search changes options and current selection isn't in options, reset or pick first
if st.session_state.selected_sat_name not in options.values:
    # Try to keep current if possible, otherwise pick first
    if len(options) > 0:
        st.session_state.selected_sat_name = options.iloc[0]

selected_name = st.sidebar.selectbox(
    "Select Satellite", 
    options, 
    key='selected_sat_name', # Links directly to session_state
    on_change=update_selection
)

st.sidebar.markdown("---")
st.sidebar.header("Visualization Options")
show_traj = st.sidebar.checkbox("Show 90min Orbit Path", value=True)

st.sidebar.markdown("---")
st.sidebar.header("Ground Station")
gs_lat = st.sidebar.number_input("Latitude", value=40.7128, min_value=-90.0, max_value=90.0)
gs_lon = st.sidebar.number_input("Longitude", value=-74.0060, min_value=-180.0, max_value=180.0)

# --- Main Logic ---

if selected_name:
    # 1. Get Selected Sat Data
    row = df[df['Name'] == selected_name].iloc[0]
    sat_idx = int(row['Index'])
    sat_obj = satellites[sat_idx]
    
    # 2. Footprint & Trajectory
    fx, fy, fz = get_footprint(row['Lat'], row['Lon'], row['Alt'])
    tx, ty, tz = [], [], []
    if show_traj:
        # Ensure we use builtin timescale for trajectory too
        ts = load.timescale(builtin=True)
        tx, ty, tz = get_trajectory(sat_obj, ts, t_now)
        
    # 3. Visibility
    is_visible, el_deg, dist_km = check_visibility(sat_obj, t_now, gs_lat, gs_lon)
    
    # 4. Proximity
    dists = np.sqrt((df['X'] - row['X'])**2 + (df['Y'] - row['Y'])**2 + (df['Z'] - row['Z'])**2)
    dists[dists == 0] = np.inf
    min_dist_idx = dists.idxmin()
    nearest_sat = df.loc[min_dist_idx]
    min_dist_km = dists[min_dist_idx]
    
    # Ground Station Cartesian
    gs_rad_lat, gs_rad_lon = np.radians(gs_lat), np.radians(gs_lon)
    gs_x = R_EARTH * np.cos(gs_rad_lat) * np.cos(gs_rad_lon)
    gs_y = R_EARTH * np.cos(gs_rad_lat) * np.sin(gs_rad_lon)
    gs_z = R_EARTH * np.sin(gs_rad_lat)

    # --- Build Visualization ---
    fig = go.Figure()
    
    # Base: Earth & Coastlines
    fig.add_trace(go.Surface(x=earth_mesh[0], y=earth_mesh[1], z=earth_mesh[2], colorscale=[[0,'#001f3f'],[1,'#001f3f']], showscale=False, opacity=1, name='Earth', hoverinfo='skip'))
    if coastlines[0]: fig.add_trace(go.Scatter3d(x=coastlines[0], y=coastlines[1], z=coastlines[2], mode='lines', line=dict(color='cyan', width=2), hoverinfo='skip', name='Coasts'))
    
    # Color Mapping
    colors = {'Starlink': '#32CD32', 'OneWeb': '#FFD700', 'GPS': '#FF4500', 'Beidou': '#FF1493', 
              'Galileo': '#00FFFF', 'GLONASS': '#FFA500', 'Iridium': '#1E90FF', 'ISS': '#FFFFFF', 'Other': '#808080'}
    
    # Satellites by Category
    for cat, color in colors.items():
        df_cat = df[df['Category'] == cat]
        if not df_cat.empty:
            fig.add_trace(go.Scatter3d(
                x=df_cat['X'], y=df_cat['Y'], z=df_cat['Z'],
                mode='markers',
                marker=dict(size=4, color=color, opacity=0.8), # INCREASED SIZE AND OPACITY
                name=cat,
                text=df_cat['Name'],
                customdata=df_cat['Name'].tolist(), # Explicit list conversion for safety
                hoverinfo='text'
            ))
    
    # Trajectory
    if show_traj:
        fig.add_trace(go.Scatter3d(x=tx, y=ty, z=tz, mode='lines', line=dict(color='orange', width=4, dash='dot'), name='Orbit Path', hoverinfo='skip'))

    # Footprint
    fig.add_trace(go.Scatter3d(x=fx, y=fy, z=fz, mode='lines', line=dict(color='yellow', width=5), name='Footprint', hoverinfo='skip'))

    # Selected Satellite
    fig.add_trace(go.Scatter3d(x=[row['X']], y=[row['Y']], z=[row['Z']], mode='markers', marker=dict(size=12, color='red', symbol='diamond'), name=selected_name, hoverinfo='text', text=selected_name))

    # Ground Station
    gs_color = 'green' if is_visible else 'gray'
    fig.add_trace(go.Scatter3d(x=[gs_x], y=[gs_y], z=[gs_z], mode='markers', marker=dict(size=8, color=gs_color, symbol='x'), name='Ground Station', hoverinfo='text', text='Ground Station'))
    if is_visible:
        fig.add_trace(go.Scatter3d(x=[gs_x, row['X']], y=[gs_y, row['Y']], z=[gs_z, row['Z']], mode='lines', line=dict(color='green', width=3), name='Line of Sight', hoverinfo='skip'))

    # Nearest Neighbor
    fig.add_trace(go.Scatter3d(x=[row['X'], nearest_sat['X']], y=[row['Y'], nearest_sat['Y']], z=[row['Z'], nearest_sat['Z']], 
                               mode='lines', line=dict(color='magenta', width=2), name=f"Nearest: {nearest_sat['Name']}", hoverinfo='skip'))

    # Layout with EXPANDED AXIS RANGE for GEO/MEO
    fig.update_layout(
        template='plotly_dark',
        dragmode='orbit',
        uirevision='constant',
        clickmode='event+select', # Allow clicking points
        scene=dict(
            aspectmode='manual', aspectratio=dict(x=1, y=1, z=1),
            # Expanded to +/- 50,000 km to include Geostationary Orbit (36,000 km alt)
            xaxis=dict(visible=False, range=[-50000, 50000]), 
            yaxis=dict(visible=False, range=[-50000, 50000]), 
            zaxis=dict(visible=False, range=[-50000, 50000]),
            camera=dict(projection=dict(type='perspective'), center=dict(x=0, y=0, z=0), eye=dict(x=1.2, y=1.2, z=1.2))
        ),
        margin=dict(r=0, t=0, l=0, b=0),
        height=800,
        modebar=dict(remove=['pan3d', 'select2d', 'lasso2d']) # Remove 2D select tools to encourage clicking
    )
    
    # --- Metrics ---
    m1, m2, m3 = st.columns(3)
    m1.metric("Altitude", f"{row['Alt']:.1f} km", f"Category: {row['Category']}")
    m2.metric("Station Visibility", "VISIBLE" if is_visible else "No Signal", f"{el_deg:.1f}° El")
    m3.metric("Nearest Neighbor", nearest_sat['Name'], f"{min_dist_km:.1f} km away")
    
    # Render with Event Capture
    # selection_mode='points' ensures we get the clicked point info
    event = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="points")
    
    # Handle Click Event
    if event and event.selection and event.selection.points:
        # Get the first clicked point
        point = event.selection.points[0]
        # Check if it has customdata (which contains the name)
        if 'customdata' in point:
            clicked_name = point['customdata']
            # If different from current, update and rerun
            if clicked_name != st.session_state.selected_sat_name:
                st.session_state.selected_sat_name = clicked_name
                st.rerun()
