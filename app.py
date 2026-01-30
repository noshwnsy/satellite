import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import os
from urllib.request import urlopen, Request

# Try importing Pillow, handle if missing
try:
    from PIL import Image
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False

from skyfield.api import load, wgs84
from skyfield.iokit import parse_tle_file
from skyfield.framelib import itrs
from datetime import timedelta

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Satellite Tracker Pro")
st.title("🛰️ Satellite Tracker Pro: Orbit, Visibility & Proximity")

# --- Constants ---
R_EARTH = 6371 # Earth radius in km

# Backup TLE data
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
    """
    Loads satellite data with a priority system:
    1. Local 'active.txt' file (Fastest, works in Cloud)
    2. CelesTrak URL (Live, requires internet)
    3. Fallback String (Emergency only)
    """
    lines = []
    source = "Unknown"

    # 1. Try Local File (Pre-cached)
    if os.path.exists("active.txt"):
        try:
            with open("active.txt", "rb") as f:
                lines = [line for line in f]
            source = "Local File"
        except Exception as e:
            print(f"Local file read error: {e}")

    # 2. Try Live Download (if no local file)
    if not lines:
        url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
        try:
            req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urlopen(req, timeout=30) as response:
                lines = [line for line in response.readlines()]
            source = "CelesTrak (Live)"
        except Exception as e:
            st.warning(f"⚠️ Could not connect to CelesTrak: {e}")

    # 3. Use Fallback
    if not lines:
        source = "Emergency Fallback"
        lines = [line.encode('ascii') for line in FALLBACK_TLE.strip().splitlines()]

    # Final Validation
    if not lines or len(lines) < 3:
        st.error("❌ No valid TLE data found.")
        return [], "None"
        
    # Ensure binary format for Skyfield
    if isinstance(lines[0], str):
        lines = [l.encode('ascii') for l in lines]

    ts = load.timescale(builtin=True)
    satellites = list(parse_tle_file(lines, ts))
    return satellites, source

@st.cache_data
def get_geometry():
    """Generates Earth mesh, texture, and coastlines."""
    # 1. Sphere Geometry
    N = 100
    phi = np.linspace(0, 2 * np.pi, N)
    theta = np.linspace(0, np.pi, N)
    phi, theta = np.meshgrid(phi, theta)
    x_earth = R_EARTH * np.sin(theta) * np.cos(phi)
    y_earth = R_EARTH * np.sin(theta) * np.sin(phi)
    z_earth = R_EARTH * np.cos(theta)

    # 2. Earth Texture (Blue Marble)
    surface_color = None
    if HAS_PILLOW:
        img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Blue_Marble_2002.png/480px-Blue_Marble_2002.png"
        try:
            req = Request(img_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urlopen(req, timeout=10) as response:
                img = Image.open(response)
                img = img.resize((N, N))
                img_gray = img.convert('L')
                surface_color = np.array(img_gray) / 255.0
        except Exception:
            pass # surface_color remains None

    # 3. Coastlines
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
    
    return (x_earth, y_earth, z_earth, surface_color), (xc, yc, zc)

@st.cache_data
def process_positions(_satellites):
    """Calculates current positions for all satellites."""
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
    satellites, source = load_satellites()
    
    if not satellites: st.stop()
    
    # Unpack updated geometry tuple
    (x_earth, y_earth, z_earth, surface_color), (xc, yc, zc) = get_geometry()
    df, t_now = process_positions(satellites)

if df.empty: st.stop()

# --- Sidebar Controls ---

st.sidebar.header("Target Selection")
st.sidebar.info(f"Data Source: {source} ({len(df)} sats)")

if 'selected_sat_name' not in st.session_state:
    sl_match = df[df['Name'].str.contains('STARLINK')]
    st.session_state.selected_sat_name = sl_match.iloc[0]['Name'] if not sl_match.empty else df.iloc[0]['Name']

def update_selection(): pass

search = st.sidebar.text_input("Search Satellite", placeholder="ISS").upper()
options = df[df['Name'].str.contains(search, case=False)]['Name'].sort_values() if search else df['Name'].sort_values()

if st.session_state.selected_sat_name not in options.values:
    if len(options) > 0: st.session_state.selected_sat_name = options.iloc[0]

selected_name = st.sidebar.selectbox("Select Satellite", options, key='selected_sat_name', on_change=update_selection)

st.sidebar.markdown("---")
st.sidebar.header("Visualization Options")
show_traj = st.sidebar.checkbox("Show 90min Orbit Path", value=True)

st.sidebar.markdown("---")
st.sidebar.header("Ground Station")
gs_lat = st.sidebar.number_input("Latitude", value=40.7128, min_value=-90.0, max_value=90.0)
gs_lon = st.sidebar.number_input("Longitude", value=-74.0060, min_value=-180.0, max_value=180.0)

# --- Main Logic ---

if selected_name:
    row = df[df['Name'] == selected_name].iloc[0]
    sat_idx = int(row['Index'])
    sat_obj = satellites[sat_idx]
    
    fx, fy, fz = get_footprint(row['Lat'], row['Lon'], row['Alt'])
    tx, ty, tz = [], [], []
    if show_traj:
        ts = load.timescale(builtin=True)
        tx, ty, tz = get_trajectory(sat_obj, ts, t_now)
        
    is_visible, el_deg, dist_km = check_visibility(sat_obj, t_now, gs_lat, gs_lon)
    
    dists = np.sqrt((df['X'] - row['X'])**2 + (df['Y'] - row['Y'])**2 + (df['Z'] - row['Z'])**2)
    dists[dists == 0] = np.inf
    min_dist_idx = dists.idxmin()
    nearest_sat = df.loc[min_dist_idx]
    min_dist_km = dists[min_dist_idx]
    
    gs_rad_lat, gs_rad_lon = np.radians(gs_lat), np.radians(gs_lon)
    gs_x = R_EARTH * np.cos(gs_rad_lat) * np.cos(gs_rad_lon)
    gs_y = R_EARTH * np.cos(gs_rad_lat) * np.sin(gs_rad_lon)
    gs_z = R_EARTH * np.sin(gs_rad_lat)

    # --- Visualization ---
    fig = go.Figure()
    
    # Earth Surface Layer
    if surface_color is not None:
        # Photorealistic Texture
        fig.add_trace(go.Surface(
            x=x_earth, y=y_earth, z=z_earth,
            surfacecolor=surface_color,
            colorscale=[[0, '#000080'], [0.1, '#1E90FF'], [0.2, '#228B22'], [0.5, '#8B4513'], [1, '#FFFFFF']],
            showscale=False,
            cmin=0, cmax=1,
            hoverinfo='skip',
            name='Earth'
        ))
    else:
        # Fallback to Lighter Blue Sphere (Fixed Black Globe issue)
        fig.add_trace(go.Surface(
            x=x_earth, y=y_earth, z=z_earth,
            colorscale=[[0, 'royalblue'], [1, 'royalblue']],
            showscale=False, opacity=1.0, hoverinfo='skip', name='Earth',
            lighting=dict(ambient=0.6, diffuse=0.5, roughness=0.1)
        ))

    # SAFE CHECK FOR COASTLINES to avoid Numpy array ambiguity
    if len(coastlines[0]) > 0: 
        fig.add_trace(go.Scatter3d(x=coastlines[0], y=coastlines[1], z=coastlines[2], 
            mode='lines', line=dict(color='cyan', width=2), hoverinfo='skip', name='Coasts'))
    
    colors = {'Starlink': '#32CD32', 'OneWeb': '#FFD700', 'GPS': '#FF4500', 'Beidou': '#FF1493', 
              'Galileo': '#00FFFF', 'GLONASS': '#FFA500', 'Iridium': '#1E90FF', 'ISS': '#FFFFFF', 'Other': '#808080'}
    for cat, color in colors.items():
        df_cat = df[df['Category'] == cat]
        if not df_cat.empty:
            fig.add_trace(go.Scatter3d(x=df_cat['X'], y=df_cat['Y'], z=df_cat['Z'], mode='markers', 
                marker=dict(size=4, color=color, opacity=0.8), name=cat, text=df_cat['Name'], 
                customdata=df_cat['Name'].tolist(), hoverinfo='text'))
    
    if show_traj: 
        fig.add_trace(go.Scatter3d(x=tx, y=ty, z=tz, mode='lines', line=dict(color='orange', width=4, dash='dot'), name='Orbit Path', hoverinfo='skip'))
    fig.add_trace(go.Scatter3d(x=fx, y=fy, z=fz, mode='lines', line=dict(color='yellow', width=5), name='Footprint', hoverinfo='skip'))
    fig.add_trace(go.Scatter3d(x=[row['X']], y=[row['Y']], z=[row['Z']], mode='markers', marker=dict(size=12, color='red', symbol='diamond'), name=selected_name, hoverinfo='text', text=selected_name))
    
    gs_col = 'green' if is_visible else 'gray'
    fig.add_trace(go.Scatter3d(x=[gs_x], y=[gs_y], z=[gs_z], mode='markers', marker=dict(size=8, color=gs_col, symbol='x'), name='Ground Station', hoverinfo='text', text='Ground Station'))
    if is_visible: 
        fig.add_trace(go.Scatter3d(x=[gs_x, row['X']], y=[gs_y, row['Y']], z=[gs_z, row['Z']], mode='lines', line=dict(color='green', width=3), name='Line of Sight', hoverinfo='skip'))
    
    fig.add_trace(go.Scatter3d(x=[row['X'], nearest_sat['X']], y=[row['Y'], nearest_sat['Y']], z=[row['Z'], nearest_sat['Z']], 
                               mode='lines', line=dict(color='magenta', width=2), name=f"Nearest: {nearest_sat['Name']}", hoverinfo='skip'))

    fig.update_layout(
        template='plotly_dark',
        dragmode='orbit',
        uirevision='constant',
        clickmode='event+select',
        scene=dict(
            aspectmode='manual', aspectratio=dict(x=1, y=1, z=1),
            xaxis=dict(visible=False, range=[-50000, 50000]), 
            yaxis=dict(visible=False, range=[-50000, 50000]), 
            zaxis=dict(visible=False, range=[-50000, 50000]),
            camera=dict(projection=dict(type='perspective'), center=dict(x=0, y=0, z=0), eye=dict(x=1.2, y=1.2, z=1.2))
        ),
        margin=dict(r=0, t=0, l=0, b=0),
        height=800,
        modebar=dict(remove=['pan3d', 'select2d', 'lasso2d'])
    )
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Altitude", f"{row['Alt']:.1f} km", f"Category: {row['Category']}")
    m2.metric("Station Visibility", "VISIBLE" if is_visible else "No Signal", f"{el_deg:.1f}° El")
    m3.metric("Nearest Neighbor", nearest_sat['Name'], f"{min_dist_km:.1f} km away")
    
    # KEY ADDED HERE to ensure chart state persists
    event = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="points", key="main_map")
    
    if event and event.selection and event.selection.points:
        point = event.selection.points[0]
        if 'customdata' in point:
            clicked = point['customdata']
            if clicked != st.session_state.selected_sat_name:
                st.session_state.selected_sat_name = clicked
                st.rerun()

    # Debug Expander
    with st.expander("Debug: Raw Event Data"):
        st.write(event)
