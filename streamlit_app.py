"""
Crystal Structure Parameter Scanner - Streamlit Web App

Interactive web interface for exploring crystal structure parameter spaces.
Allows users to configure scans, run them, and visualize/export results.

Deploy with: streamlit run streamlit_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io
import json

# Try to import the scanner and lattice modules
import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from parameter_scanner import ParameterScanner, export_to_csv
    from lattice_intersections import (
        generate_bravais_lattice,
        max_sphere_surface_intersection_fast,
    )
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    st.error(f"⚠️ Required modules not found: {str(e)}")
    st.info("Make sure these files are in the same directory as streamlit_app.py:")
    st.info("- parameter_scanner.py")
    st.info("- lattice_intersections.py")


# Page configuration
st.set_page_config(
    page_title="Crystal Structure Scanner",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'scan_result' not in st.session_state:
    st.session_state.scan_result = None
if 'results_history' not in st.session_state:
    st.session_state.results_history = []


def create_scanner():
    """Create and return a ParameterScanner instance."""
    if not MODULES_AVAILABLE:
        return None
    return ParameterScanner(
        lattice_gen_func=generate_bravais_lattice,
        k_max_func=max_sphere_surface_intersection_fast,
        verbose=False,
    )


def result_to_dataframe(result):
    """Convert ScanResult to pandas DataFrame for display."""
    data = []
    for point in result.points:
        row = {
            **point.params,
            'k_max': point.k_max,
            'n_spheres': point.n_spheres,
            'computation_time': f"{point.computation_time:.4f}s",
        }
        data.append(row)
    return pd.DataFrame(data)


def plot_1d_scan(result):
    """Create a Plotly figure for 1D scan results."""
    if len(result.parameters_varied) != 1:
        st.warning("Plot only supports 1D scans")
        return None
    
    param_name = result.parameters_varied[0]
    
    # Extract data
    param_values = [p.params[param_name] for p in result.points]
    k_max_values = [p.k_max for p in result.points]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=param_values,
        y=k_max_values,
        mode='lines+markers',
        name='k_max',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8, color='#1f77b4'),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.2)',
    ))
    
    fig.update_layout(
        title=f"Parameter Scan: {param_name}",
        xaxis_title=param_name,
        yaxis_title="k_max (Intersection Multiplicity)",
        hovermode='x unified',
        template='plotly_white',
        height=500,
    )
    
    return fig


def plot_2d_scan(result):
    """Create a heatmap for 2D scan results."""
    if len(result.parameters_varied) != 2:
        st.warning("Heatmap only supports 2D scans")
        return None
    
    param1, param2 = result.parameters_varied
    
    # Build grid
    param1_values = sorted(set(p.params[param1] for p in result.points))
    param2_values = sorted(set(p.params[param2] for p in result.points))
    
    # Create matrix
    matrix = np.zeros((len(param2_values), len(param1_values)))
    
    for point in result.points:
        i = param1_values.index(point.params[param1])
        j = param2_values.index(point.params[param2])
        matrix[j, i] = point.k_max
    
    fig = go.Figure(data=go.Heatmap(
        x=param1_values,
        y=param2_values,
        z=matrix,
        colorscale='Viridis',
        colorbar=dict(title="k_max"),
    ))
    
    fig.update_layout(
        title=f"Phase Diagram: {param1} vs {param2}",
        xaxis_title=param1,
        yaxis_title=param2,
        height=600,
        width=800,
    )
    
    return fig


def export_results(result, format='csv'):
    """Export results in various formats."""
    if format == 'csv':
        df = result_to_dataframe(result)
        return df.to_csv(index=False).encode('utf-8')
    
    elif format == 'json':
        data = {
            'scan_type': result.scan_type,
            'parameters_varied': result.parameters_varied,
            'fixed_parameters': result.fixed_parameters,
            'points': [
                {
                    'params': p.params,
                    'k_max': p.k_max,
                    'n_spheres': p.n_spheres,
                    'computation_time': p.computation_time,
                }
                for p in result.points
            ],
            'total_time': result.total_time,
        }
        return json.dumps(data, indent=2).encode('utf-8')


# ============================================================================
# MAIN APP
# ============================================================================

st.markdown("# 🔮 Crystal Structure Parameter Scanner")
st.markdown("Explore crystal structure parameter spaces interactively")

if not MODULES_AVAILABLE:
    st.stop()

# Sidebar for configuration
st.sidebar.markdown("## ⚙️ Scan Configuration")

scan_type = st.sidebar.radio(
    "Scan Type",
    options=["1D Scan", "2D Scan"],
    help="Choose between single-parameter or two-parameter scans"
)

# Lattice type selection
lattice_types = [
    'sc', 'bcc', 'fcc', 'hcp', 'body_centered_tetragonal',
    'primitive_tetragonal', 'body_centered_orthorhombic',
    'primitive_orthorhombic', 'base_centered_orthorhombic',
    'face_centered_orthorhombic', 'primitive_monoclinic',
    'base_centered_monoclinic', 'primitive_triclinic', 'rhombohedral'
]

lattice = st.sidebar.selectbox(
    "Lattice Type",
    options=lattice_types,
    index=2,  # fcc by default
    help="Select the Bravais lattice type"
)

# Fixed parameters section
st.sidebar.markdown("### Fixed Parameters")

lattice_param_a = st.sidebar.slider(
    "Lattice Parameter (a)",
    min_value=0.5,
    max_value=2.0,
    value=1.0,
    step=0.05,
    help="Lattice constant in angstroms"
)

supercell_x = st.sidebar.slider("Supercell (x)", min_value=1, max_value=5, value=2)
supercell_y = st.sidebar.slider("Supercell (y)", min_value=1, max_value=5, value=2)
supercell_z = st.sidebar.slider("Supercell (z)", min_value=1, max_value=5, value=2)

# Scan parameters
st.sidebar.markdown("### Scan Parameters")

if scan_type == "1D Scan":
    
    param_choice = st.sidebar.selectbox(
        "Parameter to Vary",
        options=['radius', 'a'],
        help="Which parameter to scan"
    )
    
    param_min, param_max = st.sidebar.slider(
        f"Range for {param_choice}",
        min_value=0.1,
        max_value=2.0,
        value=(0.3, 0.7) if param_choice == 'radius' else (0.8, 1.2),
        step=0.05,
    )
    
    n_points = st.sidebar.slider(
        "Number of Points",
        min_value=5,
        max_value=50,
        value=15,
        help="How many parameter values to sample"
    )
    
    scale = st.sidebar.radio(
        "Scale",
        options=['linear', 'log'],
        help="Linear or logarithmic spacing of points"
    )

else:  # 2D Scan
    
    param1 = st.sidebar.selectbox(
        "First Parameter",
        options=['radius', 'a'],
        help="First dimension to scan"
    )
    
    param2_options = ['a', 'radius'] if param1 == 'radius' else ['radius', 'a']
    param2 = st.sidebar.selectbox(
        "Second Parameter",
        options=param2_options,
        help="Second dimension to scan"
    )
    
    param1_min, param1_max = st.sidebar.slider(
        f"Range for {param1}",
        min_value=0.1,
        max_value=2.0,
        value=(0.3, 0.7) if param1 == 'radius' else (0.8, 1.2),
        step=0.05,
        key=f"param1_range"
    )
    
    param2_min, param2_max = st.sidebar.slider(
        f"Range for {param2}",
        min_value=0.1,
        max_value=2.0,
        value=(0.3, 0.7) if param2 == 'radius' else (0.8, 1.2),
        step=0.05,
        key=f"param2_range"
    )
    
    n_points_1d = st.sidebar.slider(
        "Points per dimension",
        min_value=3,
        max_value=20,
        value=8,
        help="Number of points along each dimension"
    )
    
    scale = 'linear'


# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Current Configuration")
    
    config_info = f"""
    **Lattice Type:** {lattice.upper()}  
    **Lattice Parameter (a):** {lattice_param_a:.3f}  
    **Supercell:** ({supercell_x} × {supercell_y} × {supercell_z})  
    """
    
    if scan_type == "1D Scan":
        config_info += f"\n**Scan:** {param_choice} from {param_min:.3f} to {param_max:.3f} ({n_points} points, {scale})"
    else:
        config_info += f"\n**Scan 2D:** {param1} [{param1_min:.3f}-{param1_max:.3f}] vs {param2} [{param2_min:.3f}-{param2_max:.3f}] ({n_points_1d}×{n_points_1d} points)"
    
    st.info(config_info)

with col2:
    st.markdown("### Run Scan")
    run_button = st.button("🚀 Start Scan", use_container_width=True, type="primary")


# Execute scan if button pressed
if run_button:
    with st.spinner("⏳ Running parameter scan..."):
        
        scanner = create_scanner()
        fixed_params = {
            'lattice_type': lattice,
            'a': lattice_param_a,
            'supercell': (supercell_x, supercell_y, supercell_z),
        }
        
        try:
            if scan_type == "1D Scan":
                result = scanner.scan_1d(
                    param_name=param_choice,
                    param_range=(param_min, param_max),
                    n_points=n_points,
                    fixed_params=fixed_params,
                    scale=scale,
                    show_progress=False,
                )
            else:
                result = scanner.scan_2d(
                    param_names=(param1, param2),
                    param_ranges=((param1_min, param1_max), (param2_min, param2_max)),
                    n_points=(n_points_1d, n_points_1d),
                    fixed_params=fixed_params,
                    show_progress=False,
                )
            
            st.session_state.scan_result = result
            st.session_state.results_history.append({
                'timestamp': datetime.now(),
                'config': {
                    'scan_type': scan_type,
                    'lattice': lattice,
                    'param_a': lattice_param_a,
                    'supercell': (supercell_x, supercell_y, supercell_z),
                },
                'result': result,
            })
            
            st.success("✅ Scan completed successfully!")
            
        except Exception as e:
            st.error(f"❌ Error during scan: {str(e)}")
            st.exception(e)


# Display results if available
if st.session_state.scan_result is not None:
    result = st.session_state.scan_result
    
    st.markdown("---")
    st.markdown("## 📊 Results")
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Points", len(result.points))
    
    with col2:
        k_max_values = [p.k_max for p in result.points]
        st.metric("k_max Range", f"{min(k_max_values)} - {max(k_max_values)}")
    
    with col3:
        st.metric("Execution Time", f"{result.total_time:.2f}s")
    
    with col4:
        best_point = max(result.points, key=lambda p: p.k_max)
        st.metric("Peak k_max", best_point.k_max)
    
    st.markdown("### Results Table")
    df = result_to_dataframe(result)
    st.dataframe(df, use_container_width=True, height=400)
    
    st.markdown("### Visualization")
    
    if result.scan_type == '1D':
        fig = plot_1d_scan(result)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    elif result.scan_type == '2D':
        fig = plot_2d_scan(result)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    # Export options
    st.markdown("### Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_data = export_results(result, format='csv')
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    
    with col2:
        json_data = export_results(result, format='json')
        st.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name=f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )
    
    with col3:
        st.write("💾 Results ready for download")


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    Crystal Structure Parameter Scanner | Built with Streamlit 🚀
</div>
""", unsafe_allow_html=True)
