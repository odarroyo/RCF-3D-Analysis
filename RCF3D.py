"""
RCF-3D Building Model Creator (Refactored)
===========================================

A comprehensive Streamlit application for creating 3D reinforced concrete frame building models
for structural analysis in OpenSeesPy. This refactored version features improved code organization,
better maintainability, and enhanced user experience.

FEATURES
--------
- **Building Geometry**: Define multi-story building layouts with customizable X-Y coordinates
- **Material Definition**: Create concrete and steel material sets with different detailing levels
- **Section Properties**: Define rectangular RC sections with reinforcement configurations
- **Element Assignment**: Assign sections to columns and beams across all floors
- **Structural Elements**: Generate OpenSeesPy elements (columns, beams) with proper connectivity
- **Slab Modeling**: Add floor slabs with material properties and self-weight
- **Load Application**: Define gravity loads on beams for analysis
- **Model Visualization**: Interactive 3D and 2D visualizations with hover information
- **Model Persistence**: Save/load complete models for analysis workflows

WORKFLOW TABS
-------------
1. **Building Geometry and Masses**: Define node coordinates and story masses
2. **Materials**: Create concrete/steel material combinations
3. **Sections**: Define cross-sectional properties and reinforcement
4. **Columns**: Assign sections to column positions per floor
5. **Beams**: Assign sections to beam spans in X and Y directions
6. **Model Visualization**: Review complete model and create structural elements
7. **Slabs**: Define slab properties and create floor diaphragms
8. **Loads**: Apply gravity loads for analysis preparation
9. **Create and Save Model**: Export model for use in analysis applications
10. **Modal Analysis**: Compute natural periods and mode shapes
11. **Gravity & Pushover**: Perform static and nonlinear analyses

TECHNICAL DETAILS
-----------------
- **Backend**: OpenSeesPy with opseestools utilities
- **Materials**: Confined/unconfined concrete + steel reinforcement
- **Elements**: Nonlinear beam-column elements with fiber sections
- **Boundary Conditions**: Fixed supports at base level
- **File Format**: Pickle (.pkl) for complete model serialization
- **Visualization**: Plotly-based interactive 3D and 2D plots

RECENT IMPROVEMENTS
-------------------
- Modular function-based architecture for better maintainability
- Enhanced error handling and user feedback
- Improved visualization with beam markers and tooltips
- Consistent UI patterns and responsive design
- Comprehensive session state management
- Better code organization and documentation

USAGE NOTES
-----------
- Follow tabs sequentially for complete model creation
- Save models after creating elements for analysis
- Use the analysis app (app_analysis_refactored.py) to load and analyze saved models
- All coordinates in meters, forces in kN, material properties in consistent units

DEPENDENCIES
------------
- streamlit
- openseespy
- opseestools
- numpy, pandas
- plotly
- pickle, os

AUTHOR
------
Refactored version with enhanced maintainability and user experience.
"""

import streamlit as st
from openseespy.opensees import *
import opseestools.analisis3D as an
import opseestools.utilidades as ut
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle
import os

# =============================================================================
# CODE ORGANIZATION:
# 1. Constants and configuration
# 2. Session state management
# 3. Material and section creation utilities
# 4. Element assignment functions
# 5. Visualization functions
# 6. Tab rendering functions (main UI)
# 7. Main application entry point
# =============================================================================

# ==================== CONSTANTS ====================
# Rebar cross-sectional areas (m²) for different bar sizes
REBAR_AREAS = {
    'As4': 0.000127,  # #4 diameter rebar
    'As5': 0.0002,    # #5 diameter rebar
    'As6': 0.000286,  # #6 diameter rebar
    'As7': 0.000387,  # #7 diameter rebar
    'As8': 0.000508   # #8 diameter rebar
}

# Concrete detailing levels for material strength calculations
DETAILING_OPTIONS = ["DES", "DMO", "PreCode"]
# DES: Ductile Earthquake Resistant (highest ductility)
# DMO: Moderate Ductility
# PreCode: Pre-1970s code level (lowest ductility)


# ==================== SESSION STATE INITIALIZATION ====================
def initialize_session_state():
    """
    Initialize all Streamlit session state variables for the application.

    This function sets up the complete state management system, ensuring all
    required variables exist with appropriate default values. The session state
    tracks the entire modeling workflow from geometry definition to model completion.

    Key state variables:
    - model_created: Whether building geometry has been defined
    - materials: Dictionary of defined material sets
    - sections: Dictionary of defined cross-sections
    - column_assignments/beam_assignments: Section assignments per floor
    - elements_created/slabs_created: Completion flags for model components
    - loads_applied: Whether gravity loads have been defined
    - Various analysis flags for workflow tracking
    """
    defaults = {
        'model_created': False,
        'materials': {},
        'sections': {},
        'column_assignments': {},
        'beamx_assignments': {},
        'beamy_assignments': {},
        'elements_created': False,
        'slabs_created': False,
        'loads_applied': False,
        'modal_analysis_done': False,
        'gravity_analysis_done': False,
        'pushover_analysis_done': False,
        'project_name': "building1",
        'coordx': None,
        'coordy': None,
        'coordz': None,
        'masas': None,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ==================== GEOMETRY FUNCTIONS ====================
def parse_coordinates(coord_input):
    """Parse comma-separated coordinates"""
    return [float(x.strip()) for x in coord_input.split(',')]


def create_opensees_model(coordx, coordy, coordz, masas):
    """Create OpenSeesPy model with geometry and masses"""
    wipe()
    model('basic', '-ndm', 3, '-ndf', 6)
    coords = ut.creategrid3D(coordx, coordy, coordz, 1, masas)
    fixZ(0.0, 1, 1, 1, 1, 1, 1)
    return coords


# ==================== MATERIAL FUNCTIONS ====================
def generate_material_tags(material_index):
    """Generate unique tags for material components"""
    base_tag = (material_index + 1) * 100
    return {
        'unctag': base_tag + 1,
        'conftag': base_tag + 2,
        'steeltag': base_tag + 3
    }


def create_material(fc, fy, detailing, material_index):
    """Create OpenSeesPy material and return tags"""
    tags = generate_material_tags(material_index)
    
    noconf, conf, acero = ut.col_materials(
        fc, fy,
        detailing=detailing,
        nps=3,
        tension='yes',
        unctag=tags['unctag'],
        conftag=tags['conftag'],
        steeltag=tags['steeltag']
    )
    
    return {
        'fc': fc,
        'fy': fy,
        'detailing': detailing,
        'noconf_tag': noconf,
        'conf_tag': conf,
        'acero_tag': acero
    }


# ==================== SECTION FUNCTIONS ====================
def generate_section_tag(section_index):
    """Generate unique tag for section"""
    return (section_index + 1) * 100 + 1000


def create_section(section_tag, H, B, cover, material_props, bars_config):
    """Create OpenSeesPy section"""
    # Handle middle bars = 0 case
    bars_middle = bars_config['bars_middle']
    area_middle = bars_config['area_middle']
    
    if bars_middle == 0:
        bars_middle = 2
        area_middle = 1e-8
    
    ut.create_rect_RC_section(
        section_tag,
        H, B, cover,
        material_props['conf_tag'],
        material_props['noconf_tag'],
        material_props['acero_tag'],
        bars_config['bars_top'],
        bars_config['area_top'],
        bars_config['bars_bottom'],
        bars_config['area_bottom'],
        bars_middle,
        area_middle
    )


# ==================== ASSIGNMENT FUNCTIONS ====================
def initialize_floor_assignments(floor, n_x, n_y, default_section):
    """Initialize assignments for a floor"""
    assignments = {}
    for x_idx in range(n_x):
        assignments[x_idx] = {}
        for y_idx in range(n_y):
            assignments[x_idx][y_idx] = default_section
    return assignments


def copy_floor_assignments(source_floor, target_floors, assignments_dict):
    """Copy floor assignments to multiple target floors"""
    for target_floor in target_floors:
        assignments_dict[target_floor] = {}
        for x_idx in assignments_dict[source_floor]:
            assignments_dict[target_floor][x_idx] = {}
            for y_idx in assignments_dict[source_floor][x_idx]:
                assignments_dict[target_floor][x_idx][y_idx] = assignments_dict[source_floor][x_idx][y_idx]


def build_element_tags_list(assignments, sections, num_floors, dim1, dim2):
    """Build list of element tags from assignments"""
    tags_list = []
    for floor in range(1, num_floors + 1):
        floor_config = []
        for idx1 in range(dim1):
            row_config = []
            for idx2 in range(dim2):
                section_name = assignments[floor][idx1][idx2]
                section_tag = sections[section_name]['tag']
                row_config.append(section_tag)
            floor_config.append(row_config)
        tags_list.append(floor_config)
    return tags_list


# ==================== MODEL SAVING ====================
def save_model_to_file(project_name):
    """
    Save the complete building model to a pickle file for later analysis.

    This function serializes all model data including geometry, materials, sections,
    element assignments, and analysis state into a .pkl file that can be loaded
    by the analysis application (app_analysis_refactored.py).

    Parameters:
    -----------
    project_name : str
        Name of the project (used as filename)

    Returns:
    --------
    str
        Path to the saved file

    Saves:
    ------
    - Building geometry (coordinates and masses)
    - Material definitions with OpenSees tags
    - Section properties and reinforcement details
    - Element assignments (columns/beams per floor)
    - Completion flags for workflow tracking
    - Slab and load parameters (if defined)
    - Analysis state flags

    File location: models/{project_name}.pkl
    """
    model_data = {
        'project_name': project_name,
        'coordx': st.session_state.coordx,
        'coordy': st.session_state.coordy,
        'coordz': st.session_state.coordz,
        'masas': st.session_state.masas,
        'materials': st.session_state.materials,
        'sections': st.session_state.sections,
        'column_assignments': st.session_state.column_assignments,
        'beamx_assignments': st.session_state.beamx_assignments,
        'beamy_assignments': st.session_state.beamy_assignments,
        'model_created': st.session_state.model_created,
        'elements_created': st.session_state.elements_created,
        'slabs_created': st.session_state.slabs_created,
        'loads_applied': st.session_state.loads_applied,
        'modal_analysis_done': st.session_state.modal_analysis_done,
        'gravity_analysis_done': st.session_state.gravity_analysis_done,
        'pushover_analysis_done': st.session_state.pushover_analysis_done,
    }
    
    # Add slab parameters if they exist
    if st.session_state.slabs_created:
        model_data['hslab'] = st.session_state.get('hslab')
        model_data['fc_slab'] = st.session_state.get('fc_slab')
        model_data['Eslab'] = st.session_state.get('Eslab')
        model_data['pois'] = st.session_state.get('pois')
    
    # Add load parameters if they exist
    if st.session_state.loads_applied:
        model_data['floorx'] = st.session_state.get('floorx')
        model_data['floory'] = st.session_state.get('floory')
        model_data['roofx'] = st.session_state.get('roofx')
        model_data['roofy'] = st.session_state.get('roofy')
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    file_path = f"models/{project_name}.pkl"
    with open(file_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    return file_path


# ==================== VISUALIZATION FUNCTIONS ====================
def create_floor_plan_figure(floor, coordx, coordy, column_assignments, beamx_assignments, beamy_assignments, sections, materials):
    """
    Create an interactive Plotly figure showing the complete floor plan with elements.

    This function generates a comprehensive 2D visualization of a building floor,
    including columns, beams, and their assigned sections with hover information.

    Parameters:
    -----------
    floor : int
        Floor number to visualize (1-based indexing)
    coordx, coordy : list
        X and Y coordinate arrays defining the building grid
    column_assignments : dict
        Section assignments for columns [floor][x_idx][y_idx] -> section_name
    beamx_assignments : dict
        Section assignments for X-direction beams [floor][span_idx][y_idx] -> section_name
    beamy_assignments : dict
        Section assignments for Y-direction beams [floor][x_idx][span_idx] -> section_name
    sections : dict
        Section definitions with properties and material references
    materials : dict
        Material definitions with properties

    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive figure with:
        - Blue squares for columns with section names
        - Green lines/arrows for X-direction beams
        - Orange lines/arrows for Y-direction beams
        - Hover tooltips showing material properties
        - Proper axis scaling and grid layout

    Features:
    ---------
    - Interactive hover information for all elements
    - Color-coded elements (columns=blue, beams X=green, beams Y=orange)
    - Section names displayed on elements
    - Material property details in tooltips
    """
    fig = go.Figure()
    
    # Add columns
    col_x, col_y, col_names, col_hover = [], [], [], []
    for x_idx, x_pos in enumerate(coordx):
        for y_idx, y_pos in enumerate(coordy):
            section_name = column_assignments.get(floor, {}).get(x_idx, {}).get(y_idx, None)
            if section_name and section_name in sections:
                col_x.append(x_pos)
                col_y.append(y_pos)
                col_names.append(section_name)
                
                section_info = sections[section_name]
                material_name = section_info['material']
                material_info = materials[material_name]
                
                col_hover.append(
                    f"<b>{section_name}</b><br>"
                    f"X: {x_pos:.2f} m<br>"
                    f"Y: {y_pos:.2f} m<br>"
                    f"Material: {material_name}<br>"
                    f"f'c: {material_info['fc']:.1f} MPa<br>"
                    f"fy: {material_info['fy']:.1f} MPa"
                )
    
    fig.add_trace(go.Scatter(
        x=col_x, y=col_y,
        mode='markers+text',
        marker=dict(size=15, color='royalblue', symbol='square', line=dict(color='darkblue', width=2)),
        text=col_names,
        textposition='top center',
        textfont=dict(size=10, color='black'),
        hovertext=col_hover,
        hoverinfo='text',
        name='Columns'
    ))
    
    # Add beams X with markers
    beamx_mid_x, beamx_mid_y, beamx_names, beamx_hover = [], [], [], []
    
    for span_idx in range(len(coordx) - 1):
        for y_idx, y_pos in enumerate(coordy):
            section_name = beamx_assignments.get(floor, {}).get(span_idx, {}).get(y_idx, None)
            if section_name and section_name in sections:
                x_start = coordx[span_idx]
                x_end = coordx[span_idx + 1]
                
                # Beam line
                fig.add_trace(go.Scatter(
                    x=[x_start, x_end],
                    y=[y_pos, y_pos],
                    mode='lines',
                    line=dict(color='green', width=3),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Endpoint circles
                fig.add_trace(go.Scatter(
                    x=[x_start, x_end],
                    y=[y_pos, y_pos],
                    mode='markers',
                    marker=dict(size=6, color='darkgreen', symbol='circle'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Prepare data for midpoint diamonds
                x_mid = (x_start + x_end) / 2
                beamx_mid_x.append(x_mid)
                beamx_mid_y.append(y_pos)
                beamx_names.append(section_name)
                
                section_info = sections[section_name]
                material_name = section_info['material']
                material_info = materials[material_name]
                
                hover_text = (
                    f"<b>{section_name}</b><br>"
                    f"Span: X {x_start:.2f} → {x_end:.2f} m<br>"
                    f"Y: {y_pos:.2f} m<br>"
                    f"Material: {material_name}<br>"
                    f"f'c: {material_info['fc']:.1f} MPa<br>"
                    f"fy: {material_info['fy']:.1f} MPa"
                )
                beamx_hover.append(hover_text)
    
    # Add beam X diamonds at midpoints
    if beamx_mid_x:
        fig.add_trace(go.Scatter(
            x=beamx_mid_x,
            y=beamx_mid_y,
            mode='markers+text',
            marker=dict(size=10, color='green', symbol='diamond', line=dict(color='darkgreen', width=2)),
            text=beamx_names,
            textposition='top center',
            textfont=dict(size=9, color='black'),
            hovertext=beamx_hover,
            hoverinfo='text',
            name='Beams X',
            showlegend=False
        ))
    
    # Add beams Y with markers
    beamy_mid_x, beamy_mid_y, beamy_names, beamy_hover = [], [], [], []
    
    for x_idx, x_pos in enumerate(coordx):
        for span_idx in range(len(coordy) - 1):
            section_name = beamy_assignments.get(floor, {}).get(x_idx, {}).get(span_idx, None)
            if section_name and section_name in sections:
                y_start = coordy[span_idx]
                y_end = coordy[span_idx + 1]
                
                # Beam line
                fig.add_trace(go.Scatter(
                    x=[x_pos, x_pos],
                    y=[y_start, y_end],
                    mode='lines',
                    line=dict(color='orange', width=3),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Endpoint circles
                fig.add_trace(go.Scatter(
                    x=[x_pos, x_pos],
                    y=[y_start, y_end],
                    mode='markers',
                    marker=dict(size=6, color='darkorange', symbol='circle'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Prepare data for midpoint diamonds
                y_mid = (y_start + y_end) / 2
                beamy_mid_x.append(x_pos)
                beamy_mid_y.append(y_mid)
                beamy_names.append(section_name)
                
                section_info = sections[section_name]
                material_name = section_info['material']
                material_info = materials[material_name]
                
                hover_text = (
                    f"<b>{section_name}</b><br>"
                    f"X: {x_pos:.2f} m<br>"
                    f"Span: Y {y_start:.2f} → {y_end:.2f} m<br>"
                    f"Material: {material_name}<br>"
                    f"f'c: {material_info['fc']:.1f} MPa<br>"
                    f"fy: {material_info['fy']:.1f} MPa"
                )
                beamy_hover.append(hover_text)
    
    # Add beam Y diamonds at midpoints
    if beamy_mid_x:
        fig.add_trace(go.Scatter(
            x=beamy_mid_x,
            y=beamy_mid_y,
            mode='markers+text',
            marker=dict(size=10, color='orange', symbol='diamond', line=dict(color='darkorange', width=2)),
            text=beamy_names,
            textposition='top center',
            textfont=dict(size=9, color='black'),
            hovertext=beamy_hover,
            hoverinfo='text',
            name='Beams Y',
            showlegend=False
        ))
    
    # Update layout
    x_min, x_max = min(coordx) - 3, max(coordx) + 3
    y_min, y_max = min(coordy) - 3, max(coordy) + 3
    
    fig.update_layout(
        title=f'Floor {floor} - Complete Structural Layout',
        xaxis_title='X Coordinate (m)',
        yaxis_title='Y Coordinate (m)',
        showlegend=False,
        height=700,
        hovermode='closest',
        plot_bgcolor='#f0f0f0',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='white', zeroline=True, range=[x_min, x_max]),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='white', zeroline=True, range=[y_min, y_max])
    )
    
    return fig


# ==================== TAB 1: GEOMETRY ====================
def render_geometry_tab():
    """Render Building Geometry and Masses tab"""
    st.header("Building Geometry and Masses")
    st.markdown("Define the coordinates of nodes and the masses of each story floor.")
    
    # Project name
    st.subheader("Project Information")
    project_name = st.text_input(
        "Project Name",
        value=st.session_state.project_name,
        help="Name for the project (will be used for saving files)"
    )
    st.session_state.project_name = project_name
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Node Coordinates")
        
        coordx_input = st.text_area(
            "**X Coordinates (m)**",
            value="0, 8.6, 12.6",
            help="Example: 0, 8.6, 12.6"
        )
        
        coordy_input = st.text_area(
            "**Y Coordinates (m)**",
            value="0, 7.3, 14.6, 21.9",
            help="Example: 0, 7.3, 14.6, 21.9"
        )
        
        coordz_input = st.text_area(
            "**Z Coordinates - Story Heights (m)**",
            value="0, 3, 6, 9, 12, 15",
            help="Example: 0, 3, 6, 9, 12, 15"
        )
    
    with col2:
        st.subheader("Story Masses")
        
        masas_input = st.text_area(
            "**Masses per Story (kN·s²/m)**",
            value="210, 210, 210, 210, 210",
            help="One mass value per floor (excluding base). Example: 210, 210, 210, 210, 210"
        )
    
    st.markdown("---")
    
    # Create model button
    if st.button("Create Model", type="primary", key="create_model"):
        try:
            # Parse inputs
            coordx = parse_coordinates(coordx_input)
            coordy = parse_coordinates(coordy_input)
            coordz = parse_coordinates(coordz_input)
            masas = parse_coordinates(masas_input)
            
            # Validate
            if len(masas) != len(coordz) - 1:
                st.error(f"⚠️ Number of masses ({len(masas)}) must equal number of floors ({len(coordz) - 1})")
            else:
                # Store in session state
                st.session_state.coordx = coordx
                st.session_state.coordy = coordy
                st.session_state.coordz = coordz
                st.session_state.masas = masas
                
                # Create model
                create_opensees_model(coordx, coordy, coordz, masas)
                
                st.session_state.model_created = True
                st.success("✅ Model created successfully!")
                
                # Display summary
                st.markdown("### Model Summary")
                col_sum1, col_sum2, col_sum3 = st.columns(3)
                with col_sum1:
                    st.metric("Total Nodes", len(coordx) * len(coordy) * len(coordz))
                with col_sum2:
                    st.metric("Number of Stories", len(coordz) - 1)
                with col_sum3:
                    st.metric("Total Mass", f"{sum(masas):.2f}")
                
        except Exception as e:
            st.error(f"❌ Error creating model: {str(e)}")
    
    # Display current configuration
    if st.session_state.model_created:
        st.markdown("---")
        st.success("✅ Model is ready!")
        
        with st.expander("View Current Configuration"):
            col_conf1, col_conf2 = st.columns(2)
            with col_conf1:
                st.write("**X Coordinates:**", st.session_state.coordx)
                st.write("**Y Coordinates:**", st.session_state.coordy)
                st.write("**Z Coordinates:**", st.session_state.coordz)
            with col_conf2:
                st.write("**Masses:**", st.session_state.masas)


# ==================== TAB 2: MATERIALS ====================
def render_materials_tab():
    """Render Material Definition tab"""
    st.header("Material Definition")
    st.markdown("Define concrete and steel materials for your structural elements.")
    
    if not st.session_state.model_created:
        st.warning("⚠️ Please create the model in the 'Building Geometry and Masses' tab first.")
        return
    
    st.subheader("Add New Material Set")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        material_name = st.text_input(
            "Material Name",
            value="",
            placeholder="e.g., Concrete_Column_1",
            help="Enter a unique name for this material"
        )
    
    with col2:
        fc = st.number_input(
            "f'c - Concrete Strength (MPa)",
            min_value=10.0,
            max_value=100.0,
            value=28.0,
            step=1.0,
            help="Concrete compressive strength"
        )
    
    with col3:
        fy = st.number_input(
            "fy - Steel Yield Strength (MPa)",
            min_value=200.0,
            max_value=600.0,
            value=420.0,
            step=10.0,
            help="Steel reinforcement yield strength"
        )
    
    detailing_level = st.selectbox(
        "Detailing Level",
        options=DETAILING_OPTIONS,
        help="DES: Special detailing, DMO: Moderate detailing, PreCode: Pre-code"
    )
    
    if st.button("Add Material", type="primary", key="add_material"):
        if material_name == "":
            st.error("⚠️ Please enter a material name")
        elif material_name in st.session_state.materials:
            st.error(f"⚠️ Material '{material_name}' already exists. Use a different name.")
        else:
            try:
                material_index = len(st.session_state.materials)
                material_props = create_material(fc, fy, detailing_level, material_index)
                st.session_state.materials[material_name] = material_props
                st.success(f"✅ Material '{material_name}' added successfully!")
            except Exception as e:
                st.error(f"❌ Error creating material: {str(e)}")
    
    # Display existing materials
    if st.session_state.materials:
        st.markdown("---")
        st.subheader("Defined Materials Sets")
        
        materials_data = []
        for name, props in st.session_state.materials.items():
            materials_data.append({
                'Material Set Name': name,
                "f'c (MPa)": props['fc'],
                'fy (MPa)': props['fy'],
                'Detailing': props['detailing'],
                'Unconfined Tag': props['noconf_tag'],
                'Confined Tag': props['conf_tag'],
                'Steel Tag': props['acero_tag']
            })
        
        df_materials = pd.DataFrame(materials_data)
        st.dataframe(df_materials, use_container_width=True)
    else:
        st.info("ℹ️ No materials defined yet. Add your first material above.")


# ==================== TAB 3: SECTIONS ====================
def render_sections_tab():
    """Render Sections tab"""
    st.header("Sections")
    st.markdown("Define cross-sections for structural elements (columns and beams).")
    
    if not st.session_state.model_created:
        st.warning("⚠️ Please create the model in the 'Building Geometry and Masses' tab first.")
        return
    elif not st.session_state.materials:
        st.warning("⚠️ Please define at least one material in the 'Materials' tab first.")
        return
    
    st.subheader("Add New Section")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        section_name = st.text_input(
            "Section Name",
            value="",
            placeholder="e.g., Col_30x40",
            help="Enter a unique name for this section"
        )
        
        H_section = st.number_input(
            "H - Section Height (m)",
            min_value=0.10,
            max_value=5.0,
            value=0.40,
            step=0.05,
            format="%.3f",
            help="Height of the section"
        )
        
        B_section = st.number_input(
            "B - Section Base (m)",
            min_value=0.10,
            max_value=5.0,
            value=0.40,
            step=0.05,
            format="%.3f",
            help="Base width of the section"
        )
    
    with col2:
        cover = st.number_input(
            "c - Cover (m)",
            min_value=0.01,
            max_value=0.20,
            value=0.05,
            step=0.01,
            format="%.3f",
            help="Concrete cover"
        )
        
        material_name_select = st.selectbox(
            "Select Material",
            options=list(st.session_state.materials.keys()),
            help="Choose the material for this section"
        )
        
        st.markdown("**Top Reinforcement**")
        bars_top = st.number_input(
            "Number of bars (top)",
            min_value=1,
            max_value=20,
            value=2,
            step=1,
            help="Number of reinforcement bars at top"
        )
        
        area_top = st.selectbox(
            "Bar size (top)",
            options=list(REBAR_AREAS.keys()),
            index=2,
            help="Select rebar size for top bars"
        )
    
    with col3:
        st.markdown("**Bottom Reinforcement**")
        bars_bottom = st.number_input(
            "Number of bars (bottom)",
            min_value=1,
            max_value=20,
            value=2,
            step=1,
            help="Number of reinforcement bars at bottom"
        )
        
        area_bottom = st.selectbox(
            "Bar size (bottom)",
            options=list(REBAR_AREAS.keys()),
            index=2,
            help="Select rebar size for bottom bars"
        )
        
        st.markdown("**Middle Reinforcement**")
        bars_middle = st.number_input(
            "Number of bars (middle)",
            min_value=0,
            max_value=20,
            value=6,
            step=2,
            help="Number of reinforcement bars at middle"
        )
        
        area_middle = st.selectbox(
            "Bar size (middle)",
            options=list(REBAR_AREAS.keys()),
            index=2,
            help="Select rebar size for middle bars"
        )
    
    if st.button("Add Section", type="primary", key="add_section"):
        if section_name == "":
            st.error("⚠️ Please enter a section name")
        elif section_name in st.session_state.sections:
            st.error(f"⚠️ Section '{section_name}' already exists. Use a different name.")
        else:
            try:
                section_index = len(st.session_state.sections)
                section_tag = generate_section_tag(section_index)
                
                selected_material = st.session_state.materials[material_name_select]
                
                bars_config = {
                    'bars_top': bars_top,
                    'area_top': REBAR_AREAS[area_top],
                    'bars_bottom': bars_bottom,
                    'area_bottom': REBAR_AREAS[area_bottom],
                    'bars_middle': bars_middle,
                    'area_middle': REBAR_AREAS[area_middle],
                }
                
                create_section(section_tag, H_section, B_section, cover, selected_material, bars_config)
                
                st.session_state.sections[section_name] = {
                    'tag': section_tag,
                    'H': H_section,
                    'B': B_section,
                    'cover': cover,
                    'material': material_name_select,
                    'bars_top': bars_top,
                    'area_top': area_top,
                    'bars_bottom': bars_bottom,
                    'area_bottom': area_bottom,
                    'bars_middle': bars_middle,
                    'area_middle': area_middle
                }
                
                st.success(f"✅ Section '{section_name}' added successfully!")
                
            except Exception as e:
                st.error(f"❌ Error creating section: {str(e)}")
    
    # Display existing sections
    if st.session_state.sections:
        st.markdown("---")
        st.subheader("Defined Sections")
        
        sections_data = []
        for name, props in st.session_state.sections.items():
            sections_data.append({
                'Section Name': name,
                'Tag': props['tag'],
                'H (m)': f"{props['H']:.3f}",
                'B (m)': f"{props['B']:.3f}",
                'Cover (m)': f"{props['cover']:.3f}",
                'Material': props['material'],
                'Top Bars': f"{props['bars_top']}x{props['area_top']}",
                'Bottom Bars': f"{props['bars_bottom']}x{props['area_bottom']}",
                'Middle Bars': f"{props['bars_middle']}x{props['area_middle']}"
            })
        
        df_sections = pd.DataFrame(sections_data)
        st.dataframe(df_sections, use_container_width=True)
    else:
        st.info("ℹ️ No sections defined yet. Add your first section above.")


# ==================== TAB 4: COLUMNS ====================
def render_columns_tab():
    """Render Columns Assignment tab"""
    st.header("Column Assignment")
    st.markdown("Assign sections to each column in the building.")
    
    if not st.session_state.model_created:
        st.warning("⚠️ Please create the model in the 'Building Geometry and Masses' tab first.")
        return
    elif not st.session_state.sections:
        st.warning("⚠️ Please define at least one section in the 'Sections' tab first.")
        return
    
    # Get building dimensions
    n_floors = len(st.session_state.coordz) - 1
    n_x_axes = len(st.session_state.coordx)
    n_y_positions = len(st.session_state.coordy)
    
    # Get list of section names
    section_names = list(st.session_state.sections.keys())
    
    st.markdown(f"""
    **Building Configuration:**
    - Number of Floors: {n_floors}
    - Number of X Axes: {n_x_axes}
    - Number of Y Positions: {n_y_positions}
    - Total Columns per Floor: {n_x_axes * n_y_positions}
    """)
    
    # Select floor to configure
    selected_floor = st.selectbox(
        "Select Floor to Configure",
        options=list(range(1, n_floors + 1)),
        format_func=lambda x: f"Floor {x}",
        help="Select which floor you want to assign column sections to"
    )
    
    st.markdown(f"### Floor {selected_floor} - Column Section Assignment")
    
    # Initialize column assignments for this floor if not exists
    if selected_floor not in st.session_state.column_assignments:
        st.session_state.column_assignments[selected_floor] = {}
        for x_idx in range(n_x_axes):
            st.session_state.column_assignments[selected_floor][x_idx] = {}
            for y_idx in range(n_y_positions):
                st.session_state.column_assignments[selected_floor][x_idx][y_idx] = section_names[0]
    
    # Create grid for column assignment
    st.markdown("---")
    
    for x_idx in range(n_x_axes):
        x_coord = st.session_state.coordx[x_idx]
        st.markdown(f"**X Axis {x_idx + 1} (X = {x_coord:.2f} m)**")
        
        cols = st.columns(n_y_positions)
        
        for y_idx in range(n_y_positions):
            y_coord = st.session_state.coordy[y_idx]
            
            with cols[y_idx]:
                current_section = st.session_state.column_assignments[selected_floor][x_idx][y_idx]
                
                new_section = st.selectbox(
                    f"Y={y_coord:.2f}m",
                    options=section_names,
                    index=section_names.index(current_section),
                    key=f"col_f{selected_floor}_x{x_idx}_y{y_idx}"
                )
                
                st.session_state.column_assignments[selected_floor][x_idx][y_idx] = new_section
        
        st.markdown("---")
    
    # Copy configuration
    st.markdown("### Copy Configuration")
    col_copy1, col_copy2 = st.columns(2)
    
    with col_copy1:
        copy_to_floors = st.multiselect(
            "Copy current floor configuration to:",
            options=[f for f in range(1, n_floors + 1) if f != selected_floor],
            format_func=lambda x: f"Floor {x}"
        )
    
    with col_copy2:
        if st.button("Copy Configuration", type="secondary", key="col_copy"):
            if copy_to_floors:
                copy_floor_assignments(selected_floor, copy_to_floors, st.session_state.column_assignments)
                st.success(f"✅ Configuration copied to {len(copy_to_floors)} floor(s)!")


# ==================== TAB 5: BEAMS ====================
def render_beams_tab():
    """Render Beams Assignment tab"""
    st.header("Beam Assignment")
    st.markdown("Assign sections to beams in X and Y directions.")
    
    if not st.session_state.model_created:
        st.warning("⚠️ Please create the model in the 'Building Geometry and Masses' tab first.")
        return
    elif not st.session_state.sections:
        st.warning("⚠️ Please define at least one section in the 'Sections' tab first.")
        return
    
    # Get building dimensions
    n_floors = len(st.session_state.coordz) - 1
    n_x_spans = len(st.session_state.coordx) - 1
    n_y_spans = len(st.session_state.coordy) - 1
    n_y_positions = len(st.session_state.coordy)
    n_x_positions = len(st.session_state.coordx)
    
    section_names = list(st.session_state.sections.keys())
    
    # Create subtabs
    beam_tab1, beam_tab2 = st.tabs(["Beams in X Direction", "Beams in Y Direction"])
    
    # BEAMS X
    with beam_tab1:
        render_beams_x_subtab(n_floors, n_x_spans, n_y_positions, section_names)
    
    # BEAMS Y
    with beam_tab2:
        render_beams_y_subtab(n_floors, n_x_positions, n_y_spans, section_names)


def render_beams_x_subtab(n_floors, n_x_spans, n_y_positions, section_names):
    """Render Beams X direction subtab"""
    st.subheader("Beams in X Direction")
    
    selected_floor_x = st.selectbox(
        "Select Floor to Configure",
        options=list(range(1, n_floors + 1)),
        format_func=lambda x: f"Floor {x}",
        key="beamx_floor_select"
    )
    
    # Initialize
    if selected_floor_x not in st.session_state.beamx_assignments:
        st.session_state.beamx_assignments[selected_floor_x] = initialize_floor_assignments(
            selected_floor_x, n_x_spans, n_y_positions, section_names[0]
        )
    
    st.markdown("---")
    
    for span_idx in range(n_x_spans):
        x_start = st.session_state.coordx[span_idx]
        x_end = st.session_state.coordx[span_idx + 1]
        st.markdown(f"**X Span {span_idx + 1} (X: {x_start:.2f} m → {x_end:.2f} m)**")
        
        cols = st.columns(n_y_positions)
        
        for y_idx in range(n_y_positions):
            y_coord = st.session_state.coordy[y_idx]
            
            with cols[y_idx]:
                current_section = st.session_state.beamx_assignments[selected_floor_x][span_idx][y_idx]
                
                new_section = st.selectbox(
                    f"Y={y_coord:.2f}m",
                    options=section_names,
                    index=section_names.index(current_section),
                    key=f"beamx_f{selected_floor_x}_s{span_idx}_y{y_idx}"
                )
                
                st.session_state.beamx_assignments[selected_floor_x][span_idx][y_idx] = new_section
        
        st.markdown("---")
    
    # Copy configuration
    st.markdown("### Copy Configuration")
    col_copy1, col_copy2 = st.columns(2)
    
    with col_copy1:
        copy_to_floors_x = st.multiselect(
            "Copy current floor configuration to:",
            options=[f for f in range(1, n_floors + 1) if f != selected_floor_x],
            format_func=lambda x: f"Floor {x}",
            key="beamx_copy_select"
        )
    
    with col_copy2:
        if st.button("Copy Configuration", type="secondary", key="beamx_copy_button"):
            if copy_to_floors_x:
                copy_floor_assignments(selected_floor_x, copy_to_floors_x, st.session_state.beamx_assignments)
                st.success(f"✅ Configuration copied to {len(copy_to_floors_x)} floor(s)!")


def render_beams_y_subtab(n_floors, n_x_positions, n_y_spans, section_names):
    """Render Beams Y direction subtab"""
    st.subheader("Beams in Y Direction")
    
    selected_floor_y = st.selectbox(
        "Select Floor to Configure",
        options=list(range(1, n_floors + 1)),
        format_func=lambda x: f"Floor {x}",
        key="beamy_floor_select"
    )
    
    # Initialize
    if selected_floor_y not in st.session_state.beamy_assignments:
        st.session_state.beamy_assignments[selected_floor_y] = initialize_floor_assignments(
            selected_floor_y, n_x_positions, n_y_spans, section_names[0]
        )
    
    st.markdown("---")
    
    for x_idx in range(n_x_positions):
        x_coord = st.session_state.coordx[x_idx]
        st.markdown(f"**X Position {x_idx + 1} (X = {x_coord:.2f} m)**")
        
        cols = st.columns(n_y_spans)
        
        for span_idx in range(n_y_spans):
            y_start = st.session_state.coordy[span_idx]
            y_end = st.session_state.coordy[span_idx + 1]
            
            with cols[span_idx]:
                current_section = st.session_state.beamy_assignments[selected_floor_y][x_idx][span_idx]
                
                new_section = st.selectbox(
                    f"Y: {y_start:.2f}→{y_end:.2f}m",
                    options=section_names,
                    index=section_names.index(current_section),
                    key=f"beamy_f{selected_floor_y}_x{x_idx}_s{span_idx}"
                )
                
                st.session_state.beamy_assignments[selected_floor_y][x_idx][span_idx] = new_section
        
        st.markdown("---")
    
    # Copy configuration
    st.markdown("### Copy Configuration")
    col_copy1, col_copy2 = st.columns(2)
    
    with col_copy1:
        copy_to_floors_y = st.multiselect(
            "Copy current floor configuration to:",
            options=[f for f in range(1, n_floors + 1) if f != selected_floor_y],
            format_func=lambda x: f"Floor {x}",
            key="beamy_copy_select"
        )
    
    with col_copy2:
        if st.button("Copy Configuration", type="secondary", key="beamy_copy_button"):
            if copy_to_floors_y:
                copy_floor_assignments(selected_floor_y, copy_to_floors_y, st.session_state.beamy_assignments)
                st.success(f"✅ Configuration copied to {len(copy_to_floors_y)} floor(s)!")


# ==================== TAB 6: MODEL VISUALIZATION ====================
def render_model_visualization_tab():
    """Render Model Visualization tab"""
    st.header("Model Visualization")
    st.markdown("Visualize the complete structural model and create elements.")
    
    if not st.session_state.model_created:
        st.warning("⚠️ Please create the model first.")
        return
    elif not st.session_state.sections:
        st.warning("⚠️ Please define sections first.")
        return
    
    # Check if all assignments are complete
    n_floors = len(st.session_state.coordz) - 1
    all_columns_configured = all(f in st.session_state.column_assignments for f in range(1, n_floors + 1))
    all_beamx_configured = all(f in st.session_state.beamx_assignments for f in range(1, n_floors + 1))
    all_beamy_configured = all(f in st.session_state.beamy_assignments for f in range(1, n_floors + 1))
    
    if not (all_columns_configured and all_beamx_configured and all_beamy_configured):
        st.warning("⚠️ Please complete all column and beam assignments before visualizing.")
        return
    
    st.success("✅ All assignments complete!")
    
    # Floor selector
    viz_floor = st.selectbox(
        "Select Floor to Visualize",
        options=list(range(1, n_floors + 1)),
        format_func=lambda x: f"Floor {x}",
        key="viz_floor_model_select"
    )
    
    # Create visualization
    fig = create_floor_plan_figure(
        viz_floor,
        st.session_state.coordx,
        st.session_state.coordy,
        st.session_state.column_assignments,
        st.session_state.beamx_assignments,
        st.session_state.beamy_assignments,
        st.session_state.sections,
        st.session_state.materials
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Create elements
    st.markdown("---")
    st.subheader("Create Structural Elements")
    
    if st.session_state.elements_created:
        st.success("✅ Elements have been created successfully!")
    else:
        if st.button("🏗️ Create Elements", type="primary", key="create_elements_button"):
            try:
                with st.spinner("Creating structural elements..."):
                    n_x_positions = len(st.session_state.coordx)
                    n_y_positions = len(st.session_state.coordy)
                    n_x_spans = len(st.session_state.coordx) - 1
                    n_y_spans = len(st.session_state.coordy) - 1
                    
                    # Build tag lists
                    coltags = build_element_tags_list(
                        st.session_state.column_assignments,
                        st.session_state.sections,
                        n_floors,
                        n_x_positions,
                        n_y_positions
                    )
                    
                    vigased = build_element_tags_list(
                        st.session_state.beamx_assignments,
                        st.session_state.sections,
                        n_floors,
                        n_x_spans,
                        n_y_positions
                    )
                    
                    vigasedY = build_element_tags_list(
                        st.session_state.beamy_assignments,
                        st.session_state.sections,
                        n_floors,
                        n_x_positions,
                        n_y_spans
                    )
                    
                    # Create elements
                    cols, vigx, vigy, sectag_col, sectag_vigx, sectag_vigy = ut.create_elements3D2(
                        st.session_state.coordx,
                        st.session_state.coordy,
                        st.session_state.coordz,
                        coltags,
                        vigased,
                        vigasedY
                    )
                    
                    st.session_state.cols = cols
                    st.session_state.vigx = vigx
                    st.session_state.vigy = vigy
                    st.session_state.elements_created = True
                    
                    st.success("✅ Elements created successfully!")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Error creating elements: {str(e)}")


# ==================== TAB 7: SLABS ====================
def render_slabs_tab():
    """Render Slabs tab"""
    st.header("Slab Definition")
    st.markdown("Define slab properties and create slabs for all floors.")
    
    if not st.session_state.model_created:
        st.warning("⚠️ Please create the model first.")
        return
    elif not st.session_state.elements_created:
        st.warning("⚠️ Please create structural elements first.")
        return
    
    # Option to skip slab creation
    skip_slabs = st.checkbox("Do not create slabs", value=False, key="skip_slabs")
    
    if skip_slabs:
        st.info("ℹ️ Slabs will not be created. You can proceed to the next tab.")
        if not st.session_state.slabs_created:
            st.session_state.slabs_created = True
            st.session_state.slabs_skipped = True
            st.success("✅ Slab creation skipped. You can now proceed to loads.")
    else:
        st.subheader("Slab Properties")
        
        col1, col2 = st.columns(2)
        
        with col1:
            hslab = st.number_input("Slab Thickness (m)", min_value=0.05, max_value=0.50, value=0.15, step=0.01, format="%.3f")
            fc_slab = st.number_input("f'c - Concrete Strength (MPa)", min_value=10.0, max_value=100.0, value=28.0, step=1.0)
            Eslab_auto = 1000 * 4400 * (fc_slab ** 0.5)
            st.info(f"📊 Auto-calculated E: {Eslab_auto:.2f} kPa")
        
        with col2:
            Eslab = st.number_input("E - Elastic Modulus (kPa)", min_value=1000.0, value=Eslab_auto, step=1000.0, format="%.2f")
            pois = st.number_input("ν - Poisson's Ratio", min_value=0.0, max_value=0.5, value=0.3, step=0.01, format="%.2f")
        
        st.markdown("---")
        
        if st.session_state.slabs_created:
            if hasattr(st.session_state, 'slabs_skipped') and st.session_state.slabs_skipped:
                st.info("ℹ️ Slab creation was skipped.")
            else:
                st.success("✅ Slabs have been created successfully!")
        else:
            if st.button("🏗️ Create Slabs", type="primary", key="create_slabs_button"):
                try:
                    with st.spinner("Creating slabs..."):
                        ut.create_slabs(
                            st.session_state.coordx,
                            st.session_state.coordy,
                            st.session_state.coordz,
                            hslab,
                            Eslab,
                            pois
                        )
                        
                        st.session_state.hslab = hslab
                        st.session_state.fc_slab = fc_slab
                        st.session_state.Eslab = Eslab
                        st.session_state.pois = pois
                        st.session_state.slabs_created = True
                        st.session_state.slabs_skipped = False
                        
                        st.success("✅ Slabs created successfully!")
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Error creating slabs: {str(e)}")


# ==================== TAB 8: LOADS ====================
def render_loads_tab():
    """Render Loads tab"""
    st.header("Loads")
    st.markdown("Define gravity loads on beams for floors and roof.")
    
    if not st.session_state.model_created:
        st.warning("⚠️ Please create the model first.")
        return
    elif not st.session_state.elements_created:
        st.warning("⚠️ Please create structural elements first.")
        return
    elif not st.session_state.slabs_created:
        st.warning("⚠️ Please create slabs first.")
        return
    
    st.subheader("Beam Load Definition")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Typical Floor Loads")
        floorx = st.number_input("Floor Load - X Beams (kN/m)", min_value=0.0, max_value=500.0, value=30.0, step=1.0)
        floory = st.number_input("Floor Load - Y Beams (kN/m)", min_value=0.0, max_value=500.0, value=30.0, step=1.0)
    
    with col2:
        st.markdown("#### Roof Loads")
        roofx = st.number_input("Roof Load - X Beams (kN/m)", min_value=0.0, max_value=500.0, value=20.0, step=1.0)
        roofy = st.number_input("Roof Load - Y Beams (kN/m)", min_value=0.0, max_value=500.0, value=20.0, step=1.0)
    
    st.markdown("---")
    
    if st.session_state.loads_applied:
        st.success("✅ Loads have been applied successfully!")
    else:
        if st.button("📊 Apply Loads", type="primary", key="apply_loads_button"):
            try:
                with st.spinner("Applying loads..."):
                    ut.load_beams3D(
                        -floorx, -floory, -roofx, -roofy,
                        st.session_state.vigx,
                        st.session_state.vigy,
                        st.session_state.coordx,
                        st.session_state.coordy
                    )
                    
                    st.session_state.floorx = floorx
                    st.session_state.floory = floory
                    st.session_state.roofx = roofx
                    st.session_state.roofy = roofy
                    st.session_state.loads_applied = True
                    
                    st.success("✅ Loads applied successfully!")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Error applying loads: {str(e)}")


# ==================== TAB 9: MODAL ANALYSIS ====================
def render_modal_analysis_tab():
    """Render Modal Analysis tab"""
    st.header("Modal Analysis")
    st.markdown("Perform eigenvalue analysis to determine natural frequencies and mode shapes.")
    
    if not st.session_state.model_created:
        st.warning("⚠️ Please create the model first.")
        return
    elif not st.session_state.loads_applied:
        st.warning("⚠️ Please apply loads first.")
        return
    
    n_floors = len(st.session_state.coordz) - 1
    n_modes = n_floors * 2
    
    st.info(f"📊 Number of modes to calculate: {n_modes} (2 per floor)")
    
    if st.session_state.modal_analysis_done:
        st.success("✅ Modal analysis has been completed!")
        
        try:
            if os.path.exists('ModalReport.txt'):
                with open('ModalReport.txt', 'r') as f:
                    modal_output = f.read()
                st.code(modal_output, language="text")
        except Exception as e:
            st.error(f"❌ Error reading modal report: {str(e)}")
    else:
        if st.button("🔬 Run Modal Analysis", type="primary", key="run_modal_button"):
            try:
                with st.spinner("Running eigenvalue analysis..."):
                    from openseespy.opensees import eigen, modalProperties
                    
                    eig = eigen('-fullGenLapack', n_modes)
                    modalProperties('-print', '-file', 'ModalReport.txt', '-unorm')
                    
                    st.session_state.eig = eig
                    st.session_state.modal_analysis_done = True
                    
                    st.success("✅ Modal analysis completed successfully!")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Error running modal analysis: {str(e)}")


# ==================== TAB 10: GRAVITY AND PUSHOVER ====================
def render_gravity_pushover_tab():
    """Render Gravity and Pushover Analysis tab"""
    st.header("Gravity and Pushover Analysis")
    st.markdown("Run gravity analysis and nonlinear pushover analysis.")
    
    if not st.session_state.model_created:
        st.warning("⚠️ Please create the model first.")
        return
    elif not st.session_state.modal_analysis_done:
        st.warning("⚠️ Please complete modal analysis first.")
        return
    
    # GRAVITY ANALYSIS
    st.subheader("1. Gravity Analysis")
    
    if st.session_state.gravity_analysis_done:
        st.success("✅ Gravity analysis completed!")
    else:
        if st.button("⚖️ Run Gravity Analysis", type="primary", key="run_gravity_button"):
            try:
                with st.spinner("Running gravity analysis..."):
                    from openseespy.opensees import loadConst
                    an.gravedad()
                    loadConst('-time', 0.0)
                    
                    st.session_state.gravity_analysis_done = True
                    st.success("✅ Gravity analysis completed successfully!")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Error running gravity analysis: {str(e)}")
    
    # PUSHOVER ANALYSIS
    st.markdown("---")
    st.subheader("2. Pushover Analysis")
    
    if not st.session_state.gravity_analysis_done:
        st.warning("⚠️ Please run gravity analysis first.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        push_direction = st.selectbox("Push Direction", options=["X", "Y"])
        dir_value = 1 if push_direction == "X" else 2
        pushdir_str = push_direction.lower()
    
    with col2:
        target_percent = st.number_input("Target Displacement (%)", min_value=0.1, max_value=10.0, value=2.5, step=0.1)
    
    col3, col4 = st.columns(2)
    
    with col3:
        dincr = st.number_input("Displacement Increment (m)", min_value=0.0001, max_value=0.1, value=0.0025, step=0.0001, format="%.4f")
    
    with col4:
        tolerance = st.number_input("Tolerance", min_value=1e-6, max_value=10.0, value=1e-2, format="%.1e")
    
    building_height = st.session_state.coordz[-1]
    pushlimit = (target_percent / 100.0) * building_height
    
    st.markdown("---")
    
    if st.session_state.pushover_analysis_done:
        st.success("✅ Pushover analysis completed!")
        
        if 'dtecho' in st.session_state and 'Vbasal' in st.session_state:
            # Plot pushover curve
            fig_push = go.Figure()
            fig_push.add_trace(go.Scatter(
                x=st.session_state.dtecho,
                y=st.session_state.Vbasal,
                mode='lines+markers',
                name='Base Shear',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ))
            
            fig_push.update_layout(
                title=f'Pushover Curve - {push_direction} Direction',
                xaxis_title='Roof Displacement (m)',
                yaxis_title='Base Shear (kN)',
                height=500
            )
            
            st.plotly_chart(fig_push, use_container_width=True)
    else:
        if st.button("🔄 Run Pushover Analysis", type="primary", key="run_pushover_button"):
            try:
                with st.spinner(f"Running pushover analysis in {push_direction} direction..."):
                    import time as time_module
                    
                    ut.pushover_loads3D(st.session_state.coordz, pushdir=pushdir_str)
                    
                    stime = time_module.time()
                    dtecho, Vbasal = an.pushover2(
                        pushlimit,
                        dincr,
                        int(len(st.session_state.coordz) - 1),
                        int(dir_value),
                        Tol=tolerance
                    )
                    etime = time_module.time()
                    ttotal = etime - stime
                    
                    st.session_state.dtecho = dtecho
                    st.session_state.Vbasal = Vbasal
                    st.session_state.pushover_time = ttotal
                    st.session_state.pushover_direction = push_direction
                    st.session_state.pushover_analysis_done = True
                    
                    st.success(f"✅ Pushover analysis completed in {ttotal:.2f} seconds!")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Error running pushover analysis: {str(e)}")


# ==================== TAB 11: MODEL SAVING ====================
def render_save_model_tab():
    """Render Save Model tab"""
    st.header("Save Model")
    st.markdown("Save the complete model configuration to a file for later analysis.")
    
    if not st.session_state.model_created:
        st.warning("⚠️ Please create a model first.")
        return
    
    if not st.session_state.materials:
        st.warning("⚠️ Please define at least one material.")
        return
    
    if not st.session_state.sections:
        st.warning("⚠️ Please define at least one section.")
        return
    
    # Check assignments
    num_floors = len(st.session_state.coordz) - 1
    all_columns_configured = all(f in st.session_state.column_assignments for f in range(1, num_floors + 1))
    all_beamx_configured = all(f in st.session_state.beamx_assignments for f in range(1, num_floors + 1))
    all_beamy_configured = all(f in st.session_state.beamy_assignments for f in range(1, num_floors + 1))
    
    if not (all_columns_configured and all_beamx_configured and all_beamy_configured):
        st.warning("⚠️ Please complete all column and beam assignments before saving.")
        return
    
    st.success("✅ Model is ready to be saved!")
    
    st.markdown("---")
    st.subheader("Save Configuration")
    
    project_name = st.text_input(
        "Filename (without extension):",
        value=st.session_state.project_name,
        help="Name for the saved model file"
    )
    
    if st.button("💾 Save Model", type="primary", key="save_model_button"):
        try:
            file_path = save_model_to_file(project_name)
            st.success(f"✅ Model saved successfully to: {file_path}")
            st.info("💡 You can now load this model in the app_analysis.py application.")
        except Exception as e:
            st.error(f"❌ Error saving model: {str(e)}")


# ==================== MAIN APPLICATION ====================
def main():
    """
    Main application entry point for the RCF-3D Building Model Creator.

    This function initializes the Streamlit application with proper configuration,
    sets up the tabbed interface, and orchestrates the complete building modeling
    workflow from geometry definition to model export.

    Application Structure:
    ---------------------
    1. Page Configuration: Wide layout, custom title and icon
    2. Session State: Initialize all required state variables
    3. Tab Interface: 11 tabs for sequential workflow
    4. Content Rendering: Call appropriate render functions for each tab
    5. Workflow Management: Ensure proper tab sequencing and dependencies

    Tab Sequence:
    -------------
    1. Building Geometry and Masses - Define structural layout
    2. Materials - Create concrete/steel material sets
    3. Sections - Define cross-sectional properties
    4. Columns - Assign sections to column positions
    5. Beams - Assign sections to beam spans
    6. Model Visualization - Review and create elements
    7. Slabs - Define floor diaphragms
    8. Loads - Apply gravity loading
    9. Save Model - Export for analysis
    10. Modal Analysis - Eigenvalue analysis
    11. Gravity + Pushover - Nonlinear analysis

    Dependencies:
    ------------
    - Streamlit for web interface
    - OpenSeesPy for structural modeling
    - Plotly for interactive visualizations
    - Session state for workflow persistence
    """
    # Page configuration
    st.set_page_config(
        page_title="3D Building Model Creator",
        page_icon="🏢",
        layout="wide"
    )
    
    st.title("🏢 3D Building Analysis with OpenSeesPy")
    st.markdown("---")
    
    # Initialize session state
    initialize_session_state()
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
        "Building Geometry and Masses",
        "Materials",
        "Sections",
        "Columns",
        "Beams",
        "Model Visualization",
        "Slabs",
        "Loads",
        "Create and Save Model",
        "Modal Analysis",
        "Gravity & Pushover"
    ])
    
    with tab1:
        render_geometry_tab()
    
    with tab2:
        render_materials_tab()
    
    with tab3:
        render_sections_tab()
    
    with tab4:
        render_columns_tab()
    
    with tab5:
        render_beams_tab()
    
    with tab6:
        render_model_visualization_tab()
    
    with tab7:
        render_slabs_tab()
    
    with tab8:
        render_loads_tab()
    
    with tab9:
        render_save_model_tab()
    
    with tab10:
        render_modal_analysis_tab()
    
    with tab11:
        render_gravity_pushover_tab()


if __name__ == "__main__":
    main()
