"""
RCF-3D Pushover Library
=======================

This module provides utility functions for working with 3D RC frame models
created by the RCF-3D Pushover application.

Functions:
----------
- load_and_create_model: Load a .pkl model file and recreate the OpenSeesPy model
"""

import pickle
import os
import openseespy.opensees as ops
import opseestools.utilidades as ut
import opseestools.analisis3D as an


def load_and_create_model(pkl_file_path):
    """
    Load a model from a .pkl file and recreate the complete OpenSeesPy model.
    
    This function reads a pickled model file created by app_modelcreator.py and 
    reconstructs the entire 3D RC frame model in OpenSeesPy, including:
    - Building geometry (grid and masses)
    - Material models (unconfined/confined concrete and steel)
    - RC sections (columns and beams)
    - Elements (columns, beams X, beams Y)
    - Slabs (if created)
    - Loads (if applied)
    
    Parameters
    ----------
    pkl_file_path : str
        Path to the .pkl file containing the model data.
        Can be absolute or relative path.
    
    Returns
    -------
    model_data : dict
        Dictionary containing all model parameters and metadata:
        - 'project_name': Name of the project
        - 'coordx', 'coordy', 'coordz': Building coordinates (lists)
        - 'masas': Floor masses (list)
        - 'materials': List of material dictionaries
        - 'sections': List of section dictionaries
        - 'column_assignments': Column-to-section assignments [floor][x][y]
        - 'beamx_assignments': Beam X assignments [floor][span][y]
        - 'beamy_assignments': Beam Y assignments [floor][x][span]
        - 'hslab', 'fc_slab', 'Eslab', 'pois': Slab parameters (if slabs_created)
        - 'floorx', 'floory', 'roofx', 'roofy': Load values (if loads_applied)
        - Status flags: model_created, elements_created, slabs_created, 
                       loads_applied, modal_analysis_done, etc.
    
    Raises
    ------
    FileNotFoundError
        If the pickle file does not exist.
    KeyError
        If required keys are missing from the model data.
    Exception
        For any other errors during model creation.
    
    Example
    -------
    >>> model_data = load_and_create_model('models/building1.pkl')
    >>> print(f"Loaded project: {model_data['project_name']}")
    >>> print(f"Number of floors: {len(model_data['coordz']) - 1}")
    >>> # Model is now ready in OpenSeesPy for analysis
    
    Notes
    -----
    - This function calls ops.wipe() to clear any existing OpenSeesPy model
    - The model is created in the same sequence as app_modelcreator.py
    - All materials, sections, and elements are created with their original tags
    - Boundary conditions (fixed supports) are automatically applied
    - The function validates that all required data is present before proceeding
    
    See Also
    --------
    save_model_data : Function in app_modelcreator.py that creates the .pkl file
    """
    
    # Validate file exists
    if not os.path.exists(pkl_file_path):
        raise FileNotFoundError(f"Model file not found: {pkl_file_path}")
    
    # Load the pickle file
    print(f"Loading model from: {pkl_file_path}")
    with open(pkl_file_path, 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"✓ Model data loaded: {model_data['project_name']}")
    
    # Validate required keys
    required_keys = ['coordx', 'coordy', 'coordz', 'masas', 'materials', 'sections',
                     'column_assignments', 'beamx_assignments', 'beamy_assignments']
    
    missing_keys = [key for key in required_keys if key not in model_data]
    if missing_keys:
        raise KeyError(f"Missing required keys in model data: {missing_keys}")
    
    # Clear any existing OpenSeesPy model
    ops.wipe()
    print("✓ OpenSeesPy workspace cleared")
    
    # Extract data
    coordx = model_data['coordx']
    coordy = model_data['coordy']
    coordz = model_data['coordz']
    masas = model_data['masas']
    materials = model_data['materials']
    sections = model_data['sections']
    column_assignments = model_data['column_assignments']
    beamx_assignments = model_data['beamx_assignments']
    beamy_assignments = model_data['beamy_assignments']
    
    print(f"  - Floors: {len(coordz) - 1}")
    print(f"  - Bays X: {len(coordx) - 1}")
    print(f"  - Bays Y: {len(coordy) - 1}")
    print(f"  - Materials: {len(materials)}")
    print(f"  - Sections: {len(sections)}")
    
    # =========================================================================
    # STEP 1: Create building geometry and masses
    # =========================================================================
    ops.wipe()
    ops.model('basic', '-ndm', 3, '-ndf', 6)
    
    print("\n[1/6] Creating building geometry...")
    coords = ut.creategrid3D(coordx, coordy, coordz, 1, masas)
    
    # Apply boundary conditions (fixed supports at base)
    ops.fixZ(0.0, 1, 1, 1, 1, 1, 1)
    print("✓ Geometry created and boundary conditions applied")
    
    # =========================================================================
    # STEP 2: Create materials
    # =========================================================================
    print("\n[2/6] Creating materials...")
    material_tags = {}  # Store for reference
    
    for i, (mat_name, mat) in enumerate(materials.items()):
        fc = mat['fc']
        fy = mat['fy']
        detailing = mat['detailing']
        noconf_tag = mat['noconf_tag']
        conf_tag = mat['conf_tag']
        acero_tag = mat['acero_tag']
        
        # Create material using opseestools
        noconf, conf, acero = ut.col_materials(
            fc,
            fy,
            detailing=detailing,
            nps=3,
            tension='yes',
            unctag=noconf_tag,
            conftag=conf_tag,
            steeltag=acero_tag
        )
        
        material_tags[mat_name] = {
            'noconf_tag': noconf_tag,
            'conf_tag': conf_tag,
            'acero_tag': acero_tag
        }
        
        print(f"  ✓ Material {i+1}/{len(materials)}: {mat_name} "
              f"(fc={fc} MPa, fy={fy} MPa, {detailing})")
    
    print(f"✓ All {len(materials)} materials created")
    
    # =========================================================================
    # STEP 3: Create sections
    # =========================================================================
    print("\n[3/6] Creating RC sections...")
    section_tags = {}  # Store for reference
    
    for i, (sec_name, sec_props) in enumerate(sections.items()):
        section_tag = sec_props['tag']
        H = sec_props['H']
        B = sec_props['B']
        cover = sec_props['cover']
        material_name = sec_props['material']
        bars_top = sec_props['bars_top']
        area_top = sec_props['area_top']
        bars_bottom = sec_props['bars_bottom']
        area_bottom = sec_props['area_bottom']
        bars_middle = sec_props['bars_middle']
        area_middle = sec_props['area_middle']
        
        # Get material tags
        mat_info = materials[material_name]
        conf_tag = mat_info['conf_tag']
        noconf_tag = mat_info['noconf_tag']
        acero_tag = mat_info['acero_tag']
        
        # Rebar areas lookup table
        rebar_areas = {
            'As4': 0.000127,
            'As5': 0.0002,
            'As6': 0.000286,
            'As7': 0.000387,
            'As8': 0.000508
        }
        
        # Convert rebar sizes to numeric areas
        area_top_value = rebar_areas[area_top]
        area_bottom_value = rebar_areas[area_bottom]
        area_middle_value = rebar_areas[area_middle]
        
        # Handle middle bars = 0 case
        bars_middle_actual = bars_middle
        area_middle_actual = area_middle_value
        if bars_middle == 0:
            bars_middle_actual = 2
            area_middle_actual = 1e-8
        
        # Create RC section with positional arguments
        ut.create_rect_RC_section(
            section_tag,
            H,
            B,
            cover,
            conf_tag,
            noconf_tag,
            acero_tag,
            bars_top,
            area_top_value,
            bars_bottom,
            area_bottom_value,
            bars_middle_actual,
            area_middle_actual
        )
        
        section_tags[sec_name] = section_tag
        
        print(f"  ✓ Section {i+1}/{len(sections)}: {sec_name} "
              f"({H}x{B} cm, tag={section_tag})")
    
    print(f"✓ All {len(sections)} sections created")
    
    # =========================================================================
    # STEP 4: Create elements (columns and beams)
    # =========================================================================
    print("\n[4/6] Creating structural elements...")
    
    # Prepare section assignment arrays
    sectag_col = []
    sectag_vigx = []
    sectag_vigy = []
    
    num_floors = len(coordz) - 1
    num_x = len(coordx)
    num_y = len(coordy)
    
    # Column assignments (floors start at 1, not 0)
    for floor in range(1, num_floors + 1):
        floor_cols = []
        for i in range(num_x):
            col_row = []
            for j in range(num_y):
                # Access nested dictionary structure
                assigned_section = column_assignments.get(floor, {}).get(i, {}).get(j, None)
                if assigned_section and assigned_section in section_tags:
                    col_row.append(section_tags[assigned_section])
                else:
                    col_row.append(None)
            floor_cols.append(col_row)
        sectag_col.append(floor_cols)
    
    # Beam X assignments (floors start at 1, not 0)
    for floor in range(1, num_floors + 1):
        floor_beamsx = []
        for i in range(num_x - 1):  # Spans in X
            beam_row = []
            for j in range(num_y):
                # Access nested dictionary structure
                assigned_section = beamx_assignments.get(floor, {}).get(i, {}).get(j, None)
                if assigned_section and assigned_section in section_tags:
                    beam_row.append(section_tags[assigned_section])
                else:
                    beam_row.append(None)
            floor_beamsx.append(beam_row)
        sectag_vigx.append(floor_beamsx)
    
    # Beam Y assignments (floors start at 1, not 0)
    for floor in range(1, num_floors + 1):
        floor_beamsy = []
        for i in range(num_x):
            beam_col = []
            for j in range(num_y - 1):  # Spans in Y
                # Access nested dictionary structure
                assigned_section = beamy_assignments.get(floor, {}).get(i, {}).get(j, None)
                if assigned_section and assigned_section in section_tags:
                    beam_col.append(section_tags[assigned_section])
                else:
                    beam_col.append(None)
            floor_beamsy.append(beam_col)
        sectag_vigy.append(floor_beamsy)
    
    # Create all elements
    cols, vigx, vigy, sectag_col_used, sectag_vigx_used, sectag_vigy_used = ut.create_elements3D2(
        coordx,
        coordy,
        coordz,
        sectag_col,
        sectag_vigx,
        sectag_vigy
    )
    
    print(f"✓ Elements created successfully")
    
    # =========================================================================
    # STEP 5: Create slabs (if they exist in the model)
    # =========================================================================
    if model_data.get('slabs_created', False):
        print("\n[5/6] Creating slabs...")
        
        hslab = model_data.get('hslab')
        fc_slab = model_data.get('fc_slab')
        Eslab = model_data.get('Eslab')
        pois = model_data.get('pois')
        
        ut.create_slabs(
            coordx,
            coordy,
            coordz,
            hslab,
            Eslab,
            pois
        )
        
        print(f"✓ Slabs created (h={hslab} cm, E={Eslab} kPa, ν={pois})")
    else:
        print("\n[5/6] Skipping slabs (not created in original model)")
    
    # =========================================================================
    # STEP 6: Apply loads (if they exist in the model)
    # =========================================================================
    if model_data.get('loads_applied', False):
        print("\n[6/6] Applying loads...")
        
        floorx = float(model_data.get('floorx'))
        floory = float(model_data.get('floory'))
        roofx = float(model_data.get('roofx'))
        roofy = float(model_data.get('roofy'))
        
        # Apply loads (negative for gravity)
        ut.load_beams3D(
            -floorx,
            -floory,
            -roofx,
            -roofy,
            vigx,
            vigy,
            coordx,
            coordy
        )
        
        print(f"✓ Loads applied:")
        print(f"  - Floor loads: X={floorx} kN/m, Y={floory} kN/m")
        print(f"  - Roof loads: X={roofx} kN/m, Y={roofy} kN/m")
    else:
        print("\n[6/6] Skipping loads (not applied in original model)")
    
    # =========================================================================
    # Model creation complete
    # =========================================================================
    print("\n" + "="*70)
    print(f"✓✓✓ MODEL SUCCESSFULLY RECREATED: {model_data['project_name']} ✓✓✓")
    print("="*70)
    print("\nThe OpenSeesPy model is now ready for analysis.")
    print("You can now run modal analysis, gravity analysis, pushover, etc.")
    print("="*70 + "\n")
    
    return model_data


if __name__ == "__main__":
    """
    Example usage and testing
    """
    import sys
    
    print("RCF-3D Pushover Library - Model Loader")
    print("=" * 70)
    
    # Check if a file path was provided
    if len(sys.argv) > 1:
        pkl_file = sys.argv[1]
    else:
        # Default example
        pkl_file = "models/building1.pkl"
        print(f"No file specified. Using default: {pkl_file}")
        print("Usage: python library.py <path_to_pkl_file>")
        print()
    
    # Try to load and create the model
    try:
        model_data = load_and_create_model(pkl_file)
        
        # Display some model information
        print("\nMODEL INFORMATION:")
        print("-" * 70)
        print(f"Project Name: {model_data['project_name']}")
        print(f"Dimensions: {len(model_data['coordx'])-1} bays X × "
              f"{len(model_data['coordy'])-1} bays Y × "
              f"{len(model_data['coordz'])-1} floors")
        print(f"Total Height: {sum(model_data['coordz'])} m")
        print(f"Materials Defined: {len(model_data['materials'])}")
        print(f"Sections Defined: {len(model_data['sections'])}")
        print(f"Slabs Created: {'Yes' if model_data.get('slabs_created') else 'No'}")
        print(f"Loads Applied: {'Yes' if model_data.get('loads_applied') else 'No'}")
        print(f"Modal Analysis Done: {'Yes' if model_data.get('modal_analysis_done') else 'No'}")
        print("-" * 70)
        
        print("\n✓ Test successful! Model loaded and created in OpenSeesPy.")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("Make sure the .pkl file exists in the specified path.")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def visualize_model_plotly(model_data):
    """
    Create an interactive 3D visualization of the structural model using Plotly.
    
    This function generates a 3D plot showing:
    - Columns (vertical elements) with section information in tooltips
    - Beams in X direction with section information in tooltips
    - Beams in Y direction with section information in tooltips
    - Interactive hover to see element details
    
    Parameters
    ----------
    model_data : dict
        Model data dictionary returned by load_and_create_model()
        Must contain: coordx, coordy, coordz, column_assignments, 
                     beamx_assignments, beamy_assignments, sections
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive 3D Plotly figure ready to display
    
    Example
    -------
    >>> model_data = load_and_create_model('models/building1.pkl')
    >>> fig = visualize_model_plotly(model_data)
    >>> fig.show()  # Or st.plotly_chart(fig) in Streamlit
    """
    import plotly.graph_objects as go
    import numpy as np
    
    # Extract coordinates
    coordx = model_data['coordx']
    coordy = model_data['coordy']
    coordz = model_data['coordz']
    
    # Extract assignments
    column_assignments = model_data['column_assignments']
    beamx_assignments = model_data['beamx_assignments']
    beamy_assignments = model_data['beamy_assignments']
    sections = model_data['sections']
    
    # coordz already contains cumulative heights from ground (index 0 = ground, 1 = floor 1, etc.)
    num_floors = len(coordz) - 1
    num_x = len(coordx)
    num_y = len(coordy)
    
    # Create figure
    fig = go.Figure()
    
    # =========================================================================
    # Add Columns
    # =========================================================================
    for floor in range(1, num_floors + 1):
        z_bottom = coordz[floor - 1]
        z_top = coordz[floor]
        
        for i in range(num_x):
            x_pos = coordx[i]
            for j in range(num_y):
                y_pos = coordy[j]
                
                # Get column assignment
                section_name = column_assignments.get(floor, {}).get(i, {}).get(j, None)
                
                if section_name and section_name in sections:
                    section_props = sections[section_name]
                    H = section_props['H']
                    B = section_props['B']
                    
                    # Create column line
                    fig.add_trace(go.Scatter3d(
                        x=[x_pos, x_pos],
                        y=[y_pos, y_pos],
                        z=[z_bottom, z_top],
                        mode='lines',
                        line=dict(color='royalblue', width=4),
                        name='Columns',
                        showlegend=(floor == 1 and i == 0 and j == 0),
                        hovertemplate=(
                            f'<b>Column</b><br>'
                            f'Floor: {floor}<br>'
                            f'Position: ({i}, {j})<br>'
                            f'Location: X={x_pos:.2f}m, Y={y_pos:.2f}m<br>'
                            f'Section: {section_name}<br>'
                            f'Dimensions: {H:.2f}×{B:.2f} m<br>'
                            f'<extra></extra>'
                        )
                    ))
    
    # =========================================================================
    # Add Beams X (parallel to X axis)
    # =========================================================================
    for floor in range(1, num_floors + 1):
        z_level = coordz[floor]
        
        for i in range(num_x - 1):  # Spans in X
            x_start = coordx[i]
            x_end = coordx[i + 1]
            
            for j in range(num_y):
                y_pos = coordy[j]
                
                # Get beam assignment
                section_name = beamx_assignments.get(floor, {}).get(i, {}).get(j, None)
                
                if section_name and section_name in sections:
                    section_props = sections[section_name]
                    H = section_props['H']
                    B = section_props['B']
                    
                    # Create beam line
                    fig.add_trace(go.Scatter3d(
                        x=[x_start, x_end],
                        y=[y_pos, y_pos],
                        z=[z_level, z_level],
                        mode='lines',
                        line=dict(color='green', width=3),
                        name='Beams X',
                        showlegend=(floor == 1 and i == 0 and j == 0),
                        hovertemplate=(
                            f'<b>Beam X</b><br>'
                            f'Floor: {floor}<br>'
                            f'Span: {i} (X-direction)<br>'
                            f'Y Position: {j}<br>'
                            f'Location: Y={y_pos:.2f}m, Z={z_level:.2f}m<br>'
                            f'Section: {section_name}<br>'
                            f'Dimensions: {H:.2f}×{B:.2f} m<br>'
                            f'Length: {x_end-x_start:.2f}m<br>'
                            f'<extra></extra>'
                        )
                    ))
    
    # =========================================================================
    # Add Beams Y (parallel to Y axis)
    # =========================================================================
    for floor in range(1, num_floors + 1):
        z_level = coordz[floor]
        
        for i in range(num_x):
            x_pos = coordx[i]
            
            for j in range(num_y - 1):  # Spans in Y
                y_start = coordy[j]
                y_end = coordy[j + 1]
                
                # Get beam assignment
                section_name = beamy_assignments.get(floor, {}).get(i, {}).get(j, None)
                
                if section_name and section_name in sections:
                    section_props = sections[section_name]
                    H = section_props['H']
                    B = section_props['B']
                    
                    # Create beam line
                    fig.add_trace(go.Scatter3d(
                        x=[x_pos, x_pos],
                        y=[y_start, y_end],
                        z=[z_level, z_level],
                        mode='lines',
                        line=dict(color='orange', width=3),
                        name='Beams Y',
                        showlegend=(floor == 1 and i == 0 and j == 0),
                        hovertemplate=(
                            f'<b>Beam Y</b><br>'
                            f'Floor: {floor}<br>'
                            f'X Position: {i}<br>'
                            f'Span: {j} (Y-direction)<br>'
                            f'Location: X={x_pos:.2f}m, Z={z_level:.2f}m<br>'
                            f'Section: {section_name}<br>'
                            f'Dimensions: {H:.2f}×{B:.2f} m<br>'
                            f'Length: {y_end-y_start:.2f}m<br>'
                            f'<extra></extra>'
                        )
                    ))
    
    # =========================================================================
    # Configure Layout
    # =========================================================================
    fig.update_layout(
        title=dict(
            text=f"3D Model Visualization: {model_data['project_name']}",
            font=dict(size=20)
        ),
        scene=dict(
            xaxis=dict(
                title='X (m)',
                gridcolor='lightgray',
                showbackground=True,
                backgroundcolor='white'
            ),
            yaxis=dict(
                title='Y (m)',
                gridcolor='lightgray',
                showbackground=True,
                backgroundcolor='white'
            ),
            zaxis=dict(
                title='Z (m)',
                gridcolor='lightgray',
                showbackground=True,
                backgroundcolor='white'
            ),
            aspectmode='data'
        ),
        height=700,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        )
    )
    
    return fig


def visualize_elevation_plotly(model_data, plane='xz', position=None):
    """
    Visualize an elevation view (X-Z or Y-Z plane) of the 3D RC frame model
    
    Parameters:
    -----------
    model_data : dict
        Dictionary containing model data from pickle file
    plane : str
        'xz' for X-Z plane (looking along Y axis) or 'yz' for Y-Z plane (looking along X axis)
    position : float or None
        Position along the perpendicular axis to slice the model.
        If None, uses the first position (index 0)
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Plotly figure object
    """
    import plotly.graph_objs as go
    
    # Extract coordinates
    coordx = model_data['coordx']
    coordy = model_data['coordy']
    coordz = model_data['coordz']
    
    # Extract assignments
    column_assignments = model_data['column_assignments']
    beamx_assignments = model_data['beamx_assignments']
    beamy_assignments = model_data['beamy_assignments']
    sections = model_data['sections']
    materials = model_data['materials']
    
    num_floors = len(coordz) - 1
    num_x = len(coordx)
    num_y = len(coordy)
    
    # Determine which axis to slice and default position
    if plane.lower() == 'xz':
        # X-Z plane (looking along Y), slice at a Y position
        if position is None:
            slice_idx = 0
            position = coordy[slice_idx]
        else:
            slice_idx = min(range(len(coordy)), key=lambda i: abs(coordy[i] - position))
            position = coordy[slice_idx]
        
        horizontal_coords = coordx
        horizontal_label = 'X (m)'
        beam_assignments = beamx_assignments
        beam_name = 'Beam X'
    else:  # 'yz'
        # Y-Z plane (looking along X), slice at an X position
        if position is None:
            slice_idx = 0
            position = coordx[slice_idx]
        else:
            slice_idx = min(range(len(coordx)), key=lambda i: abs(coordx[i] - position))
            position = coordx[slice_idx]
        
        horizontal_coords = coordy
        horizontal_label = 'Y (m)'
        beam_assignments = beamy_assignments
        beam_name = 'Beam Y'
    
    # Create figure
    fig = go.Figure()
    
    # =========================================================================
    # Add Columns
    # =========================================================================
    for floor in range(1, num_floors + 1):
        z_bottom = coordz[floor - 1]
        z_top = coordz[floor]
        
        for idx, h_pos in enumerate(horizontal_coords):
            # Get column assignment based on plane
            if plane.lower() == 'xz':
                i = idx
                j = slice_idx
            else:
                i = slice_idx
                j = idx
            
            section_name = column_assignments.get(floor, {}).get(i, {}).get(j, None)
            
            if section_name and section_name in sections:
                section_props = sections[section_name]
                H = section_props['H']
                B = section_props['B']
                material_name = section_props.get('material', None)
                
                # Get material info
                if material_name and material_name in materials:
                    fc = materials[material_name].get('fc', 'N/A')
                    fy = materials[material_name].get('fy', 'N/A')
                else:
                    fc = 'N/A'
                    fy = 'N/A'
                
                # Create column line
                fig.add_trace(go.Scatter(
                    x=[h_pos, h_pos],
                    y=[z_bottom, z_top],
                    mode='lines',
                    line=dict(color='royalblue', width=6),
                    name='Columns',
                    showlegend=(floor == 1 and idx == 0),
                    hoverinfo='skip'
                ))
                
                # Add midpoint marker for tooltip
                z_mid = (z_bottom + z_top) / 2
                fig.add_trace(go.Scatter(
                    x=[h_pos],
                    y=[z_mid],
                    mode='markers',
                    marker=dict(color='royalblue', size=8),
                    name='Columns',
                    showlegend=False,
                    hovertemplate=(
                        f'<b>Column</b><br>'
                        f'Section: {section_name}<br>'
                        f'Dimensions: {H:.2f}×{B:.2f} m<br>'
                        f"f'c: {fc} MPa<br>"
                        f'fy: {fy} MPa<br>'
                        f'<extra></extra>'
                    )
                ))
    
    # =========================================================================
    # Add Beams
    # =========================================================================
    for floor in range(1, num_floors + 1):
        z_level = coordz[floor]
        
        for idx in range(len(horizontal_coords) - 1):
            h_start = horizontal_coords[idx]
            h_end = horizontal_coords[idx + 1]
            
            # Get beam assignment based on plane
            if plane.lower() == 'xz':
                i = idx
                j = slice_idx
            else:
                i = slice_idx
                j = idx
            
            section_name = beam_assignments.get(floor, {}).get(i, {}).get(j, None)
            
            if section_name and section_name in sections:
                section_props = sections[section_name]
                H = section_props['H']
                B = section_props['B']
                material_name = section_props.get('material', None)
                
                # Get material info
                if material_name and material_name in materials:
                    fc = materials[material_name].get('fc', 'N/A')
                    fy = materials[material_name].get('fy', 'N/A')
                else:
                    fc = 'N/A'
                    fy = 'N/A'
                
                # Determine beam color based on direction
                beam_color = 'green' if plane.lower() == 'xz' else 'orange'
                
                # Create beam line
                fig.add_trace(go.Scatter(
                    x=[h_start, h_end],
                    y=[z_level, z_level],
                    mode='lines',
                    line=dict(color=beam_color, width=6),
                    name=beam_name,
                    showlegend=(floor == 1 and idx == 0),
                    hoverinfo='skip'
                ))
                
                # Add midpoint marker for tooltip
                h_mid = (h_start + h_end) / 2
                fig.add_trace(go.Scatter(
                    x=[h_mid],
                    y=[z_level],
                    mode='markers',
                    marker=dict(color=beam_color, size=8),
                    name=beam_name,
                    showlegend=False,
                    hovertemplate=(
                        f'<b>{beam_name}</b><br>'
                        f'Section: {section_name}<br>'
                        f'Dimensions: {H:.2f}×{B:.2f} m<br>'
                        f"f'c: {fc} MPa<br>"
                        f'fy: {fy} MPa<br>'
                        f'<extra></extra>'
                    )
                ))
    
    # Update layout
    plane_str = 'X-Z' if plane.lower() == 'xz' else 'Y-Z'
    perpendicular = 'Y' if plane.lower() == 'xz' else 'X'
    
    fig.update_layout(
        title=dict(
            text=f'Elevation View ({plane_str} Plane) at {perpendicular}={position:.2f}m',
            font=dict(size=18)
        ),
        xaxis_title=horizontal_label,
        yaxis_title='Z (m)',
        width=1000,
        height=800,
        hovermode='closest',
        hoverlabel=dict(font_size=14),
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=14)
        ),
        xaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=16),
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=16)
        )
    )
    
    return fig


def visualize_plan_plotly(model_data, floor):
    """
    Visualize a floor plan (X-Y plane) of the 3D RC frame model
    
    Parameters:
    -----------
    model_data : dict
        Dictionary containing model data from pickle file
    floor : int
        Floor number to visualize (1-based index)
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Plotly figure object
    """
    import plotly.graph_objs as go
    
    # Extract coordinates
    coordx = model_data['coordx']
    coordy = model_data['coordy']
    coordz = model_data['coordz']
    
    # Extract assignments
    column_assignments = model_data['column_assignments']
    beamx_assignments = model_data['beamx_assignments']
    beamy_assignments = model_data['beamy_assignments']
    sections = model_data['sections']
    materials = model_data['materials']
    
    num_floors = len(coordz) - 1
    n_x_positions = len(coordx)
    n_y_positions = len(coordy)
    n_x_spans = n_x_positions - 1
    n_y_spans = n_y_positions - 1
    
    # Validate floor number
    if floor < 1 or floor > num_floors:
        raise ValueError(f"Floor must be between 1 and {num_floors}")
    
    # Create figure
    fig = go.Figure()
    
    # =========================================================================
    # Add Columns
    # =========================================================================
    col_x = []
    col_y = []
    col_names = []
    col_hover = []
    
    for x_idx in range(n_x_positions):
        for y_idx in range(n_y_positions):
            x_pos = coordx[x_idx]
            y_pos = coordy[y_idx]
            
            section_name = column_assignments.get(floor, {}).get(x_idx, {}).get(y_idx, None)
            
            if section_name and section_name in sections:
                col_x.append(x_pos)
                col_y.append(y_pos)
                col_names.append(section_name)
                
                section_info = sections[section_name]
                H = section_info['H']
                B = section_info['B']
                material_name = section_info['material']
                
                if material_name and material_name in materials:
                    fc = materials[material_name].get('fc', 'N/A')
                    fy = materials[material_name].get('fy', 'N/A')
                else:
                    fc = 'N/A'
                    fy = 'N/A'
                
                hover_text = (
                    f"<b>Column: {section_name}</b><br>"
                    f"X: {x_pos:.2f} m<br>"
                    f"Y: {y_pos:.2f} m<br>"
                    f"Dimensions: {H:.2f}×{B:.2f} m<br>"
                    f"Material: {material_name}<br>"
                    f"f'c: {fc} MPa<br>"
                    f"fy: {fy} MPa"
                )
                col_hover.append(hover_text)
    
    fig.add_trace(go.Scatter(
        x=col_x,
        y=col_y,
        mode='markers',
        marker=dict(size=15, color='royalblue', symbol='square', line=dict(color='darkblue', width=2)),
        name='Columns',
        hovertext=col_hover,
        hoverinfo='text'
    ))
    
    # =========================================================================
    # Add Beams X (parallel to X axis)
    # =========================================================================
    beamx_mid_x = []
    beamx_mid_y = []
    beamx_names = []
    beamx_hover = []
    
    for span_idx in range(n_x_spans):
        for y_idx in range(n_y_positions):
            x_start = coordx[span_idx]
            x_end = coordx[span_idx + 1]
            y_coord = coordy[y_idx]
            
            section_name = beamx_assignments.get(floor, {}).get(span_idx, {}).get(y_idx, None)
            
            if section_name and section_name in sections:
                section_info = sections[section_name]
                H = section_info['H']
                B = section_info['B']
                material_name = section_info['material']
                
                if material_name and material_name in materials:
                    fc = materials[material_name].get('fc', 'N/A')
                    fy = materials[material_name].get('fy', 'N/A')
                else:
                    fc = 'N/A'
                    fy = 'N/A'
                
                # Beam line
                fig.add_trace(go.Scatter(
                    x=[x_start, x_end],
                    y=[y_coord, y_coord],
                    mode='lines',
                    line=dict(color='green', width=3),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Endpoint circles
                fig.add_trace(go.Scatter(
                    x=[x_start, x_end],
                    y=[y_coord, y_coord],
                    mode='markers',
                    marker=dict(size=6, color='darkgreen', symbol='circle'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Midpoint marker
                x_mid = (x_start + x_end) / 2
                beamx_mid_x.append(x_mid)
                beamx_mid_y.append(y_coord)
                beamx_names.append(section_name)
                
                hover_text = (
                    f"<b>Beam X: {section_name}</b><br>"
                    f"Span: X {x_start:.2f} → {x_end:.2f} m<br>"
                    f"Y: {y_coord:.2f} m<br>"
                    f"Dimensions: {H:.2f}×{B:.2f} m<br>"
                    f"Material: {material_name}<br>"
                    f"f'c: {fc} MPa<br>"
                    f"fy: {fy} MPa"
                )
                beamx_hover.append(hover_text)
    
    if beamx_mid_x:
        fig.add_trace(go.Scatter(
            x=beamx_mid_x,
            y=beamx_mid_y,
            mode='markers',
            marker=dict(size=2, color='green', symbol='diamond', line=dict(color='darkgreen', width=2)),
            name='Beams X',
            hovertext=beamx_hover,
            hoverinfo='text'
        ))
    
    # =========================================================================
    # Add Beams Y (parallel to Y axis)
    # =========================================================================
    beamy_mid_x = []
    beamy_mid_y = []
    beamy_names = []
    beamy_hover = []
    
    for x_idx in range(n_x_positions):
        for span_idx in range(n_y_spans):
            x_coord = coordx[x_idx]
            y_start = coordy[span_idx]
            y_end = coordy[span_idx + 1]
            
            section_name = beamy_assignments.get(floor, {}).get(x_idx, {}).get(span_idx, None)
            
            if section_name and section_name in sections:
                section_info = sections[section_name]
                H = section_info['H']
                B = section_info['B']
                material_name = section_info['material']
                
                if material_name and material_name in materials:
                    fc = materials[material_name].get('fc', 'N/A')
                    fy = materials[material_name].get('fy', 'N/A')
                else:
                    fc = 'N/A'
                    fy = 'N/A'
                
                # Beam line
                fig.add_trace(go.Scatter(
                    x=[x_coord, x_coord],
                    y=[y_start, y_end],
                    mode='lines',
                    line=dict(color='orange', width=3),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Endpoint circles
                fig.add_trace(go.Scatter(
                    x=[x_coord, x_coord],
                    y=[y_start, y_end],
                    mode='markers',
                    marker=dict(size=6, color='darkorange', symbol='circle'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Midpoint marker
                y_mid = (y_start + y_end) / 2
                beamy_mid_x.append(x_coord)
                beamy_mid_y.append(y_mid)
                beamy_names.append(section_name)
                
                hover_text = (
                    f"<b>Beam Y: {section_name}</b><br>"
                    f"X: {x_coord:.2f} m<br>"
                    f"Span: Y {y_start:.2f} → {y_end:.2f} m<br>"
                    f"Dimensions: {H:.2f}×{B:.2f} m<br>"
                    f"Material: {material_name}<br>"
                    f"f'c: {fc} MPa<br>"
                    f"fy: {fy} MPa"
                )
                beamy_hover.append(hover_text)
    
    if beamy_mid_x:
        fig.add_trace(go.Scatter(
            x=beamy_mid_x,
            y=beamy_mid_y,
            mode='markers',
            marker=dict(size=2, color='orange', symbol='diamond', line=dict(color='darkorange', width=2)),
            name='Beams Y',
            hovertext=beamy_hover,
            hoverinfo='text'
        ))
    
    # Calculate axis ranges
    x_min = min(coordx) - 3
    x_max = max(coordx) + 3
    y_min = min(coordy) - 3
    y_max = max(coordy) + 3
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Floor {floor} - Plan View',
            font=dict(size=18)
        ),
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        width=900,
        height=700,
        hovermode='closest',
        hoverlabel=dict(font_size=14),
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=14)
        ),
        plot_bgcolor='#f0f0f0',
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='white',
            zeroline=True,
            range=[x_min, x_max],
            scaleanchor="y",
            scaleratio=1,
            title_font=dict(size=18),
            tickfont=dict(size=16)
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='white',
            zeroline=True,
            range=[y_min, y_max],
            title_font=dict(size=18),
            tickfont=dict(size=16)
        )
    )
    
    return fig


def find_first_exceedance_indices(rotations, limits):
    """
    Find the first indices where each rotation limit is exceeded during pushover analysis.
    
    For each limit in the limits vector, this function finds the smallest index i 
    where rotations[i] > limit. If a limit is never exceeded, returns -1 for that limit.
    
    Parameters
    ----------
    rotations : list or array-like
        List of rotation values during the pushover analysis (one per analysis step)
    limits : list or array-like
        Vector of rotation limits to check for exceedance
    
    Returns
    -------
    indices : list
        List of indices (one per limit). Each index is the first position where 
        rotations[i] > limit, or -1 if the limit was never exceeded.
    
    Examples
    --------
    >>> rotations = [0.001, 0.002, 0.005, 0.008, 0.012]
    >>> limits = [0.003, 0.010]
    >>> find_first_exceedance_indices(rotations, limits)
    [2, 4]  # 0.005 > 0.003 at index 2, 0.012 > 0.010 at index 4
    """
    indices = []
    for limit in limits:
        exceeded = False
        for i, rot in enumerate(rotations):
            if rot > limit:
                indices.append(i)
                exceeded = True
                break
        if not exceeded:
            indices.append(-1)
    return indices
