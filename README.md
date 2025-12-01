# RCF-3D-Analysis

## Overview

RCF-3D-Analysis is a comprehensive web-based application for creating  three-dimensional reinforced concrete frame building models. Built with Streamlit and using OpenSeesPy and opseestools as backends, this tool provides an intuitive 11-tab interface for complete building model generation, from geometry definition to load application, with integrated visualization and preliminary analysis capabilities.

## Features

### 🏗️ Parametric Model Generation
- **3D Geometry**: Define building dimensions with customizable X, Y, Z coordinates
- **Grid-Based Layout**: Rectangular building layouts with variable bay spacing
- **Multi-Story Buildings**: Support for buildings up to 10+ stories
- **Flexible Configurations**: Asymmetric layouts and irregular geometries

### 🧱 Material & Section Design
- **Material Library**: Pre-defined concrete and steel materials (DES, DMO, PreCode standards)
- **Custom Materials**: Define new concrete and steel properties
- **RC Section Builder**: Create rectangular reinforced concrete sections
- **Reinforcement Configuration**: Detailed longitudinal and transverse reinforcement

### 🎯 Element Assignment
- **Interactive Grid Interface**: Visual assignment of sections to structural elements
- **Column Configuration**: Assign sections to column grid positions
- **Beam Layout**: Configure beams by floor and direction (X and Y)
- **Batch Operations**: Copy configurations across multiple floors

### 👁️ Visualization & Verification
- **3D Model Preview**: Interactive Plotly visualizations for real-time verification
- **Plan Views**: 2D floor plans with section assignments
- **Elevation Views**: Section cuts showing material distribution
- **Element Details**: Hover tooltips with section and material information

### 🏠 Structural Components
- **Rigid Diaphragms**: Define floor slabs as rigid diaphragms
- **Gravity Loads**: Apply distributed loads to beams
- **Load Cases**: Separate floor and roof load specifications
- **Mass Calculation**: Automatic mass matrix generation

### 🔬 Preliminary Analysis
- **Modal Analysis**: Eigenvalue analysis for natural frequencies
- **Model Validation**: Check for common modeling errors
- **Analysis Integration**: Seamless transition to pushover analysis

### 💾 Model Persistence
- **Pickle Serialization**: Save complete model configurations
- **Metadata Storage**: Include project information and timestamps
- **Model Library**: Organized storage in models/ directory

## System Requirements

### Software Dependencies
- Python 3.8+
- Streamlit
- OpenSeesPy
- Pandas, NumPy
- Plotly
- opseestools (custom utilities)

### Hardware Requirements
- 8 GB RAM minimum
- Multi-core processor recommended

## Installation

1. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Installation**:
   ```bash
   python -c "import streamlit, openseespy.opensees as ops; print('Dependencies OK')"
   ```

## Usage

### Running the Application

```bash
streamlit run app_modelcreator_refactored.py
```

### Workflow Overview

1. **Geometry (Tab 1)**: Define building dimensions and grid layout
2. **Materials (Tab 2)**: Configure concrete and steel properties
3. **Sections (Tab 3)**: Design reinforced concrete sections
4. **Columns (Tab 4)**: Assign sections to column positions
5. **Beams (Tab 5)**: Configure beam layouts
6. **Visualization (Tab 6)**: Review 3D model and verify assignments
7. **Slabs (Tab 7)**: Define floor diaphragms
8. **Loads (Tab 8)**: Apply gravity loads
9. **Modal Analysis (Tab 9)**: Preliminary dynamic analysis
10. **Save Model (Tab 10)**: Export model for analysis
11. **Model Info (Tab 11)**: Review complete model summary

## Application Structure

The application consists of 11 sequential tabs:

### Tab 1: Geometry Definition
- Input X, Y, Z coordinates
- Define building grid layout
- Set project metadata

### Tab 2: Material Configuration
- Select or define concrete materials
- Configure steel reinforcement properties
- Choose detailing standards

### Tab 3: Section Design
- Create rectangular RC sections
- Specify reinforcement layouts
- Define cover and spacing

### Tab 4: Column Assignment
- Grid-based section assignment
- Visual column layout
- Batch operations

### Tab 5: Beam Configuration
- Floor-by-floor beam setup
- X and Y direction beams
- Section assignment per span

### Tab 6: Model Visualization
- Interactive 3D views
- Plan and elevation visualizations
- Element verification

### Tab 7: Slab Definition
- Rigid diaphragm creation
- Material assignment
- Thickness specification

### Tab 8: Load Application
- Gravity load definition
- Floor and roof loads
- Load distribution

### Tab 9: Modal Analysis
- Eigenvalue computation
- Natural frequency display
- Mode shape visualization

### Tab 10: Model Saving
- Project naming
- File export to models/ directory
- Metadata inclusion

### Tab 11: Model Summary
- Complete model overview
- Component counts
- Validation checks

## Output

The application generates complete building models saved as `.pkl` files containing:
- Geometric coordinates and grid
- Material properties
- Section definitions and assignments
- Load configurations
- Slab definitions
- Analysis-ready OpenSees model

## Integration

Saved models are compatible with the RCF-3D-Analysis application (`app_analysis_refactored.py`) for:
- Modal analysis
- Gravity analysis
- Nonlinear pushover analysis
- Results visualization

## Tips and Best Practices

1. **Start Small**: Begin with simple 2-3 story buildings
2. **Verify Visually**: Always check 3D visualizations before saving
3. **Save Incrementally**: Export models after major changes
4. **Use Consistent Units**: All dimensions in meters, forces in kN
5. **Check Assignments**: Verify section assignments in visualization tab

## Limitations

- Rectangular grid layouts only
- Rigid diaphragm assumption for floors
- Monotonic loading (gravity only)
- Rectangular RC sections only

## Documentation

For detailed technical documentation, see Documentation.md

## Support

For issues or questions:
- Check the documentation files
- Verify Python environment and dependencies
- Ensure models/ directory exists for saving

## Citation

If you use RCF-3D-Analysis Model Creator in your research:

```
Arroyo (2025). RCF-3D-Analysis: Parametric Model Creation for 
Nonlinear Analysis of 3D Reinforced Concrete Frame Buildings.
```
