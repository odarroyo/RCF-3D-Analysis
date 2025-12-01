# RCF3D.py Documentation

## Purpose
Create comprehensive 3D RC frame building models for structural analysis with:
- **Building Geometry**: Define multi-story layouts with customizable X-Y coordinates and story masses
- **Material Definition**: Create concrete/steel material combinations with different detailing levels
- **Section Properties**: Define rectangular RC sections with reinforcement configurations
- **Element Assignment**: Assign sections to columns and beams across all floors with visual feedback
- **Structural Elements**: Generate OpenSeesPy elements (columns, beams) with proper connectivity
- **Slab Modeling**: Add floor slabs with material properties and self-weight
- **Load Application**: Define gravity loads on beams for analysis preparation
- **Model Visualization**: Interactive 3D and 2D visualizations with hover information and beam markers
- **Model Persistence**: Save complete models for use in analysis applications

## File Structure (1727 lines - Full Version)

```
├── Module Documentation (Comprehensive docstring)
├── Imports and Dependencies
├── Code Organization Guide
├── Constants Section
│   ├── REBAR_AREAS (documented rebar sizes)
│   └── DETAILING_OPTIONS (seismic detailing levels)
├── Session State Management
│   └── initialize_session_state() - Complete state initialization
├── Geometry Functions
│   ├── parse_coordinates() - Input validation and parsing
│   └── create_opensees_model() - OpenSeesPy model creation
├── Material Functions
│   ├── generate_material_tags() - Tag generation system
│   ├── create_material() - Material model creation
│   └── validate_material_input() - Input validation
├── Section Functions
│   ├── generate_section_tag() - Section tag generation
│   ├── create_section() - Fiber section creation
│   ├── validate_section_input() - Input validation
│   └── calculate_rebar_area() - Reinforcement calculations
├── Assignment Functions
│   ├── initialize_floor_assignments() - Assignment initialization
│   ├── copy_floor_assignments() - Configuration copying
│   ├── build_element_tags_list() - Tag list construction
│   └── validate_assignments_complete() - Completion checking
├── Model Operations
│   ├── save_model_to_file() - Complete model serialization
│   └── load_model_from_file() - Model deserialization
├── Visualization Functions
│   └── create_floor_plan_figure() - Interactive floor plans
├── UI Rendering Functions (11 tabs)
│   ├── render_geometry_tab() - Building layout definition
│   ├── render_materials_tab() - Material creation interface
│   ├── render_sections_tab() - Section definition interface
│   ├── render_columns_tab() - Column assignment interface
│   ├── render_beams_tab() - Beam assignment interface
│   ├── render_model_visualization_tab() - 3D model review
│   ├── render_slabs_tab() - Slab definition interface
│   ├── render_loads_tab() - Load application interface
│   ├── render_save_model_tab() - Model export interface
│   ├── render_modal_analysis_tab() - Eigenvalue analysis
│   └── render_gravity_pushover_tab() - Nonlinear analysis
└── Main Application
    └── main() - Application entry point with tab orchestration
```

## Key Functions

### Constants (Documented)

```python
REBAR_AREAS = {
    'As4': 0.000127,  # 4mm diameter rebar
    'As5': 0.0002,    # 5mm diameter rebar
    'As6': 0.000286,  # 6mm diameter rebar
    'As7': 0.000387,  # 7mm diameter rebar
    'As8': 0.000508   # 8mm diameter rebar
}
```
**Purpose**: Standard rebar cross-sectional areas for reinforcement design  
**Units**: Square meters (m²)  
**Usage**: Automatic calculation in section creation

```python
DETAILING_OPTIONS = ["DES", "DMO", "PreCode"]
```
**Purpose**: Seismic detailing levels affecting material strength models  
**Values**:
- `"DES"`: Ductile Earthquake Resistant (highest ductility, modern codes)
- `"DMO"`: Moderate Ductility (intermediate requirements)
- `"PreCode"`: Pre-1970s code level (minimal detailing, lowest ductility)

---

### Session State Management

```python
def initialize_session_state()
```
**Purpose**: Initialize complete session state for the 11-tab modeling workflow  
**Scope**: 25+ state variables covering entire application lifecycle  
**Key Variables**:
- **Geometry**: `coordx`, `coordy`, `coordz`, `masas`, `model_created`
- **Materials**: `materials` (dict of material sets)
- **Sections**: `sections` (dict of cross-section definitions)
- **Assignments**: `column_assignments`, `beamx_assignments`, `beamy_assignments`
- **Model State**: `elements_created`, `slabs_created`, `loads_applied`
- **Analysis**: `modal_analysis_done`, `gravity_analysis_done`, `pushover_analysis_done`
- **UI State**: `project_name`, visualization flags

---

### Geometry Functions

```python
def parse_coordinates(coord_input: str) -> list[float]
```
**Purpose**: Parse and validate coordinate input strings  
**Parameters**:
- `coord_input` (str): Comma-separated values like "0, 8.6, 12.6"
**Returns**: List of validated float coordinates  
**Validation**: Non-empty, numeric values, minimum spans  
**Error Handling**: Raises `ValueError` with descriptive messages

```python
def create_opensees_model(coordx, coordy, coordz, masas)
```
**Purpose**: Create complete OpenSeesPy model with geometry and boundary conditions  
**Parameters**:
- `coordx`, `coordy` (list): Plan coordinates in meters
- `coordz` (list): Story elevations in meters
- `masas` (list): Story masses in kN·s²/m
**Operations**:
- Clear previous model with `ops.wipe()`
- Create 3D model with 6 DOF per node
- Generate coordinate grid using `ut.creategrid3D()`
- Apply fixed boundary conditions at base (`ops.fixZ(0.0, 1, 1, 1, 1, 1, 1)`)

---

### Material Functions

```python
def generate_material_tags(material_index: int) -> dict[str, int]
```
**Purpose**: Generate unique OpenSeesPy material tags to prevent conflicts  
**Parameters**:
- `material_index` (int): Zero-based material sequence number
**Returns**: Dictionary with unique tags:
  - `'unctag'`: Unconfined concrete material
  - `'conftag'`: Confined concrete material
  - `'steeltag'`: Steel reinforcement material
**Algorithm**: `base_tag = (material_index + 1) * 100`

```python
def create_material(fc: float, fy: float, detailing: str, material_index: int) -> dict
```
**Purpose**: Create complete concrete-steel material set using opseestools  
**Parameters**:
- `fc` (float): Concrete compressive strength (MPa)
- `fy` (float): Steel yield strength (MPa)
- `detailing` (str): Seismic detailing level
- `material_index` (int): For tag generation
**Returns**: Material dictionary with properties and OpenSeesPy tags  
**Side Effects**: Creates `uniaxialMaterial` objects in OpenSeesPy model

---

### Section Functions

```python
def generate_section_tag(section_index: int) -> int
```
**Purpose**: Generate unique fiber section tags  
**Parameters**:
- `section_index` (int): Zero-based section sequence number
**Returns**: Unique section tag (1001, 1002, 1003, ...)  
**Algorithm**: `tag = (section_index + 1) * 100 + 1000`

```python
def create_section(section_tag: int, H: float, B: float, cover: float,
                  material_props: dict, bars_config: dict)
```
**Purpose**: Create reinforced concrete fiber section with proper discretization  
**Parameters**:
- `section_tag` (int): Unique section identifier
- `H`, `B` (float): Section dimensions in meters
- `cover` (float): Concrete cover thickness in meters
- `material_props` (dict): Material tags from `create_material()`
- `bars_config` (dict): Reinforcement layout specification
**Operations**:
- Creates fiber section with confined/unconfined concrete regions
- Places reinforcement bars at specified locations
- Uses `ut.create_rect_RC_section()` for proper discretization

---

### Assignment Functions

```python
def initialize_floor_assignments(floor: int, n_x: int, n_y: int, default_section: str) -> dict
```
**Purpose**: Initialize section assignments for a floor with default values  
**Parameters**:
- `floor` (int): Floor number (1-based)
- `n_x`, `n_y` (int): Grid dimensions
- `default_section` (str): Default section name to assign
**Returns**: Nested dictionary structure for the floor

```python
def copy_floor_assignments(source_floor: int, target_floors: list, assignments: dict)
```
**Purpose**: Copy section assignments from one floor to multiple target floors  
**Parameters**:
- `source_floor` (int): Floor to copy from
- `target_floors` (list): List of target floor numbers
- `assignments` (dict): Assignment dictionary to modify
**Use Case**: Bulk assignment of identical floor configurations

---

### Model Persistence

```python
def save_model_to_file(project_name: str) -> str
```
**Purpose**: Serialize complete model state to pickle file for analysis  
**Parameters**:
- `project_name` (str): Project identifier (becomes filename)
**Returns**: Path to saved file  
**Saves**:
- Complete geometry, materials, sections
- All element assignments and configurations
- Slab and load parameters (if defined)
- Analysis completion flags
- Project metadata
**File Format**: `models/{project_name}.pkl`

---

### Visualization

```python
def create_floor_plan_figure(floor: int, coordx: list, coordy: list,
                           column_assignments: dict, beamx_assignments: dict,
                           beamy_assignments: dict, sections: dict, materials: dict)
```
**Purpose**: Create interactive Plotly figure showing complete floor layout  
**Parameters**:
- `floor` (int): Floor number to visualize
- `coordx`, `coordy` (list): Building grid coordinates
- Assignment dictionaries for elements
- `sections`, `materials` (dict): Property definitions
**Returns**: Plotly figure with:
- Blue squares for columns with section labels
- Green lines/markers for X-direction beams
- Orange lines/markers for Y-direction beams
- Hover tooltips with material properties
- Interactive zooming and panning

---

### UI Rendering Functions

The application features 11 comprehensive tabs, each with dedicated rendering functions:

1. **`render_geometry_tab()`**: Building layout definition with coordinate input validation
2. **`render_materials_tab()`**: Material creation with property validation and preview
3. **`render_sections_tab()`**: Cross-section definition with reinforcement configuration
4. **`render_columns_tab()`**: Column assignment with visual grid and copy functionality
5. **`render_beams_tab()`**: Beam assignment for X and Y directions with visualization
6. **`render_model_visualization_tab()`**: Complete model review and element creation
7. **`render_slabs_tab()`**: Slab definition with material properties
8. **`render_loads_tab()`**: Gravity load application for analysis preparation
9. **`render_save_model_tab()`**: Model export with validation and file management
10. **`render_modal_analysis_tab()`**: Eigenvalue analysis with mode shape visualization
11. **`render_gravity_pushover_tab()`**: Nonlinear pushover analysis with customizable parameters

---

### Usage Example

```python
# Run the application
streamlit run RCF3D.py

# Workflow:
# 1. Tab 1: Enter coordinates and masses, create model
# 2. Tab 2: Define materials (concrete + steel properties)
# 3. Tab 3: Define sections (dimensions + reinforcement)
# 4. (Full version) Tab 4-5: Assign sections to columns and beams
# 5. (Full version) Tab 6: Visualize model
# 6. (Full version) Tab 7-8: Define slabs and loads
# 7. (Full version) Tab 9-10: Run analyses
# 8. Tab 10: Save model to pickle file
```

---

## Data Structures

### Material Dictionary Structure
```python
materials = {
    'Concrete_Column_1': {
        'fc': 28.0,              # MPa
        'fy': 420.0,             # MPa
        'detailing': 'DES',
        'noconf_tag': 101,       # OpenSeesPy tag
        'conf_tag': 102,
        'acero_tag': 103
    }
}
```

### Section Dictionary Structure
```python
sections = {
    'Col_40x40': {
        'tag': 1001,             # OpenSeesPy section tag
        'H': 0.40,               # m
        'B': 0.40,               # m
        'cover': 0.05,           # m
        'material': 'Concrete_Column_1',
        'bars_top': 2,
        'area_top': 'As6',
        'bars_bottom': 2,
        'area_bottom': 'As6',
        'bars_middle': 6,
        'area_middle': 'As6'
    }
}
```

### Column Assignments Structure
```python
column_assignments = {
    1: {                         # Floor 1
        0: {                     # X index 0
            0: 'Col_40x40',      # Y index 0
            1: 'Col_40x40',      # Y index 1
            2: 'Col_30x30'       # Y index 2
        },
        1: {                     # X index 1
            0: 'Col_40x40',
            1: 'Col_30x30',
            2: 'Col_30x30'
        }
    },
    2: { ... }                   # Floor 2
}
```

### Beam Assignments Structure
```python
beamx_assignments = {
    1: {                         # Floor 1
        0: {                     # Span 0 (between X[0] and X[1])
            0: 'Beam_30x50',     # At Y index 0
            1: 'Beam_30x50',     # At Y index 1
        }
    }
}

beamy_assignments = {
    1: {                         # Floor 1
        0: {                     # At X index 0
            0: 'Beam_25x45',     # Span 0 (between Y[0] and Y[1])
            1: 'Beam_25x45',     # Span 1 (between Y[1] and Y[2])
        }
    }
}
```
