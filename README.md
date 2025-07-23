# ReaxFF_tracer

Reaction Network Visualizer

A Python tool for analyzing and visualizing chemical reaction networks with interactive molecular structure display and pathway analysis.

## Features

🧪 **Chemical Reaction Network Analysis**
- Build bipartite reaction networks from CSV data
- Species nodes and reaction nodes with directional connections
- Backward pathway search from target to source species

🎨 **Interactive Visualization**
- Interactive Bokeh-based web visualization
- Hover tooltips with detailed information
- Molecular structure display (with RDKit integration)
- Pathway highlighting and subgraph extraction

🔍 **Pathway Analysis**
- Multi-reactant reaction handling
- Configurable path length and quantity limits
- Additional required species identification
- Path details and visualization

## Installation

### Prerequisites

```bash
pip install pandas networkx numpy bokeh
```

### Optional Dependencies

For molecular structure visualization:
```bash
pip install rdkit
```

## Usage

### Basic Usage

```python
from tracer5 import ReactionNetworkBuilder

# Initialize the network builder
builder = ReactionNetworkBuilder(
    spec_file="your_species.csv", 
    reac_file="your_reactions.csv"
)

# Build the network
builder.build_bipartite_network()

# Analyze pathways and visualize
source = "S1"     # Source species ID
target = "S100"   # Target species ID

paths, subgraph = builder.analyze_and_visualize_pathways(
    source=source,
    target=target,
    max_paths=3,
    max_length=50,
    output_filename="pathway_visualization.html"
)
```

## Configuration

### Customization Options

```python
# Adjust visualization parameters
builder.visualize_network(
    width=1600,           # Plot width
    height=900,           # Plot height
    output_filename="custom_network.html"
)

# Modify pathway search
paths = builder.find_pathways_backward(
    target="S100",
    source="S1", 
    max_paths=5,          # Find more paths
    max_length=30         # Shorter maximum length
)
```

### Random Seed Control

```python
# Ensure reproducible layouts
from tracer5 import set_random_seeds
set_random_seeds(42)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
