# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MapSVG is a cartographic projection system that transforms Mercator projection SVG world maps into various geographic projections, particularly globe/orthographic projections. Available in both Julia and Python implementations.

## Development Commands

### Julia Implementation
- **Run main projection script**: `julia mapsvg.jl`
- **Launch interactive 3D globe**: `julia create_globe.jl`
- **Install dependencies**: `julia -e 'using Pkg; Pkg.instantiate()'`
- **Add required packages**: `julia -e 'using Pkg; Pkg.add(["GLMakie", "GeometryBasics", "EzXML", "Observables"])'`

### Python Implementation
- **Run main projection script**: `python mapsvg.py`
- **Install dependencies**: `pip install -r requirements.txt` (currently no external dependencies)

## Architecture Overview

### Core Module System
The project uses a modular architecture available in both Julia and Python implementations:

#### Julia Files
- **coord.jl**: Coordinate transformation between SVG pixels (4170×1668) and geographic coordinates (±180°×±72°)
- **proj.jl**: Projection engine supporting Winkel Tripel and Orthographic projections with filtering and clipping
- **addgrid.jl**: Latitude/longitude grid overlay system with configurable spacing
- **shifthemi.jl**: Hemisphere alignment for combining eastern/western hemisphere data
- **create_globe.jl**: Interactive 3D visualization using GLMakie with mouse controls

#### Python Files (equivalent functionality)
- **coord.py**: Coordinate transformation functions
- **proj.py**: Projection engine with sinc function implementation
- **addgrid.py**: Grid system using built-in random module
- **shifthemi.py**: Hemisphere management with regex coordinate replacement
- **mapsvg.py**: Main orchestration script

### Data Flow
1. **Input**: Separate `InnerWorld.svg` and `OuterWorld.svg` hemisphere files
2. **Coordinate Transformation**: SVG pixels → geographic coordinates → projection coordinates
3. **Grid Enhancement**: Adds customizable lat/lon grid overlay
4. **Projection**: Transforms to target projection (orthographic, Winkel Tripel)
5. **Output**: New SVG with projected coordinates or 3D interactive visualization

### Key Technical Specifications
- **Coordinate System**: 4170×1668 pixel resolution covering ±180°×±72° geographic space
- **Projection Parameters**: Configurable center points for orthographic projections
- **Grid System**: Customizable line gaps (currently 30°) with point interpolation (currently 65 points)
- **3D Visualization**: GLMakie-based interactive globe with SVG path overlay

### File Roles
- **mapsvg.jl / mapsvg.py**: Main orchestration scripts combining all modules
- **testWorld copy.svg**: Current output file for projected maps
- **example_map.svg**: Simple test SVG for development
- Source data files: `InnerWorld.svg`, `OuterWorld.svg`
- **requirements.txt**: Python dependencies (currently no external packages needed)

## Current Development State

The project generates orthographic projections centered at 300°E, -10°N with 30° grid spacing. The main output file `testWorld copy.svg` is actively being modified during development iterations.

## Code Patterns

- Functions use descriptive names: `convline()` for coordinate conversion, `projtrans()` for projection transformation
- SVG processing uses regex patterns for coordinate extraction and replacement
- Projection functions handle edge cases with filtering (`filtproj()`) and clipping for back-face culling
- 3D visualization separates parsing, coordinate transformation, and rendering phases