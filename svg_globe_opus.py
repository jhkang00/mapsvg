#!/usr/bin/env python3
"""
Interactive 3D Globe Visualization
Parses SVG world map files and renders geographic features as vector curves
traced on a rotating sphere surface using PyVista.

Usage:
    pip install pyvista numpy
    python svg_globe.py

Input files (expected in current directory):
    - InnerWorld.svg - Eastern hemisphere map data
    - OuterWorld.svg - Western hemisphere map data
"""

import re
import math
import numpy as np
import pyvista as pv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Optional


# ============================================================================
# CONFIGURATION
# ============================================================================

# SVG coordinate system dimensions (unified after hemisphere alignment)
SVG_WIDTH = 4170.0
SVG_HEIGHT = 1668.0

# HEMISPHERE ALIGNMENT SHIFTS (CRITICAL)
# Each SVG uses local viewBox 0-2224, these shifts align them to unified space
# Reference: convcoor(-174, 0) = (69.5, 834), convcoor(0, 0) = (2085, 834)
INNER_SHIFT = -69.5      # InnerWorld (Eastern hemisphere) shifts left
OUTER_SHIFT = 2015.5     # OuterWorld (Western hemisphere) shifts right

# Geographic bounds (Mercator projection limits)
LON_RANGE = 180.0  # ±180 degrees
LAT_RANGE = 72.0   # ±72 degrees

# Sphere parameters
SPHERE_RADIUS = 1.0
SPHERE_RESOLUTION = 400

# Bézier curve sampling
BEZIER_SAMPLES = 5  # Points per curve segment

# Douglas-Peucker simplification
SIMPLIFY_ENABLED = True
SIMPLIFY_EPSILON = 0.2  # Unified SVG pixel units
SIMPLIFY_THRESHOLD = 1000  # Min points before simplifying

# Visualization settings
WINDOW_SIZE = (1600, 1600)
LINE_WIDTH = 2.0
LINE_OPACITY = 0.85
MAP_TUBE_RADIUS = 0.001    # Thin tubes for coastlines
GRID_TUBE_RADIUS = 0.003   # Slightly thicker for grid lines
EQUATOR_TUBE_RADIUS = 0.005  # Thickest for equator
SPHERE_COLOR = '#1a1a2e'  # Dark blue
MAP_COLOR = '#e0e0e0'     # Light gray/white


# ============================================================================
# SVG PARSING
# ============================================================================

def parse_svg_file(filepath: str) -> List[str]:
    """
    Parse an SVG file and extract all path 'd' attribute data.
    
    Args:
        filepath: Path to SVG file
        
    Returns:
        List of path data strings (d attributes)
    """
    if not Path(filepath).exists():
        print(f"Warning: File not found: {filepath}")
        return []
    
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
        
        # Handle SVG namespace
        namespaces = {'svg': 'http://www.w3.org/2000/svg'}
        
        paths = []
        
        # Try with namespace first
        for path in root.findall('.//svg:path', namespaces):
            d = path.get('d')
            if d:
                paths.append(d)
        
        # Try without namespace if none found
        if not paths:
            for path in root.iter():
                if path.tag.endswith('path') or path.tag == 'path':
                    d = path.get('d')
                    if d:
                        paths.append(d)
        
        print(f"  Found {len(paths)} paths in {filepath}")
        return paths
        
    except ET.ParseError as e:
        print(f"Error parsing {filepath}: {e}")
        return []


def parse_path_commands(d: str) -> List[Tuple[str, List[float]]]:
    """
    Parse SVG path 'd' attribute into list of commands with coordinates.
    
    Args:
        d: SVG path data string
        
    Returns:
        List of (command, coordinates) tuples
    """
    commands = []
    
    # Pattern to match commands: letter followed by numbers
    # Handles both space and comma separators
    pattern = r'([MCmcLlHhVvZzSsQqTtAa])([^MCmcLlHhVvZzSsQqTtAa]*)'
    
    for match in re.finditer(pattern, d):
        cmd = match.group(1)
        coords_str = match.group(2).strip()
        
        if coords_str:
            # Extract all numbers (including decimals and negatives)
            numbers = re.findall(r'-?\d+\.?\d*', coords_str)
            coords = [float(n) for n in numbers]
        else:
            coords = []
        
        commands.append((cmd, coords))
    
    return commands


# ============================================================================
# BÉZIER CURVE PROCESSING
# ============================================================================

def cubic_bezier_point(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, 
                       p3: np.ndarray, t: float) -> np.ndarray:
    """
    Calculate point on cubic Bézier curve at parameter t.
    
    P(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃
    
    Args:
        p0: Start point
        p1: First control point
        p2: Second control point
        p3: End point
        t: Parameter [0, 1]
        
    Returns:
        Point on curve at parameter t
    """
    t2 = t * t
    t3 = t2 * t
    mt = 1 - t
    mt2 = mt * mt
    mt3 = mt2 * mt
    
    return mt3 * p0 + 3 * mt2 * t * p1 + 3 * mt * t2 * p2 + t3 * p3


def linearize_bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, 
                     p3: np.ndarray, num_samples: int = 5) -> List[np.ndarray]:
    """
    Sample points along a cubic Bézier curve.
    
    Args:
        p0: Start point
        p1: First control point
        p2: Second control point
        p3: End point
        num_samples: Number of samples (including endpoints)
        
    Returns:
        List of points along the curve
    """
    t_values = np.linspace(0, 1, num_samples)
    return [cubic_bezier_point(p0, p1, p2, p3, t) for t in t_values]


def path_to_points(path_data: str) -> List[List[np.ndarray]]:
    """
    Convert SVG path data to list of point segments.
    Each M (move-to) command starts a new disconnected segment.
    
    Args:
        path_data: SVG path 'd' attribute string
        
    Returns:
        List of segments, where each segment is a list of 2D points
    """
    commands = parse_path_commands(path_data)
    segments = []
    current_segment = []
    current_pos = np.array([0.0, 0.0])
    
    for cmd, coords in commands:
        if cmd == 'M':
            # Move to (absolute) - starts a NEW segment
            if len(coords) >= 2:
                # Save previous segment if it has points
                if len(current_segment) >= 2:
                    segments.append(current_segment)
                
                # Start new segment
                current_pos = np.array([coords[0], coords[1]])
                current_segment = [current_pos.copy()]
                
                # Subsequent pairs are implicit line-to commands
                for i in range(2, len(coords) - 1, 2):
                    current_pos = np.array([coords[i], coords[i + 1]])
                    current_segment.append(current_pos.copy())
                    
        elif cmd == 'm':
            # Move to (relative) - starts a NEW segment
            if len(coords) >= 2:
                # Save previous segment if it has points
                if len(current_segment) >= 2:
                    segments.append(current_segment)
                
                # Start new segment
                current_pos = current_pos + np.array([coords[0], coords[1]])
                current_segment = [current_pos.copy()]
                
                for i in range(2, len(coords) - 1, 2):
                    current_pos = current_pos + np.array([coords[i], coords[i + 1]])
                    current_segment.append(current_pos.copy())
                    
        elif cmd == 'C':
            # Cubic Bézier (absolute)
            for i in range(0, len(coords) - 5, 6):
                p0 = current_pos
                p1 = np.array([coords[i], coords[i + 1]])
                p2 = np.array([coords[i + 2], coords[i + 3]])
                p3 = np.array([coords[i + 4], coords[i + 5]])
                
                # Linearize the Bézier curve
                bezier_points = linearize_bezier(p0, p1, p2, p3, BEZIER_SAMPLES)
                
                # Add all points except the first (it's the current position)
                current_segment.extend(bezier_points[1:])
                current_pos = p3.copy()
                
        elif cmd == 'c':
            # Cubic Bézier (relative)
            for i in range(0, len(coords) - 5, 6):
                p0 = current_pos
                p1 = current_pos + np.array([coords[i], coords[i + 1]])
                p2 = current_pos + np.array([coords[i + 2], coords[i + 3]])
                p3 = current_pos + np.array([coords[i + 4], coords[i + 5]])
                
                bezier_points = linearize_bezier(p0, p1, p2, p3, BEZIER_SAMPLES)
                current_segment.extend(bezier_points[1:])
                current_pos = p3.copy()
                
        elif cmd == 'L':
            # Line to (absolute)
            for i in range(0, len(coords) - 1, 2):
                current_pos = np.array([coords[i], coords[i + 1]])
                current_segment.append(current_pos.copy())
                
        elif cmd == 'l':
            # Line to (relative)
            for i in range(0, len(coords) - 1, 2):
                current_pos = current_pos + np.array([coords[i], coords[i + 1]])
                current_segment.append(current_pos.copy())
                
        elif cmd in ('Z', 'z'):
            # Close path - connect back to start of current segment
            if current_segment:
                current_segment.append(current_segment[0].copy())
    
    # Don't forget the last segment
    if len(current_segment) >= 2:
        segments.append(current_segment)
    
    return segments


# ============================================================================
# DOUGLAS-PEUCKER SIMPLIFICATION
# ============================================================================

def perpendicular_distance(point: np.ndarray, line_start: np.ndarray, 
                           line_end: np.ndarray) -> float:
    """
    Calculate perpendicular distance from point to line segment.
    
    Args:
        point: The point to measure from
        line_start: Start of line segment
        line_end: End of line segment
        
    Returns:
        Perpendicular distance
    """
    if np.allclose(line_start, line_end):
        return np.linalg.norm(point - line_start)
    
    # Vector from line_start to line_end
    line_vec = line_end - line_start
    line_len = np.linalg.norm(line_vec)
    line_unit = line_vec / line_len
    
    # Vector from line_start to point
    point_vec = point - line_start
    
    # Project point onto line
    projection_length = np.dot(point_vec, line_unit)
    
    if projection_length < 0:
        return np.linalg.norm(point - line_start)
    elif projection_length > line_len:
        return np.linalg.norm(point - line_end)
    else:
        projection = line_start + projection_length * line_unit
        return np.linalg.norm(point - projection)


def douglas_peucker(points: List[np.ndarray], epsilon: float) -> List[np.ndarray]:
    """
    Apply Douglas-Peucker algorithm to simplify a polyline.
    
    Args:
        points: List of 2D points
        epsilon: Maximum distance threshold
        
    Returns:
        Simplified list of points
    """
    if len(points) <= 2:
        return points
    
    # Find point with maximum distance from line between endpoints
    max_dist = 0
    max_idx = 0
    
    for i in range(1, len(points) - 1):
        dist = perpendicular_distance(points[i], points[0], points[-1])
        if dist > max_dist:
            max_dist = dist
            max_idx = i
    
    # If max distance exceeds epsilon, recursively simplify
    if max_dist > epsilon:
        # Recursively simplify both segments
        left = douglas_peucker(points[:max_idx + 1], epsilon)
        right = douglas_peucker(points[max_idx:], epsilon)
        
        # Combine results (excluding duplicate point at junction)
        return left[:-1] + right
    else:
        # All points between endpoints are within tolerance
        return [points[0], points[-1]]


def simplify_path(points: List[np.ndarray], epsilon: float = SIMPLIFY_EPSILON,
                  threshold: int = SIMPLIFY_THRESHOLD) -> List[np.ndarray]:
    """
    Optionally simplify path using Douglas-Peucker if it exceeds threshold.
    
    Args:
        points: List of 2D points
        epsilon: Maximum distance threshold for simplification
        threshold: Minimum point count before simplifying
        
    Returns:
        Simplified (or original) list of points
    """
    if not SIMPLIFY_ENABLED or len(points) < threshold:
        return points
    
    return douglas_peucker(points, epsilon)


# ============================================================================
# COORDINATE TRANSFORMATIONS
# ============================================================================

def svg_to_geographic(x: float, y: float) -> Tuple[float, float]:
    """
    Convert SVG pixel coordinates to geographic (lon, lat) coordinates.
    
    Args:
        x: SVG x coordinate (0 to 4170)
        y: SVG y coordinate (0 to 1668)
        
    Returns:
        (longitude, latitude) in degrees
    """
    # SVG origin is top-left, geographic center is at middle
    lon = (x - SVG_WIDTH / 2) * (LON_RANGE / (SVG_WIDTH / 2))
    lat = -(y - SVG_HEIGHT / 2) * (LAT_RANGE / (SVG_HEIGHT / 2))
    
    # Clip latitude to ±72°
    lat = np.clip(lat, -LAT_RANGE, LAT_RANGE)
    
    return lon, lat


def geographic_to_cartesian(lon: float, lat: float, 
                            radius: float = SPHERE_RADIUS) -> np.ndarray:
    """
    Convert geographic coordinates to 3D Cartesian on sphere surface.
    
    Args:
        lon: Longitude in degrees
        lat: Latitude in degrees
        radius: Sphere radius
        
    Returns:
        3D point (x, y, z) on sphere surface
    """
    # Convert to radians
    lon_rad = math.radians(lon)
    lat_rad = math.radians(lat)
    
    # Spherical to Cartesian
    x = radius * math.cos(lat_rad) * math.cos(lon_rad)
    y = radius * math.cos(lat_rad) * math.sin(lon_rad)
    z = radius * math.sin(lat_rad)
    
    return np.array([x, y, z])


def svg_point_to_3d(point: np.ndarray) -> np.ndarray:
    """
    Convert SVG 2D point to 3D sphere coordinates.
    
    Args:
        point: 2D point in SVG pixel coordinates
        
    Returns:
        3D point on sphere surface
    """
    lon, lat = svg_to_geographic(point[0], point[1])
    return geographic_to_cartesian(lon, lat)


# ============================================================================
# 3D GEOMETRY GENERATION
# ============================================================================

def create_sphere(resolution: int = SPHERE_RESOLUTION) -> pv.PolyData:
    """
    Create a unit sphere mesh.
    
    Args:
        resolution: Number of vertices in theta and phi directions
        
    Returns:
        PyVista sphere mesh
    """
    sphere = pv.Sphere(
        radius=SPHERE_RADIUS,
        theta_resolution=resolution,
        phi_resolution=resolution
    )
    return sphere


def create_latitude_line(lat: float, num_points: int = 200, 
                         tube_radius: float = 0.004) -> pv.PolyData:
    """
    Create a latitude (parallel) line on the sphere as tube geometry.
    
    Args:
        lat: Latitude in degrees
        num_points: Number of points in the circle
        tube_radius: Radius of the tube
        
    Returns:
        PyVista tube representing the parallel
    """
    lat_rad = math.radians(lat)
    circle_radius = SPHERE_RADIUS * math.cos(lat_rad)
    z = SPHERE_RADIUS * math.sin(lat_rad)
    
    # Create closed circle with extra point to close the loop
    angles = np.linspace(0, 2 * np.pi, num_points + 1)
    
    points = np.zeros((num_points + 1, 3))
    points[:, 0] = circle_radius * np.cos(angles)
    points[:, 1] = circle_radius * np.sin(angles)
    points[:, 2] = z
    
    # Create polyline
    line = pv.PolyData(points)
    line.lines = np.array([num_points + 1] + list(range(num_points + 1)))
    
    # Convert to tube to eliminate vertex nodes
    return line.tube(radius=tube_radius, n_sides=8)


def create_longitude_line(lon: float, num_points: int = 100,
                          tube_radius: float = 0.003) -> pv.PolyData:
    """
    Create a longitude (meridian) line on the sphere as tube geometry.
    
    Args:
        lon: Longitude in degrees
        num_points: Number of points in the arc
        tube_radius: Radius of the tube
        
    Returns:
        PyVista tube representing the meridian
    """
    lon_rad = math.radians(lon)
    
    # Go from north pole to south pole
    theta = np.linspace(0, np.pi, num_points)
    
    points = np.zeros((num_points, 3))
    points[:, 0] = SPHERE_RADIUS * np.sin(theta) * np.cos(lon_rad)
    points[:, 1] = SPHERE_RADIUS * np.sin(theta) * np.sin(lon_rad)
    points[:, 2] = SPHERE_RADIUS * np.cos(theta)
    
    line = pv.PolyData(points)
    line.lines = np.array([num_points] + list(range(num_points)))
    
    # Convert to tube to eliminate vertex nodes
    return line.tube(radius=tube_radius, n_sides=8)


def create_grid_lines() -> Tuple[List[pv.PolyData], List[pv.PolyData], pv.PolyData]:
    """
    Create latitude and longitude grid lines as tube geometry.
    
    Returns:
        Tuple of (parallels list, meridians list, equator)
    """
    parallels = []
    equator = None
    
    # Latitude lines: -60° to +60° in 30° steps
    for lat in [-60, -30, 0, 30, 60]:
        if lat == 0:
            # Equator gets thicker tube
            equator = create_latitude_line(lat, tube_radius=EQUATOR_TUBE_RADIUS)
        else:
            parallels.append(create_latitude_line(lat, tube_radius=GRID_TUBE_RADIUS))
    
    # Longitude lines: -180° to +150° in 30° steps (12 lines)
    meridians = []
    for lon in range(-180, 180, 30):
        meridians.append(create_longitude_line(lon, tube_radius=GRID_TUBE_RADIUS))
    
    return parallels, meridians, equator


def points_to_polyline(points_3d: List[np.ndarray], as_tube: bool = True, 
                       tube_radius: float = 0.003) -> Optional[pv.PolyData]:
    """
    Convert list of 3D points to a PyVista polyline or tube.
    
    Args:
        points_3d: List of 3D points
        as_tube: If True, convert to tube geometry (eliminates vertex nodes)
        tube_radius: Radius of tube if as_tube is True
        
    Returns:
        PyVista polyline/tube or None if insufficient points
    """
    if len(points_3d) < 2:
        return None
    
    points_array = np.array(points_3d)
    
    # Create line connectivity
    n_points = len(points_array)
    lines = np.array([n_points] + list(range(n_points)))
    
    polyline = pv.PolyData(points_array)
    polyline.lines = lines
    
    # Convert to tube geometry to eliminate vertex nodes
    if as_tube:
        try:
            tube = polyline.tube(radius=tube_radius, n_sides=8)
            return tube
        except Exception:
            # Fall back to polyline if tube creation fails
            return polyline
    
    return polyline


# ============================================================================
# SVG TO 3D CONVERSION
# ============================================================================

def apply_hemisphere_shift(points: List[np.ndarray], shift: float) -> List[np.ndarray]:
    """
    Apply x-coordinate shift for hemisphere alignment.
    
    Args:
        points: List of 2D points in local SVG coordinates
        shift: Pixels to add to x-coordinate (negative for left shift)
        
    Returns:
        List of shifted 2D points in unified coordinate system
    """
    return [np.array([p[0] + shift, p[1]]) for p in points]


def process_svg_paths(svg_paths: List[str], hemisphere_shift: float = 0.0,
                      tube_radius: float = 0.001) -> List[pv.PolyData]:
    """
    Process SVG paths and convert to 3D polylines with hemisphere alignment.
    Properly handles disconnected segments (islands) within the same path.
    
    Args:
        svg_paths: List of SVG path 'd' attribute strings
        hemisphere_shift: X-coordinate shift for hemisphere alignment
                         (INNER_SHIFT for InnerWorld, OUTER_SHIFT for OuterWorld)
        tube_radius: Radius for tube geometry
        
    Returns:
        List of PyVista tube meshes (one per disconnected segment)
    """
    polylines = []
    total_points_before = 0
    total_points_after = 0
    total_segments = 0
    
    for i, path_data in enumerate(svg_paths):
        # Convert path to list of 2D point segments
        # Each segment is a disconnected part (separated by M commands)
        segments = path_to_points(path_data)
        
        for segment in segments:
            if len(segment) < 2:
                continue
            
            # CRITICAL: Apply hemisphere shift BEFORE any other processing
            segment = apply_hemisphere_shift(segment, hemisphere_shift)
            
            total_points_before += len(segment)
            
            # Optional simplification (after hemisphere alignment)
            if SIMPLIFY_ENABLED and len(segment) > SIMPLIFY_THRESHOLD:
                segment = simplify_path(segment)
            
            total_points_after += len(segment)
            
            # Convert to 3D (now using unified coordinates)
            points_3d = [svg_point_to_3d(p) for p in segment]
            
            # Create tube (one per disconnected segment - no false connections)
            tube = points_to_polyline(points_3d, as_tube=True, tube_radius=tube_radius)
            if tube is not None:
                polylines.append(tube)
                total_segments += 1
    
    shift_label = f"shift={hemisphere_shift:+.1f}"
    print(f"  [{shift_label}] Segments: {total_segments}, "
          f"Points: {total_points_before:,} -> {total_points_after:,} "
          f"({100 * total_points_after / max(1, total_points_before):.1f}%)")
    
    return polylines


# ============================================================================
# VISUALIZATION
# ============================================================================

def setup_plotter() -> pv.Plotter:
    """
    Create and configure the PyVista plotter.
    
    Returns:
        Configured PyVista Plotter
    """
    plotter = pv.Plotter(window_size=WINDOW_SIZE)
    
    # Use solid black background (avoids BackgroundRenderer issues with starfield)
    plotter.set_background('black')
    
    return plotter


def main():
    """Main entry point."""
    print("=" * 60)
    print("3D Globe Visualization from SVG World Maps")
    print("=" * 60)
    
    # File paths
    inner_svg = "InnerWorld.svg"
    outer_svg = "OuterWorld.svg"
    
    # Parse SVG files
    print("\nParsing SVG files...")
    inner_paths = parse_svg_file(inner_svg)
    outer_paths = parse_svg_file(outer_svg)
    
    print(f"Total paths found: {len(inner_paths) + len(outer_paths)}")
    
    if not inner_paths and not outer_paths:
        print("\nNo SVG paths found. Please ensure InnerWorld.svg and OuterWorld.svg")
        print("are in the current directory.")
        print("\nCreating demo globe with grid only...")
    
    # Process SVG paths to 3D WITH HEMISPHERE SHIFTS
    print("\nConverting paths to 3D with hemisphere alignment...")
    map_polylines = []
    
    # Process InnerWorld (Eastern hemisphere) with left shift
    if inner_paths:
        print(f"  Processing InnerWorld ({len(inner_paths)} paths)...")
        inner_polylines = process_svg_paths(inner_paths, hemisphere_shift=INNER_SHIFT,
                                            tube_radius=MAP_TUBE_RADIUS)
        map_polylines.extend(inner_polylines)
        print(f"    Created {len(inner_polylines)} tube segments from InnerWorld")
    
    # Process OuterWorld (Western hemisphere) with right shift
    if outer_paths:
        print(f"  Processing OuterWorld ({len(outer_paths)} paths)...")
        outer_polylines = process_svg_paths(outer_paths, hemisphere_shift=OUTER_SHIFT,
                                            tube_radius=MAP_TUBE_RADIUS)
        map_polylines.extend(outer_polylines)
        print(f"    Created {len(outer_polylines)} tube segments from OuterWorld")
    
    print(f"Total polylines: {len(map_polylines)}")
    
    # Create sphere
    print("\nGenerating sphere geometry...")
    sphere = create_sphere()
    
    # Create grid (now returns tube geometry)
    print("Generating lat/lon grid...")
    parallels, meridians, equator = create_grid_lines()
    
    # Setup visualization
    print("\nSetting up visualization...")
    plotter = setup_plotter()
    
    # Add sphere
    plotter.add_mesh(
        sphere,
        color=SPHERE_COLOR,
        smooth_shading=True,
        opacity=1.0,
        name='sphere'
    )
    
    # Add map polylines (now tube geometry - no vertex nodes)
    for i, polyline in enumerate(map_polylines):
        plotter.add_mesh(
            polyline,
            color=MAP_COLOR,
            opacity=LINE_OPACITY,
            smooth_shading=True,
            name=f'map_{i}'
        )
    
    # Add grid lines (tube geometry)
    # Equator (red)
    if equator is not None:
        plotter.add_mesh(
            equator,
            color='red',
            opacity=0.7,
            smooth_shading=True,
            name='equator'
        )
    
    # Other parallels (yellow)
    for i, parallel in enumerate(parallels):
        plotter.add_mesh(
            parallel,
            color='yellow',
            opacity=0.7,
            smooth_shading=True,
            name=f'parallel_{i}'
        )
    
    # Meridians (gray)
    for i, meridian in enumerate(meridians):
        plotter.add_mesh(
            meridian,
            color='gray',
            opacity=0.7,
            smooth_shading=True,
            name=f'meridian_{i}'
        )
    
    # Configure camera for proper globe viewing
    # Camera looks at sphere from positive X direction (viewing prime meridian)
    # Z-axis is up (north pole at top)
    # This gives natural globe orientation for terrain-style rotation
    camera_distance = 3.0
    plotter.camera_position = [
        (camera_distance, 0, 0),  # Camera location (looking from +X)
        (0, 0, 0),                # Focal point (sphere center)
        (0, 0, 1)                 # View up vector (Z is up = north)
    ]
    
    # Enable terrain-style camera controls
    # This keeps the "up" direction stable while rotating
    plotter.enable_terrain_style(mouse_wheel_zooms=True)
    
    # Show instructions
    print("\n" + "=" * 60)
    print("CONTROLS (Terrain Style):")
    print("  Left-drag:  Rotate globe (horizon stays stable)")
    print("  Right-drag: Pan view")
    print("  Scroll:     Zoom in/out")
    print("  Q or ESC:   Close window")
    print("=" * 60)
    print("\nCamera setup:")
    print("  - Viewing prime meridian (lon=0°)")
    print("  - North pole at top (Z-up)")
    print("  - Terrain-style rotation maintains horizon")
    
    # Show the visualization
    print("\nOpening visualization window...")
    plotter.show(title="SVG World Globe")


if __name__ == "__main__":
    main()