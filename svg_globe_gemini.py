import sys
import os
import math
import re
import xml.etree.ElementTree as ET
import numpy as np
import pyvista as pv

# --- Configuration ---
SVG_WIDTH = 4170
SVG_HEIGHT = 1668
RADIUS = 1.0
LAT_LIMIT = 72.0
SAMPLE_STEPS = 5  # Points per Bezier curve (0.0, 0.25, 0.5, 0.75, 1.0)
RDP_EPSILON = 0.2 # Douglas-Peucker epsilon
SIMPLIFY_THRESHOLD = 1000 # Only simplify paths with more than this many points

class GlobeVisualizer:
    def __init__(self):
        self.paths_3d = [] # List of numpy arrays (N, 3)
        self.plotter = None

    # --- 1. Math & Coordinate Transformations ---

    def pixels_to_latlon(self, x, y):
        """
        Convert SVG pixel coordinates to Latitude/Longitude (degrees).
        Formula provided in requirements.
        """
        # Center points
        cx = SVG_WIDTH / 2
        cy = SVG_HEIGHT / 2
        
        # Scaling factors
        sx = 180.0 / cx
        sy = 72.0 / cy

        lon = (x - cx) * sx
        # Note: SVG Y is down, Geographic Lat is up, hence the negative sign
        lat = -(y - cy) * sy
        
        return lat, lon

    def latlon_to_cartesian(self, lat_deg, lon_deg):
        """
        Convert Lat/Lon (degrees) to 3D Cartesian coordinates on unit sphere.
        Clips latitude to +/- 72 degrees.
        """
        # Clip latitude
        lat_deg = max(-LAT_LIMIT, min(LAT_LIMIT, lat_deg))
        
        # Convert to radians
        lat_rad = math.radians(lat_deg)
        lon_rad = math.radians(lon_deg)

        # Spherical to Cartesian
        # x = r * cos(lat) * cos(lon)
        # y = r * cos(lat) * sin(lon)
        # z = r * sin(lat)
        
        x = RADIUS * math.cos(lat_rad) * math.cos(lon_rad)
        y = RADIUS * math.cos(lat_rad) * math.sin(lon_rad)
        z = RADIUS * math.sin(lat_rad)

        return x, y, z

    def cubic_bezier_sample(self, p0, p1, p2, p3, steps):
        """
        Generate points along a cubic Bezier curve.
        p0: Start [x, y]
        p1: Control 1 [x, y]
        p2: Control 2 [x, y]
        p3: End [x, y]
        steps: Number of samples
        """
        t = np.linspace(0, 1, steps)
        # Reshape for broadcasting
        # P(t) = (1-t)^3*P0 + 3(1-t)^2*t*P1 + 3(1-t)t^2*P2 + t^3*P3
        
        p0 = np.array(p0)
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)
        
        points = []
        for val in t:
            inv_t = 1 - val
            pt = (inv_t**3 * p0 + 
                  3 * inv_t**2 * val * p1 + 
                  3 * inv_t * val**2 * p2 + 
                  val**3 * p3)
            points.append(pt)
            
        return points

    # --- 2. Algorithms ---

    def douglas_peucker(self, points, epsilon):
        """
        Ramer-Douglas-Peucker algorithm for curve simplification.
        points: Array of [x, y]
        epsilon: Distance threshold
        """
        if len(points) < 3:
            return points

        dmax = 0.0
        index = 0
        end = len(points) - 1

        # Vectorized distance calculation
        # Line defined by p[0] and p[end]
        start_pt = points[0]
        end_pt = points[end]
        
        line_vec = end_pt - start_pt
        line_len = np.linalg.norm(line_vec)
        
        if line_len == 0:
            d = np.linalg.norm(points[1:end] - start_pt, axis=1)
        else:
            # Perpendicular distance in 2D
            # |(y2-y1)x0 - (x2-x1)y0 + x2y1 - y2x1| / sqrt((y2-y1)^2 + (x2-x1)^2)
            nums = np.abs(line_vec[1]*points[1:end, 0] - line_vec[0]*points[1:end, 1] + end_pt[0]*start_pt[1] - end_pt[1]*start_pt[0])
            d = nums / line_len

        if len(d) > 0:
            dmax = np.max(d)
            index = np.argmax(d) + 1 # Offset because we sliced [1:end]

        if dmax > epsilon:
            rec_results1 = self.douglas_peucker(points[:index+1], epsilon)
            rec_results2 = self.douglas_peucker(points[index:], epsilon)
            # rec_results1[:-1] removes the duplicate point at the merge
            return np.vstack((rec_results1[:-1], rec_results2))
        else:
            return np.array([points[0], points[end]])

    # --- 3. SVG Parsing ---

    def parse_svg_data(self, file_path):
        """
        Parse SVG file, extraction paths, linearizing Beziers, 
        and returning list of point arrays.
        """
        print(f"Reading {file_path}...")
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []

        # Namespace handling for SVG
        ns = {'svg': 'http://www.w3.org/2000/svg'}
        # Find all paths (try with and without namespace)
        paths = root.findall('.//svg:path', ns)
        if not paths:
            paths = root.findall('.//path')

        all_pixel_paths = []

        count_m = 0
        count_c = 0

        for path in paths:
            d = path.get('d')
            if not d:
                continue

            # Tokenize: Match commands (M, C) or numbers
            # This regex splits by letters or numbers
            tokens = re.findall(r'([a-zA-Z])|([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', d)
            # Flatten tuples from findall and remove empty strings
            tokens = [t[0] or t[1] for t in tokens if t[0] or t[1]]

            current_path_points = []
            cursor = [0, 0]
            
            idx = 0
            current_command = None

            while idx < len(tokens):
                item = tokens[idx]

                # Check if it is a command
                if item.isalpha():
                    current_command = item
                    idx += 1
                # If number, reuse last command (implicit command)
                
                if current_command == 'M':
                    # Move To: x y
                    if idx + 1 >= len(tokens): break
                    x = float(tokens[idx])
                    y = float(tokens[idx+1])
                    cursor = [x, y]
                    
                    # If we already have points, M starts a new disconnected path
                    # Save previous and start fresh
                    if current_path_points:
                        all_pixel_paths.append(np.array(current_path_points))
                        current_path_points = []
                    
                    current_path_points.append(cursor)
                    idx += 2
                    count_m += 1
                    
                    # Implicit M becomes L (LineTo), but prompt only mentions M and C.
                    # Usually map data is M -> C C C. We will assume subsequent nums are L if strict SVG,
                    # but for this specific prompt, likely M is just start. 
                    # If implicit numbers follow M, treat as L (LineTo)
                    current_command = 'L' 

                elif current_command == 'L':
                    # Line To: x y (Handling implicit after M)
                    if idx + 1 >= len(tokens): break
                    x = float(tokens[idx])
                    y = float(tokens[idx+1])
                    p0 = cursor
                    p1 = [x, y]
                    
                    # Linear interpolation (just a line)
                    # We add points to maintain structure
                    current_path_points.append(p1)
                    cursor = p1
                    idx += 2

                elif current_command == 'C':
                    # Cubic Bezier: x1 y1 x2 y2 x y
                    if idx + 5 >= len(tokens): break
                    x1 = float(tokens[idx])
                    y1 = float(tokens[idx+1])
                    x2 = float(tokens[idx+2])
                    y2 = float(tokens[idx+3])
                    x3 = float(tokens[idx+4])
                    y3 = float(tokens[idx+5])
                    
                    p0 = cursor
                    p1 = [x1, y1]
                    p2 = [x2, y2]
                    p3 = [x3, y3]
                    
                    bezier_points = self.cubic_bezier_sample(p0, p1, p2, p3, SAMPLE_STEPS)
                    
                    # Append points (skip first as it equals last of previous)
                    for pt in bezier_points[1:]:
                        current_path_points.append(pt)
                    
                    cursor = p3
                    idx += 6
                    count_c += 1
                
                elif current_command == 'Z' or current_command == 'z':
                    # Close path
                    idx += 1
                else:
                    # Unknown command or Z, skip
                    idx += 1

            if current_path_points:
                all_pixel_paths.append(np.array(current_path_points))

        print(f"  Parsed {len(all_pixel_paths)} sub-paths ({count_m} moves, {count_c} curves).")
        return all_pixel_paths

    def process_paths(self, filenames):
        """
        High level pipeline: Read -> Linearize -> Simplify -> Project -> Store
        """
        raw_paths = []
        for fn in filenames:
            if os.path.exists(fn):
                raw_paths.extend(self.parse_svg_data(fn))
            else:
                print(f"Warning: File {fn} not found.")

        if not raw_paths:
            print("No paths loaded. Please ensure InnerWorld.svg and OuterWorld.svg exist.")
            return

        print("Processing paths (Simplification & Projection)...")
        total_points_in = 0
        total_points_out = 0

        for path_2d in raw_paths:
            total_points_in += len(path_2d)
            
            # 1. Simplify (Douglas-Peucker) if heavy
            if len(path_2d) > SIMPLIFY_THRESHOLD:
                path_2d = self.douglas_peucker(path_2d, RDP_EPSILON)
            
            total_points_out += len(path_2d)

            # 2. Convert to 3D Cartesian
            path_3d = []
            for pixel in path_2d:
                lat, lon = self.pixels_to_latlon(pixel[0], pixel[1])
                x, y, z = self.latlon_to_cartesian(lat, lon)
                path_3d.append([x, y, z])
            
            self.paths_3d.append(np.array(path_3d))

        print(f"Optimization: {total_points_in} points -> {total_points_out} points.")

    # --- 4. Scene Generation ---

    def create_grid(self):
        """Generates lat/lon grid lines."""
        grid_lines = []
        
        # Parallels (Latitude lines)
        # -60 to 60 in steps of 30. Equator at 0.
        lats = [-60, -30, 0, 30, 60]
        for lat in lats:
            points = []
            # Create a circle at this latitude
            # 200 vertices
            for i in range(201):
                lon = -180 + (360 * i / 200.0)
                points.append(self.latlon_to_cartesian(lat, lon))
            
            grid_lines.append({
                'points': np.array(points),
                'color': 'red' if lat == 0 else 'yellow',
                'width': 3 if lat == 0 else 2
            })

        # Meridians (Longitude lines)
        # -180 to 150 in steps of 30 (12 lines)
        for lon in range(-180, 180, 30):
            points = []
            # Great circle arc from North Pole to South Pole
            # We iterate theta from 0 to pi, but mapping to lat -90 to 90
            # Lat range for grid: -90 to 90 (or clip to 72 if desired, usually grid goes to poles)
            # Prompt says: "Great circles from north to south pole"
            # However, map data clips at 72. We'll draw full meridians for aesthetics, 
            # or clip to 72 to match map. Let's clip to 72 to match the "window" look.
            
            for i in range(101):
                # lat from -72 to 72
                lat = -LAT_LIMIT + (2 * LAT_LIMIT * i / 100.0)
                points.append(self.latlon_to_cartesian(lat, lon))

            grid_lines.append({
                'points': np.array(points),
                'color': 'gray',
                'width': 2
            })

        return grid_lines

    def build_scene(self):
        """Constructs the PyVista scene."""
        print("Building 3D Scene...")
        
        # 1. Base Sphere
        sphere = pv.Sphere(radius=RADIUS * 0.99, theta_resolution=400, phi_resolution=400)
        
        # 2. Setup Plotter
        pl = pv.Plotter(window_size=[1600, 1600])
        pl.set_background('black')
        
        # Add Starfield
        try:
            cubemap = pv.examples.download_stars_sky_cube_map()
            pl.add_actor(cubemap.to_skybox())
        except:
            # Fallback if download fails or internet missing
            pass

        # Add Sphere
        pl.add_mesh(sphere, color='#1a2b3c', smooth_shading=True, specular=0.2)

        # 3. Add Grid
        grid_data = self.create_grid()
        for line in grid_data:
            # Create PolyData line
            pv_line = pv.lines_from_points(line['points'])
            pl.add_mesh(pv_line, color=line['color'], line_width=line['width'], 
                        opacity=0.7, render_lines_as_tubes=True)

        # 4. Add Map Paths
        # Merging lines for performance is better than adding thousands of actors
        map_lines_poly = pv.PolyData()
        
        # Construct connectivity and points arrays manually for speed
        # PyVista/VTK format: [n_points, p0, p1, ... pn] for each line
        all_points = []
        lines_list = []
        current_offset = 0
        
        for path in self.paths_3d:
            n_pts = len(path)
            if n_pts < 2: continue
            
            # Points
            all_points.append(path)
            
            # Connectivity
            # Format: [count, idx1, idx2, ... idxN]
            indices = np.arange(current_offset, current_offset + n_pts)
            line_conn = np.insert(indices, 0, n_pts)
            lines_list.append(line_conn)
            
            current_offset += n_pts

        if all_points:
            all_points_stacked = np.vstack(all_points)
            lines_stacked = np.hstack(lines_list)
            
            map_lines_poly = pv.PolyData(all_points_stacked)
            map_lines_poly.lines = lines_stacked
            
            pl.add_mesh(map_lines_poly, color='white', line_width=1.5, 
                        opacity=0.9, render_lines_as_tubes=True)

        # 5. Camera & Controls
        pl.camera_position = 'yz'
        pl.camera.azimuth = 45
        pl.camera.elevation = 20
        pl.enable_terrain_style_mouse_interaction()
        
        print("Visualization ready. Opening window...")
        pl.show(title="Interactive 3D SVG Globe")

# --- Main Execution ---

if __name__ == "__main__":
    app = GlobeVisualizer()
    
    # Input files
    inputs = ['InnerWorld.svg', 'OuterWorld.svg']
    
    # Run pipeline
    app.process_paths(inputs)
    app.build_scene()