import pyvista as pv
import numpy as np
import requests
import os
import math
from pyvista import examples

def create_textured_sphere(radius=1.0, resolution=400):
    theta_res = resolution
    phi_res = resolution

    theta = np.linspace(0, 2*np.pi, theta_res)
    phi = np.linspace(0, np.pi, phi_res)
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    x = radius * np.sin(phi_grid) * np.cos(theta_grid)
    y = radius * np.sin(phi_grid) * np.sin(theta_grid)
    z = radius * np.cos(phi_grid)

    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    points = np.column_stack((x,y,z))

    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = [theta_res, phi_res, 1]

    u = np.linspace(0, 1, theta_res)
    v = np.linspace(0, 1, phi_res)
    u_grid, v_grid = np.meshgrid(u,v)

    texture_coords = np.column_stack((u_grid.flatten(), 1-v_grid.flatten()))
    grid.active_texture_coordinates = texture_coords
    surface = grid.extract_surface()

    return surface

def download_high_res_texture(url, save_path):
    if os.path.exists(save_path):
        print(f"Using cached high-resolution texture: {save_path}")
        return save_path
    
    print(f"Downloading high-resolution texture from {url}...")
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print("High-resolution texture downloaded successfully")
            return save_path
    except Exception as e:
        print(f"error downloading high-resolution texture: {e}")
        return None
    
sphere = create_textured_sphere(radius=1.0, resolution=400)

high_res_texture_url = "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73909/world.topo.bathy.200412.3x5400x2700.jpg"
high_res_texture_path = "high_res_earth.jpg"

texture_path = download_high_res_texture(high_res_texture_url, high_res_texture_path)
texture = pv.read_texture(texture_path)

plotter = pv.Plotter(window_size=[1600,1600])

image_path = examples.planets.download_stars_sky_background(load=False)
plotter.add_background_image(image_path)

plotter.add_mesh(sphere, texture=texture, smooth_shading=True)

label_actors = {}

latitudes = list(range(-60, 61, 30))

for lat in latitudes:
    lat_rad = math.radians(lat)
    circle_radius = math.cos(lat_rad)

    circle = pv.Circle(radius = circle_radius, resolution=200)
    z_position = math.sin(lat_rad)
    translated_circle = circle.translate((0,0,z_position))

    color = 'yellow'
    opacity = 0.7
    line_width = 2
    if lat == 0:
        color = 'red'
        opacity = 0.7
        line_width = 3
    
    plotter.add_mesh(translated_circle, color=color, opacity=opacity, line_width=line_width, style='wireframe', render_lines_as_tubes = True)

longitudes = list(range(-180, 180, 30))
for lon in longitudes:
    lon_rad = math.radians(lon)

    points = []
    for theta in np.linspace(0, np.pi, 100):
        x = math.sin(theta) * math.cos(lon_rad)
        y = math.sin(theta) * math.sin(lon_rad)
        z = math.cos(theta)
        points.append([x,y,z])

    line = pv.lines_from_points(np.array(points))

    color = 'gray'
    opacity = 0.7
    line_width = 2

    plotter.add_mesh(line, color=color, opacity=opacity, line_width=line_width, render_lines_as_tubes=True)
    

plotter.enable_terrain_style()

plotter.show()