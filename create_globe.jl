using GLMakie
using GeometryBasics
using EzXML
using Observables  # for Observable, if not already available via GLMakie

# Define an alias for a 3D point with Float32 components.
const Point3f0 = GeometryBasics.Point{3,Float32}
const Point2f0 = GeometryBasics.Point{2,Float32}

function mercator_to_orthographic(lon, lat, center_lon=0, center_lat=0)
    lon, lat, center_lon, center_lat = (deg2rad(x) for x in (lon, lat, center_lon, center_lat))
    x = cos(lat) * sin(lon - center_lon)
    y = cos(center_lat) * sin(lat) - sin(center_lat) * cos(lat) * cos(lon - center_lon)
    z = sin(center_lat) * sin(lat) + cos(center_lat) * cos(lat) * cos(lon - center_lon)
    if z < 0
        return nothing
    end
    return (x, y, z)
end

# --- SVG Parsing ---
function parse_svg_paths(file_path::String)
    doc = EzXML.readxml(file_path)
    # Use XPath query to select all <path> nodes:
    path_nodes = findall("//path", doc)
    svg_paths = []
    for node in path_nodes
        d_attr = node["d"]  # Access "d" attribute via overloaded getindex
        if d_attr != ""
            push!(svg_paths, parse_path_data(d_attr))
        end
    end
    return svg_paths
end

# A simple parser for the "d" attribute (assumes only "M" and "L" commands)
function parse_path_data(d::String)
    coords = []
    tokens = split(d)
    for token in tokens
        if startswith(token, "M") || startswith(token, "L")
            token = token[2:end]  # Remove the command letter
        end
        if occursin(",", token)
            parts = split(token, ",")
            if length(parts) == 2
                x = parse(Float64, parts[1])
                y = parse(Float64, parts[2])
                push!(coords, (x, y))
            end
        end
    end
    return coords
end

# --- Globe Visualization ---
function create_interactive_globe(svg_paths)
    # Use "size" instead of "resolution"
    fig = Figure(size = (800, 800))
    ax = Axis3(fig[1, 1], limits = ((-1, 1), (-1, 1), (-1, 1)), aspect = :data)
    
    # Draw the sphere (globe)
    sphere = GeometryBasics.Sphere(Point3f0(0, 0, 0), 1f0)
    mesh!(ax, sphere, color = :lightblue, transparency = true)
    
    # For each SVG path, convert each (lon, lat) to orthographic projection.
    for path in svg_paths
        points = [mercator_to_orthographic(lon, lat) for (lon, lat) in path]
        points = filter(!isnothing, points)
        if !isempty(points)
            lines!(ax, points, color = :black, linewidth = 1)
        end
    end
    
    # Create an observable to track the last mouse position
    lastpos = Observable(Point2f0(0,0))
    # Subscribe to changes in the mouse position.
    on(events(ax.scene).mouseposition) do pos_tuple
        # Convert the tuple (Float64, Float64) to a Point2f0 (i.e. Point{2,Float32})
        pos = Point2f0(pos_tuple...)
        if ispressed(ax.scene, Mouse.left)
            delta = pos - lastpos[]
            rotate!(ax.scene, Vec3(0.0, 1.0, 0.0), delta[1] * 0.01)
            rotate!(ax.scene, Vec3(1.0, 0.0, 0.0), delta[2] * 0.01)
        end
        lastpos[] = pos
    end
    
    
    fig
end

# --- Example Usage ---
svg_file = "example_map.svg"  # Make sure the file exists.
svg_paths = parse_svg_paths(svg_file)
create_interactive_globe(svg_paths)
