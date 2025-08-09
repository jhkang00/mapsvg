def convline(x, y):
    """Convert SVG pixel coordinates to longitude/latitude."""
    lon = (x - 4170/2) * (180 / (4170/2))
    lat = -(y - 1668/2) * (72 / (1668/2))
    return lon, lat

def convcoor(lon, lat):
    """Convert longitude/latitude to SVG pixel coordinates."""
    x = lon * (4170/2/180) + 4170/2
    y = -lat * (1668/2/72) + 1668/2
    return x, y