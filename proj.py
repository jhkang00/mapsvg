import math
import re
from coord import convline, convcoor

def sinc(x):
    """Sinc function - sin(pi*x)/(pi*x), with special case for x=0."""
    if x == 0:
        return 1.0
    return math.sin(math.pi * x) / (math.pi * x)

def projtrans(lon, lat, lonz, latz, proj):
    """Transform coordinates using specified projection."""
    if proj == "Winkeltripel":
        rhlon = math.radians((lon - lonz) / 2)
        rlat = math.radians(lat)
        alpha = math.acos(math.cos(rlat) * math.cos(rhlon)) / math.pi
        newlon = math.degrees(rhlon + math.cos(rlat) * math.sin(rhlon) / sinc(alpha))
        newlat = math.degrees((rlat + math.sin(rlat) / sinc(alpha)) / 2)
    elif proj == "Orthographic":
        rlon = math.radians(lon - lonz)
        rlat = math.radians(lat)
        rlatz = math.radians(latz)
        clip = math.sin(rlatz) * math.sin(rlat) + math.cos(rlatz) * math.cos(rlat) * math.cos(rlon) < 0
        newlon = math.pi/2 * math.degrees(math.cos(rlat) * math.sin(rlon))
        newlat = math.pi/2 * math.degrees(math.cos(rlatz) * math.sin(rlat) - math.sin(rlatz) * math.cos(rlat) * math.cos(rlon))
        if clip:
            R = math.hypot(newlon, newlat)
            newlon *= 90/R
            newlat *= 90/R
    else:
        raise ValueError("No such projection.")
    return newlon, newlat

def filtproj(lonstr, latstr, lonz, latz, proj):
    """Filter and project coordinate strings."""
    lonnum, latnum = convline(float(lonstr), float(latstr))
    if (lonnum <= -180 and latnum >= 72) or (lonnum >= 180 and latnum <= -72):
        return f"{lonstr} {latstr}"
    else:
        projlon, projlat = projtrans(lonnum, latnum, lonz, latz, proj)
        newlon, newlat = convcoor(projlon, projlat)
        return f"{newlon} {newlat}"

def project(lonz, latz, proj, grid_s):
    """Apply projection to all coordinates in SVG string."""
    def replace_coords(match):
        coords = match.group(0).split()
        return filtproj(coords[0], coords[1], lonz, latz, proj)
    
    pattern = r'([.\d]+)\s+([.\d]+)'
    new_s = re.sub(pattern, replace_coords, grid_s)
    return new_s