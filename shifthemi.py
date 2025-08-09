import re

def shifthemi(orig_s, pixdis):
    """Shift hemisphere coordinates horizontally by specified pixel distance."""
    def shift_coords(match):
        coords = match.group(0).split()
        x_coord = float(coords[0]) + pixdis
        y_coord = coords[1]
        return f"{x_coord} {y_coord}"
    
    pattern = r'([.\d]+)\s+([.\d]+)'
    new_s = re.sub(pattern, shift_coords, orig_s)
    return new_s