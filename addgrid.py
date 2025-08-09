import random
from coord import convcoor

def addgrid(orig_s, linegap, pointgap, lonz, latz):
    """Add latitude/longitude grid lines to SVG string."""
    lonlist = list(range(-180, 151, linegap))
    latlist = list(range(-60, 61, linegap))
    interp = list(range(1, pointgap + 1))
    newline = ""
    
    # Add longitude lines (meridians)
    for i in lonlist:
        thislon, thislat = convcoor(i, 60)
        newline += f'<path d="M{thislon} {thislat}'
        for j in interp:
            thislon, thislat = convcoor(i, 60 * (1 - 2 * j / pointgap))
            newline += f'L{thislon} {thislat}'
        newline += '" fill="none" opacity="0.5" stroke="#000000" stroke-linecap="round" stroke-linejoin="round" stroke-width="0.5"/>\n'
    
    # Add latitude lines (parallels)
    for i in latlist:
        thislon, thislat = convcoor(-180, i)
        newline += f'<path d="M{thislon} {thislat}'
        for j in interp:
            thislon, thislat = convcoor(-180 * (1 - 2 * j / pointgap), i)
            newline += f'L{thislon} {thislat}'
        newline += '" fill="none" opacity="0.5" stroke="#000000" stroke-linecap="round" stroke-linejoin="round" stroke-width="0.5"/>\n'
    
    # Insert grid lines before closing SVG tag
    new_s = orig_s[:-7] + newline + orig_s[-7:]
    return new_s