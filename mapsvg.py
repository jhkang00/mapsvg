from coord import convcoor
from addgrid import addgrid
from shifthemi import shifthemi
from proj import project

def main():
    # Read input SVG files
    with open("InnerWorld.svg", "r") as f:
        orig_in_s = f.read()
    
    with open("OuterWorld.svg", "r") as f:
        orig_out_s = f.read()
    
    # Calculate total x pixels
    tot_x_pix = convcoor(180, 0)[0]
    
    # Parse and combine hemispheres
    svg_header = f'<svg height="100%" viewBox="0 0 {tot_x_pix} 2068" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:vectornator="http://vectornator.io">\n'
    
    # Extract content after the first </g> tag, matching Julia exactly
    # Julia: findfirst("</g>",origInS)[end]+2:end-7
    inner_end_tag_pos = orig_in_s.find("</g>")
    inner_start = inner_end_tag_pos + len("</g>") + 1  # +1 for newline
    inner_content = orig_in_s[inner_start:-7]  # Remove closing </svg>
    
    # Julia: findfirst("</g>",origOutS)[end]+2:end  
    outer_end_tag_pos = orig_out_s.find("</g>")
    outer_start = outer_end_tag_pos + len("</g>") + 1  # +1 for newline
    outer_content = orig_out_s[outer_start:]  # Keep everything to end
    
    # Shift hemispheres
    inner_shifted = shifthemi(inner_content, -convcoor(-174, 0)[0])
    outer_shifted = shifthemi(outer_content, convcoor(0, 0)[0] - convcoor(-174, 0)[0])
    
    # Combine into single SVG 
    parse_s = svg_header + inner_shifted + outer_shifted
    
    # Add grid
    grid_s = addgrid(parse_s, 30, 65, 30, 30)
    
    # Apply projection and save
    projected = project(285, -10, "Orthographic", grid_s)
    
    with open("testWorld copy.svg", "w") as f:
        f.write(projected)
    
    print("Python MapSVG processing complete. Output saved to 'testWorld copy.svg'")

if __name__ == "__main__":
    main()