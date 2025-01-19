include("coord.jl")
include("addgrid.jl")
include("shifthemi.jl")
include("proj.jl")

origInS = read(open("InnerWorld.svg", "r"), String);
origOutS = read(open("OuterWorld.svg", "r"), String);

totxpix = convcoor(180,0)[1];

parseS = """<svg height="100%" viewBox="0 0 $totxpix 2068" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:vectornator="http://vectornator.io">\n""" * 
            shifthemi(origInS[findfirst("</g>",origInS)[end]+2:end-7],-convcoor(-174,0)[1]) * 
            shifthemi(origOutS[findfirst("</g>",origOutS)[end]+2:end],convcoor(0,0)[1]-convcoor(-174,0)[1]);

gridS = addgrid(parseS,30,65,30,30);

#display("image/svg+xml", gridS);
#write(open("testWorld copy.svg", "w"), gridS);
#display("image/svg+xml", project(90, 0, "Orthographic", gridS));
write(open("testWorld copy.svg", "w"), project(90, 30, "Orthographic", gridS));

#write(open("World copy.svg", "w"), parseS);
#display("image/svg+xml", read(open("World copy.svg", "r"), String))