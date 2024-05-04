include("coord.jl")

function addgrid(origS, linegap, pointgap,lonz,latz)
    
    shadlons = lonz-180:0.1:lonz+180;
    lonlist = -180:linegap:150;
    latlist = -60:linegap:60;
    interp = 1:pointgap;
    newline = """""";
    
    for i in lonlist
        thislon, thislat = convcoor(i,60);
        newline *= """<path d="M$thislon $thislat"""; 
        for j in interp
            thislon, thislat = convcoor(i,60(1-2*j/pointgap));
            shiftlon = thislon+rand(-0.5:0.1:0.5);
            newline *= """L$thislon $thislat""";
        end
        newline *= """" fill="none" opacity="0.5" stroke="#000000" stroke-linecap="round" stroke-linejoin="round" stroke-width="0.5"/>\n""";
    end   
    
    for i in latlist
        thislon, thislat = convcoor(-180,i);
        newline *= """<path d="M$thislon $thislat""";
        for j in interp
            thislon, thislat = convcoor(-180(1-2*j/pointgap),i);
            shiftlat = thislat+rand(-0.5:0.1:0.5);
            newline *= """L$thislon $thislat""";
        end
        newline *= """" fill="none" opacity="0.5" stroke="#000000" stroke-linecap="round" stroke-linejoin="round" stroke-width="0.5"/>\n""";
    end
    
    newS = origS[1:end-7]*newline*origS[end-6:end];
    return newS
end