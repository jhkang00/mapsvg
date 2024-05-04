include("coord.jl")

function projtrans(lon, lat, lonz, latz, proj)
    if (proj == "Winkeltripel")
        rhlon = deg2rad((lon-lonz)/2);
        rlat = deg2rad(lat);
        alpha = acos(cos(rlat)*cos(rhlon))/pi;
        newlon = rad2deg(rhlon+cos(rlat)*sin(rhlon)/sinc(alpha));
        newlat = rad2deg((rlat+sin(rlat)/sinc(alpha))/2);
    elseif (proj == "Orthographic")
        rlon = deg2rad(lon-lonz);
        rlat = deg2rad(lat);
        rlatz = deg2rad(latz);
        clip = sin(rlatz)*sin(rlat)+cos(rlatz)*cos(rlat)*cos(rlon) < 0;
        newlon = pi/2*rad2deg(cos(rlat)*sin(rlon));
        newlat = pi/2*rad2deg(cos(rlatz)*sin(rlat)-sin(rlatz)*cos(rlat)*cos(rlon));
        if (clip)
            R = hypot(newlon, newlat);
            newlon *= 90/R;
            newlat *= 90/R;
        end
    else
        throw("No such projection.");
    end
    return newlon, newlat
end

function filtproj(lonstr, latstr, lonz, latz, proj)
    lonnum, latnum = convline(parse(Float64,lonstr),parse(Float64,latstr));
    if ((lonnum <= -180 && latnum >= 72) || (lonnum >= 180 && latnum <= -72))
        return "$lonstr $latstr";
    else
        projlon, projlat = projtrans(lonnum, latnum, lonz, latz, proj);
        newlon, newlat = convcoor(projlon, projlat);
        newcoor = "$newlon $newlat";
        return newcoor
    end
end

function project(lonz, latz, proj, gridS)
    newS = replace(gridS, r"(([.]|\d)+)(\s)(([.]|\d)+)" => x -> filtproj(split(x)[1],split(x)[2], lonz, latz, proj));
    return newS
end