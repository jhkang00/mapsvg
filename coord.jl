function convline(x, y)
    lon = (x-4170/2)*(180/(4170/2));
    lat = -(y-1668/2)*(72/(1668/2));
    return lon, lat;
end

function convcoor(lon, lat)
    x = lon*(4170/2/180)+4170/2;
    y = -lat*(1668/2/72)+1668/2;
    return x, y;
end