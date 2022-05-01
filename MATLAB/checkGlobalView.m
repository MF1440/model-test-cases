function isSeen = checkGlobalView(group, aol, alpha)
    earthRadius = 6378.135;

    earthAngleDimension = 2 * pi * earthRadius / 360;
    underHalfSatViewArea = group.altitude * tan(deg2rad(alpha));

    totalAlphaView = underHalfSatViewArea / earthAngleDimension;

    if (group.inclination + totalAlphaView) >= 87
        isSeen = true;
    else
        disp('Глобальная видимость не обеспечена по широтному положению группировки');
        isSeen = false;
    end

    if (group.maxRaan + totalAlphaView) >= (group.maxRaan / 2)
        isSeen = isSeen * true;
    else
        disp('Глобальная видимость не обеспечена по межвитковому разнесению КА группировки');
        isSeen = false;
    end
        
    if (aol + totalAlphaView) >= (aol / 2)
        isSeen = isSeen * true;
    else
        disp('Глобальная видимость не обеспечена по взаимному положению КА группировки');
        isSeen = false;
    end
end       
 