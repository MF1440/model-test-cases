function stationEci = calcEci(lat,lon,alt,epoch)

    % ОПИСАНИЕ:
    % Возвращает координаты в трехмерной геоцентрической инерциальной
    % системе отсчета (ECI).
    % lat - широта в геоцентрической СК в [град.] 
    % lon - долгота в геоцентрической СК в [град.] 
    % alt - высота над поверхность земного шара в геоцентрической СК в [м.] 
    % epoch - эпоха, на момент которой необходимо определить координаты
    % точки [J2000] 
    %
    % ВХОДНЫЕ ДАННЫЕ:
    % lat,lon,alt,epoch - одномерные скалярные величины.
    %
    % ВЫХОДНЫЕ ЗНАЧЕНИЯ:
    % stationEci - трехмерный вектор в геоцентрической инерциальной системе
    % координат, где каждый элемент имеет размерность метров.


    earthRadius = 6378137;      % Экваториальный радиус Земли [м]
    earthRotVel = 7.292115e-5;  % Суточная угловая скорость вращения Земли [рад/c]

    stationEcef = zeros(1,3);

    stationEcef(1) = (earthRadius + alt) * cosd(lat) * cosd(lon);
    stationEcef(2) = (earthRadius + alt) * cosd(lat) * sind(lon);
    stationEcef(3) = (earthRadius + alt) * sind(lat);

    % Вычисления угла поворота планеты необходимо для определения матрица
    % поворота, с помощью которой вычисляется кооридинаты станций в ECI
    rotationAngleRad = epoch * earthRotVel;
    rotMatrix = [cos(rotationAngleRad) -sin(rotationAngleRad) 0;...
                 sin(rotationAngleRad) cos(rotationAngleRad) 0;...
                 0 0 1];
    stationEci = stationEcef*rotMatrix;

end

