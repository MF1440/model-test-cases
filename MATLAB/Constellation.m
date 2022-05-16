classdef Constellation < handle
    properties
        totalSatCount = 0;
        groupList = {};
        state;

        % константы
        earthRadius = 6378135;           % Экваториальный радиус Земли [м]
        earthGM = 3.986004415e+14;       % Гравитационный параметр Земли [м3/с2]
        earthJ2 = 1.082626e-3;           % Вторая зональная гармоника геопотенциала [безразм. ед.]
    end
       
    methods
        function this = Constellation(varargin)

            if isempty(varargin)
                disp('Не получено имя исследуемой группировки');
                return
            end

            groupName = varargin{1}; %пояснить переменные
            this.loadFromConfigFile(groupName);
        end

        function loadFromConfigFile(this, code)
            fileName = 'constellationsTest.json';
            data = jsondecode(fileread(fileName));
            thisGroup = [];

            for i = 1:length(data)
                if strcmpi(data(i).name, code) 
                    thisGroup = data(i);
                    break
                end
            end

            if isempty(thisGroup)
                disp('Группировка не найдена в файле');
                return
            end

            for groupIdx = 1:size(thisGroup.Walkers, 1) 
                group.inclination  = deg2rad(thisGroup.Walkers(groupIdx, 1));    % наклонение орбитальной плоскости
                group.satsPerPlane    = thisGroup.Walkers(groupIdx, 2);          % число КА в каждой орбитальной плоскости группы
                group.planeCount      = thisGroup.Walkers(groupIdx, 3);          % число орбитальных плоскостей в группе
                group.f               = thisGroup.Walkers(groupIdx, 4);          % фазовый сдвиг по аргументу широты между КА в соседних плоскостях
                group.altitude        = thisGroup.Walkers(groupIdx, 5);          % высота орбиты
                group.maxRaan         = deg2rad(thisGroup.Walkers(groupIdx, 6)); % максимум прямого восхождения восходящего узла (при распределении орбитальных плоскостей)
                group.startRaan       = deg2rad(thisGroup.Walkers(groupIdx, 7)); % прямое восхождение восходящего узла для первой плоскости
                group.totalSatCount   = group.satsPerPlane * group.planeCount;

                this.groupList{length(this.groupList) + 1} = group;                
                this.totalSatCount = this.totalSatCount + group.totalSatCount;
            end
        end

        function updateInitialState(this)
            this.state.elements = zeros(this.totalSatCount, 6);
            shift = 1;

            for group = this.groupList
                for i = 1:length(group{1})
                    ending = shift + group{1}(i).totalSatCount - 1;
                    this.state.elements(shift:ending, :) = this.calcInitialElements(group{1});
                    shift = ending + 1;
                end
            end
        end

        function elements = calcInitialElements(this, group)
            raanList = linspace(group.startRaan, group.startRaan + group.maxRaan, group.planeCount + 1);
            raanList = mod(raanList(1:end-1), 2 * pi);

            elements = zeros(group.totalSatCount, 6);
            idx = 1;
            raanIdx = 0;
            sma = this.earthRadius + group.altitude * 1000;
            for raan = raanList
                for i = 0:group.satsPerPlane-1
                    aol = 2 * pi / group.satsPerPlane * i + 2 * pi / group.totalSatCount * group.f * raanIdx; % аномалия КА

                    elements(idx, :) = [sma, 0, 0, raan, group.inclination, aol];
                    idx = idx + 1;
                end
                raanIdx = raanIdx + 1;
            end
        end        

        function propagateJ2(this, epochList)
            this.state.eci = zeros(this.totalSatCount, 3, length(epochList));
            
            sma         = this.state.elements(:, 1);
            inclination = this.state.elements(:, 5);            
            raan0       = this.state.elements(:, 4);
            aol0        = this.state.elements(:, 6);

            raanPrecessionRate = -1.5 * (this.earthJ2 * this.earthGM^(1/2) * this.earthRadius^2) ./ ...
                                    (sma.^(7/2)) .* cos(inclination);
            draconicOmega      = sqrt(this.earthGM ./ sma.^3) .* ...
                                    (1 - 1.5 * this.earthJ2 .* (this.earthRadius ./ sma).^2) .* ...
                                    (1 - 4 .* cos(inclination).^2);

            for epochIdx = 1:length(epochList)
                aol = aol0 + epochList(epochIdx) * draconicOmega;
                raanOmega  = raan0  + epochList(epochIdx) * raanPrecessionRate;

                this.state.eci(:, :, epochIdx)  = [sma .* (cos(aol) .* cos(raanOmega) - sin(aol) .* cos(inclination) .* sin(raanOmega)), ...
                                                   sma .* (cos(aol) .* sin(raanOmega) + sin(aol) .* cos(inclination) .* cos(raanOmega)), ...
                                                   sma .* (sin(aol) .* sin(inclination))];
            end
        end
    end
end