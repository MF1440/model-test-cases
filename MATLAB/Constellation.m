classdef Constellation < handle
    properties
        totalSatCount = 0;
        groupArray = {};
        state;

        % константы
        earthRadius = 6378135;           % Экваториальный радиус Земли [m]
        earthGM = 3.986004415e+14;       % Гравитационный параметр Земли [m3/s2]
        earthJ2 = 1.082626e-3;           % First zonal harmonic coefficient in the expansion of the Earth's gravity field
    end

    methods
        function this = Constellation(varargin)
            if isempty(varargin)
                return
            end

            groupName = varargin{1};
            this.loadFromConfigFile(groupName);
        end

        function loadFromConfigFile(this, groupName)
            data = jsondecode(fileread('constellationsTest.json'));
            desiredGroup = [];

            for i = 1:length(data)
                if strcmpi(data(i).name, groupName)
                    desiredGroup = data(i);
                    break
                end
            end

            if isempty(desiredGroup)
                disp('Группировка не найдена в файле');
                return
            end

            for i = 1:size(desiredGroup.Walkers, 1)
                group.inclination   = deg2rad(desiredGroup.Walkers(i, 1));
                group.satsPerPlane  = desiredGroup.Walkers(i, 2);
                group.planeCount    = desiredGroup.Walkers(i, 3);
                group.f             = desiredGroup.Walkers(i, 4);
                group.altitude      = desiredGroup.Walkers(i, 5);
                group.maxRaan       = deg2rad(desiredGroup.Walkers(i, 6));
                group.startRaan     = deg2rad(desiredGroup.Walkers(i, 7));
                group.totalSatCount = group.satsPerPlane * group.planeCount;

                this.groupArray{length(this.groupArray) + 1} = group;                
                this.totalSatCount = this.totalSatCount + group.totalSatCount;
            end
        end

        function calcInitialState(this)
            this.state.elements = zeros(this.totalSatCount, 6);
            shift = 1;

            for group = this.groupArray
                ending = shift + group{1}(1).totalSatCount - 1;
                this.state.elements(shift: ending, :) = this.calcInitialElements(group{1});
                shift = ending + 1;
            end
        end

        function elements = calcInitialElements(this, group)
            raanSpace = linspace(group.startRaan, group.startRaan + group.maxRaan, group.planeCount + 1);
            raanSpace = mod(raanSpace(1: end - 1), 2 * pi);

            elements = zeros(group.totalSatCount, 6);
            elementIdx = 1;
            raanIdx = 0;
            for raan = raanSpace
                for satIdx = 0: group.satsPerPlane - 1
                    sma = this.earthRadius + group.altitude * 1000;
                    aol = 2 * pi / group.satsPerPlane * satIdx ...
                        + 2 * pi / group.totalSatCount * group.f * raanIdx;

                    elements(elementIdx, :) = [sma, 0, 0, raan, group.inclination, aol];
                    elementIdx = elementIdx + 1;
                end % конец цикла разнесения КА в группе по плоскостям
                raanIdx = raanIdx + 1;
            end
        end        

        function propagateJ2(this, epochs)
            this.state.eci = zeros(this.totalSatCount, 3, length(epochs));

            inclination = this.state.elements(:, 5);
            sma         = this.state.elements(:, 1);
            omega0      = sqrt(this.earthGM ./ sma.^3);
            aol0        = this.state.elements(:, 6);

            raanPrecessionRate = -1.5 * (this.earthJ2 * this.earthGM^(1/2) * this.earthRadius^2) ./ (sma.^(7/2)) .* cos(inclination);
            draconicOmega      = sqrt(this.earthGM ./ sma.^3) ...
                              .* (1 - 1.5 * this.earthJ2 .* (this.earthRadius ./ sma).^2) ...
                              .* (1 - 4 .* cos(inclination).^2);

            for epochIdx = 1: length(epochs)
                aol = aol0 + epochs(epochIdx) * draconicOmega;
                omega = omega0 + epochs(epochIdx) * raanPrecessionRate;

                this.state.eci(:, :, epochIdx)  = [sma .* (cos(aol) .* cos(omega) - sin(aol) .* cos(inclination) .* sin(omega)), ...
                                                   sma .* (cos(aol) .* sin(omega) + sin(aol) .* cos(inclination) .* cos(omega)), ...
                                                   sma .* (sin(aol) .* sin(inclination))];
            end 
        end
    end
end
