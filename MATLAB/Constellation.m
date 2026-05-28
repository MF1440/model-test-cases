classdef Constellation < handle

    properties
        totalSatCount = 0;
        groups = {};
        state;
    end

    properties (Constant)
        % экваториальный радиус Земли [м]
        earthRadius = 6378135;

        % гравитационный параметр Земли [м3/с2]
        earthGM = 3.986004415e+14;

        % вторая зональная гармоника геопотенциала
        earthJ2 = 1.082626e-3;

        % угловая скорость вращения Земли [рад/с]
        earthAngVel = 2 * pi / 86164
    end

    methods

        function this = Constellation(varargin)
            if isempty(varargin)
                return
            end
            this.loadFromConfigFile(varargin{1});
        end

        function loadFromConfigFile(this, groupName)
            % Загружает данные указанной спутниковой группировки из внешнего файла
            %
            % Input Arguments:
            % - groupName (string) -
            %   имя группировки для загрузки

            fileName = 'constellationsTest.json';
            stringData = fileread(fileName);
            jsonData = jsondecode(stringData);
            dataThis = [];

            for constellationIdx = 1:length(jsonData)
                constellationData = jsonData(constellationIdx);
                if strcmpi(constellationData.name, groupName)
                    dataThis = constellationData;
                    break
                end
            end

            if isempty(dataThis)
                disp('Группировка не найдена в файле');
                return
            end

            for walkerIdx = 1:size(dataThis.Walkers, 1)
                % наклонение орбитальной плоскости
                group.inclination = deg2rad(dataThis.Walkers(walkerIdx, 1));

                % число КА в каждой орбитальной плоскости группы
                group.satsPerPlane = dataThis.Walkers(walkerIdx, 2);

                % число орбитальных плоскостей в группе
                group.planeCount = dataThis.Walkers(walkerIdx, 3);

                % фазовый сдвиг по аргументу широты между КА в соседних плоскостях
                group.phaseShift = dataThis.Walkers(walkerIdx, 4);

                % высота орбиты
                group.altitude = dataThis.Walkers(walkerIdx, 5);

                % максимум прямого восхождения восходящего узла (при распределении орбитальных плоскостей)
                group.maxRaan = deg2rad(dataThis.Walkers(walkerIdx, 6));

                % прямое восхождение восходящего узла для первой плоскости
                group.startRaan = deg2rad(dataThis.Walkers(walkerIdx, 7));

                group.totalSatCount = group.satsPerPlane * group.planeCount;

                this.groups{length(this.groups) + 1} = group;
                this.totalSatCount = this.totalSatCount + group.totalSatCount;
            end
        end

        function calcInitialState(this)
            % Выполняет расчёт Кеплеровых элементы орбиты группировки (всех входящих
            % в неё групп спутников)

            this.state.elements = zeros(this.totalSatCount, 6);
            shift = 1;

            for group = this.groups
                for i = 1:length(group{1})
                    ending = shift + group{1}(i).totalSatCount - 1;
                    this.state.elements(shift:ending, :) = this.calcInitialElements(group{1});
                    shift = ending + 1;
                end
            end
        end

        function elements = calcInitialElements(this, group)
            % Выполняет расчёт Кеплеровых элементов орбиты группы спутников
            %
            % Input Arguments:
            % - group -
            %   группа спутников для загрузки
            %
            % Output Arguments:
            % - elements (double) -
            %   матрица размера [totalSatCount, 6], строки которой имеют вид
            %     [semiMajorAxis, 0, 0, raan, inclination, argumentOfLatitude]

            raans = linspace(group.startRaan, group.startRaan + group.maxRaan, group.planeCount + 1);
            raans = mod(raans(1:end-1), 2 * pi);

            % sma - Semi-Major Axis
            sma = this.earthRadius + group.altitude * 1000;

            % aol - Argument of latitude
            aolStep = 2 * pi / group.satsPerPlane;

            elements = zeros(group.totalSatCount, 6);
            idx = 1;
            raanIDX = 0;

            for raan = raans
                aolShift = 2 * pi / group.totalSatCount * group.phaseShift * raanIDX;
                for satIdx = 0:group.satsPerPlane-1
                    aol = aolStep * satIdx + aolShift;

                    elements(idx, :) = [sma, 0, 0, raan, group.inclination, aol];
                    idx = idx + 1;
                end
                raanIDX = raanIDX + 1;
            end
        end

        function propagateJ2(this, epochs)
            % Для всех указанных эпох расчитывает координаты КА в инерциальной
            % системе координат ECI (Earth-centered inertial). Результаты работы
            % сохраняются в поле this.state.eci
            %
            % Input Arguments:
            % - epochs (int) -
            %   список эпох для расчета координат

            this.state.eci = zeros(this.totalSatCount, 3, length(epochs));

            sma         = this.state.elements(:, 1);
            inclination = this.state.elements(:, 5);
            raan0       = this.state.elements(:, 4);
            aol0        = this.state.elements(:, 6);

            inclinationCos = cos(inclination);
            inclinationSin = sin(inclination);

            earthJ2Radius = -1.5 * this.earthJ2 * this.earthRadius^2;

            raanPrecessionRate = inclinationCos ./ (sma.^(7/2)) * ...
                earthJ2Radius * this.earthGM^(1/2);

            draconicOmega = (1 - 4 .* inclinationCos.^2) .* ...
                (1 + earthJ2Radius ./ sma.^2) .* sqrt(this.earthGM ./ sma.^3);

            raanOmega = raan0 + epochs .* raanPrecessionRate;
            raanOmegaCos = cos(raanOmega);
            raanOmegaSin = sin(raanOmega);

            aol = aol0 + epochs .* draconicOmega;
            aolCos = cos(aol);
            aolSin = sin(aol);

            this.state.eci(:, 1, :) = aolCos .* raanOmegaCos - aolSin .* inclinationCos .* raanOmegaSin;
            this.state.eci(:, 2, :) = aolCos .* raanOmegaSin + aolSin .* inclinationCos .* raanOmegaCos;
            this.state.eci(:, 3, :) = aolSin .* inclinationSin;
            this.state.eci = this.state.eci .* sma;

            this.eci2ecef(epochs);
        end

        function eci2ecef(this, epochs)
            % Конвертирует координаты КА из системы ECI (Earth-Centered Inertial)
            % в систему ECEF (Earth-Centered, Earth-Fixed).
            %
            % Input Arguments:
            % - epochs (int) -
            %   список эпох для расчета координат

            this.state.ecef = zeros(size(this.state.eci));

            phi = this.earthAngVel .* epochs;
            phiCos = cos(phi);
            phiSin = sin(phi);

            eciX = squeeze(this.state.eci(:, 1, :));
            eciY = squeeze(this.state.eci(:, 2, :));
            eciZ = squeeze(this.state.eci(:, 3, :));

            this.state.ecef(:, 1, :) =  phiCos .* eciX + phiSin .* eciY;
            this.state.ecef(:, 2, :) = -phiSin .* eciX + phiCos .* eciY;
            this.state.ecef(:, 3, :) =  eciZ;
        end
    end
end
