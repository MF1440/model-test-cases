classdef ConstellationR < handle

    properties
        totalSatCount = 0;
        groupList = {}; % CodeReview (1.1.6)
        state;

        % ���������
        earthRadius = 6378135;           % �������������� ������ ����� [�]  % CodeReview (3.4.1)
        earthGM = 3.986004415e+14;       % �������������� �������� ����� [�3/�2] % CodeReview (3.4.1)
        earthJ2 = 1.082626e-3;           % ������ ��������� ��������� ������������� [��] % CodeReview (3.4.9)
    end

    methods

        function this = Constellation(varargin) % CodeReview (3.4.6)
            if isempty(varargin)
			disp('��� ����������� �� �������'); %  CodeReview (���������� isempty(dataGroup))
                return
            end
            groupName = varargin{1};
            this.loadFromConfigFile(groupName); % ��� ����������� CodeReview (3.4.6)
        end

        function loadFromConfigFile(this, code)
            fileName = 'constellationsTest.json';
            str = fileread(fileName);
            data = jsondecode(str);
            dataGroup = []; % CodeReview (1.1.5) 

            for i = 1:length(data)
                if strcmpi(data(i).name, code)
                    dataGroup = data(i);
                    break
                end
            end

            if isempty(dataGroup)
                disp('����������� �� ������� � �����');
                return
            end

            for i = 1:size(dataGroup.Walkers, 1)
                group.inclination = deg2rad(dataGroup.Walkers(i, 1));         % ���������� ����������� ��������� []  CodeReview (3.4.9)
                group.satsPerPlane = dataGroup.Walkers(i, 2);				  % ����� �� � ������ ����������� ��������� ������ [��]  CodeReview (3.4.9)
                group.planeCount = dataGroup.Walkers(i, 3);					  % ����� ����������� ���������� � ������ [��]  CodeReview (3.4.9)
                group.latitudeShift = dataGroup.Walkers(i, 4);				  % ������� ����� �� ��������� ������ ����� �� � �������� ���������� [����]  CodeReview (3.4.9, 1.1.5)
                group.altitude = dataGroup.Walkers(i, 5);					  % ������ ������ [��]  CodeReview (3.4.9)
                group.maxRaan = deg2rad(dataGroup.Walkers(i, 6));             % �������� ������� ����������� ����������� ���� (��� ������������� ����������� ����������) []  CodeReview (3.4.9)
                group.startRaan = deg2rad(dataGroup.Walkers(i, 7));			  % ������ ����������� ����������� ���� ��� ������ ��������� []  CodeReview (3.4.9)
                group.totalSatCount = group.satsPerPlane * group.planeCount;  % ����� �� [��] CodeReview (3.4)

                this.groupList{length(this.groups) + 1} = group;                
                this.totalSatCount = this.totalSatCount + group.totalSatCount;
            end
        end

        function calcInitialState(this) % CodeReview (1.2.5, 1.2.7)
            this.state.elements = zeros(this.totalSatCount, 6);
            shift = 1;

            for group = this.groups
                for groupIdx = 1:length(group{1}) %  CodeReview (1.1.7)
                    ending = shift + group{1}(groupIdx).totalSatCount - 1;
                    this.state.elements(shift:ending,:) = this.calcInitialElements(group{1});
                    shift = ending + 1;
                end
            end
        end

        function elementArray = calcInitialElements(this, group)  % CodeReview (1.2.5, 1.2.7, 1.1.6)
            raanArray = linspace(group.startRaan, group.startRaan + group.maxRaan, group.planeCount + 1); % % CodeReview (1.1.6)
            raanArray = mod(raanArray(1:end-1), 2 * pi);

            elementArray = zeros(group.totalSatCount, 6);
            idx = 1;
            raanIdx = 0;  %  CodeReview (1.1.2)
            for raan = raanArray
                for satsPlaneIdx = 0:group.satsPerPlane-1 %  CodeReview (1.1.7)
                    sma = this.earthRadius + group.altitude * 1000;
                    aol = 2 * pi / group.satsPerPlane * satsPlaneIdx + 2 * pi / group.totalSatCount * group.latitudeShift * raanIdx;

                    elementArray(idx, :) = [sma, 0, 0, raan, group.inclination, aol];
                    idx = idx + 1;
                end
                raanIdx = raanIdx + 1;
            end
        end        

        function propagateJ2(this, epochsArray) % CodeReview (1.1.6)
            this.state.eci = zeros(this.totalSatCount, 3, length(epochsArray));

            sma         = this.state.elements(:, 1);    % ������ ������ [�] CodeReview (3.4)
            inclination = this.state.elements(:, 5);    % ���������� ����������� ��������� [] CodeReview (3.4)         
            raan0       = this.state.elements(:, 4);    %  [] CodeReview (3.4)
            aol0        = this.state.elements(:, 6);    %  [] CodeReview (3.4)

            raanPrecessionRate = -1.5 * (this.earthJ2 * this.earthGM^(1/2) * this.earthRadius^2) ./ (sma.^(7/2)) .* cos(inclination);
            draconicOmega      = sqrt(this.earthGM ./ sma.^3) .* (1 - 1.5 * this.earthJ2 .* (this.earthRadius ./ sma).^2) .* (1 - 4 .* cos(inclination).^2);

            for epochIdx = 1:length(epochsArray)
                aol = aol0 + epochsArray(epochIdx) * draconicOmega;
                raanOmega = raan0 + epochsArray(epochIdx) * raanPrecessionRate;

                this.state.eci(:, :, epochIdx)  = [sma .* (cos(aol) .* cos(raanOmega) - sin(aol) .* cos(inclination) .* sin(raanOmega)), ...
                                                   sma .* (cos(aol) .* sin(raanOmega) + sin(aol) .* cos(inclination) .* cos(raanOmega)), ...
                                                   sma .* (sin(aol) .* sin(inclination))];
            end
        end
    end
end