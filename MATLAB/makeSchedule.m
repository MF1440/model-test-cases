function [gateSchedule] = makeSchedule
%%%%%%%%%%% Выполнение тестового задания номер 11
%%%%%%%%%%% Автор: Александр Мурый
%
% Требования к данным
% рабочая дирректория должна содержать файлы: 
%       gatewaysTest.json,
%       constellationsTest.json
%       Constellation.m 


% Расписание gateSchedule будет иметь следующий формат: 
% матрица размером [gateways.count x antennasCount x epochsCount],
% где для каждого шлюза, в каждую эпоху времени, для каждой антенны указан
% порядковый номер спутника.
% Если в данную эпоху нет КА в зоне видимости или все видимые
% КА уже разобраны другими антеннами этого шлюза, то в расписании NaN



%%%%%%%%% Эта часть взята из файла example.m без изменения
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear

%создание объекта типа Constellation, инициализация параметрами группировки Stalink из конфига
constellation = Constellation('biWalkerGlobal');


%вычисление элементов орбиты для всех КА в начальный момент
constellation.getInitialState();

% определение точек на оси времени, в которые будут проихзводиться расчёты
epochs = (0: 1000: 6000);

% расчёт положений всех КА в заданные моменты времени
constellation.propagateJ2(epochs);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%% Здесь начинается мой код

% Определим высоту каждого КА группировки относительно центра Земли:
satAltitude = constellation.state.elements(:, 1);
satAltitudeUnique = sort(unique(satAltitude));
if length(satAltitudeUnique) ~= 2
    disp(['ERROR: по условиям задачи должно быть ровно два ешелона КА!']);
    return;
end

% Для каждого КА определить номер эшелона высоты, которой он принадлежит.
% По условиям задачи, в системе КА должно быть ровно 2 эшелона.
satEchelonIndex = ones(1, constellation.totalSatCount); 
for i=1:length(satAltitudeUnique)
    ind = satAltitude == satAltitudeUnique(i);
    satEchelonIndex(ind) = i;
end
clear ind


% Загружаем координаты шлюзовых станций из файла 'gatewaysTest.json'
fileName = 'gatewaysTest.json';
str = fileread(fileName);
data = jsondecode(str);

clear gateways
gateways.count = length(data);
% угловые координаты шлюзов
for i=1:gateways.count
    gateways.lat(i) = data(i).lat; % широта, в градусах
    gateways.lon(i) = data(i).lon; % долгота, в градусах
    gateways.altitude(i) = data(i).altitude; % высота, относительно поверхности Земли
end
gateways.altitudeFromEarthCentre = gateways.altitude + constellation.earthRadius;

% Декартовы координаты шлюзов относительно центра Земли
[x,y,z] = sph2cart(gateways.lon * pi/180, gateways.lat * pi/180, gateways.altitudeFromEarthCentre);
gateways.x = x;
gateways.y = y;
gateways.z = z;


epochsCount = length(epochs);


% критерий видимости КА шлюзом, в градусах (из задания):
epsilonThreshold = 15;

% для каждого шлюза в каждую эпоху определим видимость КА
satVisibility = false(gateways.count, epochsCount, constellation.totalSatCount); % инициируем, по дефолту false
for gateIdx = 1:gateways.count
    
    % вектор из центра Земли точку расположения шлюза
    gatePosition = [gateways.x(gateIdx), gateways.y(gateIdx), gateways.z(gateIdx)]; % вектор из центра Земли точку расположения шлюза
    gatePositionNormalized = gatePosition/norm(gatePosition); 
    
    for epochIdx = 1:epochsCount
        
        % координаты всех спутников в данную эпоху, в ECEF (т.е. относительно Земли, как и шлюзы):
        satPosEcef = constellation.state.ecef(:,:,epochIdx); 
       
        % вектор из шлюза в КА:
        gateToSatVector = satPosEcef - gatePosition;
        gateToSatVectorNorm = gateToSatVector ./ sqrt(sum(gateToSatVector.^2, 2));
        
        % угол места (эпсилон) для всех КА, в градусах:
        epsilon = 90 - acosd(dot(repmat(gatePositionNormalized, constellation.totalSatCount, 1), gateToSatVectorNorm, 2));
        
        % критерий видимости КА данному шлюзу в данную эпоху:
        satVisibility(gateIdx, epochIdx, :) = epsilon > epsilonThreshold;   
        
    end
    
end



% индексы всех КА:
satID = [1:constellation.totalSatCount]; 




% Расписание будет иметь следующий формат: 
% матрица размером [gateways.count x antennasCount x epochsCount],
% где для каждого шлюза, в каждую эпоху времени, для каждой антенны указан
% порядковый номер спутника.
% Если же в данную эпоху нет КА в зоне видимости антенны или все видимые
% КА уже разобраны другими антеннами этого шлюза, то в расписании ставим NaN

antennasCount = 10; % по условию задачи каждый шлюз содержит 10 антенн
gateSchedule = NaN(gateways.count, antennasCount, epochsCount); % инициируем матрицу расписания как NaN



% для каждого шлюза, выбрать оптимальные КА для каждой эпохи
% условия: 
% 1) приоритет тем КА, которые видны дольше
% 2) если есть такая возможность, каждый шлюз должен быть связам с обоими
%    эшелонами в каждую эпоху
for gateIdx = 1:gateways.count
    
    % видимость спутников для данного шлюза во все эпохи:
    satVisibility_currentGate = squeeze(satVisibility(gateIdx, :, :)); 
    
    for antennaIdx = 1:antennasCount
        
        % для каждого КА определить, на глубину скольки эпох он виден шлюзу
        % непрерывно, начиная с эпохи startEpoch: 
        startEpoch = 1;
        visibilityDuration = getVisDuration(satVisibility_currentGate, startEpoch, epochsCount, constellation.totalSatCount);

        % выбрать оптимальные спутники для каждой эпохи:
        satIdxPerEpoch = findBestSatPerEpoch(visibilityDuration, satVisibility_currentGate, epochsCount, constellation.totalSatCount);
        
        
                %  если это последняя антенна, то проверка на наличие обоих эшелонов в составленном до сих пор расписании шлюза;
                %  если не все эшелоны представленны, то меняем visibility КА так, чтобы дать приоритет непредставленным эшелонам
                if antennaIdx == antennasCount 
                    gateSchedule_local = squeeze(gateSchedule(gateIdx, 1:end-1, :));

                    weight = getVisibilityWeights(gateSchedule_local, satEchelonIndex, epochsCount, satID);
                    satVisibility_currentGate_weighted =  satVisibility_currentGate .* weight;

                    startEpoch = 1;
                    visibilityDuration_weighted = getVisDuration(satVisibility_currentGate_weighted, startEpoch, epochsCount, constellation.totalSatCount);
                    % выбрать оптимальные спутники для каждой эпохи:
                    [satIdxPerEpoch_weighted] = findBestSatPerEpoch(visibilityDuration_weighted, satVisibility_currentGate_weighted, epochsCount, constellation.totalSatCount);

                    ind = isnan(satIdxPerEpoch_weighted);
                    satIdxPerEpoch_weighted(ind) = satIdxPerEpoch(ind); % если условие наличия обоих эшелонов невозможно соблюсти, то возьмем любой другой доступный КА, чтобы в расписании не было NaN
                    satIdxPerEpoch = satIdxPerEpoch_weighted;
                end
                
        
        % сохраняем результат в таблицу расписания:
        gateSchedule(gateIdx, antennaIdx, :) = satIdxPerEpoch;
        
        % сделать выбранные КА недоступными для других антенн шлюза, тк они
        % уже зарезервированны для текущей антенны antennaIdx:
        for i = 1:epochsCount
            if ~isnan(satIdxPerEpoch(i))
                satVisibility_currentGate(i, satIdxPerEpoch(i)) = false;
            end
        end        
             
    end      
     
end



end
% % Sanity check:
% stppnt=1;
% squeeze(gateSchedule(1, :, :))
% 
% satEchelonIndex( squeeze(gateSchedule(2, :, :)) );


function visibilityDuration = getVisDuration(satVisibility_currentGate, startEpoch, epochsCount, satCount)
	% для каждого КА определить, на глубину скольки эпох он виден шлюзу
	% непрерывно, начиная с эпохи startEpoch

    % инициируем нулями
    visibilityDuration = zeros(epochsCount, satCount); 
    visibilityDuration(startEpoch, :) = satVisibility_currentGate(startEpoch, :);
    for epochIdx = startEpoch+1:epochsCount
        visibilityDuration(epochIdx, :) = visibilityDuration(epochIdx-1) + prod(satVisibility_currentGate(startEpoch:epochIdx, :), 1);
    end
    visibilityDuration = sum(visibilityDuration, 1);
end



function [satIdxPerEpoch] = findBestSatPerEpoch(visibilityDuration, satVisibility_currentGate, epochsCount, totalSatCount)
    % выбрать спутник, который видем дольше всех: 
    [maxVisibilityDuration, indBestSat] = max(visibilityDuration);
    satIdxPerEpoch = repmat(indBestSat, maxVisibilityDuration, 1);

    % если длительность видимости лучшего КА не покрывает все эпохи, то
    % продолжаем пока не покроем все эпохи:
    while length(satIdxPerEpoch) < epochsCount
        startEpoch = length(satIdxPerEpoch) + 1;
        visibilityDuration = getVisDuration(satVisibility_currentGate, startEpoch, epochsCount, totalSatCount);
        [maxVisibilityDuration, indBestSat] = max(visibilityDuration);

        if maxVisibilityDuration > 0 % ни одного спутника в зоне видимости
            satIdxPerEpoch = cat(1, satIdxPerEpoch, repmat(indBestSat, maxVisibilityDuration, 1));
        else
            % если нет спутников зоне видимости в этой эпохе, то в расписании ставим NaN 
            satIdxPerEpoch = cat(1, satIdxPerEpoch, NaN);
        end
    end
end

    
function [weight] = getVisibilityWeights(gateSchedule, satEchelonIndex, epochsCount, satID)
    representationEchelon1 = sum(satEchelonIndex(gateSchedule) == 1, 1) > 0; % представлен ли эшелон 1
    representationEchelon2 = sum(satEchelonIndex(gateSchedule) == 2, 1) > 0; % представлен ли эшелон 2

    echelonPriorityPerEpoch = zeros(1, epochsCount);      % 0 - no priority
    echelonPriorityPerEpoch(~representationEchelon1) = 1; % 1 - в эту эпоху не представлен эшелон 1, при поиске дать ему приоритет
    echelonPriorityPerEpoch(~representationEchelon2) = 2; % 2 - в эту эпоху не представлен эшелон 2, при поиске дать ему приоритет

    clear weight
    for i = 1:epochsCount
        if echelonPriorityPerEpoch(i) == 0
            weight(i,1:constellation.totalSatCount) = 1;
        else
            weight(i,:) = satEchelonIndex(satID) == echelonPriorityPerEpoch(i);
        end
    end
end

        


