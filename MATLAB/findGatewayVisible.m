function visiblesArray = findGatewayVisible(constellation, epochArray, epochIdx)
% ������� ���������� ������� �� ��� �������� ������� (��)  � �������� ������
% �������. ������� ���������� ������ ������ � ��������� ������ ��, ������
% ��, ���� ����� � ������ ������, � ��� �� ������������ ����� �������
% ������������.

 % ���������
    earthRadius = 6378135;           % �������������� ������ ����� [�]
    elevationMinAngle = 25;          % ����������� ���� ����� [����]
    pointSpeedLight = 299792458;     % �������� ����� [�/�]
    carrierFrequency= 433e6;         % ������� ������� [��]
        
    % ��������� �������������� ��������� �������� ������� (��) � ��������� ������ ������� [����]
    gatewaysCoordinatesGeoArray = jsondecode(fileread('gatewaysTest.json'));
    
    satellitesCount = constellation.totalSatCount;          % ����� ��
    gatewaysCount = length(gatewaysCoordinatesGeoArray);    % ����� ��
    
    % ��������� ������������ ��������� �� �� ������� �����  [�]
    for gatewayIdx = 1: gatewaysCount
        gatewaysCoordinatesIcs(gatewayIdx, 1:3) = transitionGeoToIcs([gatewaysCoordinatesGeoArray(gatewayIdx).lat, ...
                                                                      gatewaysCoordinatesGeoArray(gatewayIdx).lon],...
                                                                      epochArray(epochIdx));
    end
        
    % ��������� ������������ ��������� �� �� ������ epochIdx [�]
    satellitesCoordinatesIcsArray = constellation.state.eci(:, :, epochIdx);
    
    % ��������� �������������� ��������� �� �� ������ epochIdx [�]
    for satellitesIdx = 1:satellitesCount
        satellitesCoordinatesGeoArray(satellitesIdx, 1:2) = transitionIcsToGeo(satellitesCoordinatesIcsArray(satellitesIdx,:)); % �������������� ���������� �� 
    end
    
    maxDistanceGroupList = zeros(2,1); % ������������� ������ ������������ ��������� ��������� ��� �����������
    
    % ���������� ������������ ��������� ����� �� � �� � ���� ��������� [�]
    maxDistanceGroupList(1) = calcMaxDistance(earthRadius, constellation.groups{1}.altitude*1000, elevationMinAngle);
    maxDistanceGroupList(2) = calcMaxDistance(earthRadius, constellation.groups{2}.altitude*1000, elevationMinAngle);
    firstGroupCount  = constellation.groups{1}.satsPerPlane * constellation.groups{1}.planeCount; % ����� �� � ������ ������
    secondGroupCount = constellation.groups{2}.satsPerPlane * constellation.groups{2}.planeCount; % ����� �� �� ������ ������
    
    pairSatGatArray     = []; % �������������� ������� ������� ��� �� � �� 
    distancesSatGatList = []; % �������������� ��������� ����� �� � ��
    distanceSatGatPoint = 0;  % �������������� �������� ���������� ��������� ����� �������� �� � ��  
    maxDistancePoint    = 0;  % �������������� �������� ���������� ������������ ��������� ����� �������� �� � ��  
    
    % ���������� ��������� ����� �� � �� [�]
    for satellitesIdx = 1: satellitesCount
        for gatewayIdx = 1:  gatewaysCount           
            distanceSatGatPoint = sqrt((satellitesCoordinatesIcsArray(satellitesIdx, 1) - gatewaysCoordinatesIcs(gatewayIdx, 1)) ^ 2 ...
                                     + (satellitesCoordinatesIcsArray(satellitesIdx, 2) - gatewaysCoordinatesIcs(gatewayIdx, 2)) ^ 2 ...
                                     + (satellitesCoordinatesIcsArray(satellitesIdx, 3) - gatewaysCoordinatesIcs(gatewayIdx, 3)) ^ 2 );
                                   
            if (satellitesIdx <=  firstGroupCount)    % ����� ������������ ��������� �� �����������
                maxDistancePoint =  maxDistanceGroupList(1);
            else
                maxDistancePoint =  maxDistanceGroupList(2);
            end
        
            if (distanceSatGatPoint <= maxDistancePoint) % ������� ���������
                pairSatGatArray     = [pairSatGatArray; satellitesIdx, gatewayIdx];
                distancesSatGatList = [distancesSatGatList; distanceSatGatPoint];
            end           
        end % ��������� ����� �� ���� ���������
    end % ��������� ����� �� ���� ��
    
    pairSatGatCount         = length(pairSatGatArray);   % ����� ��� �� � ��
    antennaOrientationArray = zeros(pairSatGatCount,2);  % �������������� ������� ���������� ������ ��� ��� �� � �� (���� ����� � ������ ������)
    auxiliaryVector         = zeros(1,3);                % �������������� �������� ���������� �������� ��������
     
    % ���������� ���������� ������ ��� ��� �� � ��
    for pairSatGatIdx = 1:pairSatGatCount
        auxiliaryVector = gatewaysCoordinatesIcs   (pairSatGatArray(pairSatGatIdx,2),:) ...
                        - satellitesCoordinatesIcsArray (pairSatGatArray(pairSatGatIdx,1),:); % ��������������� ������
         % ��������� ���� ����� [����]
         antennaOrientationArray(pairSatGatIdx, 2) =  acosd ( satellitesCoordinatesIcsArray(pairSatGatArray(pairSatGatIdx,1),:) * auxiliaryVector' / ...
                                                            ( norm(auxiliaryVector) * norm(satellitesCoordinatesIcsArray(pairSatGatArray(pairSatGatIdx,1),:)))) ...
                                                             - 90;
         % ����������� ������� ������ [����]   
         antennaOrientationArray(pairSatGatIdx, 1) = 180 + atand(tand ( gatewaysCoordinatesGeoArray(pairSatGatArray(pairSatGatIdx,2)).lon ...
                                                                      - satellitesCoordinatesGeoArray(pairSatGatArray(pairSatGatIdx,1))  ...
                                                                      / sind(gatewaysCoordinatesGeoArray(pairSatGatArray(pairSatGatIdx,2)).lat)));                                                
    end % ��������� ����� �� ����� �� � ��
    
    % ����������� ������������� �������� ( ���������� ��������. 
    % ��� ����������� ����� ��������� ����� ����������� ���� �����, �������
    % ������ ������� �� ���������)
    
    satelliteSpeedPoint = 0;                        % �������������� ���������� �������� �������� �� 
    orbitRadiusPoint    = 0;                        % ������������� ���������� ������� ������ ��� �������� �� 
    orbitHeghtPoint     = 0;                        % ������������� ���������� ������� ������ ��� �������� ��
    frequencyOffsetList = zeros(pairSatGatCount,1); % �������������� ������� ������ �������
    satelliteViewAngle  = 0;                        % ���� ������ �������� ��   
    
    for pairSatGatIdx = 1:pairSatGatCount
        
      if (pairSatGatArray(pairSatGatIdx,1) <=  firstGroupCount)    % ������ ������ � ������� ������ [�]
          orbitRadiusPoint = earthRadius + constellation.groups{1}.altitude * 1000;
          orbitHeghtPoint  = constellation.groups{1}.altitude * 1000;
      else
          orbitRadiusPoint = earthRadius + constellation.groups{2}.altitude * 1000;
          orbitHeghtPoint  = constellation.groups{2}.altitude*1000;
      end 
      
      satelliteSpeedPoint = sqrt(constellation.earthGM / orbitHeghtPoint); % �������� �������� [�/�]
      satelliteViewAngle = asind( (earthRadius * sind (90 + antennaOrientationArray(pairSatGatIdx, 2)))/( earthRadius + orbitHeghtPoint) ); % ���� ������ �������� ��
      % �������� �������
      frequencyOffsetList(pairSatGatIdx,1) = 2 * carrierFrequency * satelliteSpeedPoint * cosd(90 - satelliteViewAngle) / pointSpeedLight;
    end % ��������� ����� �� ����� �� � ��

   % �������������� ��������� ������ 
    visiblesArray = struct('satelliteIdx',    pairSatGatArray(:,1),          'gatewayIdx',     pairSatGatArray(:,2), ...
                           'elevationAngle',  antennaOrientationArray(:, 2), 'antennaAzimut',  antennaOrientationArray(:, 1), ...
                           'distancesSatGat', distancesSatGatList(:,1),      'frequencyOffset', frequencyOffsetList(:,1)); 
  
end