function maxDistance = calcMaxDistance(earthRadius, orbitHeight, elevationMinAngle)
% ������� ������� ������������ ��������� ���������(��������� ���������) ����� �� � ��. 
% earthRadius - ������ �����,
% orbitHeight - ������ ������ ��,
% elevationMinAngle - ����������� ���� ����� ��.
% Maxdistance - ������������ ��������� ��������� (�������������
% ������������ ���� �����)
 
    satelliteViewAngle = asind( (earthRadius * sind (90 + elevationMinAngle))/( earthRadius + orbitHeight) ); % ���� ������ ��
    centralEarthAngle  = 180 - (90 + elevationMinAngle) - satelliteViewAngle; % ����������� ���� �� ������ �����
    maxDistance        = earthRadius * sind(centralEarthAngle) / sind(satelliteViewAngle); % ������������ ��������� ���������    
end