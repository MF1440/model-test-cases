function coordinatesIcs = transitionGeoToIcs(coordinatesGeo,epoch)
% ������� �������� �� �������������� � ������������ �������
% ��������� 
% coordinatesGeo = [lat,lon] - �������������� ����������,
% pointRadius - ������ �����. (����� ����� ������ ����� ��� ������ ������)
% coordinatesICS = [X,Y,Z] - ���������� ����������� ��� 

    % ���������
    earthRadius = 6378135;           % �������������� ������ ����� [m]
    earthRotV = 7.292115e-5;         % ������� �������� �������� ����� [���/c]
 
    % ��������� ��������� � ����������� ��� 
    coordinatesIcsn(1,1) = earthRadius ...
                           * cosd(coordinatesGeo(1)) ...
                           * cosd(coordinatesGeo(2)); % X
    coordinatesIcsn(1,2) = earthRadius ...
                           * cosd(coordinatesGeo(1)) ...
                           * sind(coordinatesGeo(2)); % Y
    coordinatesIcsn(1,3) = earthRadius ... 
                           * sind(coordinatesGeo(1)); % Z

    %  ���� �������� ����� �� ������� �����
    angleRotationEarth = epoch * earthRotV;
  
    % ������� �������� �� ����������� ��� � ��������� ���
    transferMatrixToRotationICS = [cos(angleRotationEarth) -sin(angleRotationEarth)  0;...
                                   sin(angleRotationEarth)  cos(angleRotationEarth)  0;...
                                   0                        0                        1];
    % ��������� ��������� � ��������� ���                              
    coordinatesIcs= coordinatesIcsn*transferMatrixToRotationICS';                         
end