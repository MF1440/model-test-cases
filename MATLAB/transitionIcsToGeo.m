function coordinatesGeo = transitionIcsToGeo(coordinatesICS)
% ������� �������� �� ��� � �������������� �������
% ��������� 
% coordinatesICS = [X,Y,Z] - ���������� ����������� ��� 
% coordinatesGeo = [lat,lon] - �������������� ����������,

    % ��������� 
    Radian = 57.2957795; % 1 ������ � ��������
    
    % ��������� �������������� ��������� 
    coordinatesGeo(1,1) = atand ( coordinatesICS(3) / sqrt(coordinatesICS(1)^2 + coordinatesICS(2)^2));
    coordinatesGeo(1,2) = acosd ( coordinatesICS(1) / sqrt(coordinatesICS(1)^2 + coordinatesICS(2)^2));

    % ���� ��������� 
    if (coordinatesICS(2) <= 0) 
        coordinatesGeo(1,2) = -coordinatesGeo(1,2) + 180;
    end
    
    coordinatesGeo(1,1) = coordinatesGeo(1,1); % lat    
    coordinatesGeo(1,2) = coordinatesGeo(1,2); % lon  
end