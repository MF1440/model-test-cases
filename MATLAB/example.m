clc
clear

tic
%�������� ������� ���� Constellation, ������������� ����������� ����������� Stalink �� �������
constellation = Constellation('Starlink');

%���������� ��������� ������ ��� ���� �� � ��������� ������
constellation.getInitialState();

% ����������� ����� �� ��� �������, � ������� ����� �������������� �������
epochs = (0: 1000: 6000);

% ������ ��������� ���� �� � �������� ������� �������
constellation.propagateJ2(epochs);

% ���������� ���������� �� (� ������������ ����) ����� ����� ����� ��������� �� constellation.state.eci
satIdx = ceil(constellation.totalSatCount * rand());
epochIdx = ceil(length(epochs) * rand());
disp(['��������� ��-' num2str(satIdx) ' �� ����� ' num2str(epochs(epochIdx)) ':']);
disp(constellation.state.eci(satIdx, :, epochIdx));

% ������� 3 
visibleSatellites = findGatewayVisible(constellation, epochs, epochIdx)
   
 toc