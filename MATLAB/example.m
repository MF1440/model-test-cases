% Example: load and propagate a satellite constellation.

clc
clear

tic

% Load Starlink constellation from config
constellation = Constellation('Starlink');

% Compute initial orbital elements
constellation.getInitialState();

% Time points for propagation [seconds]
epochs = 0:1001;

% Propagate all satellites (computes ECI and ECEF positions)
constellation.propagateJ2(epochs);

% Print position of a random satellite
satIdx = randi(constellation.totalSatCount);
epochIdx = randi(length(epochs));

fprintf('Всего КА: %d\n', constellation.totalSatCount);
fprintf('Положение КА-%d на эпоху %d:\n', satIdx, epochs(epochIdx));
fprintf('  ECI  (инерциальная СК): [%.2f, %.2f, %.2f]\n', constellation.state.eci(satIdx, :, epochIdx));
fprintf('  ECEF (вращающаяся СК):  [%.2f, %.2f, %.2f]\n', constellation.state.ecef(satIdx, :, epochIdx));

toc
