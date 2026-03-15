% Satellite constellation model.
%
% Loads Walker constellation parameters from a JSON config file,
% computes initial orbital elements, and propagates positions using
% a J2-perturbed two-body model.
%
% Usage:
%   c = Constellation('Starlink');
%   c.getInitialState();
%   c.propagateJ2(0:100:6000);

classdef Constellation < handle

    properties
        totalSatCount = 0;
        groups = {};
        state;
    end

    properties (Constant)
        % Physical constants for Earth
        earthRadius = 6378135;           % equatorial radius [m]
        earthGM     = 3.986004415e+14;   % gravitational parameter [m^3/s^2]
        earthJ2     = 1.082626e-3;       % second zonal harmonic of geopotential
        earthAngVel = 2 * pi / 86164;    % Earth rotation rate [rad/s]
    end

    methods

        function this = Constellation(varargin)
            if isempty(varargin)
                return
            end
            this.loadFromConfigFile(varargin{1});
        end

        function loadFromConfigFile(this, code)
            % Load constellation parameters from JSON config file.
            %
            % Args:
            %   code: constellation name (e.g. 'Starlink'), case-insensitive.

            % Resolve path relative to this script, not the working directory
            scriptDir = fileparts(mfilename('fullpath'));
            configPath = fullfile(scriptDir, '..', 'constellationsTest.json');

            if ~isfile(configPath)
                error('Config file not found: %s', configPath);
            end

            str = fileread(configPath);
            data = jsondecode(str);

            % Find constellation by name (case-insensitive)
            matchIdx = find(arrayfun(@(d) strcmpi(d.name, code), data), 1);
            if isempty(matchIdx)
                error('Constellation "%s" not found in %s', code, configPath);
            end
            dataThis = data(matchIdx);

            % Build Walker groups as struct array from Walkers matrix
            W = dataThis.Walkers;
            groups = struct( ...
                'inclination',   num2cell(deg2rad(W(:, 1))), ... % orbital plane inclination [rad]
                'satsPerPlane',  num2cell(W(:, 2)), ...          % satellites per orbital plane
                'planeCount',    num2cell(W(:, 3)), ...          % number of orbital planes
                'phaseShift',    num2cell(W(:, 4)), ...          % argument-of-latitude shift between adjacent planes
                'altitude',      num2cell(W(:, 5)), ...          % orbit altitude [km]
                'maxRaan',       num2cell(deg2rad(W(:, 6))), ... % max RAAN (Right Ascension of Ascending Node) for plane distribution [rad]
                'startRaan',     num2cell(deg2rad(W(:, 7))), ... % RAAN of the first plane [rad]
                'totalSatCount', num2cell(W(:, 2) .* W(:, 3)));
            this.groups = num2cell(groups);
            this.totalSatCount = sum(W(:, 2) .* W(:, 3));

            fprintf('Загружена группировка %s\n', code);
        end

        function getInitialState(this)
            % Compute initial orbital elements for all satellites.
            elementsList = cellfun(@(g) this.getInitialElements(g), ...
                                   this.groups, 'UniformOutput', false);
            this.state.elements = vertcat(elementsList{:});
        end

        function elements = getInitialElements(this, group)
            % Compute initial orbital elements for one Walker group.
            %
            % Returns:
            %   elements: (satCount x 6) array
            %       [sma, 0, 0, raan, inclination, argumentOfLatitude]
            %       sma in meters, angles in radians.

            raans = linspace(group.startRaan, group.startRaan + group.maxRaan, ...
                             group.planeCount + 1);
            raans = mod(raans(1:end-1), 2 * pi);

            % Vectorized: meshgrid of (planeIdx, satIdx) -> flat arrays
            satCount = group.totalSatCount;
            sma = this.earthRadius + group.altitude * 1000;  % km -> m

            [planeGrid, satGrid] = ndgrid(0:group.planeCount-1, ...
                                          0:group.satsPerPlane-1);
            planeGrid = planeGrid(:);
            satGrid = satGrid(:);

            elements = zeros(satCount, 6);
            elements(:, 1) = sma;
            elements(:, 4) = raans(planeGrid + 1);
            elements(:, 5) = group.inclination;
            elements(:, 6) = 2 * pi * (satGrid / group.satsPerPlane ...
                             + group.phaseShift * planeGrid / satCount);
        end

        function propagateJ2(this, epochs)
            % Propagate satellite positions with J2 perturbation.
            %
            % Computes ECI positions at each epoch using two J2-corrected rates:
            %   - RAAN drift: dOmega/dt = -3/2 * J2 * sqrt(mu) * Re^2 / a^(7/2) * cos(i)
            %   - Argument of latitude rate (draconic): corrected mean motion
            %
            % Args:
            %   epochs: vector of time points [seconds from initial epoch].
            %
            % Populates state.eci (totalSatCount x 3 x epochCount)
            % and state.ecef of the same shape.

            epochCount = length(epochs);
            sma         = this.state.elements(:, 1);
            inclination = this.state.elements(:, 5);
            raan0       = this.state.elements(:, 4);
            aol0        = this.state.elements(:, 6);

            % RAAN precession rate: dOmega/dt = -3/2 * J2 * sqrt(mu) * Re^2 / a^(7/2) * cos(i)
            raanRate = -1.5 * (this.earthJ2 * sqrt(this.earthGM) ...
                       * this.earthRadius^2) ./ sma.^(3.5) .* cos(inclination);

            % Draconic angular velocity with J2 correction
            draconicOmega = sqrt(this.earthGM ./ sma.^3) ...
                            .* (1 - 1.5 * this.earthJ2 .* (this.earthRadius ./ sma).^2) ...
                            .* (1 - 4 .* cos(inclination).^2);

            % Vectorized: compute all epochs simultaneously
            epochsRow = epochs(:)';  % (1 x T)
            aol  = aol0  + epochsRow .* draconicOmega;   % (N x T)
            raan = raan0 + epochsRow .* raanRate;        % (N x T)

            cosAol  = cos(aol);
            sinAol  = sin(aol);
            cosRaan = cos(raan);
            sinRaan = sin(raan);
            cosIncl = cos(inclination);   % (N x 1)
            sinIncl = sin(inclination);

            % ECI coordinates: x, y, z each (N, T)
            x = sma .* (cosAol .* cosRaan - sinAol .* cosIncl .* sinRaan);
            y = sma .* (cosAol .* sinRaan + sinAol .* cosIncl .* cosRaan);
            z = sma .* (sinAol .* sinIncl);

            % Result: (N, 3, T)
            this.state.eci = zeros(this.totalSatCount, 3, epochCount);
            this.state.eci(:, 1, :) = reshape(x, [], 1, epochCount);
            this.state.eci(:, 2, :) = reshape(y, [], 1, epochCount);
            this.state.eci(:, 3, :) = reshape(z, [], 1, epochCount);

            % Transform to Earth-fixed frame
            this.eci2ecef(epochs);
        end

        function eci2ecef(this, epochs)
            % Transform ECI coordinates to ECEF (Earth-fixed) frame.
            %
            % Applies Earth rotation matrix R_z(omega_e * t) at each epoch.

            epochCount = length(epochs);
            phi = this.earthAngVel * epochs(:)';  % (1 x T)

            cosPhi = cos(phi);   % (1 x T)
            sinPhi = sin(phi);

            eciX = squeeze(this.state.eci(:, 1, :));   % (N x T)
            eciY = squeeze(this.state.eci(:, 2, :));
            eciZ = squeeze(this.state.eci(:, 3, :));

            this.state.ecef = zeros(size(this.state.eci));
            this.state.ecef(:, 1, :) = reshape( cosPhi .* eciX + sinPhi .* eciY, [], 1, epochCount);
            this.state.ecef(:, 2, :) = reshape(-sinPhi .* eciX + cosPhi .* eciY, [], 1, epochCount);
            this.state.ecef(:, 3, :) = reshape(eciZ, [], 1, epochCount);
        end

    end
end
