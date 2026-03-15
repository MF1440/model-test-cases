"""Satellite constellation model.

Loads Walker constellation parameters from a JSON config file,
computes initial orbital elements, and propagates positions using
a J2-perturbed two-body model.

Usage:
    from constellation import Constellation
    c = Constellation('Starlink')
    c.getInitialState()
    c.propagateJ2(np.arange(1002))
"""

import os
import json
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


@dataclass(frozen=True)
class EarthConstants:
    """Physical constants for Earth."""
    radius: float = 6378135.0         # equatorial radius [m]
    gm: float = 3.986004415e14        # gravitational parameter [m^3/s^2]
    j2: float = 1.082626e-3           # second zonal harmonic of geopotential
    angularVelocity: float = 2.0 * np.pi / 86164.0  # Earth rotation rate [rad/s]


EARTH = EarthConstants()


class Walker(NamedTuple):
    """Walker constellation group parameters."""
    inclination: float      # orbital plane inclination [degrees]
    satsPerPlane: int       # satellites per orbital plane
    planeCount: int         # number of orbital planes
    phaseShift: int         # argument-of-latitude shift between adjacent planes
    altitude: float         # orbit altitude [km]
    maxRaan: float          # max RAAN (Right Ascension of Ascending Node) for plane distribution [deg]
    startRaan: float        # RAAN of the first plane [deg]


class WalkerGroup(Walker):
    """Walker group with computed properties."""

    def getTotalSatCount(self):
        """Total satellite count in this group."""
        return self.satsPerPlane * self.planeCount

    def getInitialElements(self):
        """Compute initial orbital elements for one Walker group.

        Returns:
            elements: array (satCount, 6) with columns
                [sma, 0, 0, raan, inclination, argumentOfLatitude]
                sma in meters, angles in radians.
        """
        startRaan   = np.deg2rad(self.startRaan)
        maxRaan     = np.deg2rad(self.maxRaan)
        inclination = np.deg2rad(self.inclination)
        altitude    = self.altitude * 1000.0  # km -> m
        satCount    = self.getTotalSatCount()

        raans = np.linspace(startRaan, startRaan + maxRaan, self.planeCount + 1)
        raans = raans[:-1] % (2.0 * np.pi)

        # Vectorized: meshgrid of (planeIdx, satIdx) -> flat arrays
        planeIdx = np.arange(self.planeCount)
        satIdx = np.arange(self.satsPerPlane)
        planeGrid, satGrid = np.meshgrid(planeIdx, satIdx, indexing='ij')
        planeGrid = planeGrid.ravel()
        satGrid = satGrid.ravel()

        sma = EARTH.radius + altitude
        elements = np.zeros((satCount, 6))
        elements[:, 0] = sma
        elements[:, 3] = raans[planeGrid]
        elements[:, 4] = inclination
        elements[:, 5] = 2.0 * np.pi * (satGrid / self.satsPerPlane
                                         + self.phaseShift * planeGrid / satCount)

        return elements


class Constellation:
    """Satellite constellation model.

    Loads Walker group parameters from a JSON config, computes initial
    orbital elements, and propagates positions with J2 perturbation.

    Attributes:
        totalSatCount: total number of satellites
        groups: list of WalkerGroup
        elements: array (totalSatCount, 6) of orbital elements
        stateEci: array (totalSatCount, 3, epochCount) ECI (Earth-Centered Inertial) positions [m]
        stateEcef: array (totalSatCount, 3, epochCount) ECEF (Earth-Centered Earth-Fixed) positions [m]
    """

    def __init__(self, nameCode):
        self.totalSatCount = 0
        self.groups   = []
        self.elements = np.empty((0, 6))
        self.stateEci  = np.empty(0)
        self.stateEcef = np.empty(0)
        self._loadFromConfig(nameCode)

    def _loadFromConfig(self, nameCode):
        """Load constellation parameters from JSON config file.

        Args:
            nameCode: constellation name (e.g. 'Starlink'), case-insensitive.
        """
        # Path relative to script
        scriptDir = os.path.dirname(os.path.abspath(__file__))
        configPath = os.path.join(scriptDir, '..', 'constellationsTest.json')

        if not os.path.isfile(configPath):
            raise FileNotFoundError(f"Config file not found: {configPath}")

        with open(configPath, encoding='utf-8') as configFile:
            jsonData = json.load(configFile)

        for entry in jsonData:
            if entry['name'].lower() == nameCode.lower():
                print("Загружена группировка " + nameCode)

                for walkerParams in entry['Walkers']:
                    group = WalkerGroup(*walkerParams)
                    self.groups.append(group)
                    self.totalSatCount += group.getTotalSatCount()
                return

        raise ValueError(f"Группировка '{nameCode}' не найдена в файле {configPath}")

    def getInitialState(self):
        """Compute initial orbital elements for all satellites."""
        self.elements = np.zeros((self.totalSatCount, 6))
        shift = 0

        for group in self.groups:
            ending = shift + group.getTotalSatCount()
            self.elements[shift:ending, :] = group.getInitialElements()
            shift = ending

    def propagateJ2(self, epochs):
        """Propagate satellite positions with J2 perturbation.

        Computes ECI positions at each epoch using two J2-corrected rates:
          - RAAN drift: dOmega/dt = -3/2 * J2 * sqrt(mu) * Re^2 / a^(7/2) * cos(i)
          - Argument of latitude rate (draconic): corrected mean motion

        Args:
            epochs: array or list of time points [seconds from initial epoch].

        Populates stateEci of shape (totalSatCount, 3, len(epochs))
        and stateEcef of the same shape.
        """
        epochs = np.asarray(epochs, dtype=np.float64)

        inclination = self.elements[:, 4]   # (N,)
        sma         = self.elements[:, 0]   # (N,)
        raan0       = self.elements[:, 3]   # (N,)
        aol0        = self.elements[:, 5]   # (N,)

        # RAAN precession rate: dOmega/dt = -3/2 * J2 * sqrt(mu) * Re^2 / a^(7/2) * cos(i)
        raanRate = (-1.5 * EARTH.j2 * np.sqrt(EARTH.gm)
                    * EARTH.radius**2 / sma**(3.5) * np.cos(inclination))

        # Draconic angular velocity with J2 correction
        draconicOmega = (np.sqrt(EARTH.gm / sma**3)
                         * (1.0 - 1.5 * EARTH.j2 * (EARTH.radius / sma)**2)
                         * (1.0 - 4.0 * np.cos(inclination)**2))

        # Vectorized: compute all epochs simultaneously
        # aol: (N, T),  raan: (N, T)
        aol  = aol0[:, None] + epochs[None, :] * draconicOmega[:, None]
        raan = raan0[:, None] + epochs[None, :] * raanRate[:, None]

        cosAol  = np.cos(aol)
        sinAol  = np.sin(aol)
        cosRaan = np.cos(raan)
        sinRaan = np.sin(raan)
        cosIncl = np.cos(inclination)[:, None]  # (N, 1) for broadcasting
        sinIncl = np.sin(inclination)[:, None]

        smaExp = sma[:, None]  # (N, 1)

        # ECI coordinates: x, y, z each (N, T)
        x = smaExp * (cosAol * cosRaan - sinAol * cosIncl * sinRaan)
        y = smaExp * (cosAol * sinRaan + sinAol * cosIncl * cosRaan)
        z = smaExp * (sinAol * sinIncl)

        # Result: (N, 3, T)
        self.stateEci = np.stack([x, y, z], axis=1)

        # Transform to Earth-fixed frame
        self._eci2ecef(epochs)

    def _eci2ecef(self, epochs):
        """Transform ECI coordinates to ECEF (Earth-fixed) frame.

        Applies Earth rotation matrix R_z(omega_e * t) at each epoch.

        Args:
            epochs: array of time points [seconds].
        """
        epochs = np.asarray(epochs, dtype=np.float64)
        phi = EARTH.angularVelocity * epochs  # (T,)

        cosPhi = np.cos(phi)  # (T,)
        sinPhi = np.sin(phi)

        eciX = self.stateEci[:, 0, :]  # (N, T)
        eciY = self.stateEci[:, 1, :]
        eciZ = self.stateEci[:, 2, :]

        self.stateEcef = np.empty_like(self.stateEci)
        self.stateEcef[:, 0, :] =  cosPhi * eciX + sinPhi * eciY
        self.stateEcef[:, 1, :] = -sinPhi * eciX + cosPhi * eciY
        self.stateEcef[:, 2, :] = eciZ
