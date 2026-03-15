"""Example: load and propagate a satellite constellation."""

import time
import numpy as np
from constellation import Constellation

tic = time.perf_counter()

# Load Starlink constellation from config
constellation = Constellation('Starlink')

# Compute initial orbital elements
constellation.getInitialState()

# Time points for propagation [seconds]
epochs = np.arange(1002)

# Propagate all satellites (computes ECI and ECEF positions)
constellation.propagateJ2(epochs)

# Print position of a random satellite
satIdx = np.random.randint(0, constellation.totalSatCount)
epochIdx = np.random.randint(0, len(epochs))

print(f"Всего КА: {constellation.totalSatCount}")
print(f"Положение КА-{satIdx} на эпоху {epochs[epochIdx]}:")
print(f"  ECI  (инерциальная СК): {constellation.stateEci[satIdx, :, epochIdx]}")
print(f"  ECEF (вращающаяся СК):  {constellation.stateEcef[satIdx, :, epochIdx]}")

elapsed = time.perf_counter() - tic
print(f"Elapsed time: {elapsed:.4f} s")
