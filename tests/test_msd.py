# Trajectories of a made up brownian motion, compute using a random walk
# Generate using a cumulative sum of random steps
# Now do the MSD for trajectories that are 0, 1, 2,....
from mean_square_displacement import mean_square_displacement
import numpy as np
import pytest
import ctypes


def is_cuda_available():
    try:
        libcudart = ctypes.CDLL("libcudart.so")
        device_count = ctypes.c_int()
        result = libcudart.cudaGetDeviceCount(ctypes.byref(device_count))
        return result == 0 and device_count.value > 0
    except:
        # libcudart.so could not be loaded or cudaGetDeviceCount failed
        return False


def test_cpu_gpu_selfconsistency():
    if not is_cuda_available():
        pytest.skip("GPU not available, skipping GPU tests")
    nparticles = 100
    ndim = 3
    ntimes = 100
    positions = np.random.randn(nparticles, ndim, ntimes).astype(np.float64)
    msd_cpu = mean_square_displacement(positions, "cpu")
    msd_gpu = mean_square_displacement(positions, "gpu")
    assert (
        msd_cpu.shape == msd_gpu.shape
    ), f"MSD shapes do not match: CPU {msd_cpu.shape}, GPU {msd_gpu.shape}"
    assert np.allclose(
        msd_cpu,
        msd_gpu,
        atol=1e-6,
        rtol=1e-6,
    ), "Mean square displacement on CPU and GPU should be identical, got different values"


@pytest.mark.parametrize("number_particles", [2, 10, 100, 1000])
def test_equal_trajectories_particles(number_particles):
    # Copying the same trajectory multiple times to many particles should yield identical MSDs
    ntimes = 100
    ndim = 1
    positions_0 = np.random.randn(ntimes).astype(np.float64)
    # First compute reference for one particle with ntimes
    positions = positions_0[np.newaxis, np.newaxis, :]
    assert positions.shape == (1, 1, ntimes)
    reference_msd = mean_square_displacement(positions, "cpu")
    assert reference_msd.shape == (ntimes, 1)
    # Now copy this trajectory to many particles
    positions = np.tile(positions, (number_particles, 1, 1))
    assert positions.shape == (number_particles, ndim, ntimes)
    msd = mean_square_displacement(positions, "cpu")
    assert msd.shape == (ntimes, 1)
    assert np.allclose(
        msd,
        reference_msd,
        atol=1e-6,
        rtol=1e-6,
    ), "Mean square displacement should be identical for all particles, got different values"


@pytest.mark.parametrize("number_dimensions", [2, 3])
def test_equal_trajectories_dimensions(number_dimensions):
    # Copying the same trajectory multiple times to many dimensions
    # should yield identical MSDs
    nparticles = 100
    ntimes = 100
    positions_0 = np.random.randn(nparticles, ntimes).astype(np.float64)
    positions = positions_0[:, np.newaxis, :]
    positions = np.tile(positions, (1, number_dimensions, 1))
    assert positions.shape == (nparticles, number_dimensions, ntimes)
    msd = mean_square_displacement(positions, "cpu")
    assert msd.shape == (
        ntimes,
        number_dimensions,
    ), f"Mean square displacement should have shape (ntimes, {number_dimensions}), got {msd.shape}"
    for i in range(number_dimensions):
        assert np.allclose(
            msd[:, i],
            msd[:, 0],
            atol=1e-6,
            rtol=1e-6,
        ), f"MSD for dimension {i} should be identical to the first dimension, got {msd[:, i]} vs {msd[:, 0]}"


def fit_line(times, values, discard_last_percentage=0.1):
    slope, _ = np.polyfit(
        times[: int(len(times) * (1 - discard_last_percentage))],
        values[: int(len(values) * (1 - discard_last_percentage))],
        1,
    )
    return slope


def test_random_walk():
    nparticles = 100000
    ndim = 3
    ntimes = 100
    positions = np.random.randn(nparticles, ndim, ntimes).astype(np.float64)
    positions[:, :, 0] = 0  # Start all particles at the origin
    positions = np.cumsum(positions, axis=2)  # Cumulative sum to simulate random walk
    msd = mean_square_displacement(positions, "cpu")
    assert msd.shape == (
        ntimes,
        ndim,
    ), f"Mean square displacement should have shape (ntimes, 3), got {msd.shape}"

    time = np.arange(ntimes)
    slopes = [
        fit_line(time, msd[:, i], discard_last_percentage=0.3) for i in range(ndim)
    ]

    assert np.allclose(
        slopes, np.ones(ndim), atol=0.01, rtol=0
    ), f"The slopes of the MSD should be close to 1 for Brownian motion. Got slopes: {slopes}"


def test_random_walk_different_slopes():
    # Test with different slopes for each dimension
    nparticles = 100000
    ndim = 2
    ntimes = 1000
    expected_slopes = [1, 2]  # Different slopes for each dimension
    positions = np.random.randn(nparticles, ndim, ntimes).astype(np.float64)
    positions[:, :, 0] = 0  # Start all particles at the origin
    positions = np.cumsum(positions, axis=2)  # Cumulative sum to simulate random walk
    for i in range(ndim):
        positions[:, i, :] *= np.sqrt(expected_slopes[i])
    msd = mean_square_displacement(positions, "cpu")
    assert msd.shape == (ntimes, ndim)

    time = np.arange(ntimes)
    slopes = [
        fit_line(time, msd[:, i], discard_last_percentage=0.3) for i in range(ndim)
    ]

    assert np.allclose(
        slopes, expected_slopes, atol=0.01, rtol=0
    ), f"The slopes of the MSD should be close to 1 for Brownian motion. Got slopes: {slopes}"
