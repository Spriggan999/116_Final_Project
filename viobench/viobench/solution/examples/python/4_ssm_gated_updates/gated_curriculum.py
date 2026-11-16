"""Power-of-two IMU horizons, expressed as intervals after the initial state."""
DEFAULT_HORIZON_STEPS = [1 << level for level in range(14)] + [12_000]
DEFAULT_GATED_WINDOWS = [steps + 1 for steps in DEFAULT_HORIZON_STEPS]
HORIZON_POLICY = 'imu_intervals_powers_of_two_to_12000_v1'
