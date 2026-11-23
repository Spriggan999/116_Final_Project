// learned noise weights (LEARN_P): CMA-ES on host replay of 108 training flights (nominal + IMU noise/bias x2.5), best on the choice flights
// order: (constant, speed slope) for pixel noise, depth-prior sigma, depth-prior centre (+ln), accel noise, gyro noise, velocity prior, zero-velocity sigma, outlier threshold, bias floor
static const double LP_W[18] = {-0.503849, -0.0793521, -0.323692, -0.313831, -0.127803, -0.0944896, -0.32533, 0.352669, -0.0536706, -0.227309, -0.0643941, -0.721739, 0.478917, 0.0266912, 0.183691, 0.30249, -0.5291, 0.082593};
