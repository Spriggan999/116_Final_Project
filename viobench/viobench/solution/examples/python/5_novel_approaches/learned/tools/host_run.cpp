// Host replay of one green-only feed (make_feeds.py) through init/step, like the sandbox but without fuel.
// usage: host_run FEED_DIR [estimates.f64]   prints "rmse <unaligned position RMSE> frames <n>"
// Linked with solution.cpp built for the host (see search.py). Learned weights / logs come from env LEARN_WFILE / LEARN_LFILE.
#include "viobench_abi.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static std::vector<char> read_file(const char* dir, const char* name) {
    char path[4096];
    std::snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(2); }
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<char> data((size_t)n + 1, 0);
    if (std::fread(data.data(), 1, (size_t)n, f) != (size_t)n) { std::fprintf(stderr, "short read %s\n", path); std::exit(2); }
    std::fclose(f);
    data.resize((size_t)n);
    return data;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s FEED_DIR [estimates.f64]\n", argv[0]); return 2; }
    const char* dir = argv[1];
    std::vector<char> shape = read_file(dir, "shape.txt");
    shape.push_back(0);
    long frames, height, width, channels, imu_rows;
    if (std::sscanf(shape.data(), "%ld %ld %ld %ld %ld", &frames, &height, &width, &channels, &imu_rows) != 5) return 2;
    std::vector<char> config = read_file(dir, "config.json");
    std::vector<char> green = read_file(dir, "frames.g8");
    std::vector<char> ft = read_file(dir, "frame_times.f64"), im = read_file(dir, "imu.f64"), gtv = read_file(dir, "gt.f64");
    const double* frame_times = (const double*)ft.data();
    const double* imu = (const double*)im.data();
    const double* gt = (const double*)gtv.data();
    const size_t plane = (size_t)height * (size_t)width;
    if (green.size() != plane * (size_t)frames) { std::fprintf(stderr, "frames.g8 size mismatch\n"); return 2; }
    if (init(config.data(), (uint32_t)config.size()) != VIOBENCH_STATUS_OK) { std::printf("rmse nan frames 0\n"); return 1; }
    std::vector<uint8_t> rgb(plane * 3, 0);
    std::vector<double> est((size_t)frames * VIOBENCH_STATE_VALUES, NAN);
    double state[VIOBENCH_STATE_VALUES], previous = -INFINITY, se = 0;
    long ws = 0, k = 0;
    for (; k < frames; k++) {
        const double t = frame_times[k];
        while (ws < imu_rows && imu[ws * VIOBENCH_IMU_ROW_VALUES] <= previous + 1e-6) ws++;
        long we = ws;
        while (we < imu_rows && imu[we * VIOBENCH_IMU_ROW_VALUES] <= t + 1e-6) we++;
        const uint8_t* g = (const uint8_t*)green.data() + (size_t)k * plane;
        for (size_t i = 0; i < plane; i++) rgb[3 * i + 1] = g[i];
        for (int i = 0; i < VIOBENCH_STATE_VALUES; i++) state[i] = NAN;
        if (step(t, rgb.data(), imu + ws * VIOBENCH_IMU_ROW_VALUES, (int32_t)(we - ws), state) != VIOBENCH_STATUS_OK) break;
        bool ok = true;
        for (int i = 0; i < VIOBENCH_STATE_VALUES; i++) ok = ok && std::isfinite(state[i]);
        if (!ok) break;
        std::memcpy(&est[(size_t)k * VIOBENCH_STATE_VALUES], state, sizeof state);
        const double* tr = gt + k * VIOBENCH_POSE_VALUES;
        const double dx = state[0] - tr[0], dy = state[1] - tr[1], dz = state[2] - tr[2];
        se += dx * dx + dy * dy + dz * dz;
        previous = t;
        ws = we;
    }
    if (argc > 2) { FILE* f = std::fopen(argv[2], "wb"); if (f) { std::fwrite(est.data(), sizeof(double), est.size(), f); std::fclose(f); } }
    // a failed episode scores as if the rest were zero-filled: count missing frames as the truth's distance from origin
    for (long j = k; j < frames; j++) { const double* tr = gt + j * VIOBENCH_POSE_VALUES; se += tr[0] * tr[0] + tr[1] * tr[1] + tr[2] * tr[2]; }
    std::printf("rmse %.6f frames %ld\n", std::sqrt(se / (double)frames), k);
    return 0;
}
