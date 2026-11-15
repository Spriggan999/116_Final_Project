"""Convert a real episode zip into the harness format: (cfg, [(t, frame320, imu, truth13), ...]) pickle."""
import sys, pickle, numpy as np
from viobench.tools.datasets import sources
ep = next(iter(sources.episodes([sys.argv[1]])))
cfg, frames = ep.adapt((320, 240))
gt = ep.ground_truth_state()
data = []
for i, frame in enumerate(frames):
    data.append((ep.frame_times[i], np.array(frame, dtype=np.uint8), np.array(ep.imu_window(i)), np.array(gt[i])))
pickle.dump((cfg, data), open(sys.argv[2], "wb"))
print(cfg, len(data), data[0][2].shape)
