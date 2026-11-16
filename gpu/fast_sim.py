"""viobench simulate, tuned for a laptop GPU. Same flights and same lossless frames as the stock command, but:
  - more flights rendered at once (stock: 4, fixed in the code) -> more work for the GPU per step
  - faster PNG level (lossless either way; only the file size changes, ~9 % bigger)
Usage: python fast_sim.py <normal viobench simulate arguments>   (env VIO_PAR = flights at once, default 8)"""
import os, sys
import viobench.tools.datasets.generate.simulation as sim
import viobench.tools.datasets.storage as storage
from viobench.tools.cli import main

par = int(os.environ.get("VIO_PAR", "8"))
sim.PARALLEL_FLIGHTS = par          # read when the run is configured: instances = min(this, --episodes)
sim.PNG_COMPRESSION = 1             # spool frames (lossless)
storage.PNG_COMPRESSION = 1         # episode frames (lossless)
print(f"fast_sim: {par} flights at once, PNG level 1", flush=True)
sys.exit(main(["simulate"] + sys.argv[1:]))
