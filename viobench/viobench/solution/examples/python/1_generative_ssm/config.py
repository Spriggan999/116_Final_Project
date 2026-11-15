"""Host training defaults from the HyperDrone notebook. Edit or override with train.py --config."""
from pathlib import Path
import torch

SEED = 17

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

WORKDIR = Path(__file__).resolve().parent / 'data'

H, W = 480, 640

WANDB_PROJECT = 'hyperdrone-vio-fuel-aware'

WANDB_MODE = 'online'

SCENE_OFFSET = 0

SCENE_CATALOG = 'mixed'  # ProcTHOR Train200 + HSSD Train125; see scenes.py for ordering.

NUM_SCENES = 313  # 192 ProcTHOR + 121 HSSD; final 8 + 4 scenes are held out.

COLLECT_STEPS = 2000  # Samples at 200 Hz; may exceed the native informational limit.

SOURCE_WIDTH = 640

SOURCE_HEIGHT = 480

USE_GENERATIVE_FRAME_MODEL = True  # generator choice for VISUAL_PREDICTION='rgb'

# Predict 48D visual features between captures; use 'rgb' for legacy image generation.
VISUAL_PREDICTION = 'embedding'
EMBEDDING_DIM = 48  # 96 is an opt-in capacity experiment requiring fresh SL
EMBEDDING_HIDDEN = 256
EMBEDDING_EMA = 0.99
EMBEDDING_STD_TARGET = 0.1
EMBEDDING_VARIANCE_WEIGHT = 1.0
EMBEDDING_COVARIANCE_WEIGHT = 0.01
EMBEDDING_PREDICTION_CHUNK = 1024

USE_FRAME_EDGE_LOSS = False

FRAME_EDGE_LOSS_WEIGHT = 0.05

USE_FLOW_SMOOTHNESS_LOSS = False

FRAME_FLOW_SMOOTHNESS_WEIGHT = 1e-4

# Largest profiled teacher preset with headroom below the 20 G step fuel cap.
# Resource reports live in verification/; verify the actual trained export too.
FRAME_GENERATOR_BASE = 33

# Learned non-overlapping patches consume every 640x480 pixel before the
# expensive spatial blocks. Legacy checkpoints use strides 1 (flow) / 2 (CNN).
FRAME_INPUT_STRIDE = 8

VISUAL_INPUT_STRIDE = 8

FRAME_GENERATION_CHUNK = 16

FRAME_PAIR_BATCH = 8
FRAME_VAL_PAIR_BATCH = 8  # keep joint-embedding validation batch statistics fixed
FRAME_PAIRS_UINT8 = True  # transfer sampled RGB before normalization
TORCH_CPU_THREADS = 2  # CUDA host work; avoid full-image thread-pool oversubscription

# Checkpoint the visual CNN in small chunks during sequence backpropagation.
VISUAL_CHUNK = 16

FRAME_MAX_FLOW_PIXELS = 24.0 * W / 96.0

CACHE_GENERATED_WINDOWS = False

FRAME_CACHE_BATCH = 1

MAX_FRAME_TRAIN_PAIRS = 20000

MAX_FRAME_VAL_PAIRS = 3000

USE_GENERATED_FRAMES_FOR_STATE = True

USE_LARGER_STATE_STUDENT = True

LARGE_STATE_HIDDEN = 1792

TEACHER_HIDDEN = 256

FRAME_MODEL_NAME = 'embedding_predictor' if VISUAL_PREDICTION == 'embedding' else ('flow_generative' if USE_GENERATIVE_FRAME_MODEL else 'affine')

FRAME_CHECKPOINT = f'frame_student_{FRAME_MODEL_NAME}.pt'

WINDOW = 101

BATCH = 4

EPOCHS_FRAME = 200

EPOCHS_STUDENT = 300

USE_HYBRID_SSM_TRANSFORMER = True

HYBRID_STUDENT_HIDDEN = 896

STUDENT_HIDDEN = (HYBRID_STUDENT_HIDDEN if USE_HYBRID_SSM_TRANSFORMER else
                  LARGE_STATE_HIDDEN if USE_LARGER_STATE_STUDENT else 32)

HYBRID_NUM_MAMBA_LAYERS = 3

HYBRID_ATTENTION_WINDOW = 16
HYBRID_ATTENTION_CHUNK = 512  # bounded training attention; zero restores dense masking
PARALLEL_TRAINING_PHYSICS = True  # differentiable FP32 scan; validation/deployment stay sequential

HYBRID_D_STATE = 16

HYBRID_HEAD_DIM = 32

BACKBONE_NAME = 'mamba2_hybrid' if USE_HYBRID_SSM_TRANSFORMER else 'gru'

USE_TEACHER = False

TEACHER_BACKBONE = 'mamba_hybrid' if USE_HYBRID_SSM_TRANSFORMER else 'gru'

EPOCHS_TEACHER = 60

EPOCHS_DISTILL = 25

LR_FRAME = 3e-4

LR_STUDENT = 1e-4

LR_TEACHER = 3e-4

LR_DISTILL = LR_STUDENT / 3

PATIENCE_FRAME = 8

PATIENCE_STUDENT = 6

PATIENCE_TEACHER = 8

PATIENCE_DISTILL = 6

LR_SCHEDULER_FACTOR = 0.5

LR_SCHEDULER_PATIENCE = 3

USE_LEARNED_VISUAL_ENCODER = True

STATE_VISUAL_FEATURE_DROPOUT = 0.05

STATE_CHECKPOINT = (f'state_student_{BACKBONE_NAME}_{STUDENT_HIDDEN}_'
                    f'vis{int(USE_LEARNED_VISUAL_ENCODER)}_{FRAME_MODEL_NAME}_'
                    f'generated{int(USE_GENERATED_FRAMES_FOR_STATE)}.pt')

TEACHER_CHECKPOINT = f'teacher_{TEACHER_BACKBONE}_{TEACHER_HIDDEN}.pt'

# Rolling collection: current batch + one prefetched batch + fixed validation.
STREAM_DATA = False
STREAM_SCENES = 4
STREAM_EPISODES_PER_SCENE = 2
STREAM_VALIDATION_SCENES = 12
STREAM_VALIDATION_EPISODES = 1
STREAM_MAX_MIB = 3072
STREAM_COLLECT_TIMEOUT = 1800
STREAM_MAX_PASSES_PER_BATCH = 3  # first pass plus at most two extra passes while collecting
COLLECT_INSTANCES = 2
COLLECT_COMPRESSION_LEVEL = 1  # fast lossless NPZ; 6 restores NumPy's usual compression

# Training only: neural BF16, FP32 geometry/loss/validation and export weights.
USE_AMP = True
AMP_DTYPE = 'bfloat16'
LOADER_WORKERS = 2  # CUDA state windows; CPU runs keep workers=0
FRAME_LOADER_WORKERS = 0  # trivial frame slices were faster without multiprocessing
LOADER_PREFETCH = 1
PIN_MEMORY = True
COMPACT_STATE_CAMERAS = True  # carry only the original selected ~20 Hz observations
PERSISTENT_WORKERS = True  # reuse CPU loaders; rolling data retires old workers before replacement
FUSED_ADAMW = True  # CUDA only; CPU retains ordinary AdamW

# Optional on-policy PPO refinement of the existing 9D correction head.
USE_PPO = True
PPO_ITERATIONS = 10
PPO_ROLLOUT_BATCHES = 4
PPO_UPDATE_EPOCHS = 4
PPO_MINIBATCH = 456
VECTORIZED_PPO_ROLLOUT = True  # sample/replay exogenous estimator windows in batches
PPO_LR = 3e-5
PPO_STD = 0.1  # standard deviation before the model's .25*tanh bound
PPO_GAMMA = 0.995
PPO_GAE_LAMBDA = 0.95
PPO_CLIP = 0.2
PPO_TARGET_KL = 0.02
PPO_ANCHOR_COEF = 0.01

# Optional estimator-window curriculum (IMU samples, including initial state).
# Empty keeps WINDOW. For [401, 1001, 2001, 4001], collect >=4001 samples.
# Validation stays at the longest window throughout each stage.
PPO_CURRICULUM_WINDOWS = []
PPO_CURRICULUM_MIN_ITERATIONS = 3
PPO_CURRICULUM_PATIENCE = 2
PPO_CURRICULUM_REL_IMPROVEMENT = 0.02

# Training-only curriculum for frame SL, state SL and correction-head PPO.
# Disabled preserves ordinary training and the standalone PPO window schedule.
USE_CURRICULUM = False
SL_CURRICULUM_WINDOWS = []
CURRICULUM_LEVELS = [0., .5, 1.]
CURRICULUM_MIX = [.7, .2, .1]  # current, easier, harder
CURRICULUM_ALL_EASIER = False  # staged student training enables all-prior-level replay
CURRICULUM_RANDOM_ERROR_STRENGTH = False  # staged students randomize added error amplitudes
CURRICULUM_MIN_EPOCHS = 3
CURRICULUM_PATIENCE = 2
CURRICULUM_REL_IMPROVEMENT = .02
CURRICULUM_EARLY_STOP_GRACE = 3  # fresh batches after promotions; epochs for fixed data
CURRICULUM_MAX_STAGE_EPOCHS = 6  # allow healthy plateaus to advance; zero requires improvement
CURRICULUM_PLATEAU_TOLERANCE = .1
CURRICULUM_REQUIRE_COMPLETION = True  # early stopping waits for the full SL curriculum
CURRICULUM_FINAL_STAGE_CYCLES = 3  # fresh batches at full difficulty before patience starts
CURRICULUM_MIN_FRAME_EPOCHS = 0
CURRICULUM_MIN_STATE_EPOCHS = 0
CURRICULUM_BATCH_TOKENS = 0  # zero disables sequence-length-aware batch limits
CURRICULUM_DEVICE_PHOTOMETRIC = True
CURRICULUM_PROBE_BATCHES = 2  # fixed held-out batches per difficulty axis
CURRICULUM_PRIORITIZE = True
# Extra IMU error above the native simulator noise floor; units per sample,
# bias units m/s^2 and rad/s, random-walk coefficients per sqrt(second).
CURRICULUM_ACCEL_NOISE = .14
CURRICULUM_GYRO_NOISE = .035
CURRICULUM_ACCEL_BIAS = .1
CURRICULUM_GYRO_BIAS = .01
CURRICULUM_ACCEL_WALK = .0005
CURRICULUM_GYRO_WALK = .0003
CURRICULUM_BRIGHTNESS = .4
CURRICULUM_CONTRAST = .4
CURRICULUM_BLUR = .5
