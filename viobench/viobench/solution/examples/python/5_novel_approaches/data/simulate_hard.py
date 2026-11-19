"""Harder benchmark-format episodes: faster flight and turns, more IMU noise, wind.

Runs the normal `viobench simulate` path with a modified copy of viobench's world
template (viobench itself is untouched). Levels change the autopilot (cruise speed,
acceleration, jerk, yaw rate), the IMU error model (white noise and bias turn-on,
scaled) and a constant random force per episode (wind, Gaussian std in N).

Held-out scenes (192-199) and dev scenes (60-79) as in simulate.py; never public.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from generate import environment

LEVELS = {
    'default': dict(cruise=(0.45, 0.8), accel=0.7, jerk=2.0, yaw=0.8, imu=1.0, bias=1.0, wind=0.0),
    'fast':  dict(cruise=(0.8, 1.5), accel=1.5, jerk=4.0, yaw=1.2, imu=1.0, bias=1.0, wind=0.0),
    'noisy': dict(cruise=(0.45, 0.8), accel=0.7, jerk=2.0, yaw=0.8, imu=2.0, bias=2.0, wind=1.0),
    'hard':  dict(cruise=(1.0, 2.0), accel=2.0, jerk=5.0, yaw=1.5, imu=2.0, bias=2.0, wind=1.0),
    'faster': dict(cruise=(1.5, 3.0), accel=3.0, jerk=8.0, yaw=1.5, imu=1.0, bias=1.0, wind=0.0),
    'windy': dict(cruise=(0.45, 0.8), accel=0.7, jerk=2.0, yaw=0.8, imu=1.0, bias=1.0, wind=3.0),
    'extreme': dict(cruise=(1.5, 3.0), accel=3.0, jerk=8.0, yaw=1.5, imu=3.0, bias=3.0, wind=2.0),
}
SCENES = {'dev': range(60, 64), 'heldout': range(192, 196), 'hssd-dev': range(60, 64)}


def scene_names(split):
    if split.startswith('hssd'):   # larger real-world houses (HSSD train catalog, by position)
        from viobench.tools.datasets.generate.simulation import hssd_scenes
        names = list(hssd_scenes('train'))
        return [names[i] for i in SCENES[split]]
    return [f'ProcTHOR-Train-{i}' for i in SCENES[split]]

DYNAMICS = """
    struct DYN: hyperdrone::presets::x500_fpv_imu::DynamicsStaticParameters<T, TI> {
        using BASE = hyperdrone::presets::x500_fpv_imu::DynamicsStaticParameters<T, TI>;
        using PT = typename BASE::PARAMETERS_TYPE;
        static constexpr typename PT::IMU imu = {
                {{ {0, (T)(0.28 * %(imu)s)}, {(T)(0.2 * %(bias)s), (T)0, (T)0.001} }},
                {{ {0, (T)(0.07 * %(imu)s)}, {(T)(0.02 * %(bias)s), (T)0, (T)0.0006} }}
        };
        static constexpr typename PT::Disturbances disturbances = {{0, (T)%(wind)s}, {0, 0}};
        static constexpr PT PARAMETER_VALUES = {{{BASE::dynamics, BASE::integration, BASE::mdp}, imu}, disturbances};
    };
    struct SPEC: hyperdrone::Specification<T, TI, DYN> {   // what X500FPVIMU derives from
        static constexpr TI MAX_ENTITY_SLOTS_PER_INSTANCE = 8;"""
TASK = """        static constexpr TI INITIALIZATION_HOLD_STEPS = @HOLD_STEPS@;  // hover before the route starts
        static constexpr T CRUISE_SPEED_MIN = %(cmin)s;
        static constexpr T CRUISE_SPEED_MAX = %(cmax)s;
        static constexpr T MAX_ACCELERATION = %(accel)s;
        static constexpr T MAX_JERK = %(jerk)s;
        static constexpr T MAX_YAW_RATE = %(yaw)s;"""


def render_template(level, out):
    from viobench.tools.datasets.generate import simulation
    p = LEVELS[level]
    text = simulation.TEMPLATE.read_text()
    old_spec = '    struct SPEC: hyperdrone::presets::X500FPVIMU<T, TI> {'
    old_task = '        static constexpr TI INITIALIZATION_HOLD_STEPS = @HOLD_STEPS@;  // hover before the route starts'
    assert text.count(old_spec) == 1 and text.count(old_task) == 1, 'viobench template changed'
    text = text.replace(old_spec, DYNAMICS % dict(imu=p['imu'], bias=p['bias'], wind=p['wind']))
    text = text.replace(old_task, TASK % dict(cmin=p['cruise'][0], cmax=p['cruise'][1], accel=p['accel'],
                                              jerk=p['jerk'], yaw=p['yaw']))
    out.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--level', choices=sorted(LEVELS), default='hard')
    parser.add_argument('--split', choices=sorted(SCENES), default='dev')
    parser.add_argument('--episodes', type=int, default=4, help='per scene (4 render in parallel)')
    parser.add_argument('--seed', type=int, default=1313)
    parser.add_argument('--root', type=Path, default=Path('/var/tmp')/f'novel-approaches-{os.getuid()}')
    args = parser.parse_args()
    out = args.root/'episodes'/f'{args.level}-{args.split}'
    out.mkdir(parents=True, exist_ok=True)
    template = out/'hyperdrone_world.h.in'
    render_template(args.level, template)
    scenes = scene_names(args.split)
    (out/'generation.json').write_text(json.dumps(dict(level=args.level, settings=LEVELS[args.level], scenes=scenes,
                                                       episodes_per_scene=args.episodes, seed=args.seed), indent=2))
    env = environment(args.root)
    env['VIOBENCH_SPOOL'] = str(args.root/'scratch'/'spool')
    code = ('import sys; from pathlib import Path; from viobench.tools.datasets.generate import simulation; '
            f'simulation.TEMPLATE = Path({str(template)!r}); from viobench.tools.cli import main; '
            'sys.exit(main(sys.argv[1:]))')
    subprocess.run([sys.executable, '-c', code, 'simulate', '-o', str(out), '--no-video', '--scene-names', *scenes,
                    '--episodes', str(args.episodes), '--seed', str(args.seed)], env=env, check=True)


if __name__ == '__main__':
    main()
