# SimulationMusique

GPU-accelerated fluid dynamics simulation of a recorder flute using the Lattice Boltzmann Method. Generates acoustic signals by simulating air flow through a flute geometry defined in a PNG image.

Uses OpenCL for GPU computation, outputs VTK files for ParaView visualization, and records pressure data as audio.

## Setup

Needs Python 3.8+ and an OpenCL GPU. Install with:

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```


## Running

```bash
python3 src/vonkarman_cl.py
```

Outputs VTK files to `images/` and WAV to `sounds/output.wav`. Check the console for progress (prints every 400 steps).

## Tweaking parameters

Edit `src/vonkarman_cl.py`:

- `NU` - viscosity (default 0.01). Higher = more stable but smoother
- `N_iter` - number of iterations (default 10,000,000)  
- `get_velocity()` - inlet velocity. Currently ramps from 0 to 0.005

Geometry is defined in `assets/simu_r2.png`: red pixels are inlet, blue are outlet, black is walls, green is where audio gets sampled.

## When things go wrong

If you get NaN values, the simulation diverged. Try:

- **Lower velocity**: Change the `0.05` in `get_velocity()` to `0.01`
- **Increase viscosity**: `NU = 0.05` instead of `0.01`
- Narrow passages in the geometry can make things unstable

## Looking at results

VTK files go to `images/`. Open them in ParaView.

Audio output goes to `sounds/output.wav`.
