# examples/diffusion_evolution

Overview
- Small example suite that demonstrates combined diffusion-based conditional samplers and evolutionary strategies.
- Includes the main experiment driver `parameter_search_hades.py` which implements experiments (e.g., `hades`, `charles`, `hades_GA_refined`, `hades_score_refined`).

Contents
- `nd_parameter_search.py` — experiment entry points, simple multimodal fitness (`foo`), plotting helpers, and several experiment configurations.
- A visualization of the score-function evolution for the Gaussian double-peak problem is given by [visualization_score_function.ipynb](visualization_score_function.ipynb)
- (Optional) diffusion-only example files, such as [parameter_diffusion.ipynb](parameter_diffusion.ipynb)) that show how a diffusion model learns target parameter sets.
- **Note**: for **learning non-uniform / standardized parameters** (with large mean or non-uniform STDs), preprocessing or **scaling might be required**! The [parameter_diffusion_scaling.ipynb](parameter_diffusion_scaling.ipynb) notebook shows an example; **this is WIP**.

Quick start (after `condevo` is installed): Run an example:
   - From current directory:
     - `python -m nd_parameter_search hades --generations 100`
     - `python -m nd_parameter_search charles --generations 50 --popsize 256`

Common flags (examples)
- `--generations` — number of generations to run (default per function).
- `--popsize` — population size.
- `--diffuser` — choose diffusion backend, e.g. `DDIM` or `RectFlow`.
- `--tensorboard` — enable logging to tensorboard (script-specific).
- See function docstrings in `nd_parameter_search.py` for full options.

Outputs
- Logs and TensorBoard directories: `data/logs/...`
- Trained model checkpoints: `data/models/...`
- Plots shown interactively (2D/3D animations) by the example scripts.

Notes
- Examples are intended for low-dimensional parameter spaces (2D/3D) for visualization.
- Tweak `popsize`, `sigma_init`, `elite_ratio`, and diffusion hyperparameters to explore behavior.
- For long runs or GPU use, ensure `torch` is installed with CUDA support and that models are placed on the correct device.
- For exploration / exploitation traidoff, you can tweak the initial distribution `sigma_init`, the `matthew_factor`, the `autoscaling` and `sample_uniform` flags (experimental), etc.
