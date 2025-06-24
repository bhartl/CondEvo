# Conditional Diffusion Evolution

Introduced in [Heuristically Adaptive Diffusion-Model Evolutionary Strategy](https://arxiv.org/abs/2411.13420).

The package comprises the following methods:
- [`condevo.es.HADES`](condevo/es/heuristical_diffusion_es.py): Heuristically Adaptive Diffusion-Model Evolutionary Strategy
- [`condevo.es.CHARLES`](condevo/es/conditional_diffusion_es.py): Conditional, Heuristically-Adaptive ReguLarized Evolutionary Strategy through Diffusion

## Installation

From within your desired python environment, you may install the repository [optionally in developer mode, `-e`] via
```bash
pip install -e .
```

### Example Dependencies
To install dependencies for the examples, you might first install the code depends on the following packages:
- `foobench` (see [install instructions](https://github.com/bhartl/foobench) on GitHub)
- `mincraft` (see [install instructions](https://github.com/bhartl/NeurEvo) on GitHub)

You may install the examples dependencies via
```bash
pip install -e .[examples]
```


### Mainly tested via CPU-only Usage:
Since the code is intended for evolutionary optimization, the parallelization is done across environment evaluations. Thus, we mainly applied the code without GPU usage, and you may install the cpu-only version of PyTorch  via
```bash
pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu
```

## Examples
see the [examples](examples) directory for example usage of the methods.

## Citation
If you use this code in your research, please cite the following paper:
```bibtex
@misc{hartl2024heuristicallyadaptivediffusionmodelevolutionary,
      title={Heuristically Adaptive Diffusion-Model Evolutionary Strategy}, 
      author={Benedikt Hartl and Yanbo Zhang and Hananel Hazan and Michael Levin},
      year={2024},
      eprint={2411.13420},
      archivePrefix={arXiv},
      primaryClass={cs.NE},
      url={https://arxiv.org/abs/2411.13420}, 
      note = {Code available at \url{https://github.com/bhartl/condevo}},
}
```

