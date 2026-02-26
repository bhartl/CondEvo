# Conditional Diffusion Evolution
This folder collects experiments for the *Conditional, Heuristically-Adaptive ReguLarized Evolutionary Strategy through Diffusion* (CHARLES-D) method, which is a conditional diffusion-model evolutionary strategy [^1].

## Content
The [benchmark](benchmark.py) and [run_novelty](run_novelty.py) scripts implements wrappers for testing the CHARLES-D method for simple (2D) optimization tasks via the [foobench](https://github.com/bhartl/foobench) package; the [configs](configs.py), [utils](utils.py) and [analysis](analysis.py) scripts implement helper functionalities. 

- The notebook [dynamic-environment-examples](dynamic-environment-examples.ipynb) demonstrates adaptability of the HADES method in dynamic environments.
- The notebook [conditional_evolution_examples](conditional_evolution_examples.ipynb) implements guidance in parameter space, on the minimal example of a Gaussian double-peak function in 2D.
- The notebook [conditional_evolution_examples_oscillatory-condition](conditional_evolution_examples_oscillatory-condition.ipynb) demonstrates that diffusion models can act as evolutionary memory while providing guided adaptability. I.e., time-dependent conditioning can lead to targeted jumps in parameter space, which can be used to alternate between optima in a multi-modal landscape. This is shown on the same Gaussian double-peak function as above, but with an oscillatory target condition.
- The notebooks [novelty_conditional_evolution-rastrigin](novelty_conditional_evolution-rastrigin.ipynb) and [novelty_conditional_evolution-helixrastrigin](novelty_conditional_evolution-helixrastrigin.ipynb) demonstrate that conditional sampling can be used to steer the search dynamics towards novel solutions, by conditioning on a novelty measure of genotypes. This is shown on the Rastrigin function in 2D, and a twisted variant of the Rastrigin function whose optima don't align with principal directions of local optima near the center.
- Based on the above notebooks, we analyze in [novelty_conditional_evolution-rastrigin_model-vs-buffer-sampling](novelty_conditional_evolution-rastrigin_model-vs-buffer-sampling.ipynb) and [novelty_conditional_evolution-helixrastrigin_model-vs-buffer-sampling](novelty_conditional_evolution-helixrastrigin_model-vs-buffer-sampling.ipynb) whether generating offspring with DMs exceeds naive sampling from an established evolutionary history, i.e., from the dataset buffer.

## The CHARLES-D Method

Diffusion models (DMs) provide a model-free approach for learning denoising-based generative strategies tailored to custom datasets across versatile, problem-specific domains. Once trained on statistically relevant data, they can potentially surpass traditional EAs in generating high-quality offspring genotypes. In our complementary contribution, [^2] we formally connect DMs to EAs and particularly demonstrate that the backward process in DMs can be viewed as an iterative evolutionary process across generations.

Here, we introduce a paradigm shift by sustaining and evolving a heuristic population that is sampled across successive generations via a heuristically refined DM. 
 This model is constantly refined - i.e., trained “online” - on a successively acquired dataset buffer containing elite solutions of previous generations that have been sampled by prior versions of the DM. In training the DM, we notably weight high-fitness data more heavily compared to low-fitness genotypes using a fitness weighting function. This approach increases the probability of sampling high-quality data while still maintaining diversity in the generative process.

Intriguingly, with DMs, we can apply techniques such as **classifier-free guidance to condition the generation process**. This allows us to implement an evolutionary optimizer whose search dynamics in parameter-space, fitness-space, or phenotypic-space can be controlled. 
By training the DM with additional information which numerically quantifies certain qualities or traits of genotypes $g_i\equiv x$ in their respective environments, the DM learns to associate elements in the parameter (or any other) space with corresponding features $c_i = c(g_i)$.

![conditional_evolution_examples](assets/charles.png)

Technically, this is achieved by extending the input of the DM’s ANN as $ϵ_θ(x_t, t) → ϵ_θ(x_t, t, c(x_t))$. The function $c(·)$ is a custom, not necessarily differentiable, vector-valued classifier function or a measurement of a trait of the data point $x_0$, or genotype $g$, evaluated in the parameter space, fitness space, or even phenotype space.

During sampling, the DM’s generative process can be biased towards novel high-quality data points that exhibit a
particular target trait $c^{(\textrm T)}$, by conditioning the iterative denoising process of the diffusion model as $ϵ_θ(x_t, t, c^{(\textrm T)}) → \hat x_0$ such that $c(\hat x_0) ≈ c^{(\textrm T)}$. This allows the DM to generate high-quality samples with the desired traits, similar to how Stable Diffusion and Sora generate realistic image or video content based on custom text prompts.

In our context, we propose using conditional sampling to gain exceptional control over a heuristic search process with an open, successively refining dataset. While the heuristic nature of the evolutionary process facilitates global optimum exploration, the successively refined DM-based generative process allows for diverse sampling of high-quality genotypic data points that may exhibit target traits defined independently from the fitness score, akin to prompting an image-generative DM with text input. 

We consider this CHARLES-D approach as a “Talk to your Optimizer” application [^1].

[^1]: B. Hartl, Y. Zhang, H. Hazan, M. Levin, Heuristically Adaptive Diffusion-Model Evolutionary Strategy, Advanced Science, (2026) in press, DOI [10.1002/advs.202511537](https://doi.org/10.1002/advs.202511537), [arxiv:2411.13420](https://arxiv.org/abs/2411.13420) (2024)

[^2]: Y. Zhang, B. Hartl, H. Hazan, M. Levin, ICLR (2025), [OpenReview](https://openreview.net/forum?id=xVefsBbG2O), [arxiv:2410.02543](https://arxiv.org/abs/2410.02543)