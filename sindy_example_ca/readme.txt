Sparse symbolic regression of a trained NCA update at fire_rate = 1.0
======================================================================

Layout
------
  sindy_lib.jl                    general STLS / fit_sindy (raw + normalized ξ)
  sindy_plots.jl                  coefficient + equation figure
  sindy_nca.jl                    NCA sampling, RD library, rollout eval
  SINDy.jl                        NCA application (IMAGE_SET / state-dep. diffusion)
  sindy_application_simple_fhn.jl FHN poly-CA recovery demo
  readme.txt                      this file

What SINDy.jl do
-----------------------------
The learned cell update (mask off / fire_rate=1) is a pointwise map

    y  = perceive(x; kernels)   # id/sobel/laplacian × channel
    dx = Dense2(ReLU(Dense1(y)))

We approximate dx ≈ sparse polynomial in the kernel features (degree ≤ 3).

Samples come from the NCA's deterministic (fire_rate=1) trajectory.
u_traj[:,:,1] is the visible IC for both models; hidden channels start at 0.

Usage
-----
From the NeuralCellularAutomata repo root; keep the sindy_*.jl files next
to SINDy.jl:

    julia --project=. example_scripts/SINDy.jl
    julia --project=. example_scripts/SINDy.jl 3 20000 0.01

Edit RUN_DIR / TRAJ_SOURCE (or IMAGE_SET) in the chosen script.

Optional CLI: max_degree n_samples threshold fit_steps
  fit_steps=0 (default) uses the full observation horizon.

FHN smoke test (runs in this CellularAutomata repo):

    julia --project=. sindy_application_simple_fhn.jl

Outputs (NCA app)
-----------------
Under <run_dir>/outputs/sparse_symbolic_<timestamp>/ :

  summary.txt, coefficients.csv, sparse_symbolic.png,
  sparse_model.jld2, read_fr1.0.txt, RE plots, SINDy|NCA video

Also copies key files to <run_dir>/outputs/sparse_symbolic/ (latest).
