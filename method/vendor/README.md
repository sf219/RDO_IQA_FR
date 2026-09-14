Vendored from https://github.com/balle-lab/wasserstein-distortion (Apache-2.0),
the reference implementation of

  Y. Qiu, A. B. Wagner, J. Balle, L. Theis, "Wasserstein Distortion: Unifying
  Fidelity and Realism", 2024.

Only change: `from typing import override` is guarded, because the package declares
requires-python >= 3.12 solely on that import while the rest is 3.9-compatible.
