"""
mid_price.py
============
Mid-price dynamics for the Avellaneda-Stoikov (2008) market-making model.

The mid-price follows arithmetic Brownian motion (ABM):

    dS_t = sigma * dW_t                                              [Eq. 1]

Discretised via the Euler-Maruyama scheme:

    S_{t+dt} = S_t + sigma * sqrt(dt) * Z,    Z ~ N(0, 1)

Arithmetic BM is chosen over geometric BM deliberately: it keeps the
exponential utility functionals bounded, which is a prerequisite for the
Hamilton-Jacobi-Bellman derivation in Section 3.1.  The geometric BM
alternative is treated in the paper's appendix.

The money market is assumed to pay zero interest (Section 2.1), so the
mid-price process carries no drift term.

Reference
---------
Avellaneda, M. & Stoikov, S. (2008).
"High-frequency trading in a limit order book."
Quantitative Finance, 8(3), 217-224.
DOI: 10.1080/14697680701381228
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray


class MidPriceModel:
    """
    Arithmetic Brownian motion mid-price process  dS = sigma * dW.

    Provides two complementary interfaces:

    * **Step-by-step** via ``update_price()`` — used by the simulator
      engine's event loop, which must interleave price updates with
      order-arrival sampling and agent state updates.

    * **Batch** via ``generate_path()`` — used by the Monte Carlo runner,
      which draws entire trajectories at once for speed.

    Both interfaces share the same internal random generator so that a
    single seed in ``SimConfig`` fully determines every random draw across
    an entire simulation run.

    Parameters
    ----------
    s0 : float
        Initial mid-price S_0.  Must be > 0.  Paper uses s0 = 100.
    sigma : float
        Constant volatility coefficient of the ABM.  Must be > 0.
        Paper uses sigma = 2.
    dt : float
        Simulation time step.  Must be > 0.  Paper uses dt = 0.005,
        giving 200 steps over the T = 1 horizon.
    seed : int or None, optional
        Seed for ``numpy.random.default_rng``.  Pass an integer for
        fully reproducible paths; leave ``None`` for fresh randomness.

    Attributes
    ----------
    s : float
        Current mid-price S_t.  Mutated by ``update_price``; reset to
        ``s0`` by ``reset`` or by ``generate_path(reset=True)``.
    t : float
        Current simulation time.  Incremented by ``dt`` on each call
        to ``update_price``; reset to 0.0 by ``reset``.

    Examples
    --------
    >>> model = MidPriceModel(s0=100.0, sigma=2.0, dt=0.005, seed=0)
    >>> model.update_price()           # one Euler-Maruyama step
    ...
    >>> times, prices = model.generate_path(T=1.0)
    >>> prices.shape
    (201,)
    """

    def __init__(
        self,
        s0: float = 100.0,
        sigma: float = 2.0,
        dt: float = 0.005,
        seed: int | None = None,
    ) -> None:
        # ----------------------------------------------------------------
        # Validate inputs — catch configuration errors early rather than
        # producing nonsensical prices silently.
        # ----------------------------------------------------------------
        if s0 <= 0:
            raise ValueError(f"s0 must be > 0, got {s0}")
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")

        # Store immutable model parameters.
        self.s0: float    = float(s0)
        self.sigma: float = float(sigma)
        self.dt: float    = float(dt)

        # Pre-compute the per-step standard deviation once.
        # Each Brownian increment is  dW ~ N(0, dt), so
        #   sigma * dW  ~  N(0, sigma^2 * dt)
        # and the std of each price move is sigma * sqrt(dt).
        # Caching this avoids a sqrt call on every tick across
        # 200 steps x 1000 Monte Carlo paths.
        self._sigma_sqrt_dt: float = self.sigma * np.sqrt(self.dt)

        # Random number generator — modern NumPy Generator API.
        # Thread-safe; reproducible per-instance rather than via a
        # global seed; recommended over the legacy np.random interface.
        self._rng: Generator = np.random.default_rng(seed)

        # Mutable simulation state.
        # Initialised here and reset at the start of each new path.
        self.s: float = self.s0   # current mid-price S_t
        self.t: float = 0.0       # current simulation time

    # ------------------------------------------------------------------
    # Step-by-step interface
    # ------------------------------------------------------------------

    def update_price(self) -> float:
        """
        Advance the mid-price by one time step dt.

        Applies the Euler-Maruyama discretisation of  dS = sigma * dW:

            S_{t+dt} = S_t + sigma * sqrt(dt) * Z,    Z ~ N(0, 1)

        There is no drift term — the paper assumes the agent has no
        opinion on the stock's expected return (Section 2.1, Eq. 1).

        Mutates ``self.s`` and ``self.t`` in place.

        Returns
        -------
        float
            The new mid-price S_{t+dt}.
        """
        # Draw a single standard-normal increment Z ~ N(0, 1).
        Z: float = self._rng.standard_normal()

        # Brownian increment: dW ≈ Z * sqrt(dt),
        # price increment:    dS = sigma * dW = sigma * sqrt(dt) * Z.
        self.s += self._sigma_sqrt_dt * Z

        # Advance the simulation clock by one step.
        self.t += self.dt

        return self.s

    # ------------------------------------------------------------------
    # Batch interface
    # ------------------------------------------------------------------

    def generate_path(
        self,
        T: float = 1.0,
        reset: bool = True,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Simulate a complete mid-price path over [0, T].

        All ``n_steps = round(T / dt)`` Brownian increments are drawn in
        a single ``rng.standard_normal(n_steps)`` call and accumulated
        with ``np.cumsum``.  This is ~10x faster than a Python loop over
        ``update_price`` and produces an identical distribution.

        Parameters
        ----------
        T : float
            Terminal time horizon.  Must be > 0.  Paper uses T = 1.
        reset : bool, optional
            If ``True`` (default), the internal state is reset to
            ``(s0, t=0)`` before generating the path.  Set to ``False``
            to continue from the current ``(self.s, self.t)`` — useful
            for stitching path segments together in the engine.

        Returns
        -------
        times : NDArray[np.float64], shape (n_steps + 1,)
            Equally-spaced time grid  [0, dt, 2*dt, ..., T].
        prices : NDArray[np.float64], shape (n_steps + 1,)
            Simulated mid-prices  [S_0, S_dt, S_2dt, ..., S_T].

        Raises
        ------
        ValueError
            If T <= 0.

        Notes
        -----
        After the call, ``self.s`` and ``self.t`` are updated to the
        terminal values of the path so that the object's state is always
        consistent regardless of which interface is used.
        """
        if T <= 0:
            raise ValueError(f"T must be > 0, got {T}")

        # Optionally reset to the start of a fresh path.
        if reset:
            self.s = self.s0
            self.t = 0.0

        # Number of steps: round rather than int-truncate to handle
        # floating-point representations of clean fractions (e.g. 1/0.005
        # may be 199.9999... without rounding).
        n_steps: int = int(round(T / self.dt))

        # --- Vectorised path construction ---
        # Draw all increments at once: dS_i = sigma * sqrt(dt) * Z_i
        increments: NDArray[np.float64] = (
            self._rng.standard_normal(n_steps) * self._sigma_sqrt_dt
        )

        # Allocate output array (n_steps + 1 points including S_0).
        prices: NDArray[np.float64] = np.empty(n_steps + 1, dtype=np.float64)

        # First element is the current price (either s0 or the continued
        # value when reset=False).
        prices[0] = self.s

        # cumsum of increments gives the displacement from prices[0];
        # adding prices[0] broadcasts the starting level onto the path.
        np.cumsum(increments, out=prices[1:])
        prices[1:] += prices[0]

        # Build the matching time grid.
        times: NDArray[np.float64] = np.linspace(
            self.t,                          # start (0.0 if reset, else current t)
            self.t + n_steps * self.dt,      # end
            n_steps + 1,
            dtype=np.float64,
        )

        # Sync internal state to the end of the generated path so that
        # subsequent calls to update_price or generate_path(reset=False)
        # continue consistently from here.
        self.s = float(prices[-1])
        self.t = float(times[-1])

        return times, prices

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Reset the mid-price and clock to their initial values.

        Equivalent to calling ``generate_path(reset=True)`` without
        drawing any increments.  Called by the Monte Carlo runner
        between independent simulation paths.
        """
        self.s = self.s0
        self.t = 0.0

    def theoretical_variance(self, horizon: float | None = None) -> float:
        """
        Return the theoretical variance of S_T - S_0 under ABM.

        Under  dS = sigma * dW  over a horizon h:

            Var[S_{t+h} - S_t] = sigma^2 * h

        Useful for validating that simulated paths have the correct
        second moment without having to run a full Monte Carlo.

        Parameters
        ----------
        horizon : float or None, optional
            Time horizon.  Defaults to ``self.t`` (elapsed simulation
            time since the last reset).

        Returns
        -------
        float
            sigma^2 * horizon.
        """
        h: float = horizon if horizon is not None else self.t
        return self.sigma ** 2 * h

    def theoretical_std(self, horizon: float | None = None) -> float:
        """
        Return the theoretical standard deviation of S_T - S_0.

        Convenience wrapper: ``sqrt(theoretical_variance(horizon))``.

        Parameters
        ----------
        horizon : float or None, optional
            Defaults to elapsed simulation time ``self.t``.

        Returns
        -------
        float
            sigma * sqrt(horizon).
        """
        return float(np.sqrt(self.theoretical_variance(horizon)))

    def __repr__(self) -> str:
        return (
            f"MidPriceModel("
            f"s0={self.s0}, "
            f"sigma={self.sigma}, "
            f"dt={self.dt}, "
            f"s={self.s:.4f}, "
            f"t={self.t:.4f})"
        )


# ---------------------------------------------------------------------------
# Smoke test  (python mid_price.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=== MidPriceModel smoke test ===\n")

    # Paper baseline parameters (Section 3.3)
    model = MidPriceModel(s0=100.0, sigma=2.0, dt=0.005, seed=42)
    print(f"Initialised: {model}\n")

    # ------------------------------------------------------------------
    # 1. Single-step sanity check
    # ------------------------------------------------------------------
    s_before = model.s
    s_after  = model.update_price()
    print("Single-step update:")
    print(f"  S before : {s_before:.6f}")
    print(f"  S after  : {s_after:.6f}   (delta = {s_after - s_before:+.6f})")
    print(f"  t        : {model.t:.4f}   (expected {model.dt:.4f})")
    assert model.t == model.dt, "Clock should advance by exactly dt"
    print("  ✓ clock advanced correctly\n")

    # ------------------------------------------------------------------
    # 2. Full path shape and time grid
    # ------------------------------------------------------------------
    model.reset()
    times, prices = model.generate_path(T=1.0)

    expected_steps = int(round(1.0 / 0.005))       # 200
    print("Full path (T=1.0, dt=0.005):")
    print(f"  Steps    : {len(prices) - 1}  (expected {expected_steps})")
    print(f"  S_0      : {prices[0]:.4f}   (expected {model.s0})")
    print(f"  S_T      : {prices[-1]:.4f}")
    print(f"  t[-1]    : {times[-1]:.4f}   (expected 1.0000)")
    assert len(prices) == expected_steps + 1, "Path length mismatch"
    assert prices[0] == model.s0, "Path must start at s0"
    assert abs(times[-1] - 1.0) < 1e-10, "Terminal time mismatch"
    print("  ✓ path shape and time grid correct\n")

    # ------------------------------------------------------------------
    # 3. State sync: after generate_path, self.s and self.t match path end
    # ------------------------------------------------------------------
    assert abs(model.s - prices[-1]) < 1e-12, "self.s should equal prices[-1]"
    assert abs(model.t - times[-1])  < 1e-12, "self.t should equal times[-1]"
    print("  ✓ internal state synced to end of path\n")

    # ------------------------------------------------------------------
    # 4. Monte Carlo variance check (1000 paths)
    # ------------------------------------------------------------------
    N         = 1_000
    T_horizon = 1.0
    terminals = np.empty(N, dtype=np.float64)

    for i in range(N):
        _, path    = model.generate_path(T=T_horizon, reset=True)
        terminals[i] = path[-1] - model.s0

    mc_var   = float(np.var(terminals))
    theo_var = model.theoretical_variance(T_horizon)
    rel_err  = abs(mc_var - theo_var) / theo_var * 100

    print(f"Monte Carlo variance check ({N} paths, T={T_horizon}):")
    print(f"  MC variance          : {mc_var:.4f}")
    print(f"  Theoretical (σ²T)    : {theo_var:.4f}")
    print(f"  Relative error       : {rel_err:.2f}%")

    if rel_err > 5.0:
        print(f"  WARNING: error exceeds 5% threshold.", file=sys.stderr)
        sys.exit(1)
    print("  ✓ within acceptable statistical tolerance\n")

    # ------------------------------------------------------------------
    # 5. reset() restores initial state
    # ------------------------------------------------------------------
    model.update_price()
    model.reset()
    assert model.s == model.s0,  "reset() must restore s to s0"
    assert model.t == 0.0,       "reset() must restore t to 0.0"
    print("  ✓ reset() restores initial state\n")

    # ------------------------------------------------------------------
    # 6. generate_path(reset=False) continues from current state
    # ------------------------------------------------------------------
    model.reset()
    model.update_price()           # advance one step manually
    t_mid = model.t
    s_mid = model.s

    _, continuation = model.generate_path(T=0.5, reset=False)
    assert abs(continuation[0] - s_mid) < 1e-12, (
        "Continued path must start from current self.s"
    )
    print(f"  ✓ generate_path(reset=False) continues from (s={s_mid:.4f}, t={t_mid:.4f})\n")

    # ------------------------------------------------------------------
    # 7. Theoretical helpers
    # ------------------------------------------------------------------
    theo_std = model.theoretical_std(1.0)
    print(f"Theoretical helpers (horizon=1.0):")
    print(f"  theoretical_variance : {model.theoretical_variance(1.0):.4f}  (σ²·T = {model.sigma**2:.1f})")
    print(f"  theoretical_std      : {theo_std:.4f}  (σ·√T = {model.sigma:.1f})")
    assert abs(model.theoretical_variance(1.0) - model.sigma ** 2) < 1e-12
    assert abs(theo_std - model.sigma) < 1e-12
    print("  ✓ helpers return correct analytical values\n")

    print("All checks passed.")