"""
PURKINJE CELL: LIF (Leaky Integrate-and-Fire) integration.

Biologis: Purkinje cells integrate synaptic inputs over time.
AI: Each node = LIF neuron, integrate from neighbors, fire if threshold.

Formula (ARCHITECTURE.md section 9 — Spiking Dynamics):
    tau * dU/dt = -(U - U_reset) + I_input    (membrane potential ODE)
    S = Theta(U - U_th)                        (spike threshold; S=1 iff U >= U_th)
    U = U_reset if S=1                         (hard reset after fire)

Discretisation
--------------
The ODE above is linear in U and admits a closed-form solution over a
timestep dt, which is both more stable than naive forward-Euler (Euler
oscillates when tau < dt/2 — and our default tau=0.5, dt=1.0 falls
exactly in that regime) and biologically faithful:

    U(t+dt) = U_reset
            + (U(t) - U_reset) * exp(-dt / tau)
            + I_input * (1 - exp(-dt / tau))

With this update, the membrane potential *decays* exponentially toward
U_reset when I_input = 0 (the "Membrane potential decay dengan tau"
case), and *charges* asymptotically toward U_reset + I_input when
I_input is held constant — firing periodically whenever that asymptote
exceeds U_th. This is the textbook LIF behaviour expected by
ARCHITECTURE.md.
"""

import math


class PurkinjeCell:
    """Leaky Integrate-and-Fire neuron.

    Each instance tracks its own membrane potential ``u``. The
    ``integrate_and_fire`` method advances the membrane by one timestep
    under a constant input current, then tests the spike threshold and
    (on firing) performs the hard reset prescribed by the LIF formula.

    Attributes:
        tau: Membrane time constant (seconds-equivalent). Smaller tau
            means faster leak. Default 0.5 (per ARCHITECTURE.md).
        threshold: Spike threshold U_th. The neuron fires when the
            post-update membrane potential satisfies U >= U_th.
            Default 1.0.
        u_reset: Reset potential U_reset — the value U is clamped to
            immediately after a spike, and the resting value the
            membrane decays toward under zero input. Default 0.0.
        u: Current membrane potential. Initialised to u_reset so that
            a freshly constructed cell starts at rest.
    """

    def __init__(self, tau: float = 0.5, threshold: float = 1.0, u_reset: float = 0.0):
        """Initialise LIF parameters and clamp membrane to u_reset."""
        if tau <= 0:
            raise ValueError(f"tau must be > 0 (got {tau!r}) — membrane would never leak.")
        self.tau = float(tau)
        self.threshold = float(threshold)
        self.u_reset = float(u_reset)
        # Membrane starts at the reset/resting potential.
        self.u = float(u_reset)

    def integrate_and_fire(self, input_current: float, dt: float = 1.0) -> bool:
        """Advance one LIF timestep and test the spike threshold.

        Implements the closed-form update of
        ``tau * dU/dt = -(U - U_reset) + I_input`` over the interval
        ``dt``, then applies the threshold-and-reset rule:
        ``S = Theta(U - U_th)`` and ``U <- U_reset`` iff ``S = 1``.

        Args:
            input_current: I_input for this timestep. Held constant
                across the interval ``dt``.
            dt: Time delta for this step. Default 1.0.

        Returns:
            True if the neuron fired this step (S=1), False otherwise.
            A fired neuron's membrane is left at ``u_reset`` for the
            next call.
        """
        if dt < 0:
            raise ValueError(f"dt must be >= 0 (got {dt!r}).")

        # Closed-form solution of the linear ODE over [0, dt].
        # decay in (0, 1]; equals 1.0 only in the dt -> 0 limit.
        decay = math.exp(-dt / self.tau)
        self.u = (
            self.u_reset
            + (self.u - self.u_reset) * decay
            + float(input_current) * (1.0 - decay)
        )

        # Spike threshold: Theta(U - U_th) — fire iff U >= U_th.
        spiked = self.u >= self.threshold
        if spiked:
            # Hard reset after fire.
            self.u = self.u_reset
        return bool(spiked)

    def reset(self) -> None:
        """Manually clamp the membrane potential back to u_reset.

        Useful for unit tests that want a known initial condition
        without constructing a fresh cell, and for callers that want
        to "silence" a neuron between replay episodes.
        """
        self.u = float(self.u_reset)
