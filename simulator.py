import jax
import jax.numpy as jnp
import jax.random as jr

from dales_law import generate_signed_weight_matrix

def make_network(key, n, n_input, K, p0, use_signed_matrix):
    """Create network weights with synaptic delays.

    Returns:
        W_kernel: (n, n, K) sparse delay kernel — each connection has weight at one delay slot
        W_in: (n, n_input) input weights
        delays: (n, n) integer delay matrix in {1, ..., K}
    """
    k1, k2, k3 = jr.split(key, 3)

    if use_signed_matrix:
        neuron_sign = (jnp.arange(n) > 0.2 * n) * 2 - 1
        W = generate_signed_weight_matrix(k1, neuron_sign, n, p0)
    else:
        W = jr.normal(k1, (n, n)) / n
    W_in = jr.normal(k2, (n, n_input)) / n_input

    # Random delays per connection, uniform in {1, ..., K}
    delays = jr.randint(k3, (n, n), minval=1, maxval=K + 1)

    # Build sparse kernel: W_kernel[i, j, d] = W[i,j] only at d = delays[i,j]-1
    delay_onehot = jnp.eye(K)[delays - 1]  # (n, n, K)
    W_kernel = W[:, :, None] * delay_onehot  # (n, n, K)

    return W_kernel, W_in, delays


def step(W_kernel, W_in, v, spike_buf, x, a, v_thr, dampening_factor=1.0):
    """One LIF step with synaptic delays.

    Args:
        W_kernel: (n, n, K) delay kernel
        W_in: (n, n_input) input weights
        v: (n,) membrane potential
        spike_buf: (K, n) buffer of past spikes, spike_buf[0] = z(t-1), spike_buf[K-1] = z(t-K)
        x: (n_input,) external input
    """
    # Surrogate gradient: forward pass uses Heaviside, backward uses sigmoid
    z = (v >= v_thr).astype(v.dtype)
    z_back = jax.nn.sigmoid((v - v_thr) * dampening_factor)
    z = jax.lax.stop_gradient(z - z_back) + z_back

    # Recurrent input via delayed spikes: sum_k W_kernel[:, :, k] @ spike_buf[k]
    rec_input = jnp.einsum("ijk,kj->i", W_kernel, spike_buf)

    v_next = a * v + (1 - a) * (rec_input + W_in @ x) - z * v_thr

    # Shift buffer: drop oldest, prepend new spikes
    spike_buf_next = jnp.concatenate([z[None, :], spike_buf[:-1]], axis=0)

    return v_next, spike_buf_next, z


def simulate(W_kernel, W_in, v0, inputs, a, v_thr):
    """Simulate a single trial.

    Args:
        v0: (n,)
        inputs: (n_steps, n_input)
    """
    K = W_kernel.shape[2]
    n = v0.shape[0]
    spike_buf0 = jnp.zeros((K, n))

    def scan_fn(carry, x):
        v, spike_buf = carry
        v_next, spike_buf_next, z = step(W_kernel, W_in, v, spike_buf, x, a, v_thr)
        return (v_next, spike_buf_next), z

    (v_final, _), spikes = jax.lax.scan(scan_fn, (v0, spike_buf0), inputs)
    return v_final, spikes


def simulate_batch(W_kernel, W_in, v0, inputs, a, v_thr):
    """Simulate many trials in parallel. v0: (batch, n), inputs: (batch, n_steps, n_input)."""
    return jax.vmap(simulate, in_axes=(None, None, 0, 0, None, None))(
        W_kernel, W_in, v0, inputs, a, v_thr
    )


if __name__ == "__main__":
    import jax

    n = 1000
    n_input = 1
    n_steps = 1000
    K = 10  # max synaptic delay in timesteps
    v_thr = 0.7
    dt = 0.5 # time in milliseconds
    tau_min, tau_max = 10.0, 20.0 # time in milliseconds
    p0 = 0.1
    signed = True

    key = jr.PRNGKey(0)
    key, k_tau_m = jr.split(key)
    tau = tau_min + jr.uniform(k_tau_m, shape=(n,)) * (tau_max - tau_min)
    a = jnp.exp(-dt / tau)

    W_kernel, W_in, delays = make_network(key, n, n_input, K=K, p0=p0, use_signed_matrix=signed)
    batch_size = 8
    v0 = jnp.zeros((batch_size, n))

    # OU process input: x(t+1) = alpha * x(t) + sigma * sqrt(1-alpha^2) * noise
    tau_ou = 50.0  # ms
    sigma_ou = 1.0
    key, k_ou = jr.split(key)
    noise = jr.normal(k_ou, (n_steps, batch_size, n_input))
    alpha_ou = jnp.exp(-dt / tau_ou)
    scale_ou = sigma_ou * jnp.sqrt(1 - alpha_ou**2)

    def ou_step(x, xi):
        x_next = alpha_ou * x + scale_ou * xi
        return x_next, x_next

    x0 = jnp.zeros((batch_size, n_input))
    _, inputs = jax.lax.scan(ou_step, x0, noise)  # (n_steps, batch, n_input)
    inputs = jnp.moveaxis(inputs, 0, 1)  # (batch, n_steps, n_input)

    v_final, spikes = simulate_batch(W_kernel, W_in, v0, inputs, a, v_thr)
    print(f"spikes shape: {spikes.shape}")  # (batch, n_steps, n)
    T_sec = n_steps * dt / 1000.0
    rates_per_trial = spikes.sum(axis=(1, 2)) / (n * T_sec)  # Hz per trial
    print(f"firing rate per trial (Hz): {rates_per_trial}")
    print(f"mean firing rate: {rates_per_trial.mean():.1f} Hz")

    import matplotlib.pyplot as plt

    # Plot raster for first trial
    trial_spikes = spikes[0]
    times, neurons = jnp.where(trial_spikes)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(times * dt, neurons, s=1, c="black")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron")
    ax.set_title("Spike raster plot (trial 0)")
    plt.tight_layout()
    plt.savefig("raster.png", dpi=150)
    plt.show()
