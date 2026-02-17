import jax.numpy as jnp
import jax.random as jr


def generate_signed_weight_matrix(key, neuron_sign, n_out, p0):
    n = neuron_sign.shape[0]
    k1, k2 = jr.split(key)

    # Random positive magnitudes, masked with probability p0
    abs_W = jnp.abs(jr.normal(k1, (n, n_out)))
    mask = (jr.uniform(k2, (n, n_out)) >= p0).astype(abs_W.dtype)
    abs_W = abs_W * mask

    # Separate excitatory / inhibitory contributions per column
    is_exc = (neuron_sign > 0)[:, None]
    is_inh = (neuron_sign < 0)[:, None]

    exc_sum = jnp.sum(abs_W * is_exc, axis=0)
    inh_sum = jnp.sum(abs_W * is_inh, axis=0)

    # Scale inhibitory magnitudes so each column sums to zero
    inh_scale = jnp.where(inh_sum > 0, exc_sum / inh_sum, 0.0)
    W = abs_W * is_exc - abs_W * is_inh * inh_scale[None, :]

    # Normalize so the spectral radius (largest |eigenvalue|) is exactly 1
    eigvals = jnp.linalg.eigvals(W)
    spectral_radius = jnp.max(jnp.abs(eigvals))
    W = W / spectral_radius

    return W


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    n = 1000
    p0 = 0.9
    frac_inh = 0.2
    n_inh = int(n * frac_inh)

    key = jr.PRNGKey(42)
    neuron_sign = jnp.concatenate([jnp.ones(n - n_inh), -jnp.ones(n_inh)])

    W = generate_signed_weight_matrix(key, neuron_sign, n, p0)

    # Verify properties
    print(f"Column sums (should be ~0): max abs = {jnp.abs(W.sum(axis=0)).max():.2e}")
    eigvals = jnp.linalg.eigvals(W)
    print(f"Spectral radius: {jnp.max(jnp.abs(eigvals)):.6f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Weight matrix — use nonzero percentile for color limits so zeros don't wash out
    nonzero = jnp.abs(W)[W != 0]
    vlim = jnp.percentile(nonzero, 95)
    im = axes[0].imshow(W, cmap="RdBu_r", aspect="equal",
                        vmin=-vlim, vmax=vlim)
    axes[0].set_facecolor("white")
    axes[0].set_title("Weight matrix (Dale's law)")
    axes[0].set_xlabel("Post-synaptic neuron")
    axes[0].set_ylabel("Pre-synaptic neuron")
    fig.colorbar(im, ax=axes[0], shrink=0.8)

    # Eigenvalue spectrum
    theta = jnp.linspace(0, 2 * jnp.pi, 200)
    axes[1].plot(jnp.cos(theta), jnp.sin(theta), "k--", alpha=0.3, label="unit circle")
    axes[1].scatter(eigvals.real, eigvals.imag, s=10, c="steelblue")
    axes[1].set_title("Eigenvalue spectrum")
    axes[1].set_xlabel("Real")
    axes[1].set_ylabel("Imaginary")
    axes[1].set_aspect("equal")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("dales_law.png", dpi=150)
    plt.show()
