import numpy as np
import matplotlib.pyplot as plt


st, ed, n = .0001, .02, 1000
times = np.arange(n)


def plot_helper(betas, **kwargs):
    alphas = 1-betas
    alpha_bars = np.cumprod(alphas)
    plt.semilogy(times, alpha_bars, **kwargs)

def multiplicative_factors(betas):
    alphas = 1-betas
    alpha_bars = np.cumprod(alphas)
    alpha_bars_t_minus_1 = np.hstack([[1], alpha_bars[:-1]])
    sigmas = (1 - alpha_bars_t_minus_1) / (1 - alpha_bars) * betas

    factor = betas**2 / (2 * sigmas**2 * alphas * (1 - alpha_bars))
    return factor

betas = np.linspace(st, ed, n)
plt.semilogy(times, multiplicative_factors(betas))
plt.xlabel("t")
plt.title("multiplicative factors that are dropped")
plt.savefig("plots/test.png")

# plot_helper(betas = np.linspace(st, ed, n), label="DDPM original")
# # plot_helper(betas = np.linspace(st, 1, n), label="end at one")
# plt.legend()
# plt.savefig('plots/test.png')