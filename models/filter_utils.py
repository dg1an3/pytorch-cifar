import numpy as np
import torch

def make_meshgrid(sz=7):
    return np.meshgrid(
        np.linspace(-(sz // 2), sz // 2, sz),
        np.linspace(-(sz // 2), sz // 2, sz),
    )


def complex_exp(xs, ys, freq, angle_rad):
    return np.exp(
        freq * (xs * np.sin(angle_rad) + ys * np.cos(angle_rad)) * 1.0j
    )


def gauss(xs, ys, sigma):
    return (1 / (2 * np.pi * sigma**2)) * np.exp(
        -(xs * xs + ys * ys) / (2.0 * sigma * sigma)
    )


def gabor(xs, ys, freq, angle_rad, sigma=None):
    return complex_exp(xs, ys, freq, angle_rad) * gauss(
        xs, ys, sigma if sigma else 2.0 / freq
    )


def make_gabor_bank(xs, ys, directions=3, freqs=[2.0, 1.0]):
    # type: (...) -> List[np.ndarray]
    """ """
    for freq in freqs:
        for n in range(directions):
            angle = n * np.pi / np.float32(directions)
            kernel = gabor(xs, ys, freq, angle)
            yield kernel
        yield gauss(xs, ys, 2.0 / freq)


def kernels2weights(kernels, in_channels=1, dtype=torch.float32):
    kernels = np.expand_dims(kernels, axis=1)
    kernels = np.repeat(kernels, in_channels, axis=1)
    return torch.tensor(kernels, dtype=dtype)