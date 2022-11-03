import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_meshgrid(sz=7):
    return np.meshgrid(
        np.linspace(-(sz // 2), sz // 2, sz),
        np.linspace(-(sz // 2), sz // 2, sz),
    )


def complex_exp(xs, ys, freq, angle_rad):
    return np.exp(freq * (xs * np.sin(angle_rad) + ys * np.cos(angle_rad)) * 1.0j)


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


def make_oriented_powermap(kernel_size=7, directions=9):
    """_summary_

    Args:
        kernel_size (int, optional): _description_. Defaults to 7.
        directions (int, optional): _description_. Defaults to 9.

    Returns:
        tuple: (in planes, real conv filter, imaginary conv filter)
    """
    xs, ys = make_meshgrid(sz=kernel_size)
    phi = (5**0.5 + 1) / 2  # golden ratio
    freqs = [phi**n for n in range(2, -4, -1)]
    kernels_complex = (list)(
        make_gabor_bank(xs, ys, directions=directions, freqs=freqs)
    )
    kernels_real, kernels_imag = np.real(kernels_complex), np.imag(kernels_complex)
    weights_real, weights_imag = (
        kernels2weights(kernels_real, 3),
        kernels2weights(kernels_imag, 3),
    )
    print(f"make_oriented_powermap: weights_real.shape = {weights_real.shape}")

    # NOTE: these need to be children to make sure they are handled correctly (i.e. assigning to device)
    conv_real = nn.Conv2d(
        3,
        len(kernels_complex),
        kernel_size=kernel_size,
        stride=1,
        padding=3,
        bias=False,
    )
    conv_real.weight = torch.nn.Parameter(weights_real)

    conv_imag = nn.Conv2d(
        3,
        len(kernels_complex),
        kernel_size=kernel_size,
        stride=1,
        padding=3,
        bias=False,
    )
    conv_imag.weight = torch.nn.Parameter(weights_imag)

    return len(kernels_complex), conv_real, conv_imag
