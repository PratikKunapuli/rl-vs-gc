import torch
import omni.isaac.lab.utils.math as isaac_math_utils
from typing import Tuple


@torch.jit.script
def eval_sinusoid(t: torch.Tensor, amp:float, freq:float, phase:float, offset:float):
    return (amp * torch.sin(freq * t + phase) + offset)

@torch.jit.script
def eval_sinusoid(t: torch.Tensor, amp:torch.Tensor, freq:torch.Tensor, phase:torch.Tensor, offset:torch.Tensor):
    return (amp * torch.sin(freq * t + phase) + offset)

@torch.jit.script
def eval_lissajous_curve(t: torch.Tensor, amp: torch.Tensor, freq: torch.Tensor, phase: torch.Tensor, offset: torch.Tensor, derivatives: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate Lissajous curves and their derivatives for multiple environments with local time.

    Args:
        t: Local time samples for each environment. Shape: (n_envs, n_samples).
        amp: Amplitudes. Shape: (n_envs, n_curves).
        freq: Frequencies. Shape: (n_envs, n_curves).
        phase: Phases. Shape: (n_envs, n_curves).
        offset: Offsets. Shape: (n_envs, n_curves).
        derivatives: Number of derivatives to compute (0 to 4).

    Returns:
        pos: Tensor containing the evaluated Lissajous curves and their derivatives.
             Shape: (num_derivatives + 1, n_envs, 3, n_samples).
        yaw: Tensor containing the yaw angles of the Lissajous curves.
             Shape: (num_derivatives + 1, n_envs, n_samples).
    """
    num_envs, num_samples = t.shape
    num_envs_amp, num_curves = amp.shape

    # Validate that the number of environments matches between t and other parameters
    assert num_envs == num_envs_amp, "Mismatch between number of environments in time and other parameters."

    # Reshape and expand tensors to match the time dimension
    amp = amp.unsqueeze(-1).expand(num_envs, num_curves, num_samples)
    freq = freq.unsqueeze(-1).expand(num_envs, num_curves, num_samples)
    phase = phase.unsqueeze(-1).expand(num_envs, num_curves, num_samples)
    offset = offset.unsqueeze(-1).expand(num_envs, num_curves, num_samples)

    # Compute theta, sin(theta), and cos(theta) for each environment
    theta = freq * t.unsqueeze(1) + phase  # Shape: (n_envs, n_curves, n_samples)
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)

    # Initialize the list of results with the position (0th derivative)
    curves = amp * sin_theta + offset
    results = [curves]

    # Compute derivatives up to the specified order
    if derivatives >= 1:
        first_derivative = amp * freq * cos_theta
        results.append(first_derivative)

    if derivatives >= 2:
        second_derivative = -amp * freq.pow(2) * sin_theta
        results.append(second_derivative)

    if derivatives >= 3:
        third_derivative = -amp * freq.pow(3) * cos_theta
        results.append(third_derivative)

    if derivatives >= 4:
        fourth_derivative = amp * freq.pow(4) * sin_theta
        results.append(fourth_derivative)

    # Stack the results and split into position and yaw
    full_data = torch.stack(results, dim=0)  # Shape: (num_derivatives + 1, n_envs, n_curves, n_samples)
    pos = full_data[:, :, :3, :]             # Position curves (x, y, z)
    yaw = full_data[:, :, 3, :]              # Yaw curves

    return pos, yaw

@torch.jit.script
def eval_polynomial_curve(t: torch.Tensor, coeffs: torch.Tensor, derivatives: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate polynomial curves and their derivatives for multiple environments with local time.

    Args:
        t: Local time samples for each environment. Shape: (n_envs, n_samples).
        coeffs: Polynomial coefficients. Shape: (n_envs, n_curves, degree + 1).
        derivatives: Number of derivatives to compute (0 to max degree).

    Returns:
        pos: Evaluated polynomial curves and derivatives. Shape: (num_derivatives + 1, n_envs, 3, n_samples).
        yaw: Yaw angles of the polynomial curves. Shape: (num_derivatives + 1, n_envs, n_samples).
    """
    n_envs, n_samples = t.shape
    n_envs_coeffs, n_curves, degree_plus_one = coeffs.shape
    degree = degree_plus_one - 1

    # Validate that the number of environments matches between t and coeffs
    assert n_envs == n_envs_coeffs, "Mismatch between number of environments in time and coefficients."

    # Reshape and expand coefficients to enable broadcasting
    coeffs = coeffs.unsqueeze(-1).expand(n_envs, n_curves, degree_plus_one, n_samples)

    # Compute powers of time for each environment
    t_powers = torch.stack([t.pow(i) for i in range(degree + 1)], dim=2)  # Shape: (n_envs, n_samples, degree + 1)
    t_powers = t_powers.permute(0, 2, 1)  # Shape: (n_envs, degree + 1, n_samples)

    # Precompute factorial-like coefficients for derivatives
    factorial_coeffs = torch.zeros((derivatives + 1, degree + 1), device=coeffs.device, dtype=torch.float32)
    for i in range(derivatives + 1):
        for j in range(i, degree + 1):
            factorial_coeffs[i, j] = torch.prod(torch.arange(j, j - i, -1).float())

    # Compute derivatives
    results = []
    for i in range(derivatives + 1):
        coeff_factors = factorial_coeffs[i, :].view(1, 1, -1, 1)  # Shape: (1, 1, degree + 1, 1)
        valid_coeffs = coeffs * coeff_factors
        derivative = (valid_coeffs[:, :, i:, :] * t_powers[:, i:, :].unsqueeze(1)).sum(dim=2)  # Sum along degree axis
        results.append(derivative)

    # Stack results into a single tensor
    full_data = torch.stack(results, dim=0)  # Shape: (num_derivatives + 1, n_envs, n_curves, n_samples)

    # Position and yaw separation
    pos = full_data[:, :, :3, :]  # Position curves (x, y, z)
    yaw = full_data[:, :, 3, :]   # Yaw curves

    return pos, yaw