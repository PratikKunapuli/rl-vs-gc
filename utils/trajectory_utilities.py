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
    Evaluate Lissajous curves and their derivatives for multiple environments.

    Args:
        t: Time samples. Shape: (num_samples,).
        amp: Amplitudes. Shape: (n_envs, n_curves).
        freq: Frequencies. Shape: (n_envs, n_curves).
        phase: Phases. Shape: (n_envs, n_curves).
        offset: Offsets. Shape: (n_envs, n_curves).
        derivatives: Number of derivatives to compute (0 to 4).

    Returns:
        pos: Tensor containing the evaluated Lissajous curves and their derivatives.
             Shape: (num_derivatives + 1, n_envs, 3, num_samples).
        yaw: Tensor containing the yaw angles of the Lissajous curves.
             Shape: (num_derivatives + 1, n_envs, num_samples).
    """
    if len(amp.shape) == 1:
        amp = amp.unsqueeze(0)
        freq = freq.unsqueeze(0)
        phase = phase.unsqueeze(0)
        offset = offset.unsqueeze(0)
    num_envs, num_curves = amp.shape
    num_samples = t.shape[0]

    # Reshape and expand tensors to enable broadcasting
    t = t.view(1, 1, num_samples).expand(num_envs, num_curves, num_samples)
    amp = amp.unsqueeze(-1).expand(num_envs, num_curves, num_samples)
    freq = freq.unsqueeze(-1).expand(num_envs, num_curves, num_samples)
    phase = phase.unsqueeze(-1).expand(num_envs, num_curves, num_samples)
    offset = offset.unsqueeze(-1).expand(num_envs, num_curves, num_samples)

    # Compute theta, sin(theta), and cos(theta) once for efficiency
    theta = freq * t + phase        # Shape: (n_envs, n_curves, num_samples)
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
    full_data = torch.stack(results, dim=0)  # Shape: (num_derivatives + 1, n_envs, n_curves, num_samples)
    pos = full_data[:, :, :3, :]             # Position curves (x, y, z)
    yaw = full_data[:, :, 3, :]              # Yaw curves

    return pos, yaw

@torch.jit.script
def eval_polynomial_curve(t: torch.Tensor, coeffs: torch.Tensor, derivatives: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate polynomial curves and their derivatives for multiple environments.

    Args:
        t: Time samples. Shape: (num_samples,).
        coeffs: Polynomial coefficients. Shape: (n_envs, n_curves, degree + 1).
        derivatives: Number of derivatives to compute (0 to 4).

    Returns:
        pos: Tensor containing the evaluated Polynomial curves and their derivatives.
             Shape: (num_derivatives + 1, n_envs, 3, num_samples).
        yaw: Tensor containing the yaw angles of the Polynomial curves.
             Shape: (num_derivatives + 1, n_envs, num_samples).
    """
    num_envs, num_curves, degree_plus_one = coeffs.shape
    degree = degree_plus_one - 1
    num_samples = t.shape[0]

    # Reshape and expand tensors to enable broadcasting
    t = t.view(1, 1, num_samples).expand(num_envs, num_curves, num_samples)
    coeffs = coeffs.unsqueeze(-1).expand(num_envs, num_curves, degree_plus_one, num_samples)

    # Compute powers of time for polynomial evaluation
    t_powers = torch.stack([t.pow(i) for i in range(degree + 1)], dim=2)  # Shape: (num_envs, num_curves, degree + 1, num_samples)

    # Precompute factorial-like coefficients for derivatives
    factorial_coeffs = torch.zeros((derivatives + 1, degree + 1), device=coeffs.device, dtype=torch.float32)
    for i in range(derivatives + 1):
        for j in range(i, degree + 1):
            factorial_coeffs[i, j] = torch.prod(torch.arange(j, j - i, -1).float())

    # Compute derivatives
    results = []
    for i in range(derivatives + 1):
        # Select the relevant coefficients and multiply with factorial_coeffs
        coeff_factors = factorial_coeffs[i, :].view(1, 1, -1, 1)  # Shape: (1, 1, degree + 1, 1)
        valid_coeffs = coeffs * coeff_factors
        # Multiply with time powers and sum along the degree axis
        derivative = (valid_coeffs[:, :, i:, :] * t_powers[:, :, :degree + 1 - i, :]).sum(dim=2)
        results.append(derivative)

    # Stack results into a single tensor
    full_data = torch.stack(results, dim=0)  # Shape: (num_derivatives + 1, n_envs, n_curves, num_samples)
    
    # Position and yaw separation
    pos = full_data[:, :, :3, :]  # Position curves (x, y, z)
    yaw = full_data[:, :, 3, :]   # Yaw curves

    return pos, yaw
