import torch
import omni.isaac.lab.utils.math as isaac_math_utils
from typing import Tuple

@torch.jit.script
def quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    """Get quaternion from yaw angle.

    Args:
        yaw: The yaw angle. Shape is (...,).

    Returns:
        The quaternion. Shape is (..., 4).
    """
    shape = yaw.shape
    yaw = yaw.view(-1)
    q = torch.zeros(yaw.shape[0], 4, device=yaw.device)
    q[:, 0] = torch.cos(yaw / 2.0)
    q[:, 1] = 0.0
    q[:, 2] = 0.0
    q[:, 3] = torch.sin(yaw / 2.0)
    return q.view(shape + (4,))

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
    assert num_envs == num_envs_amp, "Mismatch between number of environments in time and other parameters. num_envs_amp: {}, num_envs: {}".format(num_envs_amp, num_envs)

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

# @torch.jit.script
# def eval_polynomial_curve(t: torch.Tensor, coeffs: torch.Tensor, derivatives: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Evaluate polynomial curves and their derivatives for multiple environments with local time.

#     Args:
#         t: Local time samples for each environment. Shape: (n_envs, n_samples).
#         coeffs: Polynomial coefficients. Shape: (n_envs, n_curves, degree + 1).
#         derivatives: Number of derivatives to compute (0 to max degree).

#     Returns:
#         pos: Evaluated polynomial curves and derivatives. Shape: (num_derivatives + 1, n_envs, 3, n_samples).
#         yaw: Yaw angles of the polynomial curves. Shape: (num_derivatives + 1, n_envs, n_samples).
#     """
#     n_envs, n_samples = t.shape
#     n_envs_coeffs, n_curves, degree_plus_one = coeffs.shape
#     degree = degree_plus_one - 1

#     # Validate that the number of environments matches between t and coeffs
#     assert n_envs == n_envs_coeffs, "Mismatch between number of environments in time and coefficients."

#     # Reshape and expand coefficients to enable broadcasting
#     coeffs = coeffs.unsqueeze(-1).expand(n_envs, n_curves, degree_plus_one, n_samples)

#     # Compute powers of time for each environment
#     t_powers = torch.stack([t.pow(i) for i in range(degree + 1)], dim=2)  # Shape: (n_envs, n_samples, degree + 1)
#     t_powers = t_powers.permute(0, 2, 1)  # Shape: (n_envs, degree + 1, n_samples)

#     # Precompute factorial-like coefficients for derivatives
#     factorial_coeffs = torch.zeros((derivatives + 1, degree + 1), device=coeffs.device, dtype=torch.float32)
#     for i in range(derivatives + 1):
#         for j in range(i, degree + 1):
#             factorial_coeffs[i, j] = torch.prod(torch.arange(j, j - i, -1).float())

#     # Compute derivatives
#     results = []
#     for i in range(derivatives + 1):
#         coeff_factors = factorial_coeffs[i, :].view(1, 1, -1, 1)  # Shape: (n_envs, n_curves, n_samples)
#         valid_coeffs = coeffs * coeff_factors
#         derivative = (valid_coeffs[:, :, i:, :] * t_powers[:, i:, :].unsqueeze(1)).sum(dim=2)  # Shape: (n_envs, n_curves, n_samples)
#         results.append(derivative)

#     # Stack results into a single tensor
#     full_data = torch.stack(results, dim=0)  # Shape: (num_derivatives + 1, n_envs, n_curves, n_samples)

#     # Position and yaw separation
#     pos = full_data[:, :, :3, :]  # Position curves (x, y, z)
#     yaw = full_data[:, :, 3, :]   # Yaw curves

#     return pos, yaw
@torch.jit.script
def eval_polynomial_curve(t: torch.Tensor, coeffs: torch.Tensor, derivatives: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate polynomial curves and their derivatives for multiple environments with local time.

    Assumes that n_curves == 4: the first 3 curves correspond to (x, y, z),
    and the 4th curve corresponds to yaw.

    Args:
        t (Tensor): Local time samples for each environment.
            Shape: (n_envs, n_samples).
        coeffs (Tensor): Polynomial coefficients.
            Shape: (n_envs, n_curves, degree + 1).
        derivatives (int): Number of derivatives to compute (0 to max degree).

    Returns:
        pos (Tensor): Evaluated polynomial curves (x,y,z) and their derivatives.
            Shape: (derivatives + 1, n_envs, 3, n_samples).
        yaw (Tensor): Evaluated yaw curve and its derivatives.
            Shape: (derivatives + 1, n_envs, n_samples).
    """
    n_envs, n_samples = t.shape
    n_envs_coeffs, n_curves, degree_plus_one = coeffs.shape
    degree = degree_plus_one - 1

    # Make sure we have enough curves for pos(3) + yaw(1) = 4
    assert n_curves == 4, "This function expects exactly 4 curves (x,y,z,yaw)."
    assert n_envs == n_envs_coeffs, (
        f"Mismatch: t has {n_envs} envs, but coeffs has {n_envs_coeffs}."
    )

    # Expand coefficients so that each environment, curve, and power
    # can be multiplied by the times t (n_samples)
    # Final shape will be (n_envs, n_curves, degree+1, n_samples)
    coeffs = coeffs.unsqueeze(-1).expand(n_envs, n_curves, degree_plus_one, n_samples)

    # Precompute powers of t up to 'degree'
    # t_powers shape: (n_envs, degree+1, n_samples)
    t_powers = torch.stack([t.pow(i) for i in range(degree + 1)], dim=2)
    t_powers = t_powers.permute(0, 2, 1)  # now (n_envs, degree+1, n_samples)

    # Precompute derivative factorial-like coefficients:
    # For the i-th derivative of x^j, the multiplier is j*(j-1)*...(j-i+1).
    # If i>j, the derivative should be zero automatically.
    factorial_coeffs = torch.zeros((derivatives + 1, degree + 1),
                                   device=coeffs.device,
                                   dtype=torch.float32)
    for i in range(derivatives + 1):
        for j in range(degree + 1):
            if j < i:
                # e.g., 3rd derivative of x^2 => 0
                factorial_coeffs[i, j] = 0.0
            elif i == 0:
                # 0th derivative => multiply by 1
                factorial_coeffs[i, j] = 1.0
            else:
                # j*(j-1)*...*(j-i+1)
                factorial_coeffs[i, j] = torch.prod(
                    torch.arange(j, j - i, -1, device=coeffs.device).float()
                )

    # Compute each derivative i from 0..derivatives
    results_per_derivative = []
    for i in range(derivatives + 1):
        # for derivative i, we want factorial_coeffs[i, j] * coeffs_j * t^(j-i)

        # shape: (1,1,degree+1,1)
        coeff_factors = factorial_coeffs[i, :].view(1, 1, -1, 1)
        valid_coeffs = coeffs * coeff_factors  # (n_envs, n_curves, degree+1, n_samples)

        # Instead of slicing t_powers[:, i:, :], slice t_powers[:, 0:degree+1-i, :]
        # so that j-i goes from 0..(degree-i).
        # effectively we match j= i..degree with exponent= 0..(degree-i).
        t_powers_for_i = t_powers[:, : (degree + 1 - i), :]  # shape: (n_envs, degree+1-i, n_samples)

        d_i = (
            valid_coeffs[:, :, i:, :] *  # c_(i..degree)
            t_powers_for_i.unsqueeze(1)  # t^(0..degree-i), unsqueeze(1) for 'curve' dimension
        ).sum(dim=2)  # sum over the polynomial power dimension

        results_per_derivative.append(d_i)

    # Stack all derivatives => shape: (derivatives+1, n_envs, n_curves, n_samples)
    full_data = torch.stack(results_per_derivative, dim=0)

    # Now split out pos (the first 3 curves) and yaw (the 4th curve)
    pos = full_data[:, :, :3, :]  # (derivatives+1, n_envs, 3, n_samples)
    yaw = full_data[:, :, 3, :]   # (derivatives+1, n_envs, n_samples)

    return pos, yaw

@torch.jit.script
def eval_random_walk(pos_init:torch.Tensor, vel_init:torch.Tensor, acc_coeffs: torch.Tensor, T_max:float, step_size:float) -> torch.Tensor:
    """
    Generate random walk trajectories from acceleration coefficients. Acceleration should be in body frame, but then the random walk should be in the world frame.

    Args:
        pos_init: Initial position for each environment. Shape: (n_envs, 3).
        vel_init: Initial velocity for each environment. Shape: (n_envs, 3).
        acc_coeffs: Coefficients for the random walk acceleration.
            Shape: (n_envs, 3).
        T_max: Maximum time for the random walk.
        step_size: Time step size for integration.

    Returns:
        pos: Evaluated random walk position curves (x,y,z) and their derivatives.
            Shape: (n_envs, 3, step_size * T_max).
    """
    yaw = torch.zeros((1), device=pos_init.device, dtype=torch.float32)
    ori = quat_from_yaw(yaw) 
    
    n_envs, n_curves = acc_coeffs.shape
    
    # Initialize position and velocity
    pos = pos_init
    vel = vel_init

    # Initialize the list of results with the position (0th derivative)
    results = [pos.unsqueeze(-1)]

    # Integrate the random walk dynamics
    for _ in range(int(T_max / step_size)):
        # Compute the new velocity and position
        # acc_b = torch.randn((n_envs, 3), device=pos.device) * acc_coeffs
        acc_b = acc_coeffs
        acc_w = isaac_math_utils.quat_rotate(ori, acc_b)
        # print(acc_w)
        vel += acc_w * step_size
        pos += vel * step_size

        # Append the new position to the results
        results.append(pos.unsqueeze(-1))
    
    # Stack the results
    full_data = torch.cat(results, dim=-1)
    return full_data

    