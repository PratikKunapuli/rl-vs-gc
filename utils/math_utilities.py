import torch

@torch.jit.script
def vee_map(S):
    """Convert skew-symmetric matrix to vector.

    Args:
        S: The skew-symmetric matrix. Shape is (N, 3, 3).

    Returns:
        The vector representation of the skew-symmetric matrix. Shape is (N, 3).
    """
    return torch.stack([S[:, 2, 1], S[:, 0, 2], S[:, 1, 0]], dim=1)

@torch.jit.script
def yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Get yaw angle from quaternion.

    Args:
        q: The quaternion. Shape is (..., 4).

    Returns:
        The yaw angle. Shape is (...,).
    """
    shape = q.shape
    q = q.view(-1, 4)
    yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    return yaw.view(shape[:-1])