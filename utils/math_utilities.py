import torch
import omni.isaac.lab.utils.math as isaac_math_utils
from typing import Tuple

def exp_so3(S):
    pass

def matrix_log(S):
    pass

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
def hat_map(v):
    """Convert vector to skew-symmetric matrix.

    Args:
        v: The vector. Shape is (N, 3).

    Returns:
        The skew-symmetric matrix representation of the vector. Shape is (N, 3, 3).
    """
    return isaac_math_utils.skew_symmetric_matrix(v)

@torch.jit.script
def yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Get yaw angle from quaternion.

    Args:
        q: The quaternion. Shape is (..., 4).
        q = [w, x, y, z]

    Returns:
        The yaw angle. Shape is (...,).
    """
    shape = q.shape
    q = q.view(-1, 4)
    yaw = torch.atan2(2.0 * (q[:, 3] * q[:, 0] + q[:, 1] * q[:, 2]), -1.0 + 2.0*(q[:,0]**2 + q[:,1]**2))
    # yaw = torch.atan2(2.0 * (q[:, 2] * q[:, 3] + q[:, 0] * q[:, 1]), q[:, 0]**2 - q[:, 1]**2 - q[:, 2]**2 + q[:, 3]**2)
    # yaw3 = torch.atan2(2.0 * (q[:, 1] * q[:, 0] + q[:, 2] * q[:, 3]), 1.0 - 2.0*(q[:,0]**2 + q[:,1]**2))
    return yaw.view(shape[:-1])

def yaw_error_from_quats(q1: torch.Tensor, q2: torch.Tensor, dof:int) -> torch.Tensor:
    """Get yaw error between two quaternions.

    Args:
        q1: The first quaternion. Shape is (..., 4).
        q2: The second quaternion. Shape is (..., 4).

    Returns:
        The yaw error. Shape is (...,).
    """
    shape1 = q1.shape
    shape2 = q2.shape

    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)

    
    #Find vector "b2" that is the y-axis of the rotated frame
    b1 = isaac_math_utils.quat_rotate(q1, torch.tensor([[0.0, 1.0, 0.0]], device=q1.device).tile((q1.shape[0], 1)))
    b2 = isaac_math_utils.quat_rotate(q2, torch.tensor([[0.0, 1.0, 0.0]], device=q1.device).tile((q2.shape[0], 1)))

    if dof == 0:
        b1[:,2] = 0.0
        b2[:,2] = 0.0

    b1_norm = torch.norm(b1, dim=-1)
    b2_norm = torch.norm(b2, dim=-1)
    operand = (b1*b2).sum(dim=1) / (b1_norm * b2_norm)
    return torch.arccos(torch.clamp(operand, -1.0+1e-8, 1.0-1e-8)).view(shape1[:-1])

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
def compute_desired_pose_from_transform(
    goal_pos_w: torch.Tensor,
    goal_ori_w: torch.Tensor,
    pos_transform: torch.Tensor,
    num_joints: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the desired position and yaw from the given transform.
    
    Args:
        goal_pos_w (Tensor): Goal positions in world frame (batch_size, 3).
        goal_ori_w (Tensor): Goal orientations as quaternions (batch_size, 4).
        pos_transform (Tensor): Position transforms (batch_size, 3).
        num_joints (int): Number of joints.

    Returns:
        Tuple[Tensor, Tensor]: Desired positions and yaws.
    """
    batch_size = goal_ori_w.shape[0]

    # Rotate the y-axis vector by the goal orientations
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=goal_ori_w.device).unsqueeze(0).expand(batch_size, -1)
    b2 = isaac_math_utils.quat_rotate(goal_ori_w, y_axis)

    # Set the z-component to zero if num_joints == 0
    if num_joints == 0:
        b2 = b2.clone()  # Avoid modifying the original tensor
        b2[:, 2] = 0.0

    b2 = isaac_math_utils.normalize(b2)

    # Compute the desired yaw angle
    yaw_desired = torch.atan2(b2[:, 1], b2[:, 0]) - torch.pi / 2
    yaw_desired = isaac_math_utils.wrap_to_pi(yaw_desired)

    # Compute the desired position
    pos_transform_norm = torch.linalg.norm(pos_transform, dim=1, keepdim=True)
    displacement = pos_transform_norm * (-b2)
    pos_desired = goal_pos_w + displacement

    return pos_desired, yaw_desired

    