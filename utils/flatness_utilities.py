import torch 
from typing import Tuple
import omni.isaac.lab.utils.math as isaac_math_utils

@torch.jit.script
def s2_projection(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Compute the stereographic projection of the unit sphere S^2 onto the plane R^3. 
    The projection is defined as follows: 
    (x, y, z) = (2X/ 1+X^2+Y^2, 2Y/ 1+X^2+Y^2, (X^2 + Y^2 - 1)/ 1+X^2+Y^2)
    Args:
        X (torch.Tensor): The x-coordinate of the unit sphere S^2, of shape N, 1.
        Y (torch.Tensor): The y-coordinate of the unit sphere S^2, of shape N, 1.
    Returns:
        torch.Tensor: The projection of the unit sphere S^2 onto the plane R^3, of shape N, 3.
    """
    X2 = X**2
    Y2 = Y**2
    denom = 1 + X2 + Y2
    s = torch.stack(((2*X)/denom, (2*Y)/denom, -(X2 + Y2 - 1)/denom), dim=1)
    return isaac_math_utils.normalize(s)

@torch.jit.script
def inv_s2_projection(S: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the inverse stereographic projection of the plane R^3 onto the unit sphere S^2.
    The inverse projection is defined as follows:
    (X, Y) = (x/(1+z), y/(1+z))
    Args:
        S (torch.Tensor): The projection of the unit sphere S^2 onto the plane R^3, of shape N, 3.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The x-coordinate and y-coordinate of the unit sphere S^2, of shape N, 1 each.
    """ 

    x = S[:, 0]
    y = S[:, 1]
    z = S[:, 2]
    denom = 1 + z
    return (x/denom).view(-1, 1), (y/denom).view(-1, 1)

@torch.jit.script
def H1(psi: torch.Tensor) -> torch.Tensor:
    """
    Construct the local section of the fiber bundle according ton the paper (H1):
    "Dynamically Feasible Task Space Planning for Underactuated Aerial Manipulators" by Jake Welde, et al. [https://ieeexplore.ieee.org/document/9325015]

    H1 = [[cos(psi), -sin(psi), 0],
          [sin(psi), cos(psi), 0],
          [0, 0, 1]]

    Args:
        psi (torch.Tensor): The orientation of the quadrotor, of shape N, 1.
    Returns:
        torch.Tensor: The local section of the fiber bundle, of shape N, 3, 3
    """

    return torch.stack((torch.cos(psi), -torch.sin(psi), torch.zeros_like(psi),
                        torch.sin(psi), torch.cos(psi), torch.zeros_like(psi),
                        torch.zeros_like(psi), torch.zeros_like(psi), torch.ones_like(psi)), dim=1).view(-1, 3, 3)

@torch.jit.script
def H2(s: torch.Tensor) -> torch.Tensor:
    """ 
    Construct the local section of the fiber bundle according ton the paper (H2):
    "Dynamically Feasible Task Space Planning for Underactuated Aerial Manipulators" by Jake Welde, et al. [https://ieeexplore.ieee.org/document/9325015]

    H2 = [[1 - s_1^2 / 1+s_3, -s_1*s_2/1+s_3, s_1],
          [-s_1*s_2/1+s_3, 1 - s_2^2 / 1+s_3, s_2],
          [-s_1, -s_2, s_3]]

    Args:
        psi (torch.Tensor): The orientation of the quadrotor, of shape N, 1.
    Returns:
        torch.Tensor: The local section of the fiber bundle, of shape N, 3, 3
    """
    s = isaac_math_utils.normalize(s)
    denom = 1 + s[:, 2]
    return torch.stack((1 - (s[:, 0]**2)/denom, (-s[:, 0]*s[:, 1])/denom, s[:, 0],
                        (-s[:, 0]*s[:, 1])/denom, 1 - (s[:, 1]**2)/denom, s[:, 1],
                        -s[:, 0], -s[:, 1], s[:, 2]), dim=1).view(-1, 3, 3)

@torch.jit.script
def H1_dot(psi: torch.Tensor, psi_dot: torch.Tensor) -> torch.Tensor:
    """
    Compute the time derivative of H1 w.r.t. psi, given psi_dot.
    H1_dot = dH1/dpsi * psi_dot

    Args:
        psi (torch.Tensor): The orientation of the quadrotor, of shape N, 1.
        psi_dot (torch.Tensor): The angular velocity of the quadrotor, of shape N, 1.
    Returns:
        torch.Tensor: The time derivative of H1 w.r.t. psi, of shape N, 3, 3.
    """
    c = torch.cos(psi)
    s = torch.sin(psi)
    
    # dH1/dpsi
    dH1_dpsi = torch.stack((
        torch.stack((-s, -c, torch.zeros_like(psi)), dim=-1),
        torch.stack((c, -s, torch.zeros_like(psi)), dim=-1),
        torch.stack((torch.zeros_like(psi), torch.zeros_like(psi), torch.zeros_like(psi)), dim=-1)
    ), dim=-2)

    # H1_dot
    return dH1_dpsi * psi_dot.unsqueeze(-1).unsqueeze(-1)


@torch.jit.script
def H2_dot(s: torch.Tensor, s_dot: torch.Tensor) -> torch.Tensor:
    """
    Compute the time derivative of H2 w.r.t. s, given s_dot.
    H2_dot = sum_i dH2/ds_i * s_dot_i

    Args:
        s (torch.Tensor): The projection of the unit sphere S^2 onto the plane R^3, of shape N, 3.
        s_dot (torch.Tensor): The time derivative of s, of shape N, 3.
    Returns:
        torch.Tensor: The time derivative of H2 w.r.t. s, of shape N, 3, 3.
    """
    # We assume s is already normalized. If s changes over time, ensure s_dot is consistent.
    # Extract components
    s = isaac_math_utils.normalize(s)
    sx = s[:,0]
    sy = s[:,1]
    sz = s[:,2]
    sx_dot = s_dot[:,0]
    sy_dot = s_dot[:,1]
    sz_dot = s_dot[:,2]
    
    D = 1 + sz
    D2 = D**2
    
    # Partial derivatives for each element of H2
    # For convenience, we'll compute all partials and then combine them:
    
    # H2[0,0]
    dH2_00_sx = -2*sx/D
    dH2_00_sy = torch.zeros_like(sx)
    dH2_00_sz = (sx**2)/(D2)
    
    # H2[0,1]
    dH2_01_sx = -sy/D
    dH2_01_sy = -sx/D
    dH2_01_sz = (sx*sy)/D2
    
    # H2[0,2]
    dH2_02_sx = torch.ones_like(sx)
    dH2_02_sy = torch.zeros_like(sx)
    dH2_02_sz = torch.zeros_like(sx)
    
    # H2[1,0] = same pattern as H2[0,1]
    dH2_10_sx = -sy/D
    dH2_10_sy = -sx/D
    dH2_10_sz = (sx*sy)/D2
    
    # H2[1,1]
    dH2_11_sx = torch.zeros_like(sx)
    dH2_11_sy = -2*sy/D
    dH2_11_sz = (sy**2)/D2
    
    # H2[1,2]
    dH2_12_sx = torch.zeros_like(sx)
    dH2_12_sy = torch.ones_like(sx)
    dH2_12_sz = torch.zeros_like(sx)
    
    # H2[2,0]
    dH2_20_sx = -torch.ones_like(sx)
    dH2_20_sy = torch.zeros_like(sx)
    dH2_20_sz = torch.zeros_like(sx)
    
    # H2[2,1]
    dH2_21_sx = torch.zeros_like(sx)
    dH2_21_sy = -torch.ones_like(sx)
    dH2_21_sz = torch.zeros_like(sx)
    
    # H2[2,2]
    dH2_22_sx = torch.zeros_like(sx)
    dH2_22_sy = torch.zeros_like(sx)
    dH2_22_sz = torch.ones_like(sx)
    
    # Combine partial derivatives with s_dot
    # For each element: H2_dot[i,j] = sum_over_k (dH2[i,j]/ds_k * s_dot_k)
    
    H2_dot_00 = dH2_00_sx*sx_dot + dH2_00_sy*sy_dot + dH2_00_sz*sz_dot
    H2_dot_01 = dH2_01_sx*sx_dot + dH2_01_sy*sy_dot + dH2_01_sz*sz_dot
    H2_dot_02 = dH2_02_sx*sx_dot + dH2_02_sy*sy_dot + dH2_02_sz*sz_dot
    
    H2_dot_10 = dH2_10_sx*sx_dot + dH2_10_sy*sy_dot + dH2_10_sz*sz_dot
    H2_dot_11 = dH2_11_sx*sx_dot + dH2_11_sy*sy_dot + dH2_11_sz*sz_dot
    H2_dot_12 = dH2_12_sx*sx_dot + dH2_12_sy*sy_dot + dH2_12_sz*sz_dot
    
    H2_dot_20 = dH2_20_sx*sx_dot + dH2_20_sy*sy_dot + dH2_20_sz*sz_dot
    H2_dot_21 = dH2_21_sx*sx_dot + dH2_21_sy*sy_dot + dH2_21_sz*sz_dot
    H2_dot_22 = dH2_22_sx*sx_dot + dH2_22_sy*sy_dot + dH2_22_sz*sz_dot
    
    return torch.stack((H2_dot_00, H2_dot_01, H2_dot_02,
                        H2_dot_10, H2_dot_11, H2_dot_12,
                        H2_dot_20, H2_dot_21, H2_dot_22), dim=1).view(-1, 3, 3)

@torch.jit.script
def getRotationFromShape(s: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """
    Construct rotation matrix R as H2(s) * H1(psi) according to the paper:
    "Dynamically Feasible Task Space Planning for Underactuated Aerial Manipulators" by Jake Welde, et al. [https://ieeexplore.ieee.org/document/9325015]

    Args:
        s (torch.Tensor): The projection of the unit sphere S^2 onto the plane R^3, of shape N, 3.
        psi (torch.Tensor): The orientation of the quadrotor, of shape N, 1.
    Returns:
        torch.Tensor: The rotation matrix R, of shape N, 3, 3.
    """
    return torch.bmm(H2(s), H1(psi))

@torch.jit.script
def getAttitudeFromRotationAndYaw(R: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """
    Construct the shape s as H2^-1(R * H1^-1(psi)) according to the paper:
    "Dynamically Feasible Task Space Planning for Underactuated Aerial Manipulators" by Jake Welde, et al. [https://ieeexplore.ieee.org/document/9325015]

    Args:
        R (torch.Tensor): The rotation matrix R, of shape N, 3, 3.
        psi (torch.Tensor): The orientation of the quadrotor, of shape N, 1.
    Returns:
        torch.Tensor: The shape s, of shape N, 3.
    """
    H2 = torch.bmm(R, H1(psi).transpose(-2, -1))
    x_des, y_des = inv_s2_projection(H2[:, :, 2])
    return torch.stack((x_des, y_des, psi), dim=1)

@torch.jit.script
def getRotationDotFromShape(s: torch.Tensor, psi: torch.Tensor, s_dot: torch.Tensor, psi_dot: torch.Tensor) -> torch.Tensor:
    """
    Compute the time derivative of the rotation matrix R w.r.t. time, given s_dot and psi_dot.
    R_dot = dR/dt = H2_dot(s) * H1(psi) + H2(s) * H1_dot(psi) * psi_dot

    Args:
        s (torch.Tensor): The projection of the unit sphere S^2 onto the plane R^3, of shape N, 3.
        psi (torch.Tensor): The orientation of the quadrotor, of shape N, 1.
        s_dot (torch.Tensor): The time derivative of s, of shape N, 3.
        psi_dot (torch.Tensor): The angular velocity of the quadrotor, of shape N, 1.
    Returns:
        torch.Tensor: The time derivative of the rotation matrix R w.r.t. time, of shape N, 3, 3.
    """
    return torch.bmm(H2_dot(s, s_dot), H1(psi)) + torch.bmm(H2(s), H1_dot(psi, psi_dot))

@torch.jit.script
def getRotationDDotFromShape(s: torch.Tensor, s_dot: torch.Tensor, s_ddot: torch.Tensor, psi: torch.Tensor, psi_dot: torch.Tensor, psi_ddot: torch.Tensor) -> torch.Tensor:
    """
    Computes R_ddot for a batch of inputs.

    Args:
        s (Tensor): Shape (N, 3), normalized state vectors.
        s_dot (Tensor): Shape (N, 3), first derivative of s.
        s_ddot (Tensor): Shape (N, 3), second derivative of s.
        psi (Tensor): Shape (N, 1), orientation angles.
        psi_dot (Tensor): Shape (N, 1), first derivative of psi.
        psi_ddot (Tensor): Shape (N, 1), second derivative of psi.

    Returns:
        Tensor: R_ddot matrices of shape (N, 3, 3).
    """
    # Extract components of s, s_dot, and s_ddot
    s1, s2, s3 = s[:, 0], s[:, 1], s[:, 2]
    s1_dot, s2_dot, s3_dot = s_dot[:, 0], s_dot[:, 1], s_dot[:, 2]
    s1_ddot, s2_ddot, s3_ddot = s_ddot[:, 0], s_ddot[:, 1], s_ddot[:, 2]

    # Extract psi, psi_dot, psi_ddot
    psi = psi.squeeze(-1)
    psi_dot = psi_dot.squeeze(-1)
    psi_ddot = psi_ddot.squeeze(-1)

    # Precompute useful terms
    cos_psi = torch.cos(psi)
    sin_psi = torch.sin(psi)
    denom = 1 + s3
    denom2 = denom ** 2
    denom3 = denom ** 3

    # Initialize R_ddot
    R_ddot = torch.zeros((s.shape[0], 3, 3), device=s.device, dtype=s.dtype)

    # Compute R_ddot[0][0]
    R_ddot[:, 0, 0] = (
        -s3_dot * (psi_dot * ((s1 * s1 * sin_psi / denom2) - (s1 * s2 * cos_psi / denom2))
                   + s3_dot * ((s1 * s1 * cos_psi * 2 / denom3) + (s1 * s2 * sin_psi * 2 / denom3))
                   - s1_dot * (s1 * cos_psi * 2 / denom2 + s2 * sin_psi / denom2)
                   - s1 * s2_dot * sin_psi / denom2)
        + psi_ddot * (sin_psi * (s1 * s1 / denom - 1) - (s1 * s2 * cos_psi) / denom)
        - psi_dot * (-psi_dot * (cos_psi * (s1 * s1 / denom - 1) + (s1 * s2 * sin_psi) / denom)
                     + s3_dot * (s1 * s1 * sin_psi / denom2 - s1 * s2 * cos_psi / denom2)
                     + s1_dot * ((s2 * cos_psi) / denom - (s1 * sin_psi * 2) / denom)
                     + (s1 * s2_dot * cos_psi) / denom)
        - s2_dot * ((s1_dot * sin_psi) / denom + (psi_dot * s1 * cos_psi) / denom - (s1 * s3_dot * sin_psi) / denom2)
        + s3_ddot * ((s1 * s1 * cos_psi) / denom2 + (s1 * s2 * sin_psi) / denom2)
        - s1_dot * (psi_dot * ((s2 * cos_psi) / denom - (s1 * sin_psi * 2) / denom)
                    - s3_dot * (s1 * cos_psi * 2 / denom2 + s2 * sin_psi / denom2)
                    + (s1_dot * cos_psi * 2) / denom + (s2_dot * sin_psi) / denom)
        - s1_ddot * ((s1 * cos_psi * 2) / denom + (s2 * sin_psi) / denom)
        - (s1 * s2_ddot * sin_psi) / denom
    )

    # Compute R_ddot[0][1]
    R_ddot[:, 0, 1] = (
        psi_ddot * (cos_psi * ((s1 * s1) / denom - 1) + (s1 * s2 * sin_psi) / denom)
        + s2_dot * (-(s1_dot * cos_psi) / denom + (s1 * s3_dot * cos_psi) / denom2 + (psi_dot * s1 * sin_psi) / denom)
        - psi_dot * (psi_dot * (sin_psi * ((s1 * s1) / denom - 1) - (s1 * s2 * cos_psi) / denom)
                     + s3_dot * ((s1 * s1 * cos_psi) / denom2 + (s1 * s2 * sin_psi) / denom2)
                     - s1_dot * ((s1 * cos_psi * 2) / denom + (s2 * sin_psi) / denom)
                     - (s1 * s2_dot * sin_psi) / denom)
        - s3_ddot * ((s1 * s1 * sin_psi) / denom2 - (s1 * s2 * cos_psi) / denom2)
        + s1_dot * (psi_dot * ((s1 * cos_psi * 2) / denom + (s2 * sin_psi) / denom)
                    + s3_dot * ((s2 * cos_psi) / denom2 - (s1 * sin_psi * 2) / denom2)
                    - (s2_dot * cos_psi) / denom + (s1_dot * sin_psi * 2) / denom)
        - s1_ddot * ((s2 * cos_psi) / denom - (s1 * sin_psi * 2) / denom)
        + s3_dot * (-psi_dot * ((s1 * s1 * cos_psi) / denom2 + (s1 * s2 * sin_psi) / denom2)
                    + s3_dot * ((s1 * s1 * sin_psi * 2) / denom3 - (s1 * s2 * cos_psi * 2) / denom3)
                    + s1_dot * ((s2 * cos_psi) / denom2 - (s1 * sin_psi * 2) / denom2)
                    + (s1 * s2_dot * cos_psi) / denom2)
        - (s1 * s2_ddot * cos_psi) / denom
    )

    # Compute R_ddot[0][2]
    R_ddot[:, 0, 2] = s1_ddot

    # Compute R_ddot[1][0]
    R_ddot[:, 1, 0] = (
        -psi_ddot * (cos_psi * ((s2 * s2) / denom - 1) - (s1 * s2 * sin_psi) / denom)
        + s1_dot * (-(s2_dot * cos_psi) / denom + (s2 * s3_dot * cos_psi) / denom2 + (psi_dot * s2 * sin_psi) / denom)
        + psi_dot * (psi_dot * (sin_psi * ((s2 * s2) / denom - 1) + (s1 * s2 * cos_psi) / denom)
                     + s3_dot * ((s2 * s2 * cos_psi) / denom2 - (s1 * s2 * sin_psi) / denom2)
                     - s2_dot * ((s2 * cos_psi * 2) / denom - (s1 * sin_psi) / denom)
                     + (s2 * s1_dot * sin_psi) / denom)
        + s3_ddot * ((s2 * s2 * sin_psi) / denom2 + (s1 * s2 * cos_psi) / denom2)
        - s2_dot * (psi_dot * ((s2 * cos_psi * 2) / denom - (s1 * sin_psi) / denom)
                    - s3_dot * ((s1 * cos_psi) / denom2 + (s2 * sin_psi * 2) / denom2)
                    + (s1_dot * cos_psi) / denom + (s2_dot * sin_psi * 2) / denom)
        - s2_ddot * ((s1 * cos_psi) / denom + (s2 * sin_psi * 2) / denom)
        + s3_dot * (psi_dot * ((s2 * s2 * cos_psi) / denom2 - (s1 * s2 * sin_psi) / denom2)
                    - s3_dot * ((s2 * s2 * sin_psi * 2) / denom3 + (s1 * s2 * cos_psi * 2) / denom3)
                    + s2_dot * ((s1 * cos_psi) / denom2 + (s2 * sin_psi * 2) / denom2)
                    + (s2 * s1_dot * cos_psi) / denom2)
        - (s2 * s1_ddot * cos_psi) / denom
    )

    # Compute R_ddot[1][1]
    R_ddot[:, 1, 1] = (
        -s3_dot * (psi_dot * ((s2 * s2 * sin_psi) / denom2 + (s1 * s2 * cos_psi) / denom2)
                    + s3_dot * ((s2 * s2 * cos_psi * 2) / denom3 - (s1 * s2 * sin_psi * 2) / denom3)
                    - s2_dot * ((s2 * cos_psi * 2) / denom2 - (s1 * sin_psi) / denom2)
                    + (s2 * s1_dot * sin_psi) / denom2)
        + psi_ddot * (sin_psi * ((s2 * s2) / denom - 1) + (s1 * s2 * cos_psi) / denom)
        + psi_dot * (psi_dot * (cos_psi * ((s2 * s2) / denom - 1) - (s1 * s2 * sin_psi) / denom)
                     - s3_dot * ((s2 * s2 * sin_psi) / denom2 + (s1 * s2 * cos_psi) / denom2)
                     + s2_dot * ((s1 * cos_psi) / denom + (s2 * sin_psi * 2) / denom)
                     + (s2 * s1_dot * cos_psi) / denom)
        + s1_dot * ((s2_dot * sin_psi) / denom + (psi_dot * s2 * cos_psi) / denom - (s2 * s3_dot * sin_psi) / denom2)
        + s3_ddot * ((s2 * s2 * cos_psi) / denom2 - (s1 * s2 * sin_psi) / denom2)
        + s2_dot * (psi_dot * ((s1 * cos_psi) / denom + (s2 * sin_psi * 2) / denom)
                    + s3_dot * ((s2 * cos_psi * 2) / denom2 - (s1 * sin_psi) / denom2)
                    - (s2_dot * cos_psi * 2) / denom + (s1_dot * sin_psi) / denom)
        - s2_ddot * ((s2 * cos_psi * 2) / denom - (s1 * sin_psi) / denom)
        + (s2 * s1_ddot * sin_psi) / denom
    )

    # Compute R_ddot[1][2]
    R_ddot[:, 1, 2] = s2_ddot

    # Compute R_ddot[2][0]
    R_ddot[:, 2, 0] = (
        psi_dot * (psi_dot * (s1 * cos_psi + s2 * sin_psi) - s2_dot * cos_psi + s1_dot * sin_psi)
        - psi_ddot * (s2 * cos_psi - s1 * sin_psi)
        - s1_ddot * cos_psi - s2_ddot * sin_psi
        - psi_dot * s2_dot * cos_psi + psi_dot * s1_dot * sin_psi
    )

    # Compute R_ddot[2][1]
    R_ddot[:, 2, 1] = (
        psi_dot * (psi_dot * (s2 * cos_psi - s1 * sin_psi) + s1_dot * cos_psi + s2_dot * sin_psi)
        + psi_ddot * (s1 * cos_psi + s2 * sin_psi)
        - s2_ddot * cos_psi + s1_ddot * sin_psi
        + psi_dot * s1_dot * cos_psi + psi_dot * s2_dot * sin_psi
    )

    # Compute R_ddot[2][2]
    R_ddot[:, 2, 2] = s3_ddot

    return R_ddot

@torch.jit.script
def computeShapeDerivativesFromForce(F_des: torch.Tensor, x_dddot: torch.Tensor, x_ddddot:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Function to compute the desired shape of the Lee Geometric controller and 2 derivatives of the desired shape

    args:
        F_des: The desired force computed with PD gains. Shape is (N, 3)
        x_dddot: The reference acceleration. Shape is (N, 3)
        x_ddddot: The reference jerk. Shape is (N, 3)
    returns:
        s_des: The desired shape. Shape is (N, 3)
        s_dot_des: The desired shape derivative. Shape is (N, 3)
        s_ddot_des: The desired shape second derivative. Shape is (N, 3)
    """

    id_3 = torch.eye(3, device=F_des.device).unsqueeze(0).repeat(F_des.shape[0], 1, 1)
    denom = torch.linalg.norm(F_des, dim=1, keepdim=True)
    s_des = isaac_math_utils.normalize(F_des)
    s_dot_des = (torch.bmm(id_3 - s_des.unsqueeze(-1) * s_des.unsqueeze(1), x_dddot.unsqueeze(-1))).squeeze(-1) / denom
    num1 = (torch.bmm(id_3 - s_des.unsqueeze(-1) * s_des.unsqueeze(1), x_ddddot.unsqueeze(-1))).squeeze(-1)
    num2 = torch.bmm(2*s_dot_des.unsqueeze(-1)*s_des.unsqueeze(1) + s_des.unsqueeze(-1)*s_dot_des.unsqueeze(1), x_dddot.unsqueeze(-1)).squeeze(-1)
    s_ddot_des = (num1 - num2)/ denom

    return s_des, s_dot_des, s_ddot_des
