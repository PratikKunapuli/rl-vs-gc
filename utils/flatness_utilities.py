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