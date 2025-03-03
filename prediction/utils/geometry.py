# Copyright (c) Carizon. All rights reserved.

import math

import torch


def angle_between_2d_vectors(
    ctr_vector: torch.Tensor, nbr_vector: torch.Tensor
) -> torch.Tensor:
    """Calculate the angle between two 2D vectors in radiant.

    Args:
        ctr_vector(torch.Tensor): The 2D vector chosen to be centered.
        nbr_vector(torch.Tensor): The 2D vector chosen to be the neighbor.
    Returns:
        torch.Tensor: The angle between the vectors in radians.
    """
    return torch.atan2(
        ctr_vector[..., 0] * nbr_vector[..., 1]
        - ctr_vector[..., 1] * nbr_vector[..., 0],
        (ctr_vector[..., :2] * nbr_vector[..., :2]).sum(dim=-1),
    )


def angle_between_3d_vectors(
    ctr_vector: torch.Tensor, nbr_vector: torch.Tensor
) -> torch.Tensor:
    """Calculate the angle between two 3D vectors in radiant.

    Args:
        ctr_vector(torch.Tensor): The 3D vector chosen to be centered.
        nbr_vector(torch.Tensor): The 3D vector chosen to be the neighbor.
    Returns:
        torch.Tensor: The angle between the vectors in radians.
    """
    return torch.atan2(
        torch.cross(ctr_vector, nbr_vector, dim=-1).norm(p=2, dim=-1),
        (ctr_vector * nbr_vector).sum(dim=-1),
    )


def side_to_directed_lineseg(
    query_point: torch.Tensor,
    start_point: torch.Tensor,
    end_point: torch.Tensor,
) -> str:
    """
    Determine the relative position of a query point to a directed
    line segment.

    This function calculates the orientation of the `query_point` with
    respect to the directed line segment from `start_point` to `end_point`.
    It uses the cross product of vectors to determine if the point is
    to the left, right, or on the line segment.

    Args:
        query_point (torch.Tensor): The point whose position is being
            determined.
        start_point (torch.Tensor): The starting point of the line segment.
        end_point (torch.Tensor): The ending point of the line segment.

    Returns:
        str: "LEFT" if the query point is to the left of the line segment,
             "RIGHT" if it is to the right,
             "CENTER" if it is on the line segment.
    """
    cond = (end_point[0] - start_point[0]) * (
        query_point[1] - start_point[1]
    ) - (end_point[1] - start_point[1]) * (query_point[0] - start_point[0])
    if cond > 0:
        return "LEFT"
    elif cond < 0:
        return "RIGHT"
    else:
        return "CENTER"


def side_to_directed_lineseg_v2(
    points: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor
) -> torch.Tensor:
    """
    Vectorized version to determine if a point is on the left or right \
        of a directed line segment.

    Args:
        points: [B, 2] - target points.
        starts: [B, M, 2] - starting points of lane boundaries.
        ends: [B, M, 2] - ending points of lane boundaries.

    Returns a binary mask: 1 if the point is on the left, \
        0 if on the right or collinear.
    """
    # Compute the condition for left/right using cross product
    cond = (ends[..., 0] - starts[..., 0]) * (
        points[..., 1].unsqueeze(1) - starts[..., 1]
    ) - (ends[..., 1] - starts[..., 1]) * (
        points[..., 0].unsqueeze(1) - starts[..., 0]
    )
    return cond > 0


def wrap_angle(
    angle: torch.Tensor, min_val: float = -math.pi, max_val: float = math.pi
) -> torch.Tensor:
    """
    Wrap an angle to be within the specified range [min_val, max_val).

    This function ensures that the angle remains within the given bounds
    by wrapping it around. The result will be in the range [min_val, max_val).

    Args:
        angle (torch.Tensor): The angle(s) to be wrapped.
        min_val (float, optional): The minimum value of the wrapping range.
            Default is -math.pi.
        max_val (float, optional): The maximum value of the wrapping range.
            Default is math.pi.

    Returns:
        torch.Tensor: The wrapped angle(s) within the range [min_val, max_val).
    """
    return min_val + (angle + max_val) % (max_val - min_val)


def transform_vcs_to_acs(
    fut_traj: torch.Tensor, origin: torch.Tensor, theta: torch.Tensor
) -> torch.Tensor:
    """Transform vcs coordinates into agent-centric coordinates.

    This function ensures that the angle remains within the given bounds
    by wrapping it around. The result will be in the range [min_val, max_val).

    Args:
        fut_traj (torch.Tensor): The trajectory to be transformed.
        origin (torch.Tensor): The origin of the coordinate.
            Default is -math.pi.
        theta (torch.Tensor): The anglular theta to be tranformed.

    Returns:
        torch.Tensor: The trajectory in agent-centric coordinate system.
    """
    cos, sin = theta.cos(), theta.sin()
    rot_mat = torch.tensor([[cos, -sin], [sin, cos]], dtype=torch.float32)
    acs_fut = torch.matmul(fut_traj - origin, rot_mat)
    return acs_fut


def centerline_to_polygon(
    centerline: np.ndarray, width_scaling_factor: float = 1.0
) -> np.ndarray:
    """
    Convert a lane centerline polyline into a rough polygon of the
    lane's area.

    On average, a lane is 3.8 meters in width. Thus, we allow 1.9 m
        on each side. We use this as the length of the hypotenuse of
        a right triangle, and compute the other two legs to find the
        scaled x and y displacement.

    Args:
       centerline: Numpy array of shape (N,2).
       width_scaling_factor: Multiplier that scales 3.8 meters
                             to get the lane width.
       visualize: Save a figure showing the the output polygon.

    Returns:
       polygon: Numpy array of shape (2N+1,2), with duplicate
                first and last vertices.
    """
    # eliminate duplicates
    _, inds = np.unique(centerline, axis=0, return_index=True)
    # does not return indices in sorted order
    inds = np.sort(inds)
    centerline = centerline[inds]

    dx = np.gradient(centerline[:, 0])
    dy = np.gradient(centerline[:, 1])

    # compute the normal at each point
    slopes = dy / dx
    inv_slopes = -1.0 / slopes

    thetas = np.arctan(inv_slopes)
    x_disp = 3.8 * width_scaling_factor / 2.0 * np.cos(thetas)
    y_disp = 3.8 * width_scaling_factor / 2.0 * np.sin(thetas)

    displacement = np.hstack([x_disp[:, np.newaxis], y_disp[:, np.newaxis]])
    right_centerline = centerline + displacement
    left_centerline = centerline - displacement

    # right centerline position depends on sign of dx and dy
    subtract_cond1 = np.logical_and(dx > 0, dy < 0)
    subtract_cond2 = np.logical_and(dx > 0, dy > 0)
    subtract_cond = np.logical_or(subtract_cond1, subtract_cond2)
    left_centerline, right_centerline = swap_left_and_right(
        subtract_cond, left_centerline, right_centerline
    )

    # right centerline also depended on if we added or subtracted y
    neg_disp_cond = displacement[:, 1] > 0
    left_centerline, right_centerline = swap_left_and_right(
        neg_disp_cond, left_centerline, right_centerline
    )
    # return the polygon
    return convert_lane_boundaries_to_polygon(
        right_centerline, left_centerline
    )


def convert_lane_boundaries_to_polygon(
    right_lane_bounds: np.ndarray,
    left_lane_bounds: np.ndarray,
) -> np.ndarray:
    """
    Take a left and right lane boundary and make a polygon of the
    lane segment, closing both ends of the segment.

    These polygons have the last vertex repeated
        (that is, first vertex == last vertex).

    Args:
       right_lane_bounds: Right lane boundary points. Shape is (N, 2).
       left_lane_bounds: Left lane boundary points.

    Returns:
       polygon: Numpy array of shape (2N+1,2).
    """
    assert right_lane_bounds.shape[0] == left_lane_bounds.shape[0]
    polygon = np.vstack([right_lane_bounds, left_lane_bounds[::-1]])
    polygon = np.vstack([polygon, right_lane_bounds[0]])
    return polygon


def swap_left_and_right(
    condition: np.ndarray,
    left_centerline: np.ndarray,
    right_centerline: np.ndarray,
) -> np.ndarray:
    """
    Swap points in left and right centerline according to condition.

    Args:
       condition: Numpy array of shape (N,) of type boolean. Where true,
                  swap the values in the left and right centerlines.
       left_centerline: The left centerline, whose points should be swapped
                  with the right centerline.
       right_centerline: The right centerline.

    Returns:
       left_centerline
       right_centerline
    """

    right_swap_indices = right_centerline[condition]
    left_swap_indices = left_centerline[condition]

    left_centerline[condition] = right_swap_indices
    right_centerline[condition] = left_swap_indices
    return left_centerline, right_centerline
