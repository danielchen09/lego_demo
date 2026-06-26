import numpy as np
from scipy.spatial.transform import Rotation as R

def quat_look_at(dir_odom: np.ndarray):
    x_axis = dir_odom / np.linalg.norm(dir_odom)

    world_up = np.array([0.0, 0.0, 1.0])

    # If looking almost straight up/down, pick another up reference.
    if abs(np.dot(x_axis, world_up)) > 0.98:
        world_up = np.array([0.0, 1.0, 0.0])

    y_axis = np.cross(world_up, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    z_axis = np.cross(x_axis, y_axis)

    # Columns are the desired local hand axes expressed in odom.
    rot_mat = np.column_stack([x_axis, y_axis, z_axis])

    # scipy gives [x, y, z, w], Spot wants w, x, y, z.
    qx, qy, qz, qw = R.from_matrix(rot_mat).as_quat()
    return qw, qx, qy, qz


def quat_from_points(
    a,
    b,
    align_axis="+y",
    down_axis="+x",
    down_dir=np.array([0.0, 0.0, -1.0]),
):
    """
    Construct an end-effector quaternion from two points.

    Parameters
    ----------
    a, b:
        3D points in robot body frame.

    align_axis:
        Local EE axis that should align with direction a - b.
        Examples: '+x', '-x', '+y', '-z'

    down_axis:
        Local EE axis that should align with body-frame down direction.
        Examples: '+x', '-x', '+y', '-z'

    quat_order:
        'wxyz' or 'xyzw'

    down_dir:
        Direction considered "down" in body frame.
        Default is [0, 0, -1].

    Returns
    -------
    quat:
        Desired quaternion.

    R_body_ee:
        Rotation matrix whose columns are local EE axes expressed in body frame.
    """

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    down_dir = normalize(down_dir)

    align_idx, align_sign = parse_axis(align_axis)
    down_idx, down_sign = parse_axis(down_axis)

    if align_idx == down_idx:
        raise ValueError("align_axis and down_axis cannot refer to the same local axis")

    # Direction from b to a
    align_raw = a - b
    align_raw = normalize(align_raw)

    # Desired direction of local signed down_axis.
    #
    # Example:
    #   down_axis = '+x' means local +x should point down.
    #   Therefore local +x direction = down_dir.
    #
    #   down_axis = '-x' means local -x should point down.
    #   Therefore local +x direction = -down_dir.
    down_vec_for_positive_local_axis = down_sign * down_dir

    # Desired direction of local signed align_axis.
    #
    # Example:
    #   align_axis = '+y' means local +y should point along a-b.
    #   Therefore local +y direction = align_raw.
    #
    #   align_axis = '-y' means local -y should point along a-b.
    #   Therefore local +y direction = -align_raw.
    align_vec_for_positive_local_axis = align_sign * align_raw

    # The two local axes must be orthogonal.
    # Project align direction onto plane perpendicular to down direction.
    align_vec_for_positive_local_axis = (
        align_vec_for_positive_local_axis
        - np.dot(
            align_vec_for_positive_local_axis,
            down_vec_for_positive_local_axis,
        )
        * down_vec_for_positive_local_axis
    )

    align_vec_for_positive_local_axis = normalize(
        align_vec_for_positive_local_axis
    )

    R_body_ee = complete_basis_from_two_axes(
        down_idx,
        down_vec_for_positive_local_axis,
        align_idx,
        align_vec_for_positive_local_axis,
    )

    quat_xyzw = R.from_matrix(R_body_ee).as_quat()

    quat = np.array([
        quat_xyzw[3],
        quat_xyzw[0],
        quat_xyzw[1],
        quat_xyzw[2],
    ])

    return quat


def normalize(v, eps=1e-9):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Cannot normalize near-zero vector")
    return v / n


def parse_axis(axis):
    """
    Parse '+x', '-x', 'x', '+y', '-z', etc.

    Returns:
        idx: 0 for x, 1 for y, 2 for z
        sign: +1 or -1
    """
    axis = axis.lower().strip()

    if len(axis) == 1:
        sign = 1.0
        name = axis
    elif len(axis) == 2 and axis[0] in ["+", "-"]:
        sign = 1.0 if axis[0] == "+" else -1.0
        name = axis[1]
    else:
        raise ValueError(
            "Axis must be one of: 'x', 'y', 'z', '+x', '-x', '+y', '-y', '+z', '-z'"
        )

    if name == "x":
        idx = 0
    elif name == "y":
        idx = 1
    elif name == "z":
        idx = 2
    else:
        raise ValueError(
            "Axis must be one of: 'x', 'y', 'z', '+x', '-x', '+y', '-y', '+z', '-z'"
        )

    return idx, sign


def complete_basis_from_two_axes(axis_a_idx, axis_a_vec,
                                 axis_b_idx, axis_b_vec):
    """
    Given two local axes and their desired directions in body/world frame,
    construct a valid right-handed rotation matrix.

    Rotation matrix columns are:
        column 0 = local +x direction expressed in body frame
        column 1 = local +y direction expressed in body frame
        column 2 = local +z direction expressed in body frame
    """

    if axis_a_idx == axis_b_idx:
        raise ValueError("The two specified local axes must be different")

    basis = [None, None, None]

    basis[axis_a_idx] = normalize(axis_a_vec)
    basis[axis_b_idx] = normalize(axis_b_vec)

    remaining_idx = ({0, 1, 2} - {axis_a_idx, axis_b_idx}).pop()

    # Complete the right-handed basis.
    #
    # Required convention:
    #   x cross y = z
    #   y cross z = x
    #   z cross x = y
    if remaining_idx == 0:
        # x = y cross z
        basis[0] = np.cross(basis[1], basis[2])
    elif remaining_idx == 1:
        # y = z cross x
        basis[1] = np.cross(basis[2], basis[0])
    elif remaining_idx == 2:
        # z = x cross y
        basis[2] = np.cross(basis[0], basis[1])

    basis[remaining_idx] = normalize(basis[remaining_idx])

    # Re-orthogonalize the second specified axis to remove numerical error.
    # Keep axis_a fixed, recompute axis_b from axis_a and remaining axis.
    if axis_b_idx == 0:
        # x = y cross z
        basis[0] = np.cross(basis[1], basis[2])
    elif axis_b_idx == 1:
        # y = z cross x
        basis[1] = np.cross(basis[2], basis[0])
    elif axis_b_idx == 2:
        # z = x cross y
        basis[2] = np.cross(basis[0], basis[1])

    basis[axis_b_idx] = normalize(basis[axis_b_idx])

    R_body_ee = np.column_stack(basis)

    # Sanity check: should be a valid rotation matrix
    det = np.linalg.det(R_body_ee)
    if det < 0.0:
        raise RuntimeError("Generated a left-handed basis, which should not happen")

    return R_body_ee
