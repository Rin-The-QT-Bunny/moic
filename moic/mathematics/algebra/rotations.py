import numpy 
import math

"""
rotations use euler angle, quaternion and more.
"""

import itertools
import numpy as np

"""
Rotations
=========
Note: these have caused many subtle bugs in the past.
Be careful while updating these methods and while using them in clever ways.
See MuJoCo documentations here: http://mujoco_utils.org/book/modeling.html#COrientation
Conventions
-----------
    - All function accept batches as well as indivisual rotations
    - All rotation conventions math respective MuHoCo defaults
    - All angles are in radians
    - Matrices follow LR convention
    - Euler Angles are all relative with 'xyz' axes ordering
    - See specific representation for more information
Representations
---------------
Euler
    There are many euler angle frames -- here we will strive to use the default
        in MuJoCo, which is eulerseq = 'xyz'
    This frame is a relative rotation frame, about x, y and z axes in order.
        Relative rotating means that after we rotate about x, then new
        (rotated) y, and the same for z.
Quaternions
    These are defined in terms of rotation (angle) about a unit vector (x, y, z)
    We use the following <q0, q1, q2, q3> convention:
            q0 = cos(angle / 2)
            q1 = sin(angle / 2) * x
            q2 = sin(angle / 2) * y
            q3 = sin(angle / 2) * z
        This is also sometimes called qw, qx, qy, qz.
"""

# for testing whether a number is close to zero
_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0

def as_rotation(r):
    """convert a 3x3 matrix or a quaternion into a standard 3x3 matrix representation."""
    if isinstance(r,np.ndarray) and r.shape == (3,3): return r
    if isinstance(r,np.ndarray) and r.shape == (4,): return quat2mat(r)
    raise ValueError("Invalid rotation{}".format(r))

def mat2quat(mat):
    """ Convert Rotation Matrix to Quaternion.  See rotation.py for notes """
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    Qxx, Qyx, Qzx = mat[..., 0, 0], mat[..., 0, 1], mat[..., 0, 2]
    Qxy, Qyy, Qzy = mat[..., 1, 0], mat[..., 1, 1], mat[..., 1, 2]
    Qxz, Qyz, Qzz = mat[..., 2, 0], mat[..., 2, 1], mat[..., 2, 2]
    # Fill only lower half of symmetric matrix
    K = np.zeros(mat.shape[:-2] + (4, 4), dtype=np.float64)
    K[..., 0, 0] = Qxx - Qyy - Qzz
    K[..., 1, 0] = Qyx + Qxy
    K[..., 1, 1] = Qyy - Qxx - Qzz
    K[..., 2, 0] = Qzx + Qxz
    K[..., 2, 1] = Qzy + Qyz
    K[..., 2, 2] = Qzz - Qxx - Qyy
    K[..., 3, 0] = Qyz - Qzy
    K[..., 3, 1] = Qzx - Qxz
    K[..., 3, 2] = Qxy - Qyx
    K[..., 3, 3] = Qxx + Qyy + Qzz
    K /= 3.0
    # TODO: vectorize this -- probably could be made faster
    q = np.empty(K.shape[:-2] + (4,))
    it = np.nditer(q[..., 0], flags=['multi_index'])
    while not it.finished:
        # Use Hermitian eigenvectors, values for speed
        vals, vecs = np.linalg.eigh(K[it.multi_index])
        # Select largest eigenvector, reorder to w,x,y,z quaternion
        q[it.multi_index] = vecs[[3, 0, 1, 2], np.argmax(vals)]
        # Prefer quaternion with positive w
        # (q * -1 corresponds to same rotation as q)
        if q[it.multi_index][0] < 0:
            q[it.multi_index] *= -1
        it.iternext()
    return q

def quat2mat(quat):
    """ Convert Quaternion to Euler Angles.  See rotation.py for notes """
    quat = np.asarray(quat, dtype=np.float64)
    assert quat.shape[-1] == 4, "Invalid shape quat {}".format(quat)

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = np.sum(quat * quat, axis=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0] = 1.0 - (yY + zZ)
    mat[..., 0, 1] = xY - wZ
    mat[..., 0, 2] = xZ + wY
    mat[..., 1, 0] = xY + wZ
    mat[..., 1, 1] = 1.0 - (xX + zZ)
    mat[..., 1, 2] = yZ - wX
    mat[..., 2, 0] = xZ - wY
    mat[..., 2, 1] = yZ + wX
    mat[..., 2, 2] = 1.0 - (xX + yY)
    return np.where((Nq > _FLOAT_EPS)[..., np.newaxis, np.newaxis], mat, np.eye(3))


def quat_conjugate(q):
    inv_q = -q
    inv_q[..., 0] *= -1
    return inv_q