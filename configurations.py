import numpy as np
import scipy.ndimage as ndimage
from typing import Callable


class ConfigHelper:
    """
    Generates 3D configuration spaces from 2D obstacles and 2D robot structure
    """

    def __init__(self, obstacle_space, cs):
        """
        :param obstacle_space: A 2D numpy array of shape (rows, cols)
                            True indicates blocked space, False indicates free space
        :param cs: ConfigSpace object to be filled
        """
        self.obstacle_space = obstacle_space
        self.cs = cs
        self.dim1, self.dim2, self.dim3 = cs.blocked_space.shape

        if (self.dim1, self.dim2) != obstacle_space.shape:
            raise ValueError("Dimensions do not match")
        if 360 % self.dim3 != 0:
            raise ValueError("3rd dimension size does not divide 360 degrees")

    def make_config_space(self, get_robot: Callable[[int], np.ndarray],
                          get_row_thickness: Callable[[int], int],
                          get_col_thickness: Callable[[int], int]):
        """
        Fills the ConfigSpace object.

        :param get_robot: Function to get a robot configuration at angle_degrees
        :param get_row_thickness: Function to get a robot's row thickness at angle_degrees
        :param get_col_thickness: Function to get a robot's column thickness at angle_degrees
        """
        for angle_degrees in range(0, 360, 360 // self.dim3):
            expanded = ndimage.binary_dilation(
                self.obstacle_space, structure=get_robot(angle_degrees))

            # Restrict c-space where the rotated square would collide with the border
            row_rad = get_row_thickness(angle_degrees) // 2 + 1
            col_rad = get_col_thickness(angle_degrees) // 2 + 1
            expanded[:row_rad, :] = True
            expanded[-row_rad:, :] = True
            expanded[:, :col_rad] = True
            expanded[:, -col_rad:] = True
            self.cs.set_slice(angle_degrees, expanded)
            print(f"{angle_degrees} degree obstacle dilation complete")
        self.cs.is_blocked_space_generated = True

    def make_edf(self):
        self.cs.edf = ndimage.distance_transform_edt(~self.cs.get_blocked_space())



class ConfigSpace:
    """
    Represents the 3D configuration space of a robot (x, y, angle)
    """

    def __init__(self, dim1, dim2, dim3, blocked_space: np.ndarray = None, edf: np.ndarray = None):
        """
        :param dim1: First dimension number
        :param dim2: Second dimension number
        :param dim3: Third dimension number
        :param blocked_space: A 3D numpy array of shape (rows, cols, angles)
                            True indicates blocked space, False indicates free space
        :param edf: A 3D numpy array of shape (rows, cols, angles)
                            Indicating each entry's distance from the nearest obstacle
        """
        if blocked_space is None:
            self.blocked_space = np.zeros((dim1, dim2, dim3)).astype(bool)
            self.dim1, self.dim2, self.dim3 = dim1, dim2, dim3
            self.is_blocked_space_generated = False
        else:
            self.blocked_space = blocked_space
            self.dim1, self.dim2, self.dim3 = blocked_space.shape
            self.is_blocked_space_generated = True

        self.edf = edf

    @staticmethod
    def load(blocked_space_fn: str, edf_fn: str):
        """
        Loads a configuration space from a numpy array
        in the current working directory
        :param blocked_space_fn: the file name for the blocked space
        :param edf_fn: the file name for the Euclidean Distance Field
        """
        blocked_space = np.load(blocked_space_fn).astype(bool)
        dim1, dim2, dim3 = blocked_space.shape
        return ConfigSpace(dim1, dim2, dim3, blocked_space, np.load(edf_fn))

    def save(self, blocked_space_fn: str, edf_fn: str):
        """
        Saves a configuration space as a numpy array
        in the current working directory
        :param blocked_space_fn: the desired file name for the blocked space
        :param edf_fn: the desired file name for the Euclidean Distance Field
        """
        np.save(blocked_space_fn, self.get_blocked_space())
        np.save(edf_fn, self.get_edf())

    def get_blocked_space(self):
        if not self.is_blocked_space_generated:
            raise Exception("No blocked space generated")
        else:
            return self.blocked_space

    def get_edf(self):
        if self.edf is None:
            raise Exception("No Euclidean Distance Field")
        else:
            return self.edf

    def set_slice(self, angle_degrees: int, slice_2d: np.ndarray):
        """
        Saves a 2D slice into the static space for a particular angle.
        """
        self.blocked_space[:, :, angle_degrees] = slice_2d

    # @staticmethod
    # def distance(r1: int, c1: int, a1: int, r2: int, c2: int, a2: int) -> float:
    #     """
    #     Computes a Euclidean distance in the 3D space, treating angles with modular arithmetic
    #     """
    #     angle_diff = a2 - a1
    #     if angle_diff >= 180:
    #         angle_diff -= 360
    #     elif angle_diff < -180:
    #         angle_diff += 360
    #
    #     return np.sqrt((r2 - r1)**2 + (c2 - c1)**2 + angle_diff**2)

    @staticmethod
    def distance(r1: int, c1: int, a1: int, r2: int, c2: int, a2: int) -> float:
        """
        Computes a Euclidean distance in the 3D space with angles wrapping
        """
        angle_diff = a2 - a1
        if angle_diff >= 180:
            angle_diff -= 360
        elif angle_diff < -180:
            angle_diff += 360

        return np.sqrt((r2 - r1)**2 + (c2 - c1)**2 + angle_diff**2)

    @staticmethod
    def generate_edge(r1: int, c1: int, a1: int, r2: int, c2: int, a2: int):
        """
        Generates a 3D Bresenham-like line of lattice points from (r1, c1, a1) to (r2, c2, a2)

        :yield: (row, col, angle) tuples along the line
        """
        dr = abs(r2 - r1)
        dc = abs(c2 - c1)
        da = abs(a2 - a1)

        r, c, a = r1, c1, a1
        step_r = 1 if r2 > r1 else -1
        step_c = 1 if c2 > c1 else -1
        step_a = 1 if a2 > a1 else -1

        # Determine the dominant direction
        if dr >= dc and dr >= da:  # row dominant
            p_c = 2 * dc - dr
            p_a = 2 * da - dr
            while r != r2:
                yield r, c, a
                r += step_r
                if p_c >= 0:
                    c += step_c
                    p_c -= 2 * dr
                if p_a >= 0:
                    a += step_a
                    p_a -= 2 * dr
                p_c += 2 * dc
                p_a += 2 * da
        elif dc >= dr and dc >= da:  # col dominant
            p_c = 2 * dr - dc
            p_a = 2 * da - dc
            while c != c2:
                yield r, c, a
                c += step_c
                if p_c >= 0:
                    r += step_r
                    p_c -= 2 * dc
                if p_a >= 0:
                    a += step_a
                    p_a -= 2 * dc
                p_c += 2 * dr
                p_a += 2 * da
        else:  # angle dominant
            p_c = 2 * dc - da
            p_a = 2 * dr - da
            while a != a2:
                yield r, c, a
                a += step_a
                if p_c >= 0:
                    c += step_c
                    p_c -= 2 * da
                if p_a >= 0:
                    r += step_r
                    p_a -= 2 * da
                p_c += 2 * dc
                p_a += 2 * dr
        yield r, c, a

    def is_blocked_edge(self, r1: int, c1: int, a1: int, r2: int, c2: int, a2: int) -> bool:
        """
        Checks the edge from (r1, c1, a1) to (r2, c2, a2) to see if it is blocked by this config.

        :return: True if the edge is blocked, False otherwise.
        """
        for rr, cc, aa in ConfigSpace.generate_edge(r1, c1, a1, r2, c2, a2):
            if 0 <= rr < self.dim1 and 0 <= cc < self.dim2 and 0 <= aa < self.dim3:
                if self.get_blocked_space()[rr, cc, aa]:
                    return True
        return False