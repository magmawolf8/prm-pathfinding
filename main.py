import time
import numpy as np
from PIL import Image, ImageDraw

from configurations import ConfigHelper, ConfigSpace
from network import Network
from pathfinding import a_star


def make_square(side: int, angle_degrees: int) -> np.ndarray:
    """
    Creates a numpy array representation of a rotated square.

    :param side: Side length of the square
    :param angle_degrees: Angle to rotate by in degrees
    """
    angle = np.radians(angle_degrees)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    bounding_box_size = int(np.ceil(side * (np.abs(cos_a) + np.abs(sin_a)))) + 1
    sq = np.zeros((bounding_box_size, bounding_box_size), dtype=bool)

    # Find the center of this bounding box
    center = (bounding_box_size - 1) / 2.0
    half_size = side / 2.0

    # For each pixel in the bounding box...
    for row in range(bounding_box_size):
        for col in range(bounding_box_size):
            # Translate current pixel coordinates
            d1 = row - center
            d2 = col - center
            # Unrotate the square
            d1_unrot = d1 * cos_a + d2 * sin_a
            d2_unrot = -d1 * sin_a + d2 * cos_a
            # Check if unrotating the square put it within the original box
            if abs(d1_unrot) <= half_size and abs(d2_unrot) <= half_size:
                sq[row, col] = True
    return sq

def draw_square(draw, center, side, angle, fill=None, outline=None):
    """
    Helper function for visualizing the path by drawing a rotated square/rectangle at each node.
    """
    cx, cy = center
    corners = [
        (-side / 2, -side / 2),
        (side / 2, -side / 2),
        (side / 2, side / 2),
        (-side / 2, side / 2)
    ]
    # Convert angle from degrees to radians for trig
    rad = np.deg2rad(angle)

    rotated_corners = []
    for x, y in corners:
        ry = cy + (x * np.cos(rad) - y * np.sin(rad))
        rx = cx + (x * np.sin(rad) + y * np.cos(rad))
        rotated_corners.append((rx, ry))

    draw.polygon(rotated_corners, fill=fill, outline=outline)

def main():
    # 1. Load or build C-space

    image_path = 'FIELD.png'
    image = Image.open(image_path).convert('L')
    obstacle_space = (np.array(image) < 128)
    dim1, dim2 = obstacle_space.shape

    s = 80
    get_thickness = lambda a_d: int(s * (np.abs(np.cos(np.radians(a_d))) + np.abs(np.sin(np.radians(a_d)))))

    # Create ConfigSpace object
    try:
        cs = ConfigSpace.load('field.npy', 'edf.npy')
    except FileNotFoundError:
        print("Creating ConfigSpace object")
        start_time = time.time_ns()
        cs = ConfigSpace(dim1, dim2, 360)
        ch = ConfigHelper(obstacle_space, cs)
        ch.make_config_space(lambda a_d: make_square(s, a_d), get_thickness, get_thickness)
        end_time = time.time_ns()
        print(f"ConfigSpace creation took {(end_time - start_time) // 1000000} ms")

        print("Creating EDF")
        start_time = time.time_ns()
        ch.make_edf()
        end_time = time.time_ns()
        print(f"EDF creation took {(end_time - start_time) // 1000000} ms")

        cs.save('field.npy', 'edf.npy')


    # 2. Build or load the PRM net

    try:
        net = Network.load(cs, 'nodes.pkl', 'n_adj.pkl')
    except FileNotFoundError:
        print("Building PRM object...")
        start_time = time.time_ns()
        print("Adding random nodes...")
        net = Network(cs)
        pdf = lambda edf: np.exp(-edf ** 2.25 / 32768)
        net.add_random_nodes(200000, pdf)
        print("Building tree...")
        net.build_tree()
        print("Connecting nodes...")
        net.connect_all(7)
        end_time = time.time_ns()
        print(f"PRM object creation took {(end_time - start_time) // 1000000} ms")
        net.save('nodes.pkl', 'n_adj.pkl')

    # 3. Add start/goal nodes & run pathfinding
    start_node = net.add_node(512, 512, 60)
    goal_node  = net.add_node(108, 101, 53)
    net.build_tree()
    net.connect_node(start_node, 20)
    net.connect_node(goal_node, 20)

    start_time = time.time_ns()
    path = a_star(net, cs, start_node, goal_node)
    end_time = time.time_ns()
    print(f"A* took {(end_time - start_time) // 1000000} ms")

    print("Start node:", net.nodes[start_node], "End node:", net.nodes[goal_node])
    #
    # # -------------------------------------------------------
    # # 4. Optional: Analyze adjacency or other debug info
    # # -------------------------------------------------------
    # neighbor_counts = [len(net.n_adj[idx]) for idx in range(len(net.nodes))]
    # print("Max neighbors:", max(neighbor_counts))
    # print("Mean neighbors:", np.mean(neighbor_counts))
    # print("Median neighbors:", np.median(neighbor_counts))
    # print("Min neighbors:", min(neighbor_counts))

    # -------------------------------------------------------
    # 5. Draw the result
    # -------------------------------------------------------
    draw_obj = ImageDraw.Draw(image)
    # for node in range(len(net.nodes)):
    #         i, j, k = net.nodes[node]
    #         draw_obj.point((j, i), fill="black")
    #         for _, node_two in net.n_adj[node]:
    #             d1, d2, d3 = net.nodes[node_two]
    #             draw_obj.line([(j, i), (d2, d1)], fill="black", width = 1)
    # Draw the path as a series of rotated squares
    for node_idx in path:
        r, c, a = net.nodes[node_idx]
        draw_square(draw_obj, (c, r), s, a, outline="black")

    image.show()
    image.save('path.png')

if __name__ == "__main__":
    main()
