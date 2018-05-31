import itertools
import random
from itertools import starmap

import cv2
import numpy as np


# Perform edge detection
def hough_transform(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    kernel = np.ones((15, 15), np.uint8)

    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)  # Open (erode, then dilate)
    edges = cv2.Canny(opening, 50, 150, apertureSize=3)  # Canny edge detection

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)  # Hough line detection
    line_image = np.copy(img) * 0

    # for line in lines:
    #     print(line[0][0])
    #     for x1, y1, x2, y2 in line:
    #         cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    for l in lines:
        rho, theta = l[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
    cv2.imshow("win", lines_edges)
    cv2.waitKey(0)
    hough_lines = []
    # Lines are represented by rho, theta; converted to endpoint notation
    if lines is not None:
        for line in lines:
            hough_lines.extend(list(starmap(endpoints, line)))

    print(hough_lines)

    return hough_lines


def endpoints(rho, theta):
    a = np.cos(theta)
    b = np.sin(theta)
    x_0 = a * rho
    y_0 = b * rho
    x_1 = int(x_0 + 1000 * (-b))
    y_1 = int(y_0 + 1000 * (a))
    x_2 = int(x_0 - 1000 * (-b))
    y_2 = int(y_0 - 1000 * (a))

    return ((x_1, y_1), (x_2, y_2))


# Random sampling of lines
def sample_lines(lines, size):
    if size > len(lines):
        size = len(lines)
    return random.sample(lines, size)


def det(a, b):
    return a[0] * b[1] - a[1] * b[0]


# Find intersection point of two lines (not segments!)
def line_intersection(line1, line2):
    x_diff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    y_diff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    div = det(x_diff, y_diff)
    if div == 0:
        return None  # Lines don't cross

    d = (det(*line1), det(*line2))
    x = det(d, x_diff) / div
    y = det(d, y_diff) / div

    return x, y


# Find intersections between multiple lines (not line segments!)
def find_intersections(lines):
    intersections = []
    for i, line_1 in enumerate(lines):
        for line_2 in lines[i + 1:]:
            if not line_1 == line_2:
                intersection = line_intersection(line_1, line_2)
                if intersection:  # If lines cross, then add
                    intersections.append(intersection)

    # print("\t===== Intersections =====")
    # for inter in intersections:
    #     print(inter)
    # print("\t========== End ==========")

    return intersections


# Given intersections, find the grid where most intersections occur and treat as vanishing point
def find_vanishing_point(img, grid_size, intersections):
    # Image dimensions
    image_height = img.shape[0]
    image_width = img.shape[1]

    # Grid dimensions
    grid_rows = (image_height // grid_size) + 1
    grid_columns = (image_width // grid_size) + 1

    # Current cell with most intersection points
    max_intersections = 0
    best_cell = (0.0, 0.0)

    for i, j in itertools.product(range(grid_rows), range(grid_columns)):
        cell_left = i * grid_size
        cell_right = (i + 1) * grid_size
        cell_bottom = j * grid_size
        cell_top = (j + 1) * grid_size
        cv2.rectangle(img, (cell_left, cell_bottom), (cell_right, cell_top), (0, 0, 255), 10)

        current_intersections = 0  # Number of intersections in the current cell
        for x, y in intersections:
            if cell_left < x < cell_right and cell_bottom < y < cell_top:
                current_intersections += 1

        # Current cell has more intersections that previous cell (better)
        if current_intersections > max_intersections:
            max_intersections = current_intersections
            best_cell = ((cell_left + cell_right) / 2, (cell_bottom + cell_top) / 2)
            print("Best Cell:", best_cell)

    if best_cell[0] != None and best_cell[1] != None:
        rx1 = int(best_cell[0] - grid_size / 2)
        ry1 = int(best_cell[1] - grid_size / 2)
        rx2 = int(best_cell[0] + grid_size / 2)
        ry2 = int(best_cell[1] + grid_size / 2)
        cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (0, 255, 0), 10)
        cv2.imwrite('/pictures/output/center.jpg', img)

    return best_cell


def ransac_vanishing_point_detection(lines, distance=5, iterations=200):
    """
    Calculate the vanishing point of the road markers.
    :param lines: the lines defined as a [x1, y1, x2, y2] (4xN array, where N is the number of lines)
    :param distance: the distance (in pixels) to determine if a measurement is consistent
    :param iterations: the number of RANSAC iterations to use
    :return: Coordinates of the road vanishing point
    """

    # Number of lines
    N = len(lines)

    # Maximum number of consistant lines
    max_num_consistent_lines = 0

    # Best fit point
    best_fit = None

    # Loop through all of the iterations to find the most consistent value
    for i in range(0, iterations):

        # Randomly choosing the lines
        random_indices = np.random.choice(N, 2, replace=False)
        i1 = random_indices[0]
        i2 = random_indices[1]
        x1, y1, x2, y2 = lines[i1]
        x3, y3, x4, y4 = lines[i2]

        # Find the intersection point
        x_intersect = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
        y_intersect = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))

        this_num_consistent_lines = 0

        # Find the distance between the intersection and all of the other lines
        for i2 in range(0, N):

            tx1, ty1, tx2, ty2 = lines[i2]
            this_distance = (np.abs((ty2-ty1)*x_intersect - (tx2-tx1)*y_intersect + tx2*ty1 - ty2*tx1)
                             / np.sqrt((ty2-ty1)**2 + (tx2-tx1)**2))

            if this_distance < distance:
                this_num_consistent_lines += 1

        # If it's greater, make this the new x, y intersect
        if this_num_consistent_lines > max_num_consistent_lines:
            best_fit = int(x_intersect), int(y_intersect)
            max_num_consistent_lines = this_num_consistent_lines

    return best_fit

def get_vanishing_points(img, intersections, iterations=200, distance_threshold=5):
    """
       Calculate the vanishing point of the road markers.
       :param lines: the lines defined as a [x1, y1, x2, y2] (4xN array, where N is the number of lines)
       :param distance: the distance (in pixels) to determine if a measurement is consistent
       :param iterations: the number of RANSAC iterations to use
       :return: Coordinates of the road vanishing point
       """

    # Number of intersections
    N = len(intersections)

    # Maximum number of consistant lines
    max_num_consistent_lines = 0

    # Best fit point
    best_fit = None

    # Loop through all of the iterations to find the most consistent value
    for i in range(0, iterations):

        # Randomly choosing the lines
        random_indices = np.random.choice(N, 2, replace=False)
        i1 = random_indices[0]
        i2 = random_indices[1]
        itersec1 = intersections[random_indices[0]]
        itersec2 = intersections[random_indices[1]]
        x1, y1, x2, y2 = lines[i1]
        x3, y3, x4, y4 = lines[i2]

        # Find the intersection point
        x_intersect = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
                    (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
        y_intersect = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
                    (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))

        this_num_consistent_lines = 0

        # Find the distance between the intersection and all of the other lines
        for i2 in range(0, N):

            tx1, ty1, tx2, ty2 = lines[i2]
            this_distance = (np.abs((ty2 - ty1) * x_intersect - (tx2 - tx1) * y_intersect + tx2 * ty1 - ty2 * tx1)
                             / np.sqrt((ty2 - ty1) ** 2 + (tx2 - tx1) ** 2))

            if this_distance < distance_threshold:
                this_num_consistent_lines += 1

        # If it's greater, make this the new x, y intersect
        if this_num_consistent_lines > max_num_consistent_lines:
            best_fit = int(x_intersect), int(y_intersect)
            max_num_consistent_lines = this_num_consistent_lines

    return best_fit

def ransac_vanishing_point(edgelets, num_ransac_iter=2000, threshold_inlier=5):
    """Estimate vanishing point using Ransac.
    Parameters
    ----------
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
    num_ransac_iter: int
        Number of iterations to run ransac.
    threshold_inlier: float
        threshold to be used for computing inliers in degrees.
    Returns
    -------
    best_model: ndarry of shape (3,)
        Best model for vanishing point estimated.
    Reference
    ---------
    Chaudhury, Krishnendu, Stephen DiVerdi, and Sergey Ioffe.
    "Auto-rectification of user photos." 2014 IEEE International Conference on
    Image Processing (ICIP). IEEE, 2014.
    """
    locations, directions, strengths = edgelets
    lines = edgelet_lines(edgelets)

    num_pts = strengths.size

    arg_sort = np.argsort(-strengths)
    first_index_space = arg_sort[:num_pts // 5]
    second_index_space = arg_sort[:num_pts // 2]

    best_model = None
    best_votes = np.zeros(num_pts)

    for ransac_iter in range(num_ransac_iter):
        ind1 = np.random.choice(first_index_space)
        ind2 = np.random.choice(second_index_space)

        l1 = lines[ind1]
        l2 = lines[ind2]

        current_model = np.cross(l1, l2)

        if np.sum(current_model**2) < 1 or current_model[2] == 0:
            # reject degenerate candidates
            continue

        current_votes = compute_votes(
            edgelets, current_model, threshold_inlier)

        if current_votes.sum() > best_votes.sum():
            best_model = current_model
            best_votes = current_votes
            logging.info("Current best model has {} votes at iteration {}".format(
                current_votes.sum(), ransac_iter))

    return best_model
