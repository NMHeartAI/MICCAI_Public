import numpy as np
from skimage import measure
from shapely.ops import unary_union, split, nearest_points
from shapely.affinity import translate, rotate
from shapely.geometry import Polygon, LineString, Point, box
from shapely.geometry.base import GeometrySequence
import matplotlib.pyplot as plt
from utils.util import *

cmaps = plt.get_cmap('tab20')

def result_matrix_multi(results, num=200, path=''):
    gt_mask = np.zeros([num, 27, 512, 512])
    for i in range(num):
        category_list = results[i].boxes.cls.to('cpu').numpy()
        N = np.shape(category_list)[0]
        if N != 0:
            mask = results[i].masks.data.to('cpu').numpy()
            im_id = int(results[i].path.replace(path, '')[:-4])-1
            for j in range(N):
                gt_mask[im_id][int(category_list[j])+1] = (gt_mask[im_id][int(category_list[j])+1] > 0) | (mask[j] > 0) * (category_list[j]+1).astype(int)
    return  gt_mask

def result_matrix(results, num=200, path=''):
    gt_mask = np.zeros([num, 27, 512, 512])
    for i in range(num):
        category_list = results[i].boxes.cls.to('cpu').numpy()
        N = np.shape(category_list)[0]
        if N != 0:
            mask = results[i].masks.data.to('cpu').numpy()
            im_id = int(results[i].path.replace(path, '')[:-4])-1
            for j in range(N):
                gt_mask[im_id][int(category_list[j])+1] = (gt_mask[im_id][int(category_list[j])+1] == 1) | (mask[j] == 1)
    return  gt_mask

def result_matrix_single(results, path=''):
    gt_mask = np.zeros([27, 512, 512])

    category_list = results.boxes.cls.to('cpu').numpy()
    N = np.shape(category_list)[0]
    if N != 0:
        mask = results.masks.data.to('cpu').numpy()
        for j in range(N):
            gt_mask[int(category_list[j])+1] = (gt_mask[int(category_list[j])+1] == 1) | (mask[j] == 1)
    return gt_mask

def result_matrix_single_probability(results, prob_threshold=0.3, path=""):
    gt_mask = np.zeros([27, 512, 512])

    category_list = results.boxes.cls.to("cpu").numpy()
    class_probabilities = results.boxes.conf.to("cpu").numpy()
    N = np.shape(category_list)[0]
    if N != 0:
        mask = results.masks.data.to("cpu").numpy()
        for j in range(N):
            class_cat = category_list[j]
            class_prob = class_probabilities[j]

            if class_prob > prob_threshold:
                temp_mask = (mask[j] == 1) * class_prob
                best_mask = np.stack([gt_mask[int(class_cat) + 1], temp_mask]).max(
                    axis=0
                )
                gt_mask[int(class_cat) + 1] = best_mask

    return gt_mask

def lowest_coord(poly):
    return Point(max(poly.exterior.coords, key=lambda coord: coord[1]))

def highest_coord(poly):
    return Point(min(poly.exterior.coords, key=lambda coord: coord[1]))

def left_coord(poly):
    return Point(min(poly.exterior.coords))

def right_coord(poly):
    return Point(max(poly.exterior.coords))

def bounding_box_mask(y_min, y_max, x_min, x_max):
    mask = np.zeros((512, 512), dtype=np.int32)

    # Generate coordinates for all pixels
    y_coords, x_coords = np.indices(mask.shape)

    # Use boolean indexing to set values inside the bounding box to 1 and outside to 0
    mask[(x_coords >= x_min) & (x_coords <= x_max) & (y_coords >= y_min) & (y_coords <= y_max)] = 1

    return mask

def get_line_angle(line1, line2):
    coords1 = np.array(line1.coords)
    coords2 = np.array(line2.coords)

    # Calculate vectors along the LineStrings
    vector1 = coords1[-1] - coords1[0]
    vector2 = coords2[-1] - coords2[0]

    # Calculate the dot product and magnitudes of the vectors
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    # Calculate the cosine of the angle between the vectors
    cosine_theta = dot_product / (magnitude1 * magnitude2)

    # Calculate the angle in radians
    angle_radians = np.arccos(cosine_theta)

    # Determine the direction of rotation using the cross product
    cross_product = np.cross(vector1, vector2)

    # Check if line2 crosses under line1 (cross_product is negative)
    if cross_product > 0:
        angle_degrees = -np.degrees(angle_radians)
    else:
        # If cross_product is positive or zero, return positive angle (crosses over the top of line)
        angle_degrees = np.degrees(angle_radians)

    return angle_degrees

def get_angle(polygon1, polygon2):
    # Calculate the rotation angle of the rectangles

    # Convert polygons to minimum rotated rectangles
    rectangle1 = polygon1.minimum_rotated_rectangle
    rectangle2 = polygon2.minimum_rotated_rectangle

    angle_difference = get_line_angle(extend_linestring(get_principal_axis(rectangle1), 500),
                                      extend_linestring(get_principal_axis(rectangle2), 500))

    return angle_difference

def get_major_axis_angle(shape_1, plot_=False):
    polygon = shape_1
    # Get the coordinates of the polygon's exterior
    exterior_coords = np.array(polygon.exterior.coords)

    # Calculate the centroid of the polygon
    centroid = polygon.centroid

    # Calculate the covariance matrix
    cov_matrix = np.cov(exterior_coords, rowvar=False)

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Find the index of the eigenvalue with the maximum value
    max_eigenvalue_index = np.argmax(eigenvalues)

    # Get the corresponding eigenvector
    principal_axis = eigenvectors[:, max_eigenvalue_index]

    # Calculate the angle with the x-axis (assuming a flat line)
    angle_rad = np.arctan2(principal_axis[1], principal_axis[0])

    # Convert the angle to degrees
    angle_deg = np.degrees(angle_rad)

    if plot_:
        p("Principal Axis:", principal_axis)
        p("Angle (degrees):", angle_deg)

        # Plotting for visualization
        plt.figure(figsize=(6, 6))
        plt.plot(exterior_coords[:, 0], exterior_coords[:, 1], 'b-', alpha=0.4)
        plt.scatter(centroid.x, centroid.y, c='r', label='Centroid')
        plt.plot([centroid.x, centroid.x + principal_axis[0]],
                 [centroid.y, centroid.y + principal_axis[1]], 'ro-', label='Principal Axis')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Polygon and Principal Axis')
        plt.legend()
        plt.show()

    return angle_deg

def get_intersection_angle(p1, p2, box_size=100, plot_=False, img=None):
    # Step 1: Check if p1 and p2 intersect
    def plot_intersection(p1, p2, img):
        plot_multiple_polygons(
            [p1, p2],
            [
                extend_linestring(get_principal_axis(p1), 500),
                extend_linestring(get_principal_axis(p2), 500),
            ],
            img=img,
        )

    try:
        if p1.intersects(p2):
            # Step 2: Find the intersection region
            intersection_region = p1.intersection(p2)

            # Step 3: Create a 100x100 box centered around the intersection
            minx, miny, maxx, maxy = intersection_region.bounds
            center_x = (minx + maxx) / 2
            center_y = (miny + maxy) / 2
            roi_box = box(
                center_x - box_size / 2,
                center_y - box_size / 2,
                center_x + box_size / 2,
                center_y + box_size / 2,
            )

            # Step 4: Crop p1 and p2 using the box
            p1_cropped = p1.intersection(roi_box)
            p2_cropped = p2.intersection(roi_box)

            # Step 5: Compute minimum rotating rectangles
            p1_ = p1_cropped.minimum_rotated_rectangle
            p2_ = p2_cropped.minimum_rotated_rectangle

            angle_between_p1_and_p2 = get_angle(p1_, p2_)

            p("Angle between p1_ and p2_:", angle_between_p1_and_p2)

            if plot_:
                p("plotting")
                plot_intersection(p1_, p2_, img)

        elif segment_connected(p1, p2):
            # Step 1: Find the closest points on the edges of p1 and p2
            closest_points = nearest_points(p1, p2)

            # Step 2: Create a 100x100 box centered around the closest points
            center_x = (closest_points[0].x + closest_points[1].x) / 2
            center_y = (closest_points[0].y + closest_points[1].y) / 2
            roi_box = box(
                center_x - box_size / 2,
                center_y - box_size / 2,
                center_x + box_size / 2,
                center_y + box_size / 2,
            )

            # Step 3: Crop p1 and p2 using the box
            p1_cropped = p1.intersection(roi_box)
            p2_cropped = p2.intersection(roi_box)

            # Step 4: Compute minimum rotating rectangles
            p1_ = p1_cropped.minimum_rotated_rectangle
            p2_ = p2_cropped.minimum_rotated_rectangle

            angle_between_p1_and_p2 = get_angle(p1_, p2_)

            p("Angle between p1_ and p2_:", angle_between_p1_and_p2)

            if plot_:
                p("plotting")
                plot_intersection(p1_, p2_, img)

            return angle_between_p1_and_p2

        if angle_between_p1_and_p2 == 0:
            p("p1 and p2 do not intersect.")
            if get_principal_axis_error(
                p1_.minimum_rotated_rectangle
            ) or get_principal_axis_error(p2_.minimum_rotated_rectangle):
                return None
            else:
                return 0
        else:
            return angle_between_p1_and_p2
    except Exception as e:
        p(f"p1 and p2 do not intersect. {e}")
        return None

def find_segment_topmost_idx(segs):
    if len(segs) == 1:
        return 0
    else:
        return np.argmin([_.centroid.y for _ in segs])

def find_segment_leftmost_idx(segs):
    if len(segs) == 1:
        return 0
    else:
        return np.argmin([_.centroid.x for _ in segs])

def shortest_edges(rectangle):
    exterior_ring = rectangle.exterior

    # Initialize variables to store the two shortest edges
    shortest_edge1 = None
    shortest_edge2 = None

    # Initialize variables to store the lengths of the edges
    min_length1 = float("inf")
    min_length2 = float("inf")

    # Iterate through the coordinates of the exterior ring
    for i in range(len(exterior_ring.coords) - 1):
        # Create a LineString from two consecutive points
        edge = LineString([exterior_ring.coords[i], exterior_ring.coords[i + 1]])

        # Calculate the length of the edge
        edge_length = edge.length

        # Check if the current edge is shorter than the shortest edges found so far
        if edge_length < min_length1:
            shortest_edge2 = shortest_edge1
            min_length2 = min_length1

            shortest_edge1 = edge
            min_length1 = edge_length
        elif edge_length < min_length2:
            shortest_edge2 = edge
            min_length2 = edge_length
    return shortest_edge1, shortest_edge2

def longest_edges(rectangle):
    exterior_ring = rectangle.exterior

    # Initialize variables to store the two shortest edges
    longest_edge1 = None
    longest_edge2 = None

    # Initialize variables to store the lengths of the edges
    max_length1 = float("-inf")
    max_length2 = float("-inf")

    # Iterate through the coordinates of the exterior ring
    for i in range(len(exterior_ring.coords) - 1):
        # Create a LineString from two consecutive points
        edge = LineString([exterior_ring.coords[i], exterior_ring.coords[i + 1]])

        # Calculate the length of the edge
        edge_length = edge.length

        # Check if the current edge is shorter than the shortest edges found so far
        if edge_length > max_length1:
            longest_edge2 = longest_edge1
            max_length2 = max_length1

            longest_edge1 = edge
            max_length1 = edge_length
        elif edge_length > max_length2:
            longest_edge2 = edge
            max_length2 = edge_length
    return longest_edge1, longest_edge2


def short_edge_intersection(polygon1, polygon2, box_size=100):
    # Make sure you have polygon16b defined somewhere
    p1, p2 = polygon1, polygon2

    if p1.intersects(p2):
        # Step 2: Find the intersection region
        intersection_region = p1.intersection(p2)

        # Step 3: Create a 100x100 box centered around the intersection
        minx, miny, maxx, maxy = intersection_region.bounds
        center_x = (minx + maxx) / 2
        center_y = (miny + maxy) / 2
        roi_box = box(
            center_x - box_size / 2,
            center_y - box_size / 2,
            center_x + box_size / 2,
            center_y + box_size / 2,
        )

        # Step 4: Crop p1 and p2 using the box
        p1_cropped = p1.intersection(roi_box)
        p2_cropped = p2.intersection(roi_box)

        # Step 5: Compute minimum rotating rectangles
        p1_ = p1_cropped.minimum_rotated_rectangle
        p2_ = p2_cropped.minimum_rotated_rectangle

    elif segment_connected(p1, p2):
        # Step 1: Find the closest points on the edges of p1 and p2
        closest_points = nearest_points(p1, p2)

        # Step 2: Create a 100x100 box centered around the closest points
        center_x = (closest_points[0].x + closest_points[1].x) / 2
        center_y = (closest_points[0].y + closest_points[1].y) / 2
        roi_box = box(
            center_x - box_size / 2,
            center_y - box_size / 2,
            center_x + box_size / 2,
            center_y + box_size / 2,
        )

        # Step 3: Crop p1 and p2 using the box
        p1_cropped = p1.intersection(roi_box)
        p2_cropped = p2.intersection(roi_box)

        # Step 4: Compute minimum rotating rectangles
        p1_ = p1_cropped.minimum_rotated_rectangle
        p2_ = p2_cropped.minimum_rotated_rectangle

    # Convert polygons to minimum rotated rectangles
    rectangle1 = p1_.minimum_rotated_rectangle
    rectangle2 = p2_.minimum_rotated_rectangle

    exterior_ring = rectangle1.exterior

    # Initialize variables to store the two shortest edges
    shortest_edge1 = None
    shortest_edge2 = None

    # Initialize variables to store the lengths of the edges
    min_length1 = float("inf")
    min_length2 = float("inf")

    # Iterate through the coordinates of the exterior ring
    for i in range(len(exterior_ring.coords) - 1):
        # Create a LineString from two consecutive points
        edge = LineString([exterior_ring.coords[i], exterior_ring.coords[i + 1]])

        # Calculate the length of the edge
        edge_length = edge.length

        # Check if the current edge is shorter than the shortest edges found so far
        if edge_length < min_length1:
            shortest_edge2 = shortest_edge1
            min_length2 = min_length1

            shortest_edge1 = edge
            min_length1 = edge_length
        elif edge_length < min_length2:
            shortest_edge2 = edge
            min_length2 = edge_length

    short_edge_in_intersection = rectangle2.intersects(
        shortest_edge1
    ) or rectangle2.intersects(shortest_edge2)

    if short_edge_in_intersection:
        p("The intersection includes the short exterior edge of the rectangle.")
    else:
        p("The intersection does not include the short exterior edge of the rectangle.")

    return short_edge_in_intersection

def merge_close_polygons_with_idx(polygons, threshold=10):
    merged_polygons = []
    merged_indices = []  # List to store indices of merged polygons
    unmerged_indices = []  # List to store indices of unmerged polygons

    for i, polygon in enumerate(polygons):
        # Check if the current polygon overlaps with any merged polygon
        merged = False

        for j, merged_polygon in enumerate(merged_polygons):
            if polygon.distance(merged_polygon) < threshold:
                merged_polygons[j] = unary_union([polygon, merged_polygon])
                merged = True
                merged_indices.append(i)  # Record the index of the merged polygon
                break

        if not merged:
            merged_polygons.append(polygon)
            unmerged_indices.append(i)  # Record the index of the unmerged polygon

    return merged_polygons, merged_indices, unmerged_indices

def plot_multiple_polygons(list_polygons, list_lines=[], img=None):

    # Assuming you have a Shapely Polygon object named 'polygon'
    fig, ax = plt.subplots(figsize=(6, 6), dpi=85)
    # Reverse the y-coordinates (flip the image vertically)

    if img is None:
        pass
    else:
        reversed_image = img[::-1, :]
        ax.imshow(
            reversed_image, cmap="gray"
        )  ## Slightly adjust the figsize for better alignment
    for i, polygon in enumerate(list_polygons):

        color_ = "red" if i == 0 else cmaps(i)
        # Reverse the y-coordinates of the polygon
        reversed_polygon_coords = [(x, 512 - y) for x, y in polygon.exterior.coords]

        # Plot the polygon with reversed y-coordinates
        x, y = zip(*reversed_polygon_coords)
        ax.plot(x, y, color=color_, alpha=0.7, linewidth=2, label="Polygon")

        # Get the bounding box coordinates
        minx, miny, maxx, maxy = polygon.bounds
        bbox_corners = [
            (x, 512 - y)
            for x, y in [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
        ]

        # Plot the bounding box corners
        bbox_x, bbox_y = zip(*bbox_corners)
        ax.scatter(
            bbox_x, bbox_y, color=["red", "green", "blue", "purple"], marker="x", s=50, label="Bounding Box Corners"
        )

    for line in list_lines:
        p("line coords:", line)
        line_x, line_y = zip(*[(x, 512 - y) for x, y in line.coords])
        ax.plot(
            line_x,
            line_y,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Bisecting Line",
        )

        for i in range(len(line.coords) - 1):
            start_point = Point(line.coords[i])
            end_point = Point(line.coords[i + 1])
            dx = end_point.x - start_point.x
            dy = abs(end_point.y - start_point.y)

            if line_y[0] < line_y[1]:
                # Plot an arrow between consecutive points
                ax.annotate(
                    "",
                    xy=(line_x[0], line_y[0]),
                    xytext=(line_x[-1], line_y[-1]),
                    arrowprops={"arrowstyle": "->", "linewidth": 2, "color": "r"},
                    annotation_clip=False,
                )
            else:
                ax.annotate(
                    "",
                    xy=(line_x[-1], line_y[-1]),
                    xytext=(line_x[0], line_y[0]),
                    arrowprops={"arrowstyle": "->", "linewidth": 2, "color": "r"},
                    annotation_clip=False,
                )

    # Set plot limits
    ax.set_xlim(0, 512)
    ax.set_ylim(0, 512)

    # Set aspect ratio to be equal
    ax.set_aspect("equal")

    # Show the plot
    plt.show()
    return fig

def plot_multiple_polygons_rectangles(list_polygons, list_lines=[], img=None):
    return plot_multiple_polygons([_.minimum_rotated_rectangle for _ in list_polygons], list_lines, img)

def merge_close_polygons(polygons, threshold=10):
    merged_polygons = []

    for polygon in polygons:
        # Check if the current polygon overlaps with any merged polygon
        merged = False

        for i, merged_polygon in enumerate(merged_polygons):
            if polygon.distance(merged_polygon) < threshold:
                merged_polygons[i] = unary_union([polygon, merged_polygon])
                merged = True
                break

        if not merged:
            merged_polygons.append(polygon)

    return merged_polygons

# Step 4: Find the principal axis of polyb and create a line
def get_plot_edge_points(x1, x2, m, b):
    # Calculate y1 and y2 based on the line equation y = mx + b
    y1 = m * x1 + b
    y2 = m * x2 + b


    # Ensure y1 and y2 are within the range [0, 512] and adjust x2 accordingly
    if y1 < 0:
        y1 = 0
        x1 = (y1 - b) / m
    if y1 > 512:
        y1 = 512
        x1 = (y1 - b) / m
    if y2 < 0:
        y2 = 0
        x2 = (y2 - b) / m
    if y2 > 512:
        y2 = 512
        x2 = (y2 - b) / m

    p('slope', m)
    if m is not np.inf:
        assert round(y1) == round(m * x1 + b), f"{y1} != {m * x1 + b} | {x1}"
        assert round(y2) == round(m * x2 + b), f"{y2} != {m * x2 + b} | {x2}"

    return x1, y1, x2, y2

def get_flat_line(p):
    y = p.minimum_rotated_rectangle.centroid.y
    return LineString([(0, y), (512, y)])

def get_principal_axis(rectangle, seed=None):
    # Find the minimum and maximum points along the long edge
    longest_edge, _ = longest_edges(rectangle)

    line_x, line_y = zip(*[(x, y) for x, y in longest_edge.coords])

    if line_y[0] < line_y[1]:
        x1, x2 = line_x
        y1, y2 = line_y
    else:
        x2, x1 = line_x
        y2, y1 = line_y

    if x2 - x1 == 0:
        slope = np.inf # Vertical line, undefined slope
    else:
        slope = (y2 - y1) / (x2 - x1)

    b = rectangle.centroid.y - slope * rectangle.centroid.x

    # Calculate the points to create the line
    if ((slope is np.inf) and (y2 > y1)):
        x1, y1, x2, y2 = rectangle.centroid.x, 0, rectangle.centroid.x, 512
    elif ((slope is np.inf) and (y2 < y1)):
        x1, y1, x2, y2 = rectangle.centroid.x, 512, rectangle.centroid.x, 0
    elif slope > 0:
        x1, y1, x2, y2 = get_plot_edge_points(12, 500, slope, b)
    else:
        x1, y1, x2, y2 = get_plot_edge_points(500, 12, slope, b)

    # Create a Shapely LineString representing the desired line
    line = LineString([(x1, y1), (x2, y2)])

    return line


def get_principal_axis_error(rectangle):
    try:
        center_x, center_y = rectangle.centroid.xy

        # Find the minimum and maximum points along the long edge
        long_edge = None
        long_edge_length = 0

        for edge in zip(rectangle.exterior.coords, rectangle.exterior.coords[1:]):
            x1, y1 = edge[0]
            x2, y2 = edge[1]
            edge_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            if edge_length > long_edge_length:
                long_edge_length = edge_length
                long_edge = edge

        # Calculate the slope of the long edge
        x1, y1 = long_edge[0]
        x2, y2 = long_edge[1]

        slope = (y2 - y1) / (x2 - x1)
        return False
    except:
        return True

def extend_linestring(linestring, length_to_extend):
    # Get the coordinates of the original LineString

    coordinates = list(linestring.coords)

    # Calculate the direction of extension
    if len(coordinates) >= 2:

        # Calculate the direction vector from the last two points
        dx = coordinates[-1][0] - coordinates[-2][0]
        dy = coordinates[-1][1] - coordinates[-2][1]


        # Normalize the direction vector
        length = (dx ** 2 + dy ** 2) ** 0.5
        if length > 0:
            dx /= length
            dy /= length


        x, y = coordinates[-1][0], coordinates[-1][1]

        # Calculate the new endpoint
        new_x = x + dx * length_to_extend
        new_y = y + dy * length_to_extend

        # Extend the LineString
        coordinates.append((new_x, new_y))

        new_x = x - dx * length_to_extend
        new_y = y - dy * length_to_extend

        # Extend the LineString
        coordinates.append((new_x, new_y))

        # Create a new LineString
        extended_linestring = LineString(coordinates)
        return extended_linestring
    else:
        # If the LineString has less than two points, return the original LineString
        return linestring

def slice_polygon(ref_a, slice_x, buffer_distance=5):
    polyc, polyb = ref_a, slice_x
    # Step 1: Create a buffer around polyb to make it slightly larger
    buffer_distance = 5  # Adjust this value to make polyb larger
    buffered_polyb = polyb.buffer(buffer_distance)

    intersection = buffered_polyb.intersection(polyc)

    minx, miny, maxx, maxy = intersection.bounds
    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2
    box_size = 100
    roi_box = box(center_x - box_size / 2, center_y - box_size / 2, center_x + box_size / 2, center_y + box_size / 2)

    # Step 4: Crop polyb and polyc using the box
    polyb_cropped = polyb.intersection(roi_box)
    polyc_cropped = polyc.intersection(roi_box)

    # Step 5: Compute minimum rotating rectangles
    rotated_polyb = polyb_cropped.minimum_rotated_rectangle
    rotated_polyc = polyc_cropped.minimum_rotated_rectangle

    principal_axis_line = get_principal_axis(rotated_polyb)
    principal_axis_line = extend_linestring(principal_axis_line, 100)

    out_poly1, out_poly2 = split(polyc, principal_axis_line).geoms

    if out_poly1.centroid.x < out_poly2.centroid.x:
        polya, polyc = out_poly1, out_poly2
    else:
        polyc, polya = out_poly1, out_poly2
    return polya, polyc

def calculate_intersection_dist_to_t_edge(poly_a, poly_b):
    intersection = poly_a.intersection(poly_b)
    if intersection.area > 0:
        topmost_edge = min(coord[1] for coord in poly_a.exterior.coords)
        distance_to_topmost = abs(topmost_edge - intersection.centroid.y)
    else:
        topmost_edge_a = min(coord[1] for coord in poly_a.exterior.coords)
        closest_edge_b = min(poly_b.exterior.coords, key=lambda coord: abs(topmost_edge_a - coord[1]))
        distance_to_topmost = abs(topmost_edge_a - closest_edge_b[1])

    return distance_to_topmost

def calculate_intersection_dist_to_b_edge(poly_a, poly_b):
    intersection = poly_a.intersection(poly_b)
    if intersection.area > 0:
        bottommost_edge = max(coord[1] for coord in poly_a.exterior.coords)
        distance_to_bottommost = abs(bottommost_edge - intersection.centroid.y)
    else:
        bottommost_edge_a = max(coord[1] for coord in poly_a.exterior.coords)
        closest_edge_b = min(poly_b.exterior.coords, key=lambda coord: abs(bottommost_edge_a - coord[1]))
        distance_to_bottommost = abs(bottommost_edge_a - closest_edge_b[1])

    return distance_to_bottommost

def calculate_poly_a_c_dist_to_b_edge(poly_a, poly_b):
    poly_a_c = poly_a.centroid.y
    intersection = poly_a.intersection(poly_b)
    if intersection.area > 0:
        bottommost_edge = max(coord[1] for coord in poly_a.exterior.coords)
        distance_to_bottommost = bottommost_edge - intersection.centroid.y
    else:
        bottommost_edge_a = max(coord[1] for coord in poly_a.exterior.coords)
        closest_edge_b = min(poly_b.exterior.coords, key=lambda coord: abs(bottommost_edge_a - coord[1]))
        distance_to_bottommost = bottommost_edge_a - closest_edge_b[1]

    # if between edge and c, returns positive, if beyond edge and c, returns negative
    return distance_to_bottommost





def calculate_intersection_dist_to_r_edge(poly_a, poly_b):
    intersection = poly_a.intersection(poly_b)
    if intersection.area > 0:
        rightmost_edge = max(coord[0] for coord in poly_a.exterior.coords)
        distance_to_rightmost = abs(rightmost_edge - intersection.centroid.x)
    else:
        rightmost_edge_a = max(coord[0] for coord in poly_a.exterior.coords)
        closest_edge_b = min(poly_b.exterior.coords, key=lambda coord: abs(rightmost_edge_a - coord[0]))
        distance_to_rightmost = abs(rightmost_edge_a - closest_edge_b[0])

    return distance_to_rightmost


def calculate_intersection_dist_to_l_edge(poly_a, poly_b):
    intersection = poly_a.intersection(poly_b)
    if intersection.area > 0:
        leftmost_edge = min(coord[0] for coord in poly_a.exterior.coords)
        distance_to_left = abs(intersection.centroid.x - leftmost_edge)
    else:
        leftmost_edge_a = min(coord[0] for coord in poly_a.exterior.coords)
        closest_edge_b = min(poly_b.exterior.coords, key=lambda coord: abs(leftmost_edge_a - coord[0]))
        distance_to_left = abs(leftmost_edge_a - closest_edge_b[0])
    return distance_to_left


def calculate_intersection_dist_to_centroid(poly_a, poly_b):
    intersection = poly_a.intersection(poly_b)
    centroid_a = poly_a.centroid
    closest_edge_b = min(poly_b.exterior.coords, key=lambda coord: abs(centroid_a.x - coord[0]))
    distance_to_centroid = abs(centroid_a.x - closest_edge_b[0])
    return distance_to_centroid


def merge_close_polygons(polygons, threshold):
    merged_polygons = []

    for polygon in polygons:
        # Check if the current polygon overlaps with any merged polygon
        merged = False

        for i, merged_polygon in enumerate(merged_polygons):
            if polygon.distance(merged_polygon) < threshold:
                merged_polygons[i] = unary_union([polygon, merged_polygon])
                merged = True
                break

        if not merged:
            merged_polygons.append(polygon)

    return merged_polygons


def segment_connected(seg_a, seg_b, threshold_dist=40):
    '''
    if segment is close (within 15) or intersection then return True
    '''
    intersection = seg_a.intersection(seg_b).area > 0
    if intersection:
        return True

    distance = seg_a.distance(seg_b)
    if distance < threshold_dist:
        return True
    h_distance = seg_a.hausdorff_distance(seg_b)
    if h_distance < 20:
        return True

    p(f'not connected because distance is: {distance}, haushorff dist: {h_distance}')

    return False

def get_distance(seg_a, seg_b):
    '''
    if segment is close (within 15) or intersection then return True
    '''

    distance = seg_a.distance(seg_b)
    return distance


def get_contour_poly_multi(mask, threshold=250):
    contours = measure.find_contours(mask)

    holder = []
    for contour in contours:
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Simplify polygon
        poly = Polygon(contour)
        holder.append(poly)

    return [_ for _ in holder if _.area > threshold]


def get_contour_poly(mask):
    contours = measure.find_contours(mask)

    holder = []
    for contour in contours:
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Simplify polygon
        poly = Polygon(contour)
        holder.append(poly)

    buffered_polygon = None

    if len(holder) > 1:
        try:
            p("trying to fill in holes to make simple polygon")
            filled_polygon = unary_union(holder)

            # Apply a buffer to the filled polygon to improve the contour
            buffered_polygon = filled_polygon.buffer(distance=1).buffer(
                -5, join_style="bevel"
            )

            assert type(buffered_polygon) == Polygon

            del filled_polygon

            return buffered_polygon

        except:
            if buffered_polygon:
                del buffered_polygon

            areas = [_.area for _ in holder]

            p(
                "too many contours, merging similar ones and returning largest one:",
                areas,
            )
            return holder[np.argmax(areas)]

    return holder[0]


def poly_to_mask(poly, img_size=(512, 512)):
    segmentation = np.array(poly.exterior.coords).ravel().tolist()

    points = np.array([segmentation[::2], segmentation[1::2]], np.int32).T
    points = points.reshape((-1, 1, 2))
    tmp = np.zeros((512, 512), np.int32)
    cv2.fillPoly(tmp, [points], (1))

    return tmp

def idx_left(list_of_segs):
    nums = [_[0].centroid.x if _ else 9999 for _ in list_of_segs]

    # Use sorted() to get the sorted order and enumerate() to track the original indices
    sorted_indices = [index for index, value in sorted(enumerate(nums), key=lambda x: x[1])]

    # sorted_indices will contain the indices in the order that sorts nums
    return sorted_indices


def idx_bot(list_of_segs):
    nums = [_[0].centroid.y if _ else 9999 for _ in list_of_segs]

    # Use sorted() to get the sorted order and enumerate() to track the original indices
    sorted_indices = [index for index, value in sorted(enumerate(nums), key=lambda x: x[1])]

    # sorted_indices will contain the indices in the order that sorts nums
    return sorted_indices


def degree_to_rad(d):
    return d * np.pi / 180


def greater_than(x, threshold=15):
    r_x = degree_to_rad(x)
    while r_x >= np.pi:
        r_x = r_x - np.pi

    if abs(r_x) >= degree_to_rad(threshold):
        p(f'{r_x / np.pi * 180} > {threshold}')
        return True
    if abs(r_x - np.pi) >= degree_to_rad(threshold):
        p(f'{abs((r_x - np.pi)) / np.pi * 180} > {threshold}')
        return True
    p(r_x / np.pi * 180, (r_x - np.pi) / np.pi * 180, threshold)
    return False


def less_than(x, threshold=15):
    r_x = degree_to_rad(x)
    while r_x >= np.pi:
        r_x = r_x - np.pi

    if abs(r_x) <= degree_to_rad(threshold):
        p(f'{r_x / np.pi * 180} < {threshold}')
        p('intersection is flat')
        return True
    if abs(r_x - np.pi) <= degree_to_rad(threshold):
        p(f'{abs((r_x - np.pi)) / np.pi * 180} < {threshold}')
        p('intersection is flat')
        return True
    p(r_x / np.pi * 180, (r_x - np.pi) / np.pi * 180, threshold)
    return False


def is_flat(x, threshold=15):
    if not x:
        return None
    r_x = degree_to_rad(x)
    while r_x >= np.pi:
        r_x = r_x - np.pi

    if abs(r_x) <= degree_to_rad(threshold):
        p(f'{r_x / np.pi * 180} < {threshold}')
        p('intersection is flat')
        return True
    if abs(r_x - np.pi) <= degree_to_rad(threshold):
        p(f'{abs((r_x - np.pi)) / np.pi * 180} < {threshold}')
        p('intersection is flat')
        return True
    p(r_x / np.pi * 180, (r_x - np.pi) / np.pi * 180, threshold)
    return False

def evenly_split_poly_space(a, b):
    intersect_area = a.intersection(b).area
    p(f'evenly splitting poly space, {intersect_area}')
    if intersect_area == 0:
        return a, b
    if intersect_area > 0:
        if intersect_area < 800:
            return a, b
        elif intersect_area >= 800:
            angle = get_intersection_angle(a, b, 100)

            polygon1 = a
            polygon2 = b

            # Calculate centroids of the polygons
            centroid1 = polygon1.centroid
            centroid2 = polygon2.centroid

            intersection_points = (
                LineString(a.exterior.coords)
                .intersection(LineString(b.exterior.coords))
                .geoms
            )

            p(f"{intersection_points} and {len(intersection_points)}")
            p(f"1: {is_flat(angle, 10) and short_edge_intersection(a, b)}")
            p(f"2: {intersection_points and len(intersection_points) > 2}")
            if is_flat(angle, 10) and short_edge_intersection(a, b):
                p("bisecting by centroid angle")
                # Find the midpoint between the centroids
                midpoint = Point(
                    (centroid1.x + centroid2.x) / 2, (centroid1.y + centroid2.y) / 2
                )

                # Calculate the slope of the line passing through the midpoint
                slope = (centroid2.y - centroid1.y) / (centroid2.x - centroid1.x)

                # Calculate the negative reciprocal to get the slope of the perpendicular line
                perpendicular_slope = -1 / slope

                # Find a point on the perpendicular line that lies on the midpoint
                # You can choose any distance from the midpoint to define the point
                distance = 1  # Adjust this distance as needed
                perpendicular_point1 = Point(
                    midpoint.x + distance, midpoint.y + distance * perpendicular_slope
                )
                perpendicular_point2 = Point(
                    midpoint.x - distance, midpoint.y - distance * perpendicular_slope
                )

                # Define the perpendicular line using the slope and point
                # You can create a LineString or any other suitable geometry for your needs
                perpendicular_line = LineString(
                    [midpoint, perpendicular_point1, perpendicular_point2]
                )
                intersection_line = extend_linestring(perpendicular_line, 500.0)
            elif intersection_points and len(intersection_points) > 2:
                p("bisecting by angle")
                # Find the midpoint between the centroids
                midpoint = Point(
                    (centroid1.x + centroid2.x) / 2, (centroid1.y + centroid2.y) / 2
                )

                # Calculate the slope of the line passing through the midpoint
                slope = (centroid2.y - centroid1.y) / (centroid2.x - centroid1.x)

                # Calculate the negative reciprocal to get the slope of the perpendicular line
                perpendicular_slope = -1 / slope

                # Find a point on the perpendicular line that lies on the midpoint
                # You can choose any distance from the midpoint to define the point
                distance = 1  # Adjust this distance as needed
                perpendicular_point1 = Point(
                    midpoint.x + distance, midpoint.y + distance * perpendicular_slope
                )
                perpendicular_point2 = Point(
                    midpoint.x - distance, midpoint.y - distance * perpendicular_slope
                )

                # Define the perpendicular line using the slope and point
                # You can create a LineString or any other suitable geometry for your needs
                perpendicular_line = LineString(
                    [midpoint, perpendicular_point1, perpendicular_point2]
                )
                intersection_line = extend_linestring(perpendicular_line, 500.0)
                # _ = plot_multiple_polygons([a, b], [intersection_line])
            else:
                # since intersection is not flat, we need to get a line at the angle of intersection
                print(type(intersection_points), len(intersection_points))
                temp_line = None
                if type(intersection_points) == GeometrySequence:
                    for _ in intersection_points:
                        if type(_) == LineString:
                            temp_line = _
                        intersection_points = temp_line
                        break

                intersection_line = LineString(intersection_points)
        # intersection_line = extend_linestring(intersection_line, 500.0)

        full_poly = unary_union([polygon1, polygon2])

        split_geoms = split(full_poly, intersection_line).geoms
        p(f"number of resultant geoms {len(split_geoms)}")
        if len(split_geoms) == 1:
            return a, b
        elif len(split_geoms) == 2:
            t1, t2 = split_geoms
            _=plot_multiple_polygons([t1, t2], [intersection_line])
        elif len(split_geoms) == 3:
            t1, _, t2 = split_geoms
            if t1.area < 1:
                t1 = _
            elif t2.area < 1:
                t2 = _
        elif len(split_geoms) == 4:
            t1, t2 = split_geoms[0], sorted(split_geoms, key=lambda x: x.area)[-1]
        else:
            p(f"uhoh num geoms {len(split_geoms)}")
            return split_geoms

        # assign correct areas.
        try:
            if a.intersection(t1).area > b.intersection(t1).area:
                a, b = t1, t2
            else:
                b, a = t1, t2
        except Exception as e:
            p(e)
            pass
    return a, b


def evenly_split_poly_angle(a, b):
    if a.intersection(b).area > 0:
        angle = get_intersection_angle(a, b, 100)

        polygon1 = a
        polygon2 = b

        # Calculate centroids of the polygons
        centroid1 = polygon1.centroid
        centroid2 = polygon2.centroid

        intersection_points = (
            LineString(a.exterior.coords).intersection(LineString(b.exterior.coords)).geoms
        )

        if intersection_points and len(intersection_points) > 2:
            # Find the midpoint between the centroids
            midpoint = Point(
                (centroid1.x + centroid2.x) / 2, (centroid1.y + centroid2.y) / 2
            )

            # Calculate the slope of the line passing through the midpoint
            slope = (centroid2.y - centroid1.y) / (centroid2.x - centroid1.x)

            # Calculate the negative reciprocal to get the slope of the perpendicular line
            perpendicular_slope = -1 / slope

            # Find a point on the perpendicular line that lies on the midpoint
            # You can choose any distance from the midpoint to define the point
            distance = 1  # Adjust this distance as needed
            perpendicular_point1 = Point(
                midpoint.x + distance, midpoint.y + distance * perpendicular_slope
            )
            perpendicular_point2 = Point(
                midpoint.x - distance, midpoint.y - distance * perpendicular_slope
            )

            # Define the perpendicular line using the slope and point
            # You can create a LineString or any other suitable geometry for your needs
            perpendicular_line = LineString(
                [midpoint, perpendicular_point1, perpendicular_point2]
            )
            intersection_line = extend_linestring(perpendicular_line, 500.0)
        else:
            # since intersection is not flat, we need to get a line at the angle of intersection
            intersection_points = LineString(a.exterior.coords).intersection(LineString(b.exterior.coords)).geoms

            intersection_line = LineString(intersection_points)
        # intersection_line = extend_linestring(intersection_line, 500.0)

        full_poly = unary_union([polygon1, polygon2])

        split_geoms = split(full_poly, intersection_line).geoms
        p(f"number of resultant geoms {len(split_geoms)}")
        if len(split_geoms) == 2:
            t1, t2 = split_geoms
        elif len(split_geoms) == 3:
            t1, _, t2 = split_geoms
            if t1.area < 1:
                t1 = _
            elif t2.area < 1:
                t2 = _
        elif len(split_geoms) == 4:
            t1, t2 = split_geoms[0], sorted(split_geoms, key=lambda x: x.area)[-1]
        else:
            p(f"uhoh num geoms{len(split_geoms)}")

        # assign correct areas.
        try:
            if a.intersection(t1).area > b.intersection(t1).area:
                a, b = t1, t2
            else:
                b, a = t1, t2
        except Exception as e:
            p(e)
            pass
    return a, b


def calc_dice(a, b):
    # Calculate intersection area
    if not a.intersects(b):
        return 0

    intersection_area = a.intersection(b).area

    # Calculate union area
    union_area = a.area + b.area

    # Calculate Dice coefficient
    dice_coefficient = (2 * intersection_area) / union_area

    return dice_coefficient


def intersect_area(p1, p2):  # Calculate the area of polygon_a.
    try:
        area_p1 = p1.area

        # Calculate the area of the intersection between polygon_a and polygon_b.
        intersection_area = p1.intersection(p2).area

        # Calculate the percentage of the space of polygon_a covered by polygon_b.
        percentage_covered = (intersection_area / area_p1) * 100
    except:
        percentage_covered = 0
    # Check if polygon_b covers over 90% of the space of polygon_a.
    return percentage_covered


def find_unique_polygons(polygons):
    # Create a list to store polygons that don't intersect.
    unique_intersecting_polygons = [Polygon(polygons[0])]

    # Iterate through each polygon and check for intersection with others.
    for i, polygon1 in enumerate(polygons):
        unique = False
        for j, polygon2 in enumerate(polygons):
            if i != j:
                dice = calc_dice(
                    polygon1, polygon2
                )  # Skip comparing the polygon with itself
                if dice > 85:
                    unique = True
                elif dice > 0:
                    if (
                        intersect_area(polygon1, polygon2) > 90
                        or intersect_area(polygon2, polygon1) > 90
                    ):
                        unique = True
        if not unique:
            unique_intersecting_polygons.append(polygon1)

    return unique_intersecting_polygons


def create_temp_json(output_matrix, img_id):
    with open("json_out/empty_annotations.json") as file:
        gt = json.load(file)

    print('exporting annotations')

    gt_images = [
        {
            "id": i,
            "width": 512,
            "height": 512,
            "file_name": f"{img_id}.png",
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": 0,
        }
        for i in range(1)
    ]

    empty_submit = dict()
    empty_submit["images"] = gt_images
    empty_submit["categories"] = gt["categories"]
    empty_submit["annotations"] = []

    gt_mask = output_matrix[1:27, ...]

    count_anns = 1
    img = gt_mask

    for cls_id, cls in enumerate(img, 0):
        contours = measure.find_contours(cls)
        for contour in contours:
            for i in range(len(contour)):
                row, col = contour[i]
                contour[i] = (col - 1, row - 1)

            # Simplify polygon
            poly = Polygon(contour)

            if poly.is_empty:
                continue
            if poly.geom_type == "Polygon":
                segmentation = np.array(poly.exterior.coords).ravel().tolist()
            elif poly.geom_type == "MultiPolygon":
                poly = poly.simplify(1.0, preserve_topology=False)
                segmentation = np.array(poly.exterior.coords).ravel().tolist()
            new_ann = dict()
            # print(img_id, count_anns)
            new_ann["id"] = count_anns
            new_ann["image_id"] = img_id
            new_ann["category_id"] = cls_id + 1
            new_ann["segmentation"] = [segmentation]
            new_ann["area"] = poly.area
            x, y = contour.min(axis=0)
            w, h = contour.max(axis=0) - contour.min(axis=0)
            new_ann["bbox"] = [int(x), int(y), int(w), int(h)]
            new_ann["iscrowd"] = 0
            new_ann["attributes"] = {"occluded": False}
            count_anns += 1
            empty_submit["annotations"].append(new_ann.copy())

    with open(f"temp.json", "w") as file:
        json.dump(empty_submit, file)

def create_empty_json(num_images=30):
    with open('/opt/app/json_out/empty_annotations.json') as file:
        gt = json.load(file)

    gt_images = [{
                "id": i+1,
                "width": 512,
                "height": 512,
                "file_name": f"{i+1}.png",
                "license": 0,
                "date_captured": iso_datetime
                }
                for i in range(num_images)]

    info = {'description': 'NU Team 2023',
           'version': 'v1',
           'year': 2023,
           'contributor': 'txlmd',
            'date_created': datetime.today().strftime("%Y-%m-%d")}

    empty_submit = dict()
    empty_submit["images"] = gt_images
    empty_submit["categories"] = gt["categories"]
    empty_submit["annotations"] = []
    empty_submit["info"] = info
    empty_submit["licenses"] = []

    with open(f"/opt/app/json_out/coronary-artery-segmentation.json", "w") as file:
        json.dump(empty_submit, file)


def run_inference_test():
    print('running inference placeholder')

    create_empty_json()

    with open('/opt/app/json_out/coronary-artery-segmentation.json', encoding="utf-8") as file:
        gt = json.load(file)

    return gt