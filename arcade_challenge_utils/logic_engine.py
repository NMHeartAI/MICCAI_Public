from arcade_challenge_utils.utils import *
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

seg_dict = {i: v for i, v in enumerate(
    ['1', '2', '3', '4', '5', '6', '7', '8', '9', '9a', '10', '10a', '11', '12', '12a', '13', '14', '14a', '15',
     '16', '16a', '16b', '16c', '12b', '14b', 'stenosis'])}

seg_inverse = {v: k + 1 for k, v, in seg_dict.items()}

connections = {
    '1': ['2'],
    '2': ['3'],
    '3': ['4', '16'],
    '4': [],
    '5': ['6', '11', '12'],
    '6': ['7', '9'],
    '7': ['9a', '10', '8'],
    '8': ['10a'],
    '9': [],
    '9a': [],
    '10': [],
    '10a': [],
    '11': ['12a', '12b', '13'],
    '12': [],
    '12a': [],
    '12b': [],
    '13': ['14', '14a', '14b', '15'],
    '14': [],
    '14a': [],
    '14b': [],
    '15': [],
    '16': ['16a', '16b', '16c'],
    '16a': [],
    '16b': [],
    '16c': [],
}

parent_connections = {k: ''.join([k_ for k_, v_ in connections.items() if k in v_]) for k in connections}

left_right = {'1': ['2', '3', '4', '16', '16a', '16b', '16c'], '2': ['3', '4', '16', '16a', '16b', '16c'],
              '5': ['6', '7', '8', '9', '9a', '10', '10a', '11', '12', '12a', '12b', '13', '14', '14a', '14b', '15'],
              '6': ['6', '7', '8', '9', '9a', '10', '10a', '11', '12', '12a', '12b', '13', '14', '14a', '14b', '15']}

def calculate_distance(coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def filter_close_coordinates(coordinates, distance_threshold):
    filtered_coordinates = []

    for i in range(len(coordinates)):
        is_close = False
        for j in range(i + 1, len(coordinates)):
            if calculate_distance(coordinates[i], coordinates[j]) <= distance_threshold:
                is_close = True
                break

        if not is_close:
            filtered_coordinates.append(coordinates[i])

    return filtered_coordinates

def decide_12a_or_9(ref_5, ref_x):
    upper = Polygon([(512, 512), (0, 512), (512, 0)])
    return '9' if upper.covers(ref_x.centroid) else '12a'

def decide_closer_to_14_or_14b(ref_11, ref_x):
    distance_t = calculate_intersection_dist_to_t_edge(ref_11, ref_x)
    distance_b = calculate_intersection_dist_to_b_edge(ref_11, ref_x)
    distance_c = calculate_intersection_dist_to_centroid(ref_11, ref_x)

    if distance_t < distance_b:
        if segment_12a_should_be_12(ref_11, ref_x):
            if distance_t > distance_c:
                return '14a'
            return '14'
    elif distance_b < distance_t:
        if segment_12b_should_be_12a(ref_11, ref_x):
            if distance_b > distance_c:
                return '14a'
            return '14b'
    return '14a'


def decide_closer_to_12_or_12b(ref_11, ref_x):
    distance_t = calculate_intersection_dist_to_t_edge(ref_11, ref_x)
    distance_b = calculate_intersection_dist_to_b_edge(ref_11, ref_x)
    distance_c = calculate_intersection_dist_to_centroid(ref_11, ref_x)

    p('deciding 12 or 12_b:', distance_t, distance_b, distance_c)
    if distance_t < distance_b:
        if segment_12a_should_be_12(ref_11, ref_x):
            if distance_t > distance_c:
                return '12a'
            return '12'
    elif distance_b < distance_t:
        if segment_12b_should_be_12a(ref_11, ref_x):
            if distance_b > distance_c:
                return '12a'
            return '12b'
    return '12a'


def segment_12a_should_be_12(ref_11, ref_x):
    distance = calculate_intersection_dist_to_t_edge(ref_11, ref_x)
    if distance < 20 or short_edge_intersection(ref_11, ref_x):
        return True
    return False


def segment_12b_should_be_12a(ref_11, ref_x):
    distance = calculate_intersection_dist_to_b_edge(ref_11, ref_x)
    if distance > 20:
        if not short_edge_intersection(ref_11, ref_x):
            return True
    return False


def decide_closer_to_16a_or_16c(ref_16, ref_x):
    distance_r = calculate_intersection_dist_to_r_edge(ref_16, ref_x)
    distance_l = calculate_intersection_dist_to_l_edge(ref_16, ref_x)
    distance_c = calculate_intersection_dist_to_centroid(ref_16, ref_x)

    if distance_r < distance_l:
        if segment_16b_should_be_16c(ref_16, ref_x):
            if distance_r > distance_c:
                return '16b'
            return '16c'
    elif distance_l < distance_r:
        if segment_16a_should_be_16b(ref_16, ref_x):
            if distance_l > distance_c:
                return '16b'
            return '16a'
    return '16b'


def segment_16b_should_be_16c(ref_16, ref_16b):
    distance = calculate_intersection_dist_to_r_edge(ref_16, ref_16b)
    if distance < 20 or short_edge_intersection(ref_16, ref_16b):
        return True
    return False


def segment_16a_should_be_16b(ref_16, ref_16a):
    distance = calculate_intersection_dist_to_l_edge(ref_16, ref_16a)
    if distance > 20:
        if not short_edge_intersection(ref_16, ref_16a):
            return True
    return False

def parent_engine(mask, segs, seg_history, category_list, parent):
    p('before', mask.shape, category_list.shape)
    try:
        if parent == '5_11' and not seg_history['5']:
            idx6 = segs.get(6)[0];
            idx13 = segs.get(13)[0]
            seg6 = get_contour_poly(mask[idx6])
            seg11 = get_contour_poly(mask[idx13])
            seg5, seg6 = slice_polygon(seg6, seg11)
            seg_history['5'] = [seg5]
            seg_history['6'] = [seg6]
            seg_history['11'] = [seg11]

            category_list = np.append(category_list, seg_inverse['5'] - 1)
            new_array = poly_to_mask(seg5).reshape(1, 512, 512)
            # Use np.concatenate to combine the existing_array and new_array along axis 0
            mask = np.concatenate((mask, new_array), axis=0)
            segs[5] += [len(category_list) - 1]

    except Exception as e1:
        p(e1)
    p('after', mask.shape, category_list.shape)
    return mask, segs, seg_history, category_list

def multiple_segment_engine(seg_a, sub_segments, current_seg, candidate):
    """returns idx of sub_segments based on rules"""

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

    # check if sub_segments are overlapped
    p(f"multiple segment engine for {candidate}")

    if len(sub_segments) >= 2:
        # Create Shapely polygons
        try:
            a, b = sub_segments

            # Calculate intersection area
            intersection_area = a.intersection(b).area

            # Calculate union area
            union_area = a.area + b.area

            # Calculate Dice coefficient
            dice_coefficient = (2 * intersection_area) / union_area

            p(f"possibly merged segments. Dice: {dice_coefficient}")
            p(
                f"difference in angle between shapes: {abs(get_major_axis_angle(a) - get_major_axis_angle(b))}"
            )
            if dice_coefficient > 0.9:
                p(f"recommend merging {candidate}")
                return "merge", 0
            elif (
                dice_coefficient > 0.75
                and abs(get_major_axis_angle(a) - get_major_axis_angle(b)) < 10
            ):
                p(f"recommend merging {candidate}")
                return "merge", 0
            elif max(intersect_area(a, b), intersect_area(b, a)) > 80.0:
                p(f"recommend splitting {candidate}")
                return "split", 0
            p('segments are not overlapping enough to make a decision, moving to main body')

        except:
            connected_ = [segment_connected(seg_a, _) for _ in sub_segments]
            a_idx = np.argmax(connected_)
            a_ = sub_segments[a_idx]
            temp_idx = [a_idx]
            p(connected_, a_idx, temp_idx)
            for i, b_ in enumerate(sub_segments):
                dice_coefficient = calc_dice(a_, b_)
                intersection_angle = get_intersection_angle(a_, b_)
                p(
                    f"deciding merging {candidate}, {dice_coefficient}, {intersection_angle}"
                )
                if (i == a_idx) | (dice_coefficient < 0.4):
                    pass
                elif dice_coefficient > 0.9:
                    temp_idx += [i]
                elif dice_coefficient > 0.75 and is_flat(intersection_angle, 5):
                    temp_idx += [i]
            if sum(connected_) == 0:
                return "push", []

            if decision_engine(seg_a, a_, current_seg, candidate):
                return "push", temp_idx

            return "cycle", [i for i in range(sub_segments)]

    p("figuring out multiple segments here")
    temp_sub_segments, temp_idx = sub_segments, []
    p([get_major_axis_angle(seg) for seg in sub_segments])
    if candidate == "16":
        for i, seg in enumerate(sub_segments):
            if get_major_axis_angle(seg) < -10:
                temp_idx += [i]

        if len(temp_idx) == 0:
            return "merge", 0

        return "push", temp_idx
    if candidate in ["16a", "16b", "16c"]:
        p([segment_connected(seg_a, _) for _ in sub_segments])
        which_class = [decide_closer_to_16a_or_16c(seg_a, _) for _ in sub_segments]
        p([get_intersection_angle(seg_a, _) for _ in sub_segments])
        candidate_dict = {k: [] for k in ["16a", "16b", "16c"]}
        for i, k in enumerate(which_class):
            candidate_dict[k] += [i]
        return candidate_dict, 0

    if candidate in ["12", "12a", "12b"]:
        p([segment_connected(seg_a, _) for _ in sub_segments])
        which_class = [decide_closer_to_12_or_12b(seg_a, _) for _ in sub_segments]
        p([get_intersection_angle(seg_a, _) for _ in sub_segments])
        candidate_dict = {k: [] for k in ["12", "12a", "12b"]}
        for i, k in enumerate(which_class):
            candidate_dict[k] += [i]
        return candidate_dict, 0

    if candidate in ["14", "14a", "14b"]:
        p([segment_connected(seg_a, _) for _ in sub_segments])
        which_class = [decide_closer_to_14_or_14b(seg_a, _) for _ in sub_segments]
        p([get_intersection_angle(seg_a, _) for _ in sub_segments])
        candidate_dict = {k: [] for k in ["14", "14a", "14b"]}
        for i, k in enumerate(which_class):
            candidate_dict[k] += [i]
        return candidate_dict, 0

    if candidate in ["5", "6"]:
        p([segment_connected(seg_a, _) for _ in sub_segments])
        connected_ = [segment_connected(seg_a, _) for _ in sub_segments]
        if sum(connected_) == len(sub_segments):
            return "merge", temp_idx
        else:
            p("only one is connected")
            temp_idx = np.argmax(connected_)
            return "best", temp_idx

    if candidate == "2":
        connected_ = [segment_connected(seg_a, _) for _ in sub_segments]
        if sum(connected_) == len(sub_segments):
            temp_idx = find_segment_topmost_idx(sub_segments)
            return "top", temp_idx
        else:
            p("only one is connected")
            temp_idx = np.argmax(connected_)
            return "best", temp_idx

    if candidate == "3":
        connected_ = [segment_connected(seg_a, _) for _ in sub_segments]
        if sum(connected_) == len(sub_segments):
            closest_to_straight = np.argmin(
                [180 - get_intersection_angle(seg_a, _) for _ in sub_segments]
            )
            return "best", closest_to_straight
        else:
            return "best", np.argmax(connected_)

    if candidate == "4":
        connected_ = [
            segment_connected(seg_a, _)
            for _ in sub_segments
            if get_major_axis_angle(_) > 0
        ]

        if sum(connected_) == len(sub_segments):
            temp_idx = find_segment_leftmost_idx(sub_segments)
            return "merge", temp_idx
        else:
            p("only one is connected")
            temp_idx = np.argmax(connected_)
            return "best", temp_idx

    if candidate == "7":
        connected_ = [
            segment_connected(seg_a, _)
            for _ in sub_segments
            if get_major_axis_angle(_) > 0
        ]
        angles = [get_intersection_angle(seg_a, _) for _ in sub_segments]
        p("angles", angles)

        if sum(connected_) == len(sub_segments):
            return "merge", 0

        for i, angle in enumerate(angles):
            if (is_flat(angle, 30) == False) | (abs(180 - angle) > 30):
                temp_idx += [i]
        if len(temp_idx) == 0:
            return "merge", 0
        else:
            return "push", temp_idx

    if candidate == "8":
        connected_ = [segment_connected(seg_a, _) for _ in sub_segments]
        if sum(connected_) == len(sub_segments):
            closest_to_straight = np.argmin(
                [180 - get_angle(seg_a, _) for _ in sub_segments]
            )
            return "best", closest_to_straight
        else:
            return "best", np.argmax(connected_)

    if candidate == "9a":
        connected_ = [segment_connected(seg_a, _) for _ in sub_segments]
        if sum(connected_) == len(sub_segments):
            correct_angle = [
                get_intersection_angle(seg_a, _, 100) for _ in sub_segments
            ]
            p(correct_angle)
            return "best", np.argmax([_ > 15 for _ in correct_angle])
        else:
            return "best", np.argmax(connected_)

    if candidate == "10":
        connected_ = [segment_connected(seg_a, _) for _ in sub_segments]
        if sum(connected_) == len(sub_segments):
            correct_angle = [
                get_intersection_angle(seg_a, _, 100) for _ in sub_segments
            ]
            p(correct_angle)
            return "best", np.argmax([_ > 15 for _ in correct_angle])
        else:
            return "best", np.argmax(connected_)

    if candidate == "10a":
        connected_ = [segment_connected(seg_a, _) for _ in sub_segments]
        if sum(connected_) == len(sub_segments):
            correct_angle = [
                get_intersection_angle(seg_a, _, 100) for _ in sub_segments
            ]
            p(correct_angle)
            return "best", np.argmax([_ > 15 for _ in correct_angle])
        else:
            return "best", np.argmax(connected_)

    if candidate == "13":
        connected_ = [segment_connected(seg_a, _) for _ in sub_segments]
        angles = [get_intersection_angle(seg_a, _) for _ in sub_segments]
        p(angles)
        temp_idx = []
        p([(is_flat(_, 15)) for _ in angles])
        for i, angle in enumerate(angles):
            if angle and ((angle > 0) and (is_flat(angle, 15) == False)):
                temp_idx += [i]
        if len(temp_idx) == 0:
            return "merge", 0
        else:
            return "push", temp_idx

    if candidate == "14":
        connected_ = [segment_connected(seg_a, _) for _ in sub_segments]
        if sum(connected_) == len(sub_segments):
            closest_to_straight = np.argmin(
                [180 - get_intersection_angle(seg_a, _) for _ in sub_segments]
            )
            p(closest_to_straight)
            return "merge", 0

    if candidate == "15":
        connected_ = [
            segment_connected(seg_a, _)
            for _ in sub_segments
            if get_major_axis_angle(_) > -10
        ]
        if sum(connected_) == len(sub_segments):
            return "best", 0
        else:
            return "best", np.argmax(connected_)

    if candidate == "14b":
        connected_ = [segment_connected(seg_a, _) for _ in sub_segments]
        if sum(connected_) == len(sub_segments):
            closest_to_straight = np.argmin(
                [180 - get_intersection_angle(seg_a, _) for _ in sub_segments]
            )
            return "best", closest_to_straight
        else:
            return "best", np.argmax(connected_)

    if candidate == "16":
        connected_ = [
            segment_connected(seg_a, _)
            for _ in sub_segments
            if get_major_axis_angle(_) < -10
        ]
        if sum(connected_) == len(sub_segments):
            return "best", 0
        else:
            return "best", np.argmax(connected_)

    if candidate == "16a":
        connected_ = [
            segment_connected(seg_a, _)
            for _ in sub_segments
            if get_major_axis_angle(_) > -10
        ]
        if sum(connected_) == len(sub_segments):
            return "best", 0
        else:
            return "best", np.argmax(connected_)

def seg7_or_seg9(seg_a, seg_b):
    angle = get_intersection_angle(seg_a, seg_b, 150)
    closer_angle = get_intersection_angle(seg_a, seg_b, 20) or get_intersection_angle(
        seg_a, seg_b, 30
    )

    if (angle >= -180 and angle < -90) or (angle >= 0 and angle < 100):
        return '7'
    else:
        return '9'

def seg7_connection_to_seg9a_should_be_9(seg7, seg9):
    distance_t = calculate_intersection_dist_to_t_edge(seg7, seg9)
    distance_b = calculate_intersection_dist_to_b_edge(seg7, seg9)
    distance_c = calculate_intersection_dist_to_centroid(seg7, seg9)

    p("deciding 9 or 9a connection to 7:", distance_t, distance_b, distance_c)
    if distance_t < distance_b:
        if distance_t > distance_c:
            return "9"
    return "9a"

def list_vertecies(img, reference_segment):

    # Display the image using Matplotlib
    plt.imshow(img, cmap="gray")
    # plt.axis('off')  # Turn off axis labels
    plt.show()

    # try:
    # perform skeletonization
    skeleton = skeletonize(img)

    # Convert the binary image to uint8 data type
    skeleton = skeleton.astype("uint8")

    # Display the image using Matplotlib
    plt.imshow(skeleton, cmap="gray")
    # plt.axis('off')  # Turn off axis labels
    plt.show()

    # Find the coordinates of skeleton points
    skeleton_points = np.argwhere(skeleton == 1)

    # Initialize a list to store bifurcation points
    bifurcation_points = []

    # Define a structuring element for neighborhood analysis
    neighborhood = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)

    # Iterate through each skeleton point
    for point in skeleton_points:
        x, y = point[1], point[0]  # Extract x and y coordinates

        # Extract the neighborhood around the current point
        neighborhood_image = skeleton[y - 1 : y + 2, x - 1 : x + 2]

        # Count the number of nonzero pixels (branch points) in the neighborhood
        branch_points = cv2.countNonZero(
            cv2.bitwise_and(neighborhood_image, neighborhood)
        )

        # If there are three or more nonzero pixels, it's a bifurcation point
        if branch_points >= 4:
            bifurcation_points.append((x, y))

    # Specify the radius of the circle for visulization
    radius = 20

    fig, ax = plt.subplots()

    ax.imshow(skeleton)

    # Draw green circles at the specified coordinates
    for x, y in bifurcation_points:
        circle = patches.Circle((x, y), radius, fill=False, color="green", lw=2)
        ax.add_patch(circle)

    # Display the image with green circles
    plt.show()

    distance_threshold = 3  # Adjust this threshold to filter bigger search area

    filtered_list = filter_close_coordinates(bifurcation_points, distance_threshold)

    print(filtered_list)

    #output will be a list of points that are inside reference segment
    filtered_list = [point for point in filtered_list if reference_segment.contains(Point(point))]
    print(filtered_list, len(filtered_list))
    return filtered_list

def count_6_vertex(seg_history, vessel_map):
    all_history = [_ for _ in seg_history.values() if _]; all_history = [item for sublist in all_history for item in sublist]
    plt.imshow(np.array([poly_to_mask(_) for _ in all_history]).sum(axis=0))
    plt.show()
    plt.imshow(vessel_map); plt.show()

    #check if multiple vertecies on 6 and 7

    seg_6 = seg_history.get("6", [])
    seg_7 = seg_history.get("7", [])
    seg_9a = seg_history.get("9a", [])
    seg_9 = seg_history.get("9", [])

    vertexes = list_vertecies(vessel_map, unary_union([seg_6[0], seg_7[0]]))

    if len(vertexes) == 1:
        return '9'

    #intersection centroid closest to the vertex is the one we want.
    print(seg_6, seg_9)
    print(seg_6[0].buffer(0.5).intersects(seg_9[0].buffer(0.5)))
    i_c_6 = seg_6[0].buffer(1).intersection(seg_9[0].buffer(1)).centroid
    print(i_c_6)

    #determine if vertexes are higher or lower, if higher it is 9 if lower it is 9a
    the_vertex = Point(min(vertexes, key=lambda x: i_c_6.distance(Point(x))))
    other_vertex = Point(max(vertexes, key=lambda x: i_c_6.distance(Point(x))))

    print(vertexes, len(vertexes))

    print(the_vertex, other_vertex, i_c_6)
    if the_vertex.y >= other_vertex.y:
        return '9a'
    else:
        return '9'

def seg6_connection_to_seg9_should_be_9a(seg6, seg9a):
    distance_t = calculate_intersection_dist_to_t_edge(seg6, seg9a)
    distance_b = calculate_intersection_dist_to_b_edge(seg6, seg9a)
    distance_c = calculate_intersection_dist_to_centroid(seg6, seg9a)

    p("deciding 9 or 9a connection to 6:", distance_t, distance_b, distance_c)
    if distance_b < distance_t:
        if calculate_poly_a_c_dist_to_b_edge(seg6, seg9a) <= 0:
            return "9a"
        if short_edge_intersection(seg6, seg9a) and seg6.intersects(seg9a):
            return "9"
        if distance_b > distance_c:
            return "9"
        return "9a"
    return '9'


def decision_engine(seg_a, seg_b, segment_a, segment_b):
    """ """

    p(f'examining significant overlap between {segment_a} and {segment_b}')
    dice = calc_dice(seg_a, seg_b)
    p(f'dice: {dice}')
    max_intersection_area_a = intersect_area(seg_b, seg_a)
    max_intersection_area_b = intersect_area(seg_a, seg_b)
    p(f'max a to b intersection area: {max_intersection_area_a}')
    p(f'max b to a intersection area: {max_intersection_area_b}')
    if segment_a in ['1', '2', '5', '6']:
        if (dice > 0.5) | (max_intersection_area_a > 50) | (max_intersection_area_b > 50):
            return 'split'
    if max_intersection_area_a > 85:
        return 'split'
    if dice > 0.9:
        try:
            seg_a, seg_b = evenly_split_poly_space(seg_a, seg_b)
            return 'merge'
        except:
            pass
    if (max_intersection_area_a > 50.0) | (max_intersection_area_b > 70):
        return 'split'

    p(f'deciding on segment {segment_a} connection with {segment_b}')

    if segment_connected(seg_a, seg_b):
        if segment_a in ['1', '2', '3']:
            if segment_connected(seg_a, seg_b, threshold_dist=10):
                return True

        angle = get_intersection_angle(seg_a, seg_b, 150)
        closer_angle = get_intersection_angle(seg_a, seg_b, 20) or get_intersection_angle(seg_a, seg_b, 30)
        angle_direction = abs(angle) - abs(
            closer_angle) if closer_angle else 0  # if positive means angle is getting smaller as we close in meaning it is flatter.
        p(f'angles: {angle} | closer angle {closer_angle} | angle decreasing as getting closer {angle_direction > 0}')

        if segment_a == '4':
            '''look for segment at 4'''
            p(f'testing segment {segment_a} connection to {segment_b} with angle: {angle} | closer angle {closer_angle} | angle decreasing as getting closer {angle_direction > 0}')
            if segment_connected(seg_a, seg_b, threshold_dist=10) and angle < 0:
                return True

        if segment_a == '5':
            '''look for segment 6, 11 or 12'''
            if segment_b == '11':
                p(f'segment {segment_a}  with angle > 60 or < 0: {angle}')
                if (angle < 0) or (angle > 60):
                    return True
                return False

            elif segment_b == '6':
                p(f'segment {segment_a} connected with angle: {angle}')
                return True

            elif segment_b == '12':
                p(f'segment {segment_a} with angle: {angle}')
                p(f'will need to fix this area')
                if (angle < -150) or ((angle > -60) and (angle < 10)) or (angle > 135):
                    if angle_direction > 0:
                        return False
                    return True
                else:
                    return decide_12a_or_9(seg_a, seg_b)
            else:
                return False

        if segment_a == '6':
            '''look for segment 9 or 7'''
            if segment_b in ['9']:
                p(
                    f"testing segment {segment_a} connection to {segment_b} with angle not 10-60 or less than > 135: {closer_angle} | {angle}"
                )
                p((angle < 15 and angle > 60), (abs(angle) > 135), (angle < 15 and angle > 60) | (abs(angle) > 135))
                if (angle < 15) | (angle > 60) | (abs(angle) > 135):
                    return True
                return False
            elif segment_b == '7':
                p(f'testing segment {segment_a} connection to {segment_b} with angle no criteria: {angle}')
                return True
            else:
                return False

        if segment_a in ['7', '8']:
            p(f'testing segment {segment_a} connection to {segment_b}')
            if segment_b in ['9a', '10', '10a']:
                if (closer_angle and (closer_angle > 20)) | (angle and (abs(angle) > 20)):
                    p(f'testing segment {segment_a} connection to {segment_b} with angle > 15: {closer_angle} | {angle}')
                    return True
            if segment_b in ['8']:
                if is_flat(angle, 40):
                    p(f'testing segment {segment_a} connection to {segment_b} with angle < 20: {angle}')
                    return True
                return False
            else:
                return False

        if segment_a == '11':
            if segment_b in ['12a', '12b'] and (is_flat(angle, 15) == False and angle_direction < 0):
                predicted_seg = decide_closer_to_12_or_12b(seg_a, seg_b)
                p(f'testing segment {segment_a} connection to {segment_b} with angle: {angle} that should not be flat\npred_seg {predicted_seg} | {segment_b}\ncloser angle is {closer_angle}')
                if predicted_seg == segment_b:
                    return True
                return predicted_seg
            if segment_b in ['12a'] and (is_flat(angle, 35) == False):
                predicted_seg = decide_closer_to_12_or_12b(seg_a, seg_b)
                p(
                    f"testing segment {segment_a} connection to {segment_b} with angle: {angle} that should not be flat\npred_seg {predicted_seg} | {segment_b}\ncloser angle is {closer_angle}"
                )
                if predicted_seg == segment_b:
                    return True
                return predicted_seg
            if segment_b in ['13'] and any([is_flat(angle, 15) == False, abs(180 - angle) > 30]):
                p(f'testing segment {segment_a} connection is not flat: {angle}')
                return True
            return False

        if segment_a in ['13', '16']:
            p('13 decisions', is_flat(angle, 15) == False, angle_direction < 0)
            if segment_b in ['14a', '14'] and angle:
                return True
            if segment_b in ['14b']:
                p(f'testing segment {segment_a} connection to {segment_b} with angle < 20: {angle}')
                return True
            return False

    else:
        # if segment_a == "3" and segment_b == "16":
        #     angle = get_major_axis_angle(seg_b)
        #     if 35 <= angle <= 50:
        #         return True
        # if segment_a == "5" and segment_b == "12":
        #     angle = get_intersection_angle(seg_a, seg_b)
        #     p(
        #         f"segment {segment_a} not connected with {segment_b} with angle: {angle}, centroid: {seg_b.centroid}"
        #     )
        #     p(f"will need to fix this area")
        #     return "12a" if seg_a.covers(seg_b.centroid) else "9"

        return False

def post_connection_engine(root_seg, output_matrix, vessel_map_mask):
    # vessel_map_mask is 27x512x512

    if root_seg == "1":
        # get segs 1, 2, 3, 16
        segs1 = [get_contour_poly(output_matrix[1])] if (output_matrix[1].sum() > 0) else []
        segs2 = [get_contour_poly(output_matrix[2])] if (output_matrix[2].sum() > 0) else []
        segs3 = [get_contour_poly(output_matrix[3])] if (output_matrix[3].sum() > 0) else []
        segs4 = [get_contour_poly(output_matrix[4])] if (output_matrix[4].sum() > 0) else []
        segs16 = [get_contour_poly(output_matrix[20])] if (output_matrix[20].sum() > 0) else []

        print(
            f"seg 1 {True if segs1 else False}, seg 2 {True if segs2 else False}, seg 3 {True if segs3 else False}, seg 4 {True if segs4 else False}, seg 16 {True if segs16 else False}")

        # find segment2 if 1 and 3 exist but 2 does not
        if (segs1 and segs3) and not segs2:
            print(f'filling in segment 2')
            # fill in the space between 1 and 3 using the vessel map, segment 2 should be between 1 and 3 and to the left in the space
            a, b = segs1[0], segs3[0]
            y_min, y_max = lowest_coord(a).y, highest_coord(b).y
            x_min, x_max = (
                min(lowest_coord(a).x, highest_coord(b).x) - 50,
                max(lowest_coord(a).x, highest_coord(b).x) + 50,
            )

            roi = bounding_box_mask(y_min, y_max, x_min, x_max)

            output_matrix[2][roi == 1] = vessel_map_mask[roi == 1]

            return output_matrix

        # find segment 1 if 2 exists but 1 does not
        if (segs2) and not segs1:
            print(f"filling in segment 1 {vessel_map_mask[0].sum()}")
            a = segs2[0]
            # a.centroid is lowest corner
            c = a.centroid
            y_min, y_max = 0, c.y
            x_min, x_max = c.x, c.x + 200

            roi = bounding_box_mask(y_min, y_max, x_min, x_max)

            output_matrix[1][roi == 1] = vessel_map_mask[roi == 1]

            return output_matrix

        # if seg 2 and 4 not 3
        if (segs2 and segs4) and not segs3:
            print(f"filling in segment 3")

            # shoudl go from right most of seg 2 to left most of seg 4
            a, b = segs2[0], segs4[0]

            y_min, y_max = lowest_coord(a).y, highest_coord(b).y
            x_min, x_max = (
                min(lowest_coord(a).x, highest_coord(b).x),
                max(lowest_coord(a).x, highest_coord(b).x),
            )

            roi = bounding_box_mask(y_min - 10, y_max + 10, x_min, x_max)

            output_matrix[3][roi == 1] = vessel_map_mask[roi == 1] - output_matrix[[1, 2], ...].max(axis=0)[roi == 1]

            return output_matrix
    else:
        segs5 = [get_contour_poly(output_matrix[5])] if (output_matrix[5].sum() > 0) else []
        segs6 = [get_contour_poly(output_matrix[6])] if (output_matrix[6].sum() > 0) else []
        segs7 = [get_contour_poly(output_matrix[7])] if (output_matrix[7].sum() > 0) else []
        segs8 = [get_contour_poly(output_matrix[8])] if (output_matrix[8].sum() > 0) else []
        if segs6 and not segs5:
            print(f"filling in segment 5")
            a = segs6[0]
            # a.centroid is lowest corner
            y_min, y_max = 0, left_coord(a).y + 50
            x_min, x_max = 0, left_coord(a).x + 15

            roi = bounding_box_mask(y_min, y_max, x_min, x_max)

            output_matrix[5][roi == 1] = vessel_map_mask[roi == 1]

            return output_matrix

        if (segs5 and segs7) and not segs6:
            print(f"filling in segment 6")

            a, b = segs5[0], segs7[0]
            y_max, y_min = (left_coord(b).y, right_coord(a).y) if left_coord(b).y > right_coord(a).y else (
            right_coord(a).y, left_coord(b).y)
            x_min, x_max = (
                min(right_coord(a).x, left_coord(b).x),
                max(right_coord(a).x, left_coord(b).x),
            )

            roi = bounding_box_mask(y_min - 10, y_max + 10, x_min, x_max)

            output_matrix[6][roi == 1] = vessel_map_mask[roi == 1]


        if (segs6 and segs8) and not segs7:
            print(f"filling in segment 7")

            a, b = segs6[0], segs8[0]
            y_max, y_min = (left_coord(b).y, right_coord(a).y) if left_coord(b).y > right_coord(a).y else (
            right_coord(a).y, left_coord(b).y)
            x_min, x_max = (
                min(right_coord(a).x, left_coord(b).x),
                max(right_coord(a).x, left_coord(b).x),
            )

            roi = bounding_box_mask(y_min, y_max, x_min - 10, x_max + 10)

            output_matrix[7][roi == 1] = vessel_map_mask[roi == 1]

    return output_matrix

def logic_engine(seg_a, sub_segments, candidate, category_list, segs, seg_history, vessel_map):
    seg_index = {v: k + 1 for k, v in seg_dict.items()}
    p(f"logic engine for {candidate}")

    if candidate == "16":
        seg_a = seg_history["16"][0]
        sub_segments = ["16a", "16b", "16c"]

        segments = [seg_history.get(_) for _ in sub_segments]

        category_list_out = category_list.copy()
        seg_history_out = {k: v for k, v in seg_history.items()}

        idx_16a = seg_inverse["16a"]
        idx_16b = seg_inverse["16b"]
        idx_16c = seg_inverse["16c"]

        left_order = idx_left(segments)
        p(left_order, np.argmin(left_order), segments)
        if segments[np.argmin(left_order)]:
            if segment_16a_should_be_16b(seg_a, segments[np.argmin(left_order)][0]):
                seg_idx = seg_index.get(sub_segments[np.argmin(left_order)])

                mask_ids = segs[seg_idx]

                for m_id in mask_ids:
                    category_list_out[m_id] = idx_16b - 1
                    seg_history_out["16b"] = seg_history["16a"]
                    seg_history["16a"] = []
                    p(f"switching seg 16a to 16b")

        if segment_16b_should_be_16c(seg_a, segments[1][0]):
            seg_idx = seg_index.get(sub_segments[1])

            mask_ids = segs[seg_idx]

            for m_id in mask_ids:
                category_list_out[m_id] = idx_16c - 1
                seg_history_out["16c"] = seg_history["16b"]
                seg_history["16b"] = []
                p(f"switching seg 16b to 16c")

        p("returning", category_list_out, seg_history_out)
        return category_list_out, seg_history

    if candidate == "6_11":
        seg_6 = seg_history.get("6", [])
        seg_11 = seg_history.get("11", [])
        category_list_out = category_list.copy()
        if seg_6 and seg_11:
            overlap = calc_dice(seg_6[0], seg_11[0])
            if overlap > 0.9:
                # check if seg should be 6 or 11
                seg_7 = seg_history.get("7", [])
                seg_13 = seg_history.get("13", [])
                if seg_7 and segment_connected(seg_7[0], seg_11[0]):
                    print(
                        "seg 11 should not be connected to seg 7, because significant overlap, assigning to seg 6"
                    )
                    seg_6 = unary_union([seg_6[0], seg_11[0]])
                    seg_history["6"] = [seg_6]
                    seg_history["11"] = []
                elif seg_13 and segment_connected(seg_6[0], seg_13[0]):
                    print(
                        "seg 6 should not be connected to seg 13, because significant overlap, assigning to seg 11"
                    )
                    seg_11 = unary_union([seg_11[0], seg_6[0]])
                    seg_history["11"] = [seg_11]
                    seg_history["6"] = []
        return category_list_out, seg_history

    if candidate == "11":
        seg_a = seg_history.get("11", [])
        sub_segments = ["12", "12a", "12b"]

        segments = [seg_history.get(_) for _ in sub_segments]

        category_list_out = category_list.copy()
        seg_history_out = {k: v for k, v in seg_history.items()}

        idx_12 = seg_inverse["12"]
        idx_12a = seg_inverse["12a"]
        idx_12b = seg_inverse["12b"]

        bottom_order = idx_bot(segments)
        p(bottom_order, np.argmin(bottom_order), segments)
        if seg_a and segments[np.argmin(bottom_order)]:
            if segment_12a_should_be_12(seg_a, segments[np.argmin(bottom_order)][0]):
                seg_idx = seg_index.get(sub_segments[np.argmin(bottom_order)])

                mask_ids = segs[seg_idx]

                for m_id in mask_ids:
                    category_list_out[m_id] = idx_12 - 1
                    seg_history_out["12"] = seg_history["12a"]
                    seg_history["12a"] = []
                    p(f"switching seg 12a to 12")

        if seg_a and segment_12b_should_be_12a(seg_a, segments[1][0]):
            seg_idx = seg_index.get(sub_segments[1])

            mask_ids = segs[seg_idx]

            for m_id in mask_ids:
                category_list_out[m_id] = idx_12b - 1
                seg_history_out["12a"] = seg_history["12b"]
                seg_history["12b"] = []
                p(f"switching seg 12b to 12a")

        return category_list_out, seg_history

    # if candidate == "6":
    #     segments = {k: v[0] for k, v in seg_history.items() if v}
    #     # if segments includes '5', '6', '11', '13':
    #     if all([k in segments.keys() for k in ["5", "6", "11", "13"]]):
    #         # delete 6
    #         seg_6 = seg_history.get("6", [])
    #         seg_5 = seg_history.get("5", [])
    #         if seg_6 and seg_5:
    #             seg_6_angle = get_intersection_angle(seg_5[0], seg_6[0])
    #             if seg_6_angle > 10:
    #                 seg_history["6"] = []
    #                 if seg_history.get('7', []):
    #                     seg_history["7"] = []

    if candidate == "9":
        seg_6 = seg_history.get("6", [])
        seg_7 = seg_history.get("7", [])
        seg_9a = seg_history.get("9a", [])
        seg_9 = seg_history.get("9", [])
        if seg_6 and seg_7 and seg_9:
            seg_name = seg6_connection_to_seg9_should_be_9a(seg_6[0], seg_7[0], seg_9[0])
            #
            seg_name = count_6_vertex(seg_history, vessel_map)
            if seg_name == "9a":
                if not seg_9a:
                    print("transposing 9 to 9a which is a connection to 6 not 7")
                    seg_history["9a"] = seg_9
                    seg_history["9"] = []

    # if candidate == "9":
    #     seg_5 = seg_history.get("5", [])
    #     seg_6 = seg_history.get("6", [])
    #     seg_7 = seg_history.get("7", [])
    #     seg_8 = seg_history.get("8", [])
    #     seg_9 = seg_history.get("9", [])
    #     # if all segs 5, 6, 7, 8 ,9 are true and all other segments are empty
    #     if seg_5 and seg_6 and seg_7 and seg_8 and seg_9:
    #         if any([seg_history.get(_, []) for _ in
    #                 ['9a', '10', '10a', '13', '14', '14a', '14b', '15', '16', '16a', '16b', '16c']]):
    #             print('not all other segments are empty')
    #             pass
    #         else:
    #             print('since all segments are true, we will delete 9')
    #             seg_history["9"] = []

    return category_list, seg_history

def logical_constructor_ensemble(results, vessel_map, category_list, mask, filter_=200):
    gt_mask = np.zeros([27, 512, 512])
    p('initial indicies: ', category_list);
    original_cat_list = category_list.copy()

    # all possible segments
    segs = {n + 1: list(np.where(np.isin(category_list, [n]))[0].flatten()) for n in range(26)};
    p(segs);
    p({seg_dict[k - 1]: v for k, v in segs.items()})
    seg_index = {v: k + 1 for k, v in seg_dict.items()}

    if len(segs[seg_index["1"]]) > 0:
        segment = "1"
        visited_segments = ["1"]
    elif len(segs[seg_inverse["5"]]) > 0:
        segment = "5"
        visited_segments = ["5"]
    elif len(segs[seg_inverse["6"]]) > 0:
        segment = "6"
        visited_segments = ["5", "6"]
    elif len(segs[seg_inverse["2"]]) > 0:
        segment = "2"
        visited_segments = ["1", "2"]
    else:
        p('no seed segment')
        return result_matrix_single(results)

    seed_segment = segment

    filtered_segs = {k: v for k, v in segs.items() if seg_dict.get(k - 1) in (left_right.get(segment, []) + [segment])}
    empty_segs = {k: [] for k in segs.keys() - filtered_segs.keys()}
    result = {**filtered_segs, **empty_segs}
    segs = result
    p(segs);
    p({seg_dict[k - 1]: v for k, v in segs.items()})
    seg_index = {v: k + 1 for k, v in seg_dict.items()}

    assert ((len(segs[seg_index['5']]) > 0 | len(segs[seg_index['6']]) > 0) and (
            len(segs[seg_index['1']]) > 0 | len(segs[seg_index['2']]) > 0)) == False, 'two seeding segments'
    assert ((len(segs[seg_index['5']]) == 0 | len(segs[seg_index['6']]) == 0) and (
            len(segs[seg_index['1']]) == 0 | len(segs[seg_index['2']]) == 0)) == False, 'empty segments'

    seg_idx = seg_index[segment]
    mask_ids = segs[seg_idx]
    seg_history = {k: [] for k, v in seg_index.items()}

    p(f'starting at segment {segment} | num_objects: {len(segs[seg_index.get(segment)])}')
    reference_segment = get_contour_poly(mask[mask_ids[0]]);
    seg_history[segment] += [reference_segment];
    p(f'adding_segment {segment}')

    next_segs = visited_segments

    possible_segs = [_ for _ in left_right[segment] if _ not in next_segs]

    possible_segments = possible_segs + [k for k, v in seg_inverse.items() if (len(segs[v]) > 0) and (
            k not in next_segs + visited_segments + possible_segs)];
    p('possible segments: ', possible_segments)

    seg_queue = next_segs + possible_segments

    while len(seg_queue) > 0:
        segment = seg_queue.pop(0)
        p("looping segment:", segment, "| queue:", seg_queue)

        seg_idx = seg_index[segment]

        next_segs = connections[segment]
        p("upcoming segs", next_segs, mask_ids)

        mask_ids = segs[seg_idx]

        if len(mask_ids) == 0:
            continue

        #     if len(seg_queue) == 0:
        #         # Check lesion consistency
        #         # seg_history[segment] += [get_contour_poly(mask[mask_idx[0]])]
        #         break

        # len(measure.find_contours(poly_to_mask(unary_union([get_contour_poly(mask[_]) for _ in mask_ids]))))
        p(f"current segment {segment}, {seg_idx}, {mask_ids}")
        current_segment = get_contour_poly(mask[mask_ids, ...].max(axis=0))

        for ii, next_segment in enumerate(next_segs, 1):
            p(f"Next Segment list {next_segs}")
            p(
                f"{ii} checking for segment {parent_connections.get(segment)}_{segment} connection"
            )

            mask, segs, seg_history, category_list = parent_engine(
                mask,
                segs,
                seg_history,
                category_list,
                f"{parent_connections.get(segment)}_{segment}",
            )
            p(segs)

            p(f"{ii} checking for segment {segment} connection with {next_segment}")

            seg_idx = seg_index.get(next_segment)

            mask_ids = segs[seg_idx]
            p("next seg id", mask_ids, seg_idx)

            if next_segment in seg_queue:
                queue_idx = seg_queue.index(next_segment)
            else:
                p("segment not found in queue, putting in line")
                queue_idx = 0
            if queue_idx + 1 == len(seg_queue):
                next_in_queue = seg_queue[queue_idx]
            else:
                next_in_queue = seg_queue[queue_idx + 1]

            if len(mask_ids) > 1:
                p('multple segments available engine')
                action, temp_idx = multiple_segment_engine(current_segment,
                                                           [get_contour_poly(mask[_]) for _ in mask_ids], segment,
                                                           next_segment)
                p(action, 'mask index:', temp_idx, type(action))
                if type(action) == type(dict()):
                    temp_dict = {k: v for k, v in action.items()}
                    for k, (seg_, idx_) in enumerate(action.items()):
                        idx_ = [mask_ids[_] for _ in idx_]
                        p(f'{k} moving {idx_} from {next_segment} to {seg_}')
                        p('cat_list: ', category_list)
                        for j in idx_:
                            category_list[j] = seg_inverse[seg_] - 1
                        p('cat_list: ', category_list)
                        temp = segs[seg_inverse[seg_]] + idx_
                        p(segs)
                        segs[seg_inverse[seg_]] = idx_ if seg_ == next_segment else [_ for _ in temp]
                        p(segs)
                        p(f'{idx_} moved from {next_segment} to {seg_}')
                    mask_ids = segs.get(seg_inverse[next_segment], [])
                    p(f'new mask_ids {mask_ids}')

                    if len(mask_ids) > 0:
                        p(f'post action write merging code for segment {next_segment}, {mask_ids}, {seg_idx}')
                        merged_mask = mask[mask_ids, ...].sum(axis=0)
                        seg_history[next_segment] = [get_contour_poly(merged_mask)];
                        p(f'adding_segment {next_segment}');
                        for m_id in mask_ids:
                            category_list[m_id] = seg_idx - 1
                            mask[m_id] = merged_mask
                            p(f'adding_segment {next_segment}, {seg_idx - 1}, {m_id} and updating mask')
                            # seg_history[next_segment] += [get_contour_poly(mask[i])]
                        mask_ids = [mask_ids[0]]

                elif action == 'cycle':
                    p(f'recycling {next_segment} to next_segs')
                    # next_segs.append(next_segment)
                    if seg_queue[0] != segment:
                        seg_queue.insert(0, segment)

                    p(f'previous seg_idx {segs[seg_idx]}')
                    segs[seg_idx] = temp_idx[1:] + [temp_idx[0]]
                    p(f'new seg_idx {segs[seg_idx]}')


                elif action == 'push':
                    p('mask:', mask_ids, 'new:', temp_idx)
                    mask_ids, temp_idx = [mask_ids[i] for i in range(len(mask_ids)) if i in temp_idx], [mask_ids[i] for
                                                                                                        i in range(
                            len(mask_ids)) if i not in temp_idx]

                    p(f'mask for {next_segment}: {mask_ids}, new to {next_in_queue}: {temp_idx}')  # seg_queue[ii]
                    p('cat_list: ', category_list)
                    for j in temp_idx:
                        category_list[j] = seg_inverse[next_in_queue] - 1
                        p(f'pushing {temp_idx} to {next_in_queue}, leaving {mask_ids} with {next_segment}')
                    p('cat_list: ', category_list)
                    p(segs)
                    segs[seg_inverse[next_in_queue]] = segs[seg_inverse[next_in_queue]] + temp_idx
                    segs[seg_inverse[next_segment]] = mask_ids
                    p(segs)

                    if len(mask_ids) > 0:
                        p(f'write merging code for segment {next_segment}, {mask_ids}, {seg_idx}')
                        merged_mask = mask[mask_ids, ...].sum(axis=0)
                        seg_history[next_segment] = [get_contour_poly(merged_mask)];
                        p(f'adding_segment {next_segment}');
                        for m_id in mask_ids:
                            category_list[m_id] = seg_idx - 1
                            mask[m_id] = merged_mask
                            p(f'adding_segment {next_segment}, {seg_idx - 1}, {m_id} and updating mask')
                            # seg_history[next_segment] += [get_contour_poly(mask[i])]
                        mask_ids = [mask_ids[0]]
                        segs[seg_inverse[next_segment]] = mask_ids

                elif action == 'merge':
                    p(f'write merging code for segment {next_segment}, {mask_ids}, {seg_idx}')
                    merged_mask = mask[mask_ids, ...].max(axis=0)
                    seg_history[next_segment] = [get_contour_poly(merged_mask)];
                    p(f'adding_segment {next_segment}');
                    for m_id in mask_ids:
                        category_list[m_id] = seg_idx - 1
                        mask[m_id] = merged_mask
                        p(f'adding_segment {next_segment}, {seg_idx - 1}, {m_id} and updating mask')
                        # seg_history[next_segment] += [get_contour_poly(mask[i])]
                    mask_ids = [mask_ids[0]]
                    segs[seg_inverse[next_segment]] = mask_ids

                else:
                    temp_idx = [mask_ids.pop(temp_idx)]
                    for j in mask_ids:
                        category_list[j] = seg_inverse[next_in_queue] - 1
                        p(f'pushing {mask_ids} to {next_in_queue}, leaving {temp_idx} with {next_segment}')
                    segs[seg_inverse[next_in_queue]] = segs[seg_inverse[next_in_queue]] + mask_ids
                    mask_ids = temp_idx
                    segs[seg_inverse[next_segment]] = mask_ids

                p('len of mask_ids after merge operation: ', len(mask_ids))

            if len(mask_ids) == 1:

                p(f'trying to extend {segment} with {next_segment}, {mask_ids}, {seg_queue}')
                seg_next = get_contour_poly(mask[mask_ids[0]])
                decision = decision_engine(current_segment, seg_next, segment, next_segment)
                if isinstance(decision, bool) == False:
                    if decision == 'merge':
                        current_segment = unary_union([current_segment, seg_next])
                        seg_history[segment] = [current_segment]
                        p(f'segment merged into {segment}')
                        mask[mask_ids[0]] = poly_to_mask(current_segment)
                    elif decision == 'split':
                        current_segment, seg_next = evenly_split_poly_space(current_segment, seg_next)
                        seg_history[segment] = [current_segment]
                        seg_history[next_segment] = [seg_next]
                        p(f'segment split into {segment} and {next_segment}')
                        mask[mask_ids[0]] = poly_to_mask(seg_next)
                        seg_queue.insert(0, next_segment)
                        p(f'recycling segment {next_segment} inserted to queue')
                    else:
                        p(f'moving {next_segment} to {decision}')
                        p('cat_list: ', category_list)
                        category_list[mask_ids[0]] = seg_inverse[decision] - 1
                        p('cat_list: ', category_list)
                        temp = segs[seg_inverse[decision]] + mask_ids
                        segs[seg_inverse[next_segment]] = [_ for _ in segs[seg_inverse[next_segment]] if _ not in temp]
                        p(segs, decision, temp, seg_inverse[decision])
                        segs[seg_inverse[decision]] = mask_ids if decision == next_segment else [_ for _ in temp]
                        p(segs)
                        p(f'{mask_ids} moved from {next_segment} to {decision}')
                        seg_queue.insert(0, decision)

                elif decision:
                    p(f'segment {next_segment} connected to {segment} as continuous segment')
                    seg_history[next_segment] = [seg_next] if not seg_history[next_segment] else [
                        unary_union([seg_next, seg_history[next_segment][0]])];
                    p(f'adding_segment {next_segment}');
                elif decision == False:
                    mask_ids, temp_idx = [], mask_ids
                    p('mask:', mask_ids, 'new:', temp_idx)
                    p(seg_queue)
                    p(f'mask for {next_segment}: {mask_ids}, new to {next_in_queue}: {temp_idx}')
                    p('cat_list: ', category_list)
                    for j in temp_idx:
                        category_list[j] = seg_inverse[next_in_queue] - 1
                        p(f'pushing {temp_idx} to {next_in_queue}, leaving {mask_ids} with {next_segment}')
                    p('cat_list: ', category_list)
                    p(segs)
                    if next_in_queue != next_segment:
                        segs[seg_inverse[next_in_queue]] = segs[seg_inverse[next_in_queue]] + temp_idx
                        segs[seg_inverse[next_segment]] = mask_ids
                    p(segs)

            if len(mask_ids) == 0:
                pass

    # write code to validate segments are in correct order

    try:
        category_list, seg_history = logic_engine(current_segment, next_segs, '16', category_list, segs, seg_history)
        segs = {n + 1: list(np.where(np.isin(category_list, [n]))[0].flatten()) for n in range(26)}
    except Exception as e:
        print(f'16 logic failed: {e}')

    try:
        category_list, seg_history = logic_engine(current_segment, next_segs, '11', category_list, segs, seg_history)
        segs = {n + 1: list(np.where(np.isin(category_list, [n]))[0].flatten()) for n in range(26)}
    except Exception as e:
        print(f'11 logic failed: {e}')

    try:
        category_list, seg_history = logic_engine(current_segment, next_segs, "9", category_list, segs, seg_history)
        segs = {n + 1: list(np.where(np.isin(category_list, [n]))[0].flatten()) for n in range(26)}
    except Exception as e:
        print(f'9 logic failed: {e}')

    try:
        category_list, seg_history = logic_engine(current_segment, next_segs, "6", category_list, segs, seg_history)
        segs = {n + 1: list(np.where(np.isin(category_list, [n]))[0].flatten()) for n in range(26)}
    except Exception as e:
        print(f'6 logic failed: {e}')


    p(original_cat_list)
    p(category_list)
    p(seg_inverse)
    p(seg_history)

    polygon_list_out = []
    gt_mask = np.zeros([27, 512, 512])
    number_filtered_out = 0
    for i in range(26):
        polygon_list = seg_history[seg_dict[i]]
        if len(polygon_list) > 0:
            for j, poly in enumerate(polygon_list):
                polygon_list_out.append(poly)
                poly_ = poly
                next_segs = connections[seg_dict[i]]
                for k, child_seg in enumerate(next_segs):
                    child_seg_poly = seg_history.get(child_seg)
                    if len(child_seg_poly) > 0:
                        child_seg_poly_ii = unary_union(child_seg_poly)
                        for ii, child_seg_poly_ii in enumerate(child_seg_poly):
                            print(i, j, seg_dict[i], next_segs, child_seg)
                            try:
                                poly_, child_seg_poly_ii = evenly_split_poly_space(
                                    poly_, child_seg_poly_ii
                                )
                            except Exception as e:
                                print(e)
                            child_seg_poly[ii] = child_seg_poly_ii
                        seg_history[child_seg] = [child_seg_poly_ii]
                if poly_.area > filter_:
                    try:
                        new_mask = poly_to_mask(poly_.buffer(1, join_style=1))
                    except:
                        print("not buffered")
                        new_mask = poly_to_mask(poly_.buffer(1, join_style=1))
                    gt_mask[i + 1] += (gt_mask[i + 1] > 0) | (new_mask > 0)
                    polygon_list[j] = poly_
                else:
                    number_filtered_out += 1

            seg_history[seg_dict[i]]

    print('predicted classes:\t', [seg_dict[i - 1] for i in range(1, 27) if gt_mask[i].max() > 0])
    print('filtered out:\t', number_filtered_out)

    return gt_mask

def logical_constructor(results):
    gt_mask = np.zeros([27, 512, 512])

    # category map
    category_list = results.boxes.cls.to('cpu').numpy()
    p('initial indicies: ', category_list);
    original_cat_list = category_list.copy();
    mask = results.masks.data.to('cpu').numpy()
    N = len(category_list)

    # all possible segments
    segs = {n + 1: list(np.where(np.isin(category_list, [n]))[0].flatten()) for n in range(26)};
    p(segs);
    p({seg_dict[k - 1]: v for k, v in segs.items()})
    seg_index = {v: k + 1 for k, v in seg_dict.items()}

    if len(segs[seg_index['1']]) > 0:
        segment = '1'
        visited_segments = ['1']
    elif len(segs[seg_inverse['5']]) > 0:
        segment = '5'
        visited_segments = ['5']
    elif len(segs[seg_inverse['6']]) > 0:
        segment = '6'
        visited_segments = ['5', '6']
    # elif len(segs[seg_inverse['2']]) > 0:
    #     segment = '2'
    #     finished_segments = ['2']
    else:
        p('no seed segment')
        return result_matrix_single(results)

    filtered_segs = {k: v for k, v in segs.items() if seg_dict.get(k - 1) in (left_right.get(segment, []) + [segment])}
    empty_segs = {k: [] for k in segs.keys() - filtered_segs.keys()}
    result = {**filtered_segs, **empty_segs}
    segs = result
    p(segs);
    p({seg_dict[k - 1]: v for k, v in segs.items()})
    seg_index = {v: k + 1 for k, v in seg_dict.items()}

    assert ((len(segs[seg_index['5']]) > 0 | len(segs[seg_index['6']]) > 0) and (
            len(segs[seg_index['1']]) > 0 | len(segs[seg_index['2']]) > 0)) == False, 'two seeding segments'
    assert ((len(segs[seg_index['5']]) == 0 | len(segs[seg_index['6']]) == 0) and (
            len(segs[seg_index['1']]) == 0 | len(segs[seg_index['1']]) == 0)) == False, 'empty segments'

    seg_idx = seg_index[segment]
    mask_ids = segs[seg_idx]
    seg_history = {k: [] for k, v in seg_index.items()}

    p(f'starting at segment {segment} | num_objects: {len(segs[seg_index.get(segment)])}')
    reference_segment = get_contour_poly(mask[mask_ids[0]]);
    seg_history[segment] += [reference_segment];
    p(f'adding_segment {segment}')

    next_segs = visited_segments

    possible_segs = [_ for _ in left_right[segment] if _ not in next_segs]

    possible_segments = possible_segs + [k for k, v in seg_inverse.items() if (len(segs[v]) > 0) and (
            k not in next_segs + visited_segments + possible_segs)];
    p('possible segments: ', possible_segments)

    seg_queue = next_segs + possible_segments

    while len(seg_queue) > 0:
        segment = seg_queue.pop(0)
        p("looping segment:", segment, "| queue:", seg_queue)

        seg_idx = seg_index[segment]

        next_segs = connections[segment]
        p("upcoming segs", next_segs, mask_ids)

        mask_ids = segs[seg_idx]

        if len(mask_ids) == 0:
            continue

        p(f'current segment {segment}, {seg_idx}, {mask_ids}')
        current_segment = get_contour_poly(mask[mask_ids[0]])

        for ii, next_segment in enumerate(next_segs, 1):
            p(f"Next Segment list {next_segs}")
            p(
                f"{ii} checking for segment {parent_connections.get(segment)}_{segment} connection"
            )

            mask, segs, seg_history, category_list = parent_engine(
                mask,
                segs,
                seg_history,
                category_list,
                f"{parent_connections.get(segment)}_{segment}",
            )
            p(segs)

            p(f"{ii} checking for segment {segment} connection with {next_segment}")

            seg_idx = seg_index.get(next_segment)

            mask_ids = segs[seg_idx]
            p("next seg id", mask_ids, seg_idx)

            if next_segment in seg_queue:
                queue_idx = seg_queue.index(next_segment)
            else:
                p("segment not found in queue, putting in line")
                queue_idx = 0
            if queue_idx + 1 == len(seg_queue):
                next_in_queue = seg_queue[queue_idx]
            else:
                next_in_queue = seg_queue[queue_idx + 1]

            if len(mask_ids) > 1:
                p('multple segments available engine')
                action, temp_idx = multiple_segment_engine(current_segment,
                                                           [get_contour_poly(mask[_]) for _ in mask_ids], segment,
                                                           next_segment)
                p(action, 'mask index:', temp_idx, type(action))
                if type(action) == type(dict()):
                    temp_dict = {k: v for k, v in action.items()}
                    for k, (seg_, idx_) in enumerate(action.items()):
                        idx_ = [mask_ids[_] for _ in idx_]
                        p(f'{k} moving {idx_} from {next_segment} to {seg_}')
                        p('cat_list: ', category_list)
                        for j in idx_:
                            category_list[j] = seg_inverse[seg_] - 1
                        p('cat_list: ', category_list)
                        temp = segs[seg_inverse[seg_]] + idx_
                        p(segs)
                        segs[seg_inverse[seg_]] = idx_ if seg_ == next_segment else [_ for _ in temp]
                        p(segs)
                        p(f'{idx_} moved from {next_segment} to {seg_}')
                    mask_ids = segs.get(seg_inverse[next_segment], [])
                    p(f'new mask_ids {mask_ids}')

                    if len(mask_ids) > 0:
                        p(f'post action write merging code for segment {next_segment}, {mask_ids}, {seg_idx}')
                        merged_mask = mask[mask_ids, ...].sum(axis=0)
                        seg_history[next_segment] = [get_contour_poly(merged_mask)];
                        p(f'adding_segment {next_segment}');
                        for m_id in mask_ids:
                            category_list[m_id] = seg_idx - 1
                            mask[m_id] = merged_mask
                            p(f'adding_segment {next_segment}, {seg_idx - 1}, {m_id} and updating mask')
                            # seg_history[next_segment] += [get_contour_poly(mask[i])]
                        mask_ids = [mask_ids[0]]

                elif action == 'cycle':
                    p(f'recycling {next_segment} to next_segs')
                    # next_segs.append(next_segment)
                    if seg_queue[0] != segment:
                        seg_queue.insert(0, segment)

                    p(f'previous seg_idx {segs[seg_idx]}')
                    segs[seg_idx] = temp_idx[1:] + [temp_idx[0]]
                    p(f'new seg_idx {segs[seg_idx]}')

                elif action == "push":
                    p("mask:", mask_ids, "new:", temp_idx)
                    mask_ids, temp_idx = [
                        mask_ids[i] for i in range(len(mask_ids)) if i in temp_idx
                    ], [mask_ids[i] for i in range(len(mask_ids)) if i not in temp_idx]

                elif action == 'push':
                    p('mask:', mask_ids, 'new:', temp_idx)
                    mask_ids, temp_idx = [mask_ids[i] for i in range(len(mask_ids)) if i in temp_idx], [mask_ids[i] for
                                                                                                        i in range(
                            len(mask_ids)) if i not in temp_idx]

                    p(f'mask for {next_segment}: {mask_ids}, new to {next_in_queue}: {temp_idx}')  # seg_queue[ii]
                    p('cat_list: ', category_list)
                    for j in temp_idx:
                        category_list[j] = seg_inverse[next_in_queue] - 1
                        p(
                            f"pushing {temp_idx} to {next_in_queue}, leaving {mask_ids} with {next_segment}"
                        )
                    p("cat_list: ", category_list)
                    p(segs)
                    segs[seg_inverse[next_in_queue]] = segs[seg_inverse[next_in_queue]] + temp_idx
                    segs[seg_inverse[next_segment]] = mask_ids
                    p(segs)

                    if len(mask_ids) > 0:
                        p(f'write merging code for segment {next_segment}, {mask_ids}, {seg_idx}')
                        merged_mask = mask[mask_ids, ...].sum(axis=0)
                        seg_history[next_segment] = [get_contour_poly(merged_mask)];
                        p(f'adding_segment {next_segment}');
                        for m_id in mask_ids:
                            category_list[m_id] = seg_idx - 1
                            mask[m_id] = merged_mask
                            p(f'adding_segment {next_segment}, {seg_idx - 1}, {m_id} and updating mask')
                            # seg_history[next_segment] += [get_contour_poly(mask[i])]
                        mask_ids = [mask_ids[0]]
                        segs[seg_inverse[next_segment]] = mask_ids

                elif action == 'merge':
                    p(f'write merging code for segment {next_segment}, {mask_ids}, {seg_idx}')
                    merged_mask = mask[mask_ids, ...].max(axis=0)
                    seg_history[next_segment] = [get_contour_poly(merged_mask)];
                    p(f'adding_segment {next_segment}');
                    for m_id in mask_ids:
                        category_list[m_id] = seg_idx - 1
                        mask[m_id] = merged_mask
                        p(f'adding_segment {next_segment}, {seg_idx - 1}, {m_id} and updating mask')
                        # seg_history[next_segment] += [get_contour_poly(mask[i])]
                    mask_ids = [mask_ids[0]]
                    segs[seg_inverse[next_segment]] = mask_ids

                else:
                    temp_idx = [mask_ids.pop(temp_idx)]
                    for j in mask_ids:
                        category_list[j] = seg_inverse[next_in_queue] - 1
                        p(f'pushing {mask_ids} to {next_in_queue}, leaving {temp_idx} with {next_segment}')
                    segs[seg_inverse[next_in_queue]] = segs[seg_inverse[next_in_queue]] + mask_ids
                    mask_ids = temp_idx
                    segs[seg_inverse[next_segment]] = mask_ids

                p('len of mask_ids after merge operation: ', len(mask_ids))

            if len(mask_ids) == 1:

                p(f'trying to extend {segment} with {next_segment}, {mask_ids}, {seg_queue}')
                seg_next = get_contour_poly(mask[mask_ids[0]])
                decision = decision_engine(current_segment, seg_next, segment, next_segment)
                if isinstance(decision, bool) == False:
                    if decision == 'merge':
                        current_segment = unary_union([current_segment, seg_next])
                        seg_history[segment] = [current_segment]
                        p(f'segment merged into {segment}')
                        mask[mask_ids[0]] = poly_to_mask(current_segment)
                    elif decision == 'split':
                        current_segment, seg_next = evenly_split_poly_space(current_segment, seg_next)
                        seg_history[segment] = [current_segment]
                        seg_history[next_segment] = [seg_next]
                        p(f'segment split into {segment} and {next_segment}')
                        mask[mask_ids[0]] = poly_to_mask(seg_next)
                        seg_queue.insert(0, next_segment)
                        p(f'recycling segment {next_segment} inserted to queue')
                    else:
                        p(f'moving {next_segment} to {decision}')
                        p('cat_list: ', category_list)
                        category_list[mask_ids[0]] = seg_inverse[decision] - 1
                        p('cat_list: ', category_list)
                        temp = segs[seg_inverse[decision]] + mask_ids
                        segs[seg_inverse[next_segment]] = [_ for _ in segs[seg_inverse[next_segment]] if _ not in temp]
                        p(segs, decision, temp, seg_inverse[decision])
                        segs[seg_inverse[decision]] = mask_ids if decision == next_segment else [_ for _ in temp]
                        p(segs)
                        p(f'{mask_ids} moved from {next_segment} to {decision}')
                        seg_queue.insert(0, decision)

                elif decision:
                    p(f'segment {next_segment} connected to {segment} as continuous segment')
                    seg_history[next_segment] = [seg_next] if not seg_history[next_segment] else [
                        unary_union([seg_next, seg_history[next_segment][0]])];
                    p(f'adding_segment {next_segment}');
                elif decision == False:
                    mask_ids, temp_idx = [], mask_ids
                    p('mask:', mask_ids, 'new:', temp_idx)
                    p(seg_queue)
                    p(f'mask for {next_segment}: {mask_ids}, new to {next_in_queue}: {temp_idx}')
                    p('cat_list: ', category_list)
                    for j in temp_idx:
                        category_list[j] = seg_inverse[next_in_queue] - 1
                        p(f'pushing {temp_idx} to {next_in_queue}, leaving {mask_ids} with {next_segment}')
                    p('cat_list: ', category_list)
                    p(segs)
                    if next_in_queue != next_segment:
                        segs[seg_inverse[next_in_queue]] = segs[seg_inverse[next_in_queue]] + temp_idx
                        segs[seg_inverse[next_segment]] = mask_ids
                    p(segs)

            if len(mask_ids) == 0:
                pass


    polygon_list_out = []
    number_filtered_out = 0
    gt_mask = np.zeros([27, 512, 512])
    for i in range(26):
        polygon_list = seg_history[seg_dict[i]]
        if len(polygon_list) > 0:
            for j, poly in enumerate(polygon_list):
                polygon_list_out.append(poly)
                poly_ = poly
                next_segs = connections[seg_dict[i]]
                for k, child_seg in enumerate(next_segs):
                    child_seg_poly = seg_history.get(child_seg)
                    if len(child_seg_poly) > 0:
                        for ii, child_seg_poly_ii in enumerate(child_seg_poly):
                            p(i, j, seg_dict[i], next_segs, child_seg)
                            if poly_.intersection(child_seg_poly_ii).area > 400:
                                try:
                                    poly_, child_seg_poly_ii = evenly_split_poly_space(poly_, child_seg_poly_ii)
                                except Exception as e:
                                    print(e)
                            child_seg_poly[ii] = child_seg_poly_ii
                        seg_history[child_seg] = [child_seg_poly_ii]
                if poly_.area > 200:
                    try:
                        new_mask = poly_to_mask(poly_.buffer(-2))
                    except:
                        print("not buffered")
                        new_mask = poly_to_mask(poly_)
                    gt_mask[i + 1] += (gt_mask[i + 1] == 1) | (new_mask == 1)
                    polygon_list[j] = poly_
                else:
                    number_filtered_out += 1

            seg_history[seg_dict[i]]

    print('predicted classes:\t', [seg_dict[i - 1] for i in range(1, 27) if gt_mask[i].max() > 0])
    print('filtered out:\t', number_filtered_out)

    return gt_mask