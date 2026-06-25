import numpy as np
import json
import cv2

input_image_path = 'output/hand_image.png'
board_seg_cfg_path = 'configs/board_seg.json'
green_seg_cfg_path = 'configs/green_seg.json'
magenta_seg_cfg_path = 'configs/magenta_seg.json'
board_rows = 6
board_cols = 7
EMPTY = 0
MAGENTA = 1
GREEN = 2

def load_config(path):
    with open(path, 'r') as config_file:
        return json.load(config_file)

def apply_morphology(mask, morphology_cfg):
    if not morphology_cfg.get('enabled', False):
        return mask

    kernel_size = max(1, morphology_cfg.get('kernel_size', 3))
    if kernel_size % 2 == 0:
        kernel_size += 1

    iterations = morphology_cfg.get('iterations', 1)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    operation = morphology_cfg.get('operation', '').lower()

    if operation == 'open':
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    if operation == 'close':
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    if operation == 'erode':
        return cv2.erode(mask, kernel, iterations=iterations)
    if operation == 'dilate':
        return cv2.dilate(mask, kernel, iterations=iterations)

    return mask

def get_segmentation_mask(bgr_image, seg_cfg):
    if seg_cfg.get('color_space') != 'HSV':
        raise ValueError(f"Unsupported color space: {seg_cfg.get('color_space')}")

    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)

    for range_cfg in seg_cfg['ranges']:
        lower = np.array(range_cfg['lower'], dtype=np.uint8)
        upper = np.array(range_cfg['upper'], dtype=np.uint8)
        range_mask = cv2.inRange(hsv_image, lower, upper)
        mask = cv2.bitwise_or(mask, range_mask)

    return apply_morphology(mask, seg_cfg.get('morphology', {}))

def flood_fill_board_from_border(board_mask):
    flood_mask = board_mask.copy()
    height, width = flood_mask.shape[:2]
    fill_value = 128

    for x in range(width):
        cv2.floodFill(flood_mask, None, (x, 0), fill_value)
        cv2.floodFill(flood_mask, None, (x, height - 1), fill_value)

    for y in range(height):
        cv2.floodFill(flood_mask, None, (0, y), fill_value)
        cv2.floodFill(flood_mask, None, (width - 1, y), fill_value)

    return np.where(flood_mask == fill_value, 255, 0).astype(np.uint8)

def get_black_clusters(mask):
    black_mask = np.where(mask == 0, 255, 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(black_mask)
    clusters = []
    for label in range(1, num_labels):
        x, y, width, height, area = stats[label]
        clusters.append(
            {
                'label': label,
                'x': int(x),
                'y': int(y),
                'width': int(width),
                'height': int(height),
                'area': int(area),
                'center_x': float(x + width / 2),
                'center_y': float(y + height / 2),
            }
        )
    return labels, clusters

def arrange_board_clusters(clusters, rows, cols):
    expected_count = rows * cols
    if len(clusters) < expected_count:
        selected_clusters = clusters
    else:
        selected_clusters = sorted(clusters, key=lambda cluster: cluster['area'], reverse=True)[:expected_count]

    selected_clusters = sorted(selected_clusters, key=lambda cluster: cluster['center_y'])
    board_grid = []
    for row_index in range(0, len(selected_clusters), cols):
        row = selected_clusters[row_index:row_index + cols]
        board_grid.append(sorted(row, key=lambda cluster: cluster['center_x']))

    return board_grid

def get_board_cluster_display(bgr_image, labels, board_grid, green_circles, magenta_circles):
    cluster_display = bgr_image.copy()
    board_state = []

    for row_index, row in enumerate(board_grid):
        board_state.append([])
        for col_index, cluster in enumerate(row):
            x = cluster['x']
            y = cluster['y']
            width = cluster['width']
            height = cluster['height']
            label = cluster['label']

            state = get_cluster_state(labels, label, green_circles, magenta_circles)
            board_state[row_index].append(state_to_value(state))

            color = (255, 255, 255)
            if state == 'G':
                color = (0, 255, 0)
            elif state == 'M':
                color = (255, 0, 255)

            cv2.rectangle(cluster_display, (x, y), (x + width, y + height), color, 2)
            cv2.putText(
                cluster_display,
                f'{row_index},{col_index}',
                (x + 5, max(0, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )

            if state is not None:
                cv2.putText(
                    cluster_display,
                    state,
                    (x + 5, y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    color,
                    2,
                    cv2.LINE_AA,
                )

    return cluster_display, board_state

def state_to_value(state):
    if state == 'M':
        return MAGENTA
    if state == 'G':
        return GREEN
    return EMPTY

def circle_center_in_cluster(labels, label, circles):
    height, width = labels.shape[:2]
    for x, y, radius in circles:
        if 0 <= x < width and 0 <= y < height and labels[y, x] == label:
            return True
    return False

def get_cluster_state(labels, label, green_circles, magenta_circles):
    if circle_center_in_cluster(labels, label, green_circles):
        return 'G'
    if circle_center_in_cluster(labels, label, magenta_circles):
        return 'M'
    return None

def detect_circles(bgr_image, mask, circle_cfg):
    if not circle_cfg.get('enabled', False):
        return []

    if circle_cfg.get('source', 'mask') == 'image':
        source = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    else:
        source = mask

    blurred = cv2.medianBlur(source, 5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=circle_cfg['dp'],
        minDist=circle_cfg['min_dist'],
        param1=circle_cfg['param1'],
        param2=circle_cfg['param2'],
        minRadius=circle_cfg['min_radius'],
        maxRadius=circle_cfg['max_radius'],
    )

    if circles is None:
        return []

    return np.round(circles[0]).astype(int).tolist()

def draw_circles(display_image, circles, color, label):
    for index, (x, y, radius) in enumerate(circles, start=1):
        cv2.circle(display_image, (x, y), radius, color, 2)
        cv2.circle(display_image, (x, y), 2, color, 3)
        cv2.putText(
            display_image,
            f'{label}{index}',
            (x - radius, max(0, y - radius - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

def detect_board_state(image, debug=False):
    if image is None:
        raise ValueError('detect_board_state expected a valid image, got None')

    board_seg_cfg = load_config(board_seg_cfg_path)
    green_seg_cfg = load_config(green_seg_cfg_path)
    magenta_seg_cfg = load_config(magenta_seg_cfg_path)

    board_mask = get_segmentation_mask(image, board_seg_cfg)
    board_mask = flood_fill_board_from_border(board_mask)

    green_mask = get_segmentation_mask(image, green_seg_cfg)
    magenta_mask = get_segmentation_mask(image, magenta_seg_cfg)
    green_circles = detect_circles(image, green_mask, green_seg_cfg['circle_detection'])
    magenta_circles = detect_circles(image, magenta_mask, magenta_seg_cfg['circle_detection'])
    cluster_labels, board_clusters = get_black_clusters(board_mask)
    board_grid = arrange_board_clusters(board_clusters, board_rows, board_cols)
    board_cluster_display, board_state = get_board_cluster_display(
        image,
        cluster_labels,
        board_grid,
        green_circles,
        magenta_circles,
    )

    if debug:
        circle_display = image.copy()
        draw_circles(circle_display, green_circles, (0, 255, 0), 'G')
        draw_circles(circle_display, magenta_circles, (255, 0, 255), 'M')
        print(f'Black clusters: {len(board_clusters)}')
        if len(board_grid) != board_rows or any(len(row) != board_cols for row in board_grid):
            print(f'Warning: arranged board as {[len(row) for row in board_grid]} clusters per row')
        print(f'Green circles: {len(green_circles)}')
        print(f'Magenta circles: {len(magenta_circles)}')
        print('Board state:')
        for row in board_state:
            print(' '.join(str(value) for value in row))

        cv2.imshow('Board Mask', board_mask)
        cv2.imshow('Green Mask', green_mask)
        cv2.imshow('Magenta Mask', magenta_mask)
        cv2.imshow('Board Clusters', board_cluster_display)
        cv2.imshow('Circle Detections', circle_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return board_state

if __name__ == '__main__':
    image = cv2.imread(input_image_path)
    detect_board_state(image, debug=True)
