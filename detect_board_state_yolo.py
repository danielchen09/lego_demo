from ultralytics import YOLO

import numpy as np
from typing import List

def center_from_xyxy(xyxy):
    return np.array([
        (xyxy[0] + xyxy[2]) / 2,
        (xyxy[1] + xyxy[3]) / 2
    ])

def detect_board_state(yolo_model, image):
    # 0 empty 1 magenta 2 green 
    # need to swap 1 and 2 since yolo is trained with 1 green 2 magenta
    results = yolo_model([image])
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.save(filename="output/yolo.jpg")

    board_raw: List[List] = []

    for result in results:
        boxes = result.boxes.cpu()

        boxes_y_sorted = sorted(boxes, key=lambda x: center_from_xyxy(x.xyxy[0])[1])
        while len(boxes_y_sorted) > 0:
            if len(board_raw) >= 6:
                break
            board_raw.append(boxes_y_sorted[:min(7, len(boxes_y_sorted))])
            boxes_y_sorted = boxes_y_sorted[min(7, len(boxes_y_sorted)):] 
    
    for row in board_raw:
        row.sort(key=lambda x: center_from_xyxy(x.xyxy[0])[0])
    
    board = []
    for row in board_raw:
        board.append([])
        for col in row:
            state = int(col.cls.item())
            if state == 1:
                state = 2
            elif state == 2:
                state = 1
            board[-1].append(state)
    
    return board


if __name__ == '__main__':
    model = YOLO('models/exp-4.pt')
    board = detect_board_state(model, 'data/board/1782502491.png')
    for row in board:
        print(' '.join(map(str, row)))