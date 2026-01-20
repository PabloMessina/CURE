from typing import List, Dict
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union
# Remember to pip install shapely if not already installed:


# Mapping from common bounding box format names to Albumentations format strings
# See Albumentations documentation for details: https://albumentations.ai/docs/
BBOX_FORMAT_TO_ALBUMENTATIONS_FORMAT: Dict[str, str] = {
    "xyxy": "albumentations", # (x_min, y_min, x_max, y_max) normalized [0, 1]
    "cxcywh": "yolo", # (cx, cy, w, h) normalized [0, 1]
    "xywh": "coco", # (x_min, y_min, w, h)
    "xyxy_nonnormalized": "pascal_voc", # (x_min, y_min, x_max, y_max) non-normalized
    # Identity formats
    "albumentations": "albumentations",
    "yolo": "yolo",
    "coco": "coco",
    "pascal_voc": "pascal_voc",
}

def xyxy_to_cxcywh(xyxy):
    """
    Convert bounding box from (x1, y1, x2, y2) format to (cx, cy, w, h) format.
    
    Args:
        xyxy (list or tuple): Bounding box in (x1, y1, x2, y2) format.
        
    Returns:
        list: Bounding box in (cx, cy, w, h) format.
    """
    x1, y1, x2, y2 = xyxy
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]

def cxcywh_to_xyxy(cxcywh):
    """
    Convert bounding box from (cx, cy, w, h) format to (x1, y1, x2, y2) format.
    
    Args:
        cxcywh (list or tuple): Bounding box in (cx, cy, w, h) format.
        
    Returns:
        list: Bounding box in (x1, y1, x2, y2) format.
    """
    cx, cy, w, h = cxcywh
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return [x1, y1, x2, y2]


def calculate_bbox_union_iou(
    bboxes_list1: List[List[float]],
    bboxes_list2: List[List[float]],
    bbox_format: str = "xyxy",
    img_width: int = 1, # For unnormalized coordinates, otherwise keep 1
    img_height: int = 1 # For unnormalized coordinates, otherwise keep 1
) -> float:
    """
    Calculates the Intersection over Union (IoU) of the unions of two lists
    of bounding boxes.

    Args:
        bboxes_list1: First list of bounding boxes.
                      Each box is [x1, y1, x2, y2] or [cx, cy, w, h].
        bboxes_list2: Second list of bounding boxes.
        bbox_format: Format of the input bounding boxes ("xyxy" or "cxcywh").
        img_width: Width of the image, only used if boxes are not normalized
                   and need scaling for shapely. For normalized boxes, keep 1.
        img_height: Height of the image, only used if boxes are not normalized.
                    For normalized boxes, keep 1.

    Returns:
        The IoU of the two unioned regions. Returns 1.0 if both lists are
        empty (representing a perfect match of "no objects"). Returns 0.0 if
        one list is empty and the other is not.
    """

    # Handle empty lists
    polygons1 = []
    for bbox_orig in bboxes_list1:
        bbox = list(bbox_orig) # Ensure it's a list
        if bbox_format == "cxcywh":
            bbox = cxcywh_to_xyxy(bbox)
        if (bbox[0] < bbox[2] and bbox[1] < bbox[3]): # Ensure valid bounding box
            polygons1.append(shapely_box(
                bbox[0] * img_width,
                bbox[1] * img_height,
                bbox[2] * img_width,
                bbox[3] * img_height
            ))

    polygons2 = []
    for bbox_orig in bboxes_list2:
        bbox = list(bbox_orig) # Ensure it's a list
        if bbox_format == "cxcywh":
            bbox = cxcywh_to_xyxy(bbox)
        if (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
            polygons2.append(shapely_box(
                bbox[0] * img_width,
                bbox[1] * img_height,
                bbox[2] * img_width,
                bbox[3] * img_height
            ))

    # If both lists are empty, return 1.0 (perfect match of "no objects")
    if not polygons1 and not polygons2:
        return 1.0
    # If one list is empty, return 0.0 (no overlap)
    if not polygons1 or not polygons2:
        return 0.0
        
    union1 = unary_union(polygons1)
    union2 = unary_union(polygons2)

    intersection_area = union1.intersection(union2).area
    union_area = union1.area + union2.area - intersection_area
    assert union_area > 0
    iou = intersection_area / union_area
    return iou