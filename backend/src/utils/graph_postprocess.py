import cv2
import numpy as np
import scipy.ndimage as ndimage
from typing import List, Tuple

try:
    from skimage.morphology import skeletonize
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

def hysteresis_threshold(probs: np.ndarray, high_thresh: float = 0.35, low_thresh: float = 0.12) -> np.ndarray:
    """
    Hysteresis Thresholding: Retains weak road predictions (>= low_thresh) if they are connected
    to high-confidence road regions (>= high_thresh). Vectorized for ultra-fast, memory-efficient execution.
    """
    strong = probs >= high_thresh
    weak = (probs >= low_thresh) & (probs < high_thresh)
    
    total_candidate = (strong | weak).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(total_candidate, connectivity=8)
    
    if num_labels <= 1:
        return (strong * 255).astype(np.uint8)
        
    strong_labels = np.unique(labels[strong])
    strong_labels = strong_labels[strong_labels > 0]
    
    keep_mask = np.isin(labels, strong_labels)
    output_mask = np.zeros_like(total_candidate, dtype=np.uint8)
    output_mask[keep_mask] = 255
            
    return output_mask

def find_skeleton_endpoints(skel: np.ndarray) -> List[Tuple[int, int, np.ndarray]]:
    """
    Finds endpoints in a 1-pixel binary skeleton and computes their outward tangent direction vectors.
    Returns list of (y, x, direction_vector).
    """
    skel_bool = (skel > 0).astype(np.uint8)
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    filtered = cv2.filter2D(skel_bool, -1, kernel)
    ey, ex = np.where(filtered == 11)
    
    endpoints = []
    h, w = skel.shape
    
    for y, x in zip(ey, ex):
        patch_y1, patch_y2 = max(0, y - 4), min(h, y + 5)
        patch_x1, patch_x2 = max(0, x - 4), min(w, x + 5)
        
        py, px = np.where(skel_bool[patch_y1:patch_y2, patch_x1:patch_x2] > 0)
        py = py + patch_y1
        px = px + patch_x1
        
        if len(py) > 1:
            dy = float(y - np.mean(py[py != y])) if any(py != y) else 0.0
            dx = float(x - np.mean(px[px != x])) if any(px != x) else 0.0
            norm = np.hypot(dx, dy)
            if norm > 1e-5:
                dir_vec = np.array([dx / norm, dy / norm])
            else:
                dir_vec = np.array([0.0, 0.0])
        else:
            dir_vec = np.array([0.0, 0.0])
            
        endpoints.append((y, x, dir_vec))
        
    return endpoints

def connect_canopy_gaps(
    binary_mask: np.ndarray,
    max_gap_dist: float = 220.0,
    max_angle_deg: float = 65.0,
    road_width: int = 6,
    border_margin: int = 16,
) -> np.ndarray:
    """
    Multi-Strategy Graph-Based Post-Processing:
    1. Extracts topological 1-pixel skeleton using scikit-image skeletonize (or cv2.ximgproc as fallback).
    2. Bridges facing dead-end endpoints across wide tree canopy gaps (up to max_gap_dist).
    3. Connects dead-end endpoints to nearby main road segments (T-junction completion).

    Endpoints within ``border_margin`` px of the tile edge are ignored: there a road simply
    leaves the tile, it is not interrupted. Treating those as dead ends made the bridging
    draw spurious straight "roads" along the tile border between neighbouring exits.
    ``border_margin=0`` reproduces the original behaviour.
    """
    mask_out = binary_mask.copy()
    h, w = mask_out.shape
    
    # 1. Extract 1-pixel topological skeleton
    if HAS_SKIMAGE:
        skel = skeletonize(mask_out > 0).astype(np.uint8)
    elif hasattr(cv2, 'ximgproc'):
        skel = cv2.ximgproc.thinning(mask_out)
    else:
        # Morphological fallback
        skel = (mask_out > 0).astype(np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        done = False
        skel_acc = np.zeros(mask_out.shape, dtype=np.uint8)
        img_temp = skel.copy()
        while not done:
            eroded = cv2.erode(img_temp, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img_temp, temp)
            skel_acc = cv2.bitwise_or(skel_acc, temp)
            img_temp = eroded.copy()
            if cv2.countNonZero(img_temp) == 0:
                done = True
        skel = skel_acc

    endpoints = [(y, x, v) for y, x, v in find_skeleton_endpoints(skel)
                 if border_margin <= y < h - border_margin and border_margin <= x < w - border_margin]
    n_pts = len(endpoints)
    if n_pts == 0:
        return mask_out

    connected_pairs = []
    connected_endpoints = set()
    
    cos_threshold = np.cos(np.radians(max_angle_deg))
    
    # --- Strategy A: Endpoint-to-Endpoint Collinear & Facing Pair Connection ---
    for i in range(n_pts):
        y1, x1, v1 = endpoints[i]
        best_j = None
        best_score = float('inf')
        
        for j in range(i + 1, n_pts):
            y2, x2, v2 = endpoints[j]
            dist = float(np.hypot(x2 - x1, y2 - y1))
            if dist > max_gap_dist or dist < 5.0:
                continue
                
            gap_vec = np.array([(x2 - x1) / dist, (y2 - y1) / dist])
            
            dot1 = float(np.dot(v1, gap_vec)) if np.linalg.norm(v1) > 0 else 0.8
            dot2 = float(np.dot(v2, -gap_vec)) if np.linalg.norm(v2) > 0 else 0.8
            
            if dot1 >= cos_threshold and dot2 >= cos_threshold:
                alignment_penalty = (2.0 - dot1 - dot2) * 50.0
                score = dist + alignment_penalty
                if score < best_score:
                    best_score = score
                    best_j = j
                    
        if best_j is not None:
            y2, x2, _ = endpoints[best_j]
            connected_pairs.append(((x1, y1), (x2, y2)))
            connected_endpoints.add(i)
            connected_endpoints.add(best_j)

    # --- Strategy B: Endpoint-to-Road Edge Connection (T-Junction Completion) ---
    skel_y, skel_x = np.where(skel > 0)
    if len(skel_x) > 0:
        skel_pts = np.column_stack((skel_x, skel_y))
        
        for i in range(n_pts):
            if i in connected_endpoints:
                continue
            y1, x1, v1 = endpoints[i]
            if np.linalg.norm(v1) == 0:
                continue
                
            dists_to_skel = np.hypot(skel_pts[:, 0] - x1, skel_pts[:, 1] - y1)
            valid_idx = dists_to_skel > 25.0  # Outside local 25px radius
            
            if not np.any(valid_idx):
                continue
                
            cand_pts = skel_pts[valid_idx]
            cand_dists = dists_to_skel[valid_idx]
            
            within_dist = cand_dists <= max_gap_dist
            if not np.any(within_dist):
                continue
                
            cand_pts = cand_pts[within_dist]
            cand_dists = cand_dists[within_dist]
            
            best_cand = None
            best_cand_score = float('inf')
            
            for (cx, cy), d in zip(cand_pts, cand_dists):
                g_vec = np.array([(cx - x1) / d, (cy - y1) / d])
                dot = float(np.dot(v1, g_vec))
                if dot >= cos_threshold:
                    score = d + (1.0 - dot) * 60.0
                    if score < best_cand_score:
                        best_cand_score = score
                        best_cand = (cx, cy)
                        
            if best_cand is not None:
                connected_pairs.append(((x1, y1), best_cand))

    # Draw connecting road strokes
    for pt1, pt2 in connected_pairs:
        cv2.line(mask_out, pt1, pt2, 255, thickness=road_width)
        
    return mask_out

