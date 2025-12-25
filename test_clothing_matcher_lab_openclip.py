#!/usr/bin/env python3
"""
Clothing Matcher Test Script - Lab + OpenCLIP Version

Supports comparing 2 or 3 images at once.

Usage:
    # Compare 2 images
    python test_clothing_matcher_lab_openclip.py --img1 person1.jpg --img2 person2.jpg
    
    # Compare 3 images (pairwise)
    python test_clothing_matcher_lab_openclip.py --img1 p1.jpg --img2 p2.jpg --img3 p3.jpg
    
    # Camera capture mode
    python test_clothing_matcher_lab_openclip.py --capture [--realsense]
"""

import argparse
import sys
import cv2
import numpy as np
from clothing_matcher_lab_openclip import ClothingMatcher, SegmentationError


def parse_args():
    p = argparse.ArgumentParser(description="Clothing Matcher Test - Lab + OpenCLIP")
    p.add_argument('--img1', type=str, help='Path to first image')
    p.add_argument('--img2', type=str, help='Path to second image')
    p.add_argument('--img3', type=str, help='Path to third image (optional)')
    p.add_argument('--yolo-seg', type=str, 
                   default='/home/openmind/Desktop/wendy-work-station/yolo11s-seg.raw.engine')
    p.add_argument('--capture', action='store_true', help='Camera capture mode')
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--realsense', action='store_true')
    p.add_argument('--debug', action='store_true')
    p.add_argument('--clip-model', type=str, default='ViT-B-16')
    p.add_argument('--clip-pretrained', type=str, default='laion2b_s34b_b88k')
    p.add_argument('--clothing-threshold', type=float, default=0.5)
    p.add_argument('--clip-threshold', type=float, default=0.7)
    return p.parse_args()


def visualize_features(image: np.ndarray, mask: np.ndarray, features: dict, title: str) -> np.ndarray:
    """Create visualization of extracted Lab features"""
    H, W = image.shape[:2]
    canvas = np.zeros((H + 120, W, 3), dtype=np.uint8)
    canvas[:H, :W] = image.copy()
    
    # Draw mask overlay
    if mask is not None:
        mask_overlay = np.zeros_like(image)
        mask_overlay[:, :, 1] = mask * 255
        canvas[:H, :W] = cv2.addWeighted(canvas[:H, :W], 0.7, mask_overlay, 0.3, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas[:H, :W], contours, -1, (0, 255, 0), 2)
    
    cv2.putText(canvas, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if features is None:
        cv2.putText(canvas, "No features", (10, H + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return canvas
    
    # Average color swatch
    avg_color = features['avg_color']
    cv2.rectangle(canvas, (10, H + 10), (60, H + 60), tuple(map(int, avg_color)), -1)
    cv2.rectangle(canvas, (10, H + 10), (60, H + 60), (255, 255, 255), 1)
    cv2.putText(canvas, "Avg", (20, H + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    
    # Lab values
    cv2.putText(canvas, f"a: {features['avg_a']:.0f}", (70, H + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(canvas, f"b: {features['avg_b']:.0f}", (70, H + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(canvas, f"edge: {features['edge_density']:.3f}", (70, H + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    
    # Draw a-channel histogram (green-red axis)
    bar_w, bar_h = min(W - 140, 150), 15
    bar_x, bar_y = 130, H + 15
    
    a_hist = features['a_hist']
    a_norm = a_hist / (a_hist.max() + 1e-6)
    for i, val in enumerate(a_norm):
        x = bar_x + int(i * bar_w / len(a_norm))
        h = int(val * bar_h)
        color = (0, int(255 * (1 - i/len(a_norm))), int(255 * i/len(a_norm)))
        cv2.rectangle(canvas, (x, bar_y + bar_h - h), (x + bar_w // len(a_norm), bar_y + bar_h), color, -1)
    cv2.putText(canvas, "a (G-R)", (bar_x, bar_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
    
    # Draw b-channel histogram (blue-yellow axis)
    bar_y2 = H + 50
    b_hist = features['b_hist']
    b_norm = b_hist / (b_hist.max() + 1e-6)
    for i, val in enumerate(b_norm):
        x = bar_x + int(i * bar_w / len(b_norm))
        h = int(val * bar_h)
        color = (int(255 * (1 - i/len(b_norm))), int(255 * i/len(b_norm)), int(255 * i/len(b_norm)))
        cv2.rectangle(canvas, (x, bar_y2 + bar_h - h), (x + bar_w // len(b_norm), bar_y2 + bar_h), color, -1)
    cv2.putText(canvas, "b (B-Y)", (bar_x, bar_y2 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
    
    return canvas


def extract_features(matcher: ClothingMatcher, img: np.ndarray, name: str, debug: bool = False):
    """Extract all features from an image"""
    print(f"\nExtracting features from {name}...")
    try:
        mask = matcher.extract_person_mask_from_crop(img, debug=debug)
        coverage = mask.sum() / mask.size * 100
        print(f"  ✓ Mask coverage: {coverage:.1f}%")
        
        feat = matcher.extract_clothing_features(img, mask)
        print(f"  ✓ Lab features extracted")
        
        clip_emb = None
        if matcher.clip_matcher:
            clip_emb = matcher.extract_clip_embedding(img, mask)
            print(f"  ✓ CLIP embedding extracted ({clip_emb.shape[0]} dim)")
        
        return mask, feat, clip_emb, coverage
    except (SegmentationError, ValueError) as e:
        print(f"  ✗ Failed: {e}")
        return None, None, None, 0.0


def compare_pair(matcher: ClothingMatcher, 
                 feat1: dict, clip1: np.ndarray, 
                 feat2: dict, clip2: np.ndarray,
                 name1: str, name2: str,
                 clothing_thresh: float = 0.5,
                 clip_thresh: float = 0.7):
    """Compare a pair of features and return similarities"""
    lab_sim = 0.0
    clip_sim = 0.0
    
    if feat1 and feat2:
        lab_sim = matcher.compute_clothing_similarity(feat1, feat2)
    
    if clip1 is not None and clip2 is not None:
        clip_sim = matcher.compute_clip_similarity(clip1, clip2)
    
    # Determine match status
    lab_pass = lab_sim >= clothing_thresh
    clip_pass = clip_sim >= clip_thresh if (clip1 is not None and clip2 is not None) else True
    
    if lab_pass and clip_pass:
        status = "MATCH"
        color = (0, 255, 0)
    elif lab_pass and clip_sim >= clip_thresh * 0.85:  # Close to threshold
        status = "LIKELY"
        color = (0, 200, 255)
    elif lab_pass:
        status = "UNCERTAIN"
        color = (0, 165, 255)
    else:
        status = "NO MATCH"
        color = (0, 0, 255)
    
    return {
        'lab_sim': lab_sim,
        'clip_sim': clip_sim,
        'lab_pass': lab_pass,
        'clip_pass': clip_pass,
        'status': status,
        'color': color,
        'name1': name1,
        'name2': name2
    }


def compare_images(matcher: ClothingMatcher, images: list, names: list, 
                   clothing_thresh: float = 0.5, clip_thresh: float = 0.7, debug: bool = False):
    """Compare multiple images and display results"""
    print("\n" + "=" * 60)
    print("CLOTHING MATCHER - Lab + OpenCLIP")
    print("=" * 60)
    
    # Extract features from all images
    features = []
    for img, name in zip(images, names):
        mask, feat, clip_emb, coverage = extract_features(matcher, img, name, debug)
        features.append({
            'img': img,
            'mask': mask,
            'feat': feat,
            'clip': clip_emb,
            'coverage': coverage,
            'name': name
        })
    
    # Pairwise comparisons
    print("\n" + "=" * 60)
    print("PAIRWISE COMPARISONS")
    print("=" * 60)
    
    comparisons = []
    n = len(features)
    for i in range(n):
        for j in range(i + 1, n):
            f1, f2 = features[i], features[j]
            result = compare_pair(
                matcher,
                f1['feat'], f1['clip'],
                f2['feat'], f2['clip'],
                f1['name'], f2['name'],
                clothing_thresh, clip_thresh
            )
            comparisons.append(result)
            
            # Print result
            print(f"\n{f1['name']} vs {f2['name']}:")
            print(f"  Lab:  {result['lab_sim']:.4f} {'✓' if result['lab_pass'] else '✗'} (threshold: {clothing_thresh})")
            if f1['clip'] is not None and f2['clip'] is not None:
                print(f"  CLIP: {result['clip_sim']:.4f} {'✓' if result['clip_pass'] else '✗'} (threshold: {clip_thresh})")
            print(f"  → {result['status']}")
    
    # Visualization
    display_h = 400
    
    def resize_with_aspect(img, target_h):
        scale = target_h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), target_h))
    
    # Create visualizations for each image
    vis_list = []
    for f in features:
        img_r = resize_with_aspect(f['img'], display_h)
        mask_r = None
        if f['mask'] is not None:
            mask_r = cv2.resize(f['mask'], (img_r.shape[1], img_r.shape[0]), interpolation=cv2.INTER_NEAREST)
        vis = visualize_features(img_r, mask_r, f['feat'], f['name'])
        vis_list.append(vis)
    
    # Match heights
    max_h = max(v.shape[0] for v in vis_list)
    for i, v in enumerate(vis_list):
        if v.shape[0] < max_h:
            vis_list[i] = np.vstack([v, np.zeros((max_h - v.shape[0], v.shape[1], 3), dtype=np.uint8)])
    
    # Create gap panels with similarity scores
    gap_w = 100
    result_panels = [vis_list[0]]
    
    comp_idx = 0
    for i in range(1, len(vis_list)):
        # Find comparison between previous and current image
        gap = np.zeros((max_h, gap_w, 3), dtype=np.uint8)
        
        # Find the relevant comparison
        for comp in comparisons:
            if (comp['name1'] == features[i-1]['name'] and comp['name2'] == features[i]['name']) or \
               (comp['name2'] == features[i-1]['name'] and comp['name1'] == features[i]['name']):
                
                y_offset = max_h // 2 - 60
                
                cv2.putText(gap, "Lab", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                lab_color = (0, 255, 0) if comp['lab_pass'] else (0, 0, 255)
                cv2.putText(gap, f"{comp['lab_sim']:.3f}", (10, y_offset + 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, lab_color, 2)
                
                cv2.putText(gap, "CLIP", (10, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                clip_color = (0, 255, 0) if comp['clip_pass'] else (0, 0, 255)
                cv2.putText(gap, f"{comp['clip_sim']:.3f}", (10, y_offset + 85), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, clip_color, 2)
                
                cv2.putText(gap, comp['status'], (10, y_offset + 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, comp['color'], 2)
                break
        
        result_panels.append(gap)
        result_panels.append(vis_list[i])
    
    combined = np.hstack(result_panels)
    
    # Title bar with all comparisons
    title_bar_h = 50
    title_bar = np.zeros((title_bar_h, combined.shape[1], 3), dtype=np.uint8)
    
    title_parts = []
    for comp in comparisons:
        status_str = f"{comp['name1'][-5:]}-{comp['name2'][-5:]}: {comp['status']}"
        title_parts.append((status_str, comp['color']))
    
    x_offset = 10
    for text, color in title_parts:
        cv2.putText(title_bar, text, (x_offset, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        x_offset += len(text) * 12 + 30
    
    result = np.vstack([title_bar, combined])
    
    # If 3 images, also show the 1-3 comparison
    if len(features) == 3:
        # Find 1-3 comparison
        comp_1_3 = None
        for comp in comparisons:
            if (comp['name1'] == features[0]['name'] and comp['name2'] == features[2]['name']) or \
               (comp['name2'] == features[0]['name'] and comp['name1'] == features[2]['name']):
                comp_1_3 = comp
                break
        
        if comp_1_3:
            print(f"\n{features[0]['name']} vs {features[2]['name']} (diagonal):")
            print(f"  Lab:  {comp_1_3['lab_sim']:.4f} {'✓' if comp_1_3['lab_pass'] else '✗'}")
            if features[0]['clip'] is not None and features[2]['clip'] is not None:
                print(f"  CLIP: {comp_1_3['clip_sim']:.4f} {'✓' if comp_1_3['clip_pass'] else '✗'}")
            print(f"  → {comp_1_3['status']}")
            
            # Add diagonal comparison info at bottom
            diag_bar = np.zeros((30, result.shape[1], 3), dtype=np.uint8)
            diag_text = f"Diagonal ({features[0]['name'][-8:]}<->{features[2]['name'][-8:]}): Lab={comp_1_3['lab_sim']:.3f} CLIP={comp_1_3['clip_sim']:.3f} -> {comp_1_3['status']}"
            cv2.putText(diag_bar, diag_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, comp_1_3['color'], 1)
            result = np.vstack([result, diag_bar])
    
    cv2.imshow("Clothing Comparison - Lab + OpenCLIP", result)
    print("\n" + "=" * 60)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save result
    output_path = "clothing_comparison_openclip_result.png"
    cv2.imwrite(output_path, result)
    print(f"Result saved to: {output_path}")


def capture_mode(matcher: ClothingMatcher, use_realsense: bool = False, camera_idx: int = 0,
                 clothing_thresh: float = 0.5, clip_thresh: float = 0.7):
    """Interactive capture mode"""
    print("\n" + "=" * 50)
    print("CAPTURE MODE")
    print("=" * 50)
    print("'1' - Capture image 1")
    print("'2' - Capture image 2")
    print("'3' - Capture image 3 (optional)")
    print("'c' - Compare captured images")
    print("'r' - Reset all")
    print("'q' - Quit")
    print("=" * 50 + "\n")
    
    if use_realsense:
        import pyrealsense2 as rs
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        get_frame = lambda: np.asanyarray(pipeline.wait_for_frames().get_color_frame().get_data())
    else:
        cap = cv2.VideoCapture(camera_idx)
        if not cap.isOpened():
            print(f"✗ Failed to open camera {camera_idx}")
            return
        get_frame = lambda: cap.read()[1]
    
    images = [None, None, None]
    
    try:
        while True:
            frame = get_frame()
            if frame is None:
                continue
            
            display = frame.copy()
            H, W = display.shape[:2]
            
            cv2.putText(display, "'1'/'2'/'3': capture | 'c': compare | 'r': reset | 'q': quit", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            
            status = f"[{'✓' if images[0] is not None else '○'}]1  [{'✓' if images[1] is not None else '○'}]2  [{'✓' if images[2] is not None else '○'}]3"
            cv2.putText(display, status, (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Thumbnails
            thumb_sz = 60
            for i, img in enumerate(images):
                if img is not None:
                    x_pos = W - (3 - i) * (thumb_sz + 10)
                    display[10:10+thumb_sz, x_pos:x_pos+thumb_sz] = cv2.resize(img, (thumb_sz, thumb_sz))
                    cv2.putText(display, str(i+1), (x_pos + 5, 10 + thumb_sz - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            cv2.imshow("Capture", display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('1'):
                images[0] = frame.copy()
                print("✓ Captured Image 1")
            elif key == ord('2'):
                images[1] = frame.copy()
                print("✓ Captured Image 2")
            elif key == ord('3'):
                images[2] = frame.copy()
                print("✓ Captured Image 3")
            elif key == ord('c'):
                # Collect non-None images
                valid_images = [(img, f"Image_{i+1}") for i, img in enumerate(images) if img is not None]
                if len(valid_images) >= 2:
                    imgs, names = zip(*valid_images)
                    compare_images(matcher, list(imgs), list(names), clothing_thresh, clip_thresh)
                else:
                    print("✗ Need at least 2 images to compare")
            elif key == ord('r'):
                images = [None, None, None]
                print("✓ Reset all images")
    finally:
        if use_realsense:
            pipeline.stop()
        else:
            cap.release()
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("CLOTHING MATCHER TEST - Lab + OpenCLIP")
    print("=" * 60)
    print(f"Segmentation: {args.yolo_seg}")
    print(f"OpenCLIP: {args.clip_model} / {args.clip_pretrained}")
    print(f"Thresholds: Lab >= {args.clothing_threshold}, CLIP >= {args.clip_threshold}")
    
    try:
        matcher = ClothingMatcher(
            args.yolo_seg, 
            device='cuda', 
            use_clip=True,
            clip_model=args.clip_model, 
            clip_pretrained=args.clip_pretrained
        )
    except Exception as e:
        print(f"✗ Init failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    if args.capture:
        capture_mode(matcher, args.realsense, args.camera, 
                     args.clothing_threshold, args.clip_threshold)
    elif args.img1 and args.img2:
        # Load images
        images = []
        names = []
        
        for path, name in [(args.img1, "Image_1"), (args.img2, "Image_2"), (args.img3, "Image_3")]:
            if path:
                img = cv2.imread(path)
                if img is None:
                    print(f"✗ Failed to load: {path}")
                    sys.exit(1)
                images.append(img)
                names.append(path.split('/')[-1])  # Use filename as name
                print(f"Loaded {path}: {img.shape[1]}x{img.shape[0]}")
        
        compare_images(matcher, images, names, 
                       args.clothing_threshold, args.clip_threshold, args.debug)
    else:
        print("\nUsage:")
        print("  # Compare 2 images")
        print("  python test_clothing_matcher_lab_openclip.py --img1 p1.jpg --img2 p2.jpg")
        print("")
        print("  # Compare 3 images")
        print("  python test_clothing_matcher_lab_openclip.py --img1 p1.jpg --img2 p2.jpg --img3 p3.jpg")
        print("")
        print("  # Camera capture mode")
        print("  python test_clothing_matcher_lab_openclip.py --capture [--realsense]")
        print("")
        print("  # Custom thresholds")
        print("  python test_clothing_matcher_lab_openclip.py --img1 p1.jpg --img2 p2.jpg --clothing-threshold 0.6 --clip-threshold 0.8")


if __name__ == "__main__":
    main()