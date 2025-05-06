import subprocess
from pathlib import Path
import numpy as np
import json
import os
from matplotlib import pyplot as plt
import cv2
import ipywidgets as widgets
from pycocotools import mask as mask_utils
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
from ipywidgets import (
    Button, Dropdown, HBox, VBox, IntSlider, Output, ToggleButtons, Label
)
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path as MplPath
from copy import deepcopy
from PIL import Image
from IPython.display import display
from ipywidgets import FloatSlider
from collections import defaultdict
from tqdm import tqdm
import shutil

def extract_frames_ffmpeg(video_path, output_dir, quality=2):
    """
    Extracts frames from a video as high-quality JPEGs using ffmpeg.
    
    Args:
        video_path (str or Path): Path to the input .mp4 video.
        output_dir (str or Path): Directory where the JPEG frames will be saved.
        quality (int): JPEG quality (2 = high quality, 31 = low). Lower is better in ffmpeg.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output file pattern
    output_pattern = "%05d.jpg"

    # Run ffmpeg to extract frames
    command = [
        "ffmpeg",
        "-i", str(video_path),
        "-q:v", str(quality),  # lower is better; 2 is considered high quality
        "-vsync", "vfr",       # Very important: prevent duplicate frames
        "-start_number", "0",  # Start numbering frames from 0
        f"{output_dir}/{output_pattern}",
    ]

    print("Running command:", " ".join(command))
    subprocess.run(command) #, check=True)

    print(f"Frames saved to: {output_dir}")


def create_coco_json(frames_path, output_path):
    categories = [
    {
      "id": 1,
      "name": "Chinook",
      "supercategory": ""
    },
    {
      "id": 2,
      "name": "Coho",
      "supercategory": ""
    },
    {
      "id": 3,
      "name": "Atlantic",
      "supercategory": ""
    },
    {
      "id": 4,
      "name": "Rainbow Trout",
      "supercategory": ""
    },
    {
      "id": 5,
      "name": "Brown Trout",
      "supercategory": ""
    }
    ]
    images = []
    annotations = []
    coco_json = {}

    # Fill out the license information
    coco_json["licenses"] = [{"name": "", "id": 0, "url": ""}]
    coco_json["info"] = {
        "contributor": "",
        "date_created": "",
        "description": "",
        "url": "",
        "version": "",
        "year": "",
    }

    # Image entries
    output_pattern = "%05d.jpg"

    # Loop through all frames
    for frame_idx in range(len(os.listdir(frames_path))):
        file_name = output_pattern % frame_idx
        # Add the image information
        images.append(
            {
                "id": frame_idx + 1,
                "license": 0,
                "file_name": file_name,
                "height": 960,
                "width": 1280,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": 0,
            }
        )

    # Update the coco_json dictionary
    coco_json.update(
        {"categories": categories, "images": images, "annotations": annotations}
    )

    # Save the json file in the frames directory and the json directory
    with open(output_path, "w") as json_file:
        json.dump(coco_json, json_file, indent=2)

    print("COCO JSON Saved")
    return coco_json


def decode_rle(rle):
    """
    Decode RLE mask encoding, handling empty masks.
    """
    if not isinstance(rle, dict) or not rle:
        return np.array([])
    
    # Check for empty counts
    if not rle.get('counts', []):
        return np.array([])
    
    decoded_mask = mask_utils.decode({
        'size': rle['size'],
        'counts': bytes(rle['counts']) if isinstance(rle['counts'][0], int) else rle['counts']
    })
    return decoded_mask

def clean_clicks(clicks):
    """
    Remove empty objects (no pos or neg clicks), and clean up categories and frames
    if they become empty as a result.
    
    Args:
        clicks (dict): Nested dict {frame_id: {category: {track_id: {"pos": [...], "neg": [...]}}}}

    Returns:
        dict: Cleaned version of the same dictionary.
    """
    cleaned_clicks = {}

    for frame_id, frame_data in clicks.items():
        new_frame = {}
        for category, obj_dict in frame_data.items():
            new_cat = {}
            for obj_id, clicks_dict in obj_dict.items():
                pos = clicks_dict.get("pos", [])
                neg = clicks_dict.get("neg", [])
                if pos or neg:
                    new_cat[obj_id] = {"pos": pos, "neg": neg}
            if new_cat:
                new_frame[category] = new_cat
        if new_frame:
            cleaned_clicks[frame_id] = new_frame

    return cleaned_clicks


def sam_mask_to_uncompressed_rle(mask_tensor, is_binary=False):
    """
    Converts a SAM2 mask tensor (shape [1, H, W]) into COCO uncompressed RLE.
    """
    # Step 1: Convert to binary mask (uint8, 0/1)
    if is_binary:
      binary_mask = mask_tensor.astype(np.uint8)
    else:
      binary_mask = (mask_tensor > 0).astype(np.uint8)  # Threshold at 0

    # Step 2: Fortran-contiguous layout (required by pycocotools)
    binary_mask_fortran = np.asfortranarray(binary_mask)

    # Step 3: Encode to RLE (compressed by default)
    rle = mask_utils.encode(binary_mask_fortran)

    # Calculate area of the mask
    area = float(mask_utils.area(rle))

    # Calculate bounding box
    bbox = mask_utils.toBbox(rle).tolist()

    # Step 4: Convert counts from bytes to list (uncompressed-style for COCO/YTVIS)
    rle["counts"] = list(rle["counts"])
    
    return rle, area, bbox

def create_annotation_id_map(coco_dict):
    """
    Create a mapping of (image_id, track_id) to annotation index.
    """
    ann_index_map = {}
    for idx, ann in enumerate(coco_dict["annotations"]):
        ann_index_map[(ann["image_id"], ann["attributes"]["track_id"])] = idx
    return ann_index_map

def create_data_maps(coco_dict):
    """
    Create a mapping of image_id to file_name and a mapping of category_id to name.
    """
    image_id_to_filename = {img["id"]: img["file_name"] for img in coco_dict["images"]}
    image_id_to_data = {img["id"]: img for img in coco_dict["images"]}
    category_id_to_name = {cat["id"]: cat["name"] for cat in coco_dict["categories"]}
    categories = list(category_id_to_name.values())
    category_name_to_id = {v: k for k, v in category_id_to_name.items()}
    
    return image_id_to_filename, image_id_to_data, categories, category_id_to_name, category_name_to_id

# A widget for adding click prompts for object annotations
class ImageAnnotationWidget:
    def __init__(self, coco_dict, image_dir, start_frame=0, predictor=None, inference_state=None, output_json_path=None):
        # Close any existing figures to prevent memory issues
        plt.close('all')
        # add arguments to the class output
        self.image_dir = image_dir
        self.predictor = predictor
        self.inference_state = inference_state
        self.coco = coco_dict
        self.coco_json_path = output_json_path
        # Create data maps of coco info
        self.image_id_to_filename, self.image_id_to_data, self.categories, self.category_id_to_name, self.category_name_to_id = create_data_maps(self.coco)
        self.image_ids = sorted(self.image_id_to_data.keys())
        self.annotations_by_image = self._group_annotations(self.coco["annotations"])
        self.ann_index_map = create_annotation_id_map(self.coco)
        self.cat_to_color = self._assign_colors()
        # Initialize some variables
        self.clicks = {}
        self.current_frame_idx = start_frame
        self.current_xlim = None
        self.current_ylim = None
        self.mask_history = {} # Dictionary to store mask history per frame
        self.active_category = self.categories[0]
        self.active_track_id = 1
        self.show_clicks = True  # Default to showing clicks

        # Create UI elements
        self.category_selector = widgets.Dropdown(options=self.categories, 
                                                  value = self.active_category,
                                                  description="Species")
        self.track_id_selector = widgets.BoundedIntText(value=self.active_track_id, 
                                                        min=1, 
                                                        max=100, 
                                                        step=1, 
                                                        description="Track ID")
        self.prev_button = widgets.Button(description="Previous Frame")
        self.next_button = widgets.Button(description="Next Frame")
        self.generate_mask_button = widgets.Button(description="Generate Mask")
        self.undo_mask_button = widgets.Button(description="Undo Mask")
        self.delete_button = widgets.Button(description="Delete Annotation")
        self.show_clicks_toggle = widgets.ToggleButton(
            value=True,
            description='Show Clicks',
            tooltip='Toggle click visibility'
        )
        # Add button and target frame box for forward and backward mask propagation
        self.propagate_button = widgets.Button(description="Propagate Mask")
        self.target_frame = widgets.BoundedIntText(
            value=len(self.image_ids)-1,
            min=0,
            max=len(self.image_ids)-1,
            description="Target Frame"
        )
        self.save_button = widgets.Button(description="Save JSON")

        self.output = widgets.Output()

        self.fig, self.ax = plt.subplots(figsize=(9, 7))
        self.fig.canvas.header_visible = False  # This hides the "Figure #" text
        
        self._connect_events()
        self._display_ui()

    def _group_annotations(self, annotations):
        """
        Groups annotations by image_id and then by track_id for faster lookup.
        
        Returns:
            dict: {
                image_id: {
                    track_id: annotation,
                    ...
                },
                ...
            }
        """
        grouped = {}
        for ann in annotations:
            image_id = ann["image_id"]
            track_id = ann["attributes"]["track_id"]
            
            # Initialize nested dictionaries if they don't exist
            if image_id not in grouped:
                grouped[image_id] = {}
                
            # Store annotation by track_id
            grouped[image_id][track_id] = ann
            
        return grouped

    def _assign_colors(self):
        cmap = plt.get_cmap("Set1")
        num_colors = cmap.N
        return {track_id: cmap(i % num_colors) for i, track_id in enumerate(range(1, 10))}

    def _connect_events(self):
        self.fig.canvas.mpl_connect('button_press_event', self._on_click())
        self.fig.canvas.mpl_connect('key_press_event', self._on_key())
        self.prev_button.on_click(lambda b: self.update_frame(-1))
        self.next_button.on_click(lambda b: self.update_frame(+1))
        self.generate_mask_button.on_click(self.generate_mask_for_current_frame)
        self.undo_mask_button.on_click(self.undo_mask_for_current_frame)
        self.delete_button.on_click(self.delete_annotation_for_current_track)  
        self.propagate_button.on_click(self._propagate_mask)
        self.save_button.on_click(self._save_annotations)

        # Dropdown/selector events using observe
        self.category_selector.observe(self._update_category, names='value')
        self.track_id_selector.observe(self._update_track_id, names='value')
        self.show_clicks_toggle.observe(self._update_click_visibility, names='value')

    def _update_category(self, change):
        """Handler for category selection changes"""
        self.active_category = change['new']
        
    def _update_track_id(self, change):
        """Handler for track ID selection changes"""
        self.active_track_id = change['new']
    
    def _update_click_visibility(self, change):
        """Handler for click visibility toggle"""
        self.show_clicks = change['new']
        self.plot_frame()  # Refresh display

    def _display_ui(self):
        controls = VBox([
            HBox([self.prev_button, self.next_button, 
                  self.category_selector, self.track_id_selector]),
            HBox([self.generate_mask_button, self.undo_mask_button, 
                  self.delete_button, self.show_clicks_toggle]),
            HBox([self.propagate_button, self.target_frame, self.save_button])
        ])
        
        display(widgets.VBox([controls, self.output]))
        self.plot_frame()

    def plot_frame(self):
        image_id = self.image_ids[self.current_frame_idx]
        image_info = self.image_id_to_data[image_id]
        image_path = os.path.join(self.image_dir, image_info["file_name"])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_anns = self.annotations_by_image.get(image_id, {})

        self.output.clear_output(wait=True)
        with self.output:
            self.ax.clear()
            self.ax.imshow(image)
            self.ax.set_title(f"{image_info['file_name']}")

            # Draw annotations
            for track_id, ann in image_anns.items():
                cat_id = ann["category_id"]
                cat_name = self.category_id_to_name[cat_id]
                track_id = ann["attributes"]["track_id"]
                color = self.cat_to_color[track_id] # Use track-based color

                x, y, w, h = ann["bbox"]

                if "segmentation" in ann:
                    rle = ann["segmentation"]
                    mask = decode_rle(rle)
                    if mask.size > 0:  # Only plot if mask is not empty
                        self.ax.text(int(w/2), y - 20, f"{cat_name} T{track_id}", color=color, fontsize=10, weight='bold')
                        self.ax.imshow(np.ma.masked_where(mask == 0, mask),
                                    cmap=mcolors.ListedColormap([color]), alpha=0.2)

            # Draw clicks
            if self.show_clicks:
                frame_clicks = self.clicks.get(image_id, {})
                for cat, tracks in frame_clicks.items():
                    for track_id, data in tracks.items():
                        pos = np.array(data.get("pos", []))
                        neg = np.array(data.get("neg", []))
                        if len(pos):
                            self.ax.scatter(pos[:, 0], pos[:, 1], c="green", marker="o")
                            for (x, y) in pos:
                                self.ax.text(x + 5, y, f"{cat} T{track_id}", color="green", fontsize=10)
                        if len(neg):
                            self.ax.scatter(neg[:, 0], neg[:, 1], c="red", marker="x")
                            for (x, y) in neg:
                                self.ax.text(x + 5, y, f"{cat} T{track_id}", color="red", fontsize=10)

            if self.current_xlim and self.current_ylim:
                self.ax.set_xlim(self.current_xlim)
                self.ax.set_ylim(self.current_ylim)

            self.fig.tight_layout()

    def update_frame(self, direction):
        self.current_xlim = self.ax.get_xlim()
        self.current_ylim = self.ax.get_ylim()
        new_idx = self.current_frame_idx + direction
        if 0 <= new_idx < len(self.image_ids):
            self.current_frame_idx = new_idx
            self.plot_frame()
    
    def generate_mask_for_current_frame(self, b):
        image_id = self.image_ids[self.current_frame_idx]
        frame_clicks = self.clicks.get(image_id, {})

        # Save current state before generating new masks
        if image_id not in self.mask_history:
            self.mask_history[image_id] = {}
        
        # Remove empty clicks
        frame_clicks = clean_clicks({image_id: frame_clicks})[image_id]

        if not frame_clicks:
            print("No valid clicks on this frame.")
            return
        
        # Rebuild ann_index_map to ensure it's in sync
        self.ann_index_map = create_annotation_id_map(self.coco)

        for category, track in frame_clicks.items():
            category_id = self.category_name_to_id[category]
            for track_id, data in track.items():
                # Store history per track
                if track_id not in self.mask_history[image_id]:
                    self.mask_history[image_id][track_id] = []

                # Store current state before changes
                current_ann = next((ann.copy() for ann in self.coco["annotations"] 
                                if ann["image_id"] == image_id and 
                                ann["attributes"]["track_id"] == track_id), None)
                if current_ann:
                    self.mask_history[image_id][track_id].append(current_ann)

                # Prepare points
                pos = np.array(data.get("pos", []), dtype=np.float32)
                neg = np.array(data.get("neg", []), dtype=np.float32)
                if len(pos) == 0 and len(neg) == 0:
                    continue
                labels = [1] * len(pos) + [0] * len(neg)
                points = np.concatenate([pos, neg], axis=0) if len(neg) else pos
                labels = np.array(labels, dtype=np.int32)

                # Generate mask
                _, out_ids, masks = self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=self.current_frame_idx,
                    obj_id=track_id,
                    points=points,
                    labels=labels,
                )

                if track_id not in out_ids:
                    print(f"Failed to generate mask for {category} (track {track_id})")
                    continue

                mask_idx = out_ids.index(track_id)
                mask_tensor = masks[mask_idx].cpu().numpy()[0]
                rle, area, bbox = sam_mask_to_uncompressed_rle(mask_tensor)

                key = (image_id, track_id)
                if key in self.ann_index_map:
                    ann_idx = self.ann_index_map[key]
                    ann = self.coco["annotations"][ann_idx]
                    ann.update({"segmentation": rle, "area": area, "bbox": bbox})
                else:
                    self.ann_index_map[key] = max(self.ann_index_map.values()) + 1 if self.ann_index_map else 1
                    new_ann = {
                        "id": self.ann_index_map[key],
                        "image_id": image_id,
                        "category_id": category_id,
                        "segmentation": rle,
                        "area": area,
                        "bbox": bbox,
                        "iscrowd": 0,
                        "attributes": {
                            "occluded": False,
                            "rotation": 0.0,
                            "track_id": track_id,
                            "keyframe": True,
                        },
                    }
                    self.coco["annotations"].append(new_ann)

        self.annotations_by_image = self._group_annotations(self.coco["annotations"])
        self.plot_frame()

    def undo_mask_for_current_frame(self, b):
        image_id = self.image_ids[self.current_frame_idx]

        # Check if there's any annotation for this frame and track_id
        current_ann = next((ann for ann in self.coco["annotations"] 
                        if ann["image_id"] == image_id and 
                        ann["attributes"]["track_id"] == self.active_track_id), None)
        
        # If no annotation exists for this frame/track_id, do nothing
        if not current_ann:
            print(f"No mask to undo for track {self.active_track_id} in frame {image_id}")
            return
        
        # Check if we have history for this track
        if (image_id not in self.mask_history or 
            self.active_track_id not in self.mask_history[image_id] or 
            not self.mask_history[image_id][self.active_track_id]):
            # If no history, set empty segmentation but keep annotation entry
            current_ann["segmentation"]["counts"] = []
            current_ann["area"] = 0
            current_ann["bbox"] = [0, 0, 0, 0]

            # Update annotations_by_image
            if image_id in self.annotations_by_image:
                # for i, ann in enumerate(self.annotations_by_image[image_id]):
                #     if ann["attributes"]["track_id"] == self.active_track_id:
                #         self.annotations_by_image[image_id][i] = current_ann
                #         break
                self.annotations_by_image[image_id][self.active_track_id] = current_ann
        else:
            # Restore previous state for this track only
            previous_ann = self.mask_history[image_id][self.active_track_id].pop()
            
            # Update current annotation in coco["annotations"]
            for i, ann in enumerate(self.coco["annotations"]):
                if (ann["image_id"] == image_id and 
                    ann["attributes"]["track_id"] == self.active_track_id):
                    self.coco["annotations"][i] = previous_ann
                    break
            
            # Update annotations_by_image
            if image_id in self.annotations_by_image:
                self.annotations_by_image[image_id][self.active_track_id] = previous_ann
        
            
            # Rebuild ann_index_map
            self.ann_index_map = {}
            for idx, ann in enumerate(self.coco["annotations"]):
                key = (ann["image_id"], ann["attributes"]["track_id"])
                self.ann_index_map[key] = idx

        # Update display
        self.plot_frame()

    def delete_annotation_for_current_track(self, b):
        """Delete annotation for current track in current frame."""
        image_id = self.image_ids[self.current_frame_idx]
        
        # Check if there's any annotation for this frame and track_id
        current_ann = next((ann for ann in self.coco["annotations"] 
                        if ann["image_id"] == image_id and 
                        ann["attributes"]["track_id"] == self.active_track_id), None)
        
        if not current_ann:
            print(f"No annotation to delete for track {self.active_track_id} in frame {image_id}")
            return
        
        # Remove from coco annotations
        self.coco["annotations"] = [ann for ann in self.coco["annotations"]
                                if not (ann["image_id"] == image_id and 
                                        ann["attributes"]["track_id"] == self.active_track_id)]
        
        # Remove from annotations_by_image
        if image_id in self.annotations_by_image:
            if self.active_track_id in self.annotations_by_image[image_id]:
                del self.annotations_by_image[image_id][self.active_track_id]
        
        # Remove from ann_index_map
        key = (image_id, self.active_track_id)
        if key in self.ann_index_map:
            del self.ann_index_map[key]
        
        # Rebuild ann_index_map to ensure indices are correct
        self.ann_index_map = create_annotation_id_map(self.coco)
        
        # Clear history for this track in this frame
        if image_id in self.mask_history and self.active_track_id in self.mask_history[image_id]:
            del self.mask_history[image_id][self.active_track_id]
        
        # Update display
        self.plot_frame()
    
    def _propagate_mask(self, b):
        """Propagate masks forward from current frame"""
        image_id = self.image_ids[self.current_frame_idx]
        target_frame = self.target_frame.value
        
        with self.output:
            if target_frame == self.current_frame_idx:
                print("Target frame must be different from current frame")
                return
                
            # Get all tracks in current frame
            current_anns = self.annotations_by_image.get(image_id, {})
            if not current_anns:
                print("No masks to propagate in current frame")
                return
            
            # Propagate masks
            video_segments = track_masks(
                cur_frame_idx=self.current_frame_idx,
                predictor=self.predictor,
                inference_state=self.inference_state,
                max_frame2propagate=target_frame
            )
            
            # Update annotations with propagated masks
            trackID_to_category = {
                track_id: self.category_id_to_name[ann["category_id"]]
                for track_id, ann in current_anns.items()
            }
            print("Updating annotations...")
            self.coco = add_propagated_masks_to_annotations(
                video_segments=video_segments,
                ann_index_map=self.ann_index_map,
                trackID_to_category=trackID_to_category,
                category_name_to_id=self.category_name_to_id,
                coco_dict=self.coco
            )
            
            # Update internal state
            self.annotations_by_image = self._group_annotations(self.coco["annotations"])
            self.ann_index_map = create_annotation_id_map(self.coco)
            print("✅ Propagation complete!")

        self.plot_frame()

    def _on_click(self):
        def handler(event):
            if not event.inaxes:
                return

            xdata, ydata = round(event.xdata), round(event.ydata)
            image_id = self.image_ids[self.current_frame_idx]

            self.clicks.setdefault(image_id, {}).setdefault(self.active_category, {}).setdefault(self.active_track_id, {"pos": [], "neg": []})
            click_type = "pos" if event.button == 1 else "neg" if event.button == 3 else None
            if click_type is None:
                return

            points = self.clicks[image_id][self.active_category][self.active_track_id][click_type]
            for i, (px, py) in enumerate(points):
                if abs(px - xdata) <= 10 and abs(py - ydata) <= 10:
                    points.pop(i)
                    self.plot_frame()
                    return

            points.append([xdata, ydata])
            self.plot_frame()

        return handler

    def _on_key(self):
        def handler(event):
            if event.key == 'right':
                self.update_frame(+1)
            elif event.key == 'left':
                self.update_frame(-1)
            elif event.key == 'p':
                self.copy_clicks_from_previous()
                self.plot_frame()
        return handler

    def copy_clicks_from_previous(self):
        idx = self.current_frame_idx
        if idx == 0:
            return
        prev_id = self.image_ids[idx - 1]
        curr_id = self.image_ids[idx]
        if prev_id not in self.clicks:
            return
        prev_clicks = self.clicks[prev_id]
        curr_clicks = self.clicks.setdefault(curr_id, {})
        for cat, tracks in prev_clicks.items():
            for track_id, data in tracks.items():
                curr_clicks.setdefault(cat, {}).setdefault(track_id, {"pos": [], "neg": []})
                curr_clicks[cat][track_id]["pos"].extend([pt[:] for pt in data.get("pos", [])])
                curr_clicks[cat][track_id]["neg"].extend([pt[:] for pt in data.get("neg", [])])
    
    def _save_annotations(self, b):
        """Save current annotations to JSON file"""
        with self.output:
            try:
                output_path = self.coco_json_path
                # Save the COCO JSON
                with open(output_path, "w") as f:
                    json.dump(self.coco, f, indent=2)
                
                print(f"✅ Annotations saved to: {output_path}")
            except Exception as e:
                print(f"❌ Error saving annotations: {str(e)}")

    def __del__(self):
        """Cleanup when widget is destroyed"""
        plt.close(self.fig)

# A function to track masks over multiple frames
def track_masks(
    cur_frame_idx,
    predictor,
    inference_state,
    max_frame2propagate=None,
):
    """
    Track masks either forward or backward through video frames.
    
    Args:
        cur_frame_idx (int): Current frame index (where mask exists)
        predictor: SAM2 predictor instance
        inference_state: SAM2 inference state
        max_frame2propagate (int): Target frame to propagate to
        reverse (bool): If True, propagate backward; if False, propagate forward
    """
    # Dictionary to save the generated masks for each frame
    video_segments = {}

    if max_frame2propagate < cur_frame_idx:
        # For backward propagation
        num_frames_to_track = cur_frame_idx - max_frame2propagate + 1
        reverse = True
        print(f"Starting from frame {cur_frame_idx}, tracking backwards {num_frames_to_track} frames")
    else:
        # For forward propagation
        num_frames_to_track = max_frame2propagate - cur_frame_idx + 1
        reverse = False
        print(f"Starting from frame {cur_frame_idx}, tracking forwards {num_frames_to_track} frames")

    # Propage the video and save the masks
    for (
        out_frame_idx,
        out_obj_ids,
        out_mask_logits,
    ) in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=cur_frame_idx,
        reverse=reverse,
        max_frame_num_to_track=num_frames_to_track,
    ):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    
    return video_segments

def add_propagated_masks_to_annotations(
    video_segments,
    ann_index_map,
    trackID_to_category,
    category_name_to_id,
    coco_dict,
):
    coco_out = coco_dict.copy()
    for ann_frame_idx, masks in video_segments.items():
        for ann_obj_id, mask_tensor in masks.items():
            # Convert to uncompressed rle
            rle, area, bbox = sam_mask_to_uncompressed_rle(mask_tensor, is_binary=True)

            # Add to annotation
            if (ann_frame_idx + 1, ann_obj_id) in ann_index_map:
                ann_index = ann_index_map[(ann_frame_idx + 1, ann_obj_id)]
                coco_out["annotations"][ann_index]["segmentation"] = rle
                coco_out["annotations"][ann_index]["area"] = area
                coco_out["annotations"][ann_index]["bbox"] = bbox
            else:
                coco_out["annotations"].append(
                    {
                        "id": len(coco_out["annotations"]) + 1,
                        "image_id": ann_frame_idx + 1,
                        "category_id": category_name_to_id[trackID_to_category[ann_obj_id]],  # Placeholder, will be updated later
                        "segmentation": rle,
                        "area": area,
                        "bbox": bbox,
                        "iscrowd": 0,
                        "attributes": {
                            "occluded": False,
                            "rotation": 0.0,
                            "track_id": ann_obj_id,
                            "keyframe": False,
                        },
                    },
                )
    return coco_out

def adjust_brightness_contrast(img, brightness=1.0, contrast=1.0):
        img = img.astype(np.float32)
        # Apply contrast (centered at 128)
        img = (img - 128) * contrast + 128
        # Apply brightness
        img = img * brightness
        # Clip and convert back
        return np.clip(img, 0, 255).astype(np.uint8)

## Drawing widget -------
class MaskEditor:
    def __init__(self, coco_json_path, frames_dir, start_frame):
        self.frames_dir = Path(frames_dir)
        # Load the COCO JSON file
        self.coco_json_path = Path(coco_json_path)
        with open(coco_json_path) as f:
            self.coco = json.load(f)
        # Create name maps
        self.image_id_to_filename, self.image_id_to_data, self.categories, self.category_id_to_name, self.category_name_to_id = create_data_maps(self.coco)

        self.annotations_by_image = {}
        self.track_to_category = {}
        for ann in self.coco["annotations"]:
            # Store original annotations
            self.annotations_by_image.setdefault(ann["image_id"], []).append(ann)
            # Create mapping of track_id to category_name
            track_id = ann["attributes"]["track_id"]
            category_name = self.category_id_to_name[ann["category_id"]]
            self.track_to_category[track_id] = category_name

        # Create dropdown options in "Object track_id: category_name" format
        self.track_options = [f"Object {track_id}: {cat_name}" 
                            for track_id, cat_name in sorted(self.track_to_category.items())]
        
        if self.track_options:
            self.active_track_id = int(self.track_options[0].split(':')[0].split()[-1])
            category_name = self.track_to_category[self.active_track_id]
            self.active_category_id = self.category_name_to_id[category_name]
        
        self.image_ids = sorted(self.image_id_to_filename.keys())
        self.current_index = start_frame
        self.mode = "draw"
        self.brush_size = 10

        self.drawing_mode = "lasso"  # Add this to track drawing tool
        self.polygon_vertices = []    # Store vertices while drawing polygon
        self.temp_line = None        # Store temporary line while drawing

        self.mask_history = {}  # Dictionary to hold history for each image_id
        self.last_click_pos = None
        self.zoom_mode = False  # Add this to track if we're waiting for a zoom click
        self.img = None  # Initialize img attribute
        self.fig, self.ax = plt.subplots(figsize=(9, 7))
        self.fig.canvas.header_visible = False

        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

        # Connect both click types
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)

        self._setup_ui()
        self._update_canvas()

        # Return the widget for display
        display(self.ui)
        
    def _setup_ui(self):
        self.output = Output()

        self.prev_btn = Button(description="Previous")
        self.next_btn = Button(description="Next")
        self.save_btn = Button(description="Save JSON")
        self.undo_btn = Button(description="Undo")
        self.smooth_btn = Button(description="Smooth Mask")
        self.zoom_in_btn = Button(description="Zoom In")
        self.reset_zoom_btn = Button(description="Reset Zoom")

        self.mode_toggle = ToggleButtons(options=["draw", "erase"], value="draw")
        #self.brush_slider = IntSlider(description="Brush Size", min=1, max=50, value=self.brush_size)

        # Track-based dropdown
        self.object_dropdown = Dropdown(
            options=self.track_options,
            value=self.track_options[0] if self.track_options else None,
            description="Track"
        )

        # Add drawing tool dropdown
        self.drawing_tool = Dropdown(
            options=['lasso', 'polygon'],
            value='lasso',
            description='Tool:'
        )

        self.brightness_slider = FloatSlider(description="Brightness", min=0.0, max=2.0, value=1.0, step=0.01)
        self.contrast_slider = FloatSlider(description="Contrast", min=0.0, max=2.0, value=1.0, step=0.01)

        self.prev_btn.on_click(lambda _: self._change_frame(-1))
        self.next_btn.on_click(lambda _: self._change_frame(1))
        self.save_btn.on_click(lambda _: self._save_json())
        self.undo_btn.on_click(lambda _: self._undo_last_action())
        self.smooth_btn.on_click(lambda _: self._smooth_mask())
        self.zoom_in_btn.on_click(self._zoom_in)
        self.reset_zoom_btn.on_click(self._reset_zoom)

        self.drawing_tool.observe(self._update_drawing_tool, names='value')
        self.mode_toggle.observe(self._update_mode, names="value")
        self.object_dropdown.observe(self._update_object, names="value")
        #self.brush_slider.observe(self._update_brush, names="value")

        self.brightness_slider.observe(self._update_canvas_from_slider, names="value")
        self.contrast_slider.observe(self._update_canvas_from_slider, names="value")

        controls = VBox([
            HBox([self.prev_btn, self.next_btn, self.object_dropdown, self.drawing_tool]),
            HBox([Label("Mode:"), self.mode_toggle, self.smooth_btn, self.undo_btn, self.save_btn]),
            HBox([self.zoom_in_btn, self.reset_zoom_btn,
                  self.brightness_slider, self.contrast_slider]),
        ])

        self.output = Output()
        self.ui = VBox([controls, self.output])

        #display(VBox([controls, self.output]))
    
    def _update_drawing_tool(self, change):
        self.drawing_mode = change['new']
        if self.drawing_mode == 'lasso':
            self.lasso.set_active(True)
            # Clear any in-progress polygon
            self.polygon_vertices = []
            if self.temp_line:
                self.temp_line.remove()
                self.temp_line = None
                self.fig.canvas.draw_idle()
        else:
            self.lasso.set_active(False)

    def _update_mode(self, change):
        self.mode = change["new"]

    def _update_object(self, change):
        # Parse track ID and category from selected option
        track_id = int(change["new"].split(':')[0].split()[-1])
        category_name = change["new"].split(': ')[1]
        
        self.active_track_id = track_id
        self.active_category_id = next(cat_id for cat_id, name 
                                     in self.categories.items() 
                                     if name == category_name)

    #def _update_brush(self, change):
        #self.brush_size = change["new"]
    
    def _update_canvas_from_slider(self, change):
        self._update_canvas()

    def _change_frame(self, direction):
        self.current_index = np.clip(self.current_index + direction, 0, len(self.image_ids) - 1)
        self._update_canvas()
    
    def _undo_last_action(self):
        image_id = self.image_ids[self.current_index]
        if image_id in self.mask_history and self.mask_history[image_id]:
            # Restore mask + annotations
            last_mask, last_anns = self.mask_history[image_id].pop()
            self.mask = last_mask
            self.annotations_by_image[image_id] = deepcopy(last_anns)

            # Update the global coco annotations list
            self.coco["annotations"] = [
                ann for ann in self.coco["annotations"]
                if ann["image_id"] != image_id
            ]
            self.coco["annotations"].extend(deepcopy(last_anns))

            # Refresh display
            masked_display = np.ma.masked_where(self.mask == 0, self.mask * 40)
            self.img_plot.set_data(masked_display)
            self.img_plot.figure.canvas.draw_idle()
    
    def _zoom_in(self, b):
        self.zoom_mode = "in"
        print("Click where you want to zoom in")

    def _reset_zoom(self, b):
        # Reset view to full image
        self.ax.set_xlim(0, self.img.shape[1])
        self.ax.set_ylim(self.img.shape[0], 0)
        self.fig.canvas.draw_idle()
    
    def _on_click(self, event):
        if not event.inaxes == self.ax:
            return
            
        if self.zoom_mode:
            # Handle zoom functionality
            self.last_click_pos = (event.xdata, event.ydata)
            if self.zoom_mode == "in":
                self._perform_zoom(zoom_in=True)
                self.zoom_mode = False
            return

        if self.drawing_mode == 'polygon':
            if event.dblclick:
                # Complete polygon
                if len(self.polygon_vertices) >= 3:
                    self.polygon_vertices.append(self.polygon_vertices[0])  # Close the polygon
                    verts = np.array(self.polygon_vertices)
                    self._on_select(verts)
                    
                # Clear temporary drawing
                self.polygon_vertices = []
                if self.temp_line:
                    self.temp_line.remove()
                    self.temp_line = None
                self.fig.canvas.draw_idle()
            else:
                # Add vertex
                self.polygon_vertices.append([event.xdata, event.ydata])
                self._update_temp_polygon()
    
    def _on_mouse_move(self, event):
        if not event.inaxes == self.ax:
            return
            
        if self.drawing_mode == 'polygon' and len(self.polygon_vertices) > 0:
            # Update temporary line
            temp_vertices = self.polygon_vertices + [[event.xdata, event.ydata]]
            self._update_temp_polygon(temp_vertices)
    
    def _update_temp_polygon(self, vertices=None):
        if vertices is None:
            vertices = self.polygon_vertices
            
        if self.temp_line:
            self.temp_line.remove()
        
        if len(vertices) > 0:
            verts = np.array(vertices)
            self.temp_line, = self.ax.plot(verts[:, 0], verts[:, 1], 'r-', linewidth=1)
            self.fig.canvas.draw_idle()
    
    def _perform_zoom(self, zoom_in=True):
        if self.last_click_pos is None:
            return

        x, y = self.last_click_pos
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()
        
        # Calculate zoom factor
        factor = 0.5 if zoom_in else 2.0
        
        # Calculate new width and height
        new_width = (current_xlim[1] - current_xlim[0]) * factor
        new_height = (current_ylim[0] - current_ylim[1]) * factor
        
        # Calculate initial zoom window centered on click
        xmin = x - new_width/2
        xmax = x + new_width/2
        ymin = y - new_height/2
        ymax = y + new_height/2
        
        # Get image boundaries
        img_width = self.img.shape[1]
        img_height = self.img.shape[0]
            
        # Adjust if zoom window exceeds boundaries
        if xmin < 0:
            xmax = min(new_width, img_width)
            xmin = 0
        elif xmax > img_width:
            xmin = max(img_width - new_width, 0)
            xmax = img_width
            
        if ymin < 0:
            ymax = min(new_height, img_height)
            ymin = 0
        elif ymax > img_height:
            ymin = max(img_height - new_height, 0)
            ymax = img_height
        
        # Set new limits
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymax, ymin)  # Reversed for matplotlib's coordinate system
        self.fig.canvas.draw_idle()

    def _update_canvas(self):
        self.output.clear_output(wait=True)
        with self.output:
            #fig, ax = plt.subplots(figsize=(9, 7))
            # Clear the current axes instead of creating new ones
            self.ax.clear()
            image_id = self.image_ids[self.current_index]
            image_path = self.frames_dir / self.image_id_to_filename[image_id]
            self.img = np.array(Image.open(image_path))

            # Apply brightness and contrast adjustments
            self.img = adjust_brightness_contrast(self.img, self.brightness_slider.value, self.contrast_slider.value)

            self.ax.imshow(self.img)
            self.ax.set_title(f"Image ID: {image_id}")
            self.ax.axis("off")

            # Initialize mask as zero (no mask yet)
            self.mask = np.zeros((self.img.shape[0], self.img.shape[1]), dtype=np.uint8)

            # Apply existing annotations as masks (if any)
            anns = self.annotations_by_image.get(image_id, [])
            for ann in anns:
                rle = ann["segmentation"]
                mask = decode_rle(rle)
                #self.mask = np.ma.masked_where(mask == 0, mask * ann["category_id"])
                self.mask[mask == 1] = ann["category_id"]

            # Only plot the mask if it's not empty (i.e., there are masks for the current frame)
            if np.any(self.mask):
                # Display only non-zero regions in the mask
                mask_display = np.ma.masked_where(self.mask == 0, self.mask)
                self.img_plot = self.ax.imshow(mask_display * 40, cmap="jet", alpha=0.3)

            # Draw the lasso selector to allow drawing/erasing
            self.lasso = LassoSelector(self.ax, onselect=self._on_select)
            if self.drawing_mode != 'lasso':
                self.lasso.set_active(False)

            self.fig.canvas.draw_idle()  # Update the canvas instead of displaying new figure
            self.fig.tight_layout()
            # display(self.fig)
            # plt.close()

    def _on_select(self, verts):
        image_id = self.image_ids[self.current_index]
        anns = self.annotations_by_image.setdefault(image_id, [])
        
        # Save a deepcopy of mask and annotations *before* any changes
        previous_mask = self.mask.copy()
        previous_anns = deepcopy(anns)
        self.mask_history.setdefault(image_id, []).append((previous_mask, previous_anns))

        path = MplPath(verts)
        y, x = np.meshgrid(np.arange(self.mask.shape[0]), np.arange(self.mask.shape[1]), indexing='ij')
        points = np.vstack((x.flatten(), y.flatten())).T
        inside = path.contains_points(points).reshape(self.mask.shape)

        if self.mode == "draw":
            self.mask[inside] = self.active_category_id
        elif self.mode == "erase":
            # Only erase pixels belonging to the active category
            self.mask[inside & (self.mask == self.active_category_id)] = 0

        # Update both internal state and COCO annotations
        self._update_annotations(image_id)
        
        # Force redraw of the mask
        self._refresh_display()

    def _update_annotations(self, image_id):
        # Extract updated binary mask for the current category
        category_mask = (self.mask == self.active_category_id)
        # Get current annotations for this image
        anns = self.annotations_by_image.setdefault(image_id, [])
        # Find existing annotation for this track/category
        existing = next((a for a in anns 
                        if a["category_id"] == self.active_category_id 
                        and a["attributes"]["track_id"] == self.active_track_id), None)

        # Remove the existing annotation from both lists if it exists
        if existing:
            anns.remove(existing)
            self.coco["annotations"] = [a for a in self.coco["annotations"] 
                                    if not (a["image_id"] == image_id and 
                                            a["category_id"] == self.active_category_id and
                                            a["attributes"]["track_id"] == self.active_track_id)]

        # Only create new annotation if there are pixels in the mask
        if np.any(category_mask):
            encoded_rle, area, bbox = sam_mask_to_uncompressed_rle(category_mask, is_binary=True)
            
            new_ann = {
                "id": max([a["id"] for a in self.coco["annotations"]] + [0]) + 1,
                "image_id": image_id,
                "category_id": self.active_category_id,
                "segmentation": encoded_rle,
                "area": int(area),
                "bbox": bbox,
                "iscrowd": 0,
                "attributes": {
                    "occluded": False,
                    "rotation": 0.0,
                    "track_id": self.active_track_id,
                    "keyframe": False
                }
            }
            anns.append(new_ann)
            self.coco["annotations"].append(new_ann)

    def _smooth_mask(self):
        image_id = self.image_ids[self.current_index]

        # Save current state before smoothing — this makes undo work!
        anns = self.annotations_by_image.setdefault(image_id, [])
        previous_mask = self.mask.copy()
        previous_anns = deepcopy(anns)
        self.mask_history.setdefault(image_id, []).append((previous_mask, previous_anns))

        binary_mask = (self.mask == self.active_category_id).astype(np.uint8) * 255

        # Morphological closing to smooth edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)


        # Fill holes by flood-filling the background and inverting
        h, w = closed.shape
        flood_fill = closed.copy()
        mask_flood = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(flood_fill, mask_flood, (0, 0), 255)
        holes_filled = cv2.bitwise_or(closed, cv2.bitwise_not(flood_fill))

        # Remove small components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(holes_filled, connectivity=8)
        cleaned_mask = np.zeros_like(holes_filled)
        min_size = 500
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                cleaned_mask[labels == i] = 255

        self.mask[cleaned_mask > 0] = self.active_category_id
        self._update_annotations(image_id)
        
        # Force redraw of the mask
        self._refresh_display()
    
    def _refresh_display(self):
        """Helper method to update the display"""
        if hasattr(self, 'img_plot'):
            masked_display = np.ma.masked_where(self.mask == 0, self.mask * 40)
            self.img_plot.set_data(masked_display)
            self.fig.canvas.draw_idle()

    def _save_json(self):
        file_name = self.coco_json_path.name
        if file_name.startswith("edited"):
            output_path = self.coco_json_path.parent / f"{self.coco_json_path.name}"
        else:
            output_path = self.coco_json_path.parent / f"edited_{self.coco_json_path.name}"
        # Save the edited COCO JSON
        with open(output_path, "w") as f:
            json.dump(self.coco, f, indent=2)
        print(f"Saved to {output_path}")

def convert_annotations_to_uncompressed(coco_data):
    """
    Converts all compressed RLE segmentations in the COCO JSON to uncompressed RLE.
    """
    for ann in coco_data["annotations"]:
        seg = ann.get("segmentation")
        if isinstance(seg, dict) and isinstance(seg["counts"], str):
            # Decode compressed RLE to binary mask
            rle = {
                "size": seg["size"],
                "counts": seg["counts"].encode("utf-8")
            }
            binary_mask = mask_utils.decode(rle).astype(np.uint8)

            # Re-encode using your uncompressed RLE function
            uncompressed_rle, area, bbox = sam_mask_to_uncompressed_rle(binary_mask, is_binary=True)

            # Replace in annotation
            ann["segmentation"] = uncompressed_rle
            ann["area"] = int(area)
            ann["bbox"] = bbox

    return coco_data


def convert_all_coco_to_ytvis(coco_root_dir, output_json_path, indentation=2):
    """
    Convert COCO-style annotations (with track_ids) for multiple videos into one
    YTVIS-style JSON.
    
    Parameters:
        coco_root_dir (str): Path to the folders of video annotations.
        output_json_path (str): Path to write the YTVIS-style JSON.
    """

    ytvis_json = {
        "info": {"description": "Converted from COCO to YTVIS"},
        "categories": [],
        "videos": [],
        "annotations": []
    }

    video_id = 0
    annotation_id_offset = 0
    categories_added = False

    for video_folder in sorted(os.listdir(coco_root_dir)):
        video_path = os.path.join(coco_root_dir, video_folder)
        images_path = os.path.join(video_path, "images")
        coco_json_path = os.path.join(video_path, "annotations", "edited_instances_default.json")

        if not os.path.exists(images_path) or not os.path.exists(coco_json_path):
            continue

        with open(coco_json_path, "r") as f:
            coco_data = json.load(f)

        if not categories_added:
            ytvis_json["categories"] = coco_data["categories"]
            categories_added = True

        # Sort images by filename and build image_id -> frame index mapping
        images_sorted = sorted(coco_data["images"], key=lambda x: x["file_name"])
        image_id_to_frame_index = {img["id"]: idx for idx, img in enumerate(images_sorted)}
        frame_filenames = [os.path.join(video_folder, "images", img["file_name"]) for img in images_sorted]

        width = images_sorted[0]["width"]
        height = images_sorted[0]["height"]
        length = len(images_sorted)

        ytvis_json["videos"].append({
            "id": video_id,
            "width": width,
            "height": height,
            "length": length,
            "file_names": frame_filenames
        })

        # Organize annotations by track_id
        track_segments = {}
        for anno in coco_data["annotations"]:
            track_id = anno["attributes"]["track_id"]
            image_id = anno["image_id"]
            frame_index = image_id_to_frame_index.get(image_id)

            if frame_index is None:
                continue  # skip unmatched image_ids

            if track_id not in track_segments:
                track_segments[track_id] = {
                    "id": track_id + annotation_id_offset,
                    "video_id": video_id,
                    "category_id": anno["category_id"],
                    "segmentations": [None] * length,
                    "areas": [None] * length,
                    "bboxes": [None] * length,
                    "iscrowd": anno.get("iscrowd", 0)
                }

            track_segments[track_id]["segmentations"][frame_index] = anno["segmentation"]
            track_segments[track_id]["areas"][frame_index] = anno["area"]
            track_segments[track_id]["bboxes"][frame_index] = anno["bbox"]

        ytvis_json["annotations"].extend(track_segments.values())

        max_track_id = max(track_segments.keys(), default=0)
        annotation_id_offset += max_track_id + 1
        video_id += 1

    with open(output_json_path, "w") as f:
        json.dump(ytvis_json, f, indent=indentation)

    print(f"YTVIS JSON saved to {output_json_path}")


def find_short_segmentations(json_path, threshold=10):
    with open(json_path, 'r') as f:
        data = json.load(f)

    if "images" in data:
        # COCO format
        print("Detected COCO format.")
        id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

        for anno in data["annotations"]:
            seg = anno.get("segmentation")
            if isinstance(seg, dict) and "counts" in seg:
                counts = seg["counts"]
                if isinstance(counts, list) and len(counts) < threshold:
                    image_id = anno["image_id"]
                    print(f"[COCO] Short segmentation: image_id={image_id}, file_name={id_to_filename.get(image_id, 'unknown')}, counts_len={len(counts)}")

    elif "videos" in data:
        # YTVIS format
        print("Detected YTVIS format.")
        video_id_to_names = {v["id"]: v["file_names"] for v in data["videos"]}

        for anno in data["annotations"]:
            segs = anno.get("segmentations", [])
            video_id = anno["video_id"]
            file_names = video_id_to_names.get(video_id, [])
            for frame_idx, seg in enumerate(segs):
                if isinstance(seg, dict) and "counts" in seg:
                    counts = seg["counts"]
                    if isinstance(counts, list) and len(counts) < threshold:
                        file_name = file_names[frame_idx] if frame_idx < len(file_names) else "unknown"
                        print(f"[YTVIS] Short segmentation: video_id={video_id}, frame={frame_idx}, file_name={file_name}, counts_len={len(counts)}")

    else:
        print("Unsupported format: JSON must contain 'images' or 'videos' key.")


def fix_video_folders_safe(source_root, target_root):
    """
    Create a safe copy of Fishway_Data as Fishway_Data_NoDup, then:
    - Remove duplicated frames.
    - Renumber frames to be continuous (00000.jpg, 00001.jpg, ...)
    - Update all COCO JSON files inside annotations/ folders accordingly.

    Args:
        source_root (str or Path): Original dataset directory (e.g., Fishway_Data).
        target_root (str or Path): New safe directory (e.g., Fishway_Data_NoDup).
    """
    source_root = Path(source_root)
    target_root = Path(target_root)

    if target_root.exists():
        raise ValueError(f"Target directory {target_root} already exists. Please delete it first to avoid accidental overwrite.")

    print(f"Copying {source_root} -> {target_root} (this might take a few minutes)...")
    shutil.copytree(source_root, target_root)
    print("✅ Copy complete.")

    print("Processing videos in the copied directory...")
    for video_folder in tqdm(list(target_root.glob("*"))):
        if not video_folder.is_dir():
            continue

        images_dir = video_folder / "images"
        annotations_dir = video_folder / "annotations"

        if not images_dir.exists() or not annotations_dir.exists():
            print(f"Skipping {video_folder.name}: missing images/ or annotations/")
            continue

        # Step 1: Detect duplicated frames
        image_files = sorted(images_dir.glob("*.jpg"))
        to_delete = []
        last_img = None

        for img_file in image_files:
            img = cv2.imread(str(img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if last_img is not None and np.array_equal(img, last_img):
                to_delete.append(img_file)
            else:
                last_img = img

        # Step 2: Remove duplicated images
        for dup in to_delete:
            os.remove(dup)

        # Step 3: Build renaming dict
        remaining_images = sorted(images_dir.glob("*.jpg"))
        rename_dict = {}
        for idx, img_file in enumerate(remaining_images):
            new_name = f"{idx:05d}.jpg"
            if img_file.name != new_name:
                rename_dict[img_file.name] = new_name
                img_file.rename(images_dir / new_name)

        deleted_filenames = {dup.name for dup in to_delete}

        # Step 4: Update all COCO files
        json_files = list(annotations_dir.glob("*.json"))
        for coco_path in json_files:
            with open(coco_path, "r") as f:
                coco = json.load(f)

            # Build filename -> image entry mapping
            filename_to_image = {img["file_name"]: img for img in coco["images"]}

            deleted_image_ids = set()
            new_images = []
            old_id_to_new_id = {}
            next_image_id = 0

            # Update images
            for img in coco["images"]:
                filename = Path(img["file_name"]).name
                if filename in deleted_filenames:
                    deleted_image_ids.add(img["id"])
                    continue
                # Rename if necessary
                if filename in rename_dict:
                    new_filename = rename_dict[filename]
                    img["file_name"] = str(Path(img["file_name"]).parent / new_filename)
                img["id"] = next_image_id
                old_id_to_new_id[img["id"]] = next_image_id
                new_images.append(img)
                next_image_id += 1

            # Update annotations
            new_annotations = []
            next_ann_id = 0
            for ann in coco["annotations"]:
                if ann["image_id"] in deleted_image_ids:
                    continue
                ann["image_id"] = old_id_to_new_id.get(ann["image_id"], ann["image_id"])
                ann["id"] = next_ann_id
                new_annotations.append(ann)
                next_ann_id += 1

            coco["images"] = new_images
            coco["annotations"] = new_annotations

            with open(coco_path, "w") as f:
                json.dump(coco, f, indent=2)

        print(f"✅ {video_folder.name} fixed.")
