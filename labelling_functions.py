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

# def create_mask(
#     frame_data,
#     frame_idx,
#     created_masks,
#     points,
#     points_type,
#     label,
#     frame_mask_data,
#     num_obj_id,
#     id_2_label,
#     is_edit,
#     selected_obj,
#     predictor,
#     inference_state,
# ):
#     print("\n--- create_mask called ---")
#     print(f"Frame Index: {frame_idx}")
#     print(f"Label: {label}")
#     print(f"id_2_label before create_mask: {id_2_label}")

#     labels = np.array(points_type, dtype=np.int32)

#     # Create mask using predictor
#     _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
#         inference_state=inference_state,
#         frame_idx=frame_idx,
#         obj_id=obj_id,
#         points=points,
#         labels=labels,
#     )

#     for i, out_obj_id in enumerate(out_obj_ids):
#         mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
        
#         if frame_idx not in frame_mask_data:
#             frame_mask_data[frame_idx] = {}

#         frame_mask_data[frame_idx][out_obj_id] = mask

#         print("frame_data[frame_idx]: ", frame_data[frame_idx])

#         # Save mask separately per ID
#         # save_mask(frame_data[frame_idx], mask, label, obj_id)
#         save_mask(frame_data[frame_idx], mask, label, obj_id)


#     # created_masks.setdefault(frame_idx, {})[obj_id] = {
#     #     "points": points,
#     #     "points_type": points_type,
#     # }
#     created_masks.setdefault(frame_idx, {})[obj_id] = {
#         "points": points,
#         "points_type": points_type,
#     }


#     points, points_type = [], []

#     blank_image_pil = Image.open(frame_data[frame_idx]).convert("RGB")
#     blank_image = np.array(blank_image_pil)
#     image_w_mask = apply_mask(blank_image, frame_mask_data, frame_idx)

#     print("--- create_mask completed ---\n")

#     return (
#         image_w_mask,
#         points,
#         points_type,
#         frame_mask_data,
#         created_masks,
#         num_obj_id,
#         id_2_label,
#         is_edit,
#         button_msg,
#         [],
#     )

# def save_mask(frame_path, mask, label, obj_id):
#     """Save mask into a separate directory for each object"""
#     obj_dir = CONFIG["mask_dir"] / f"object_{obj_id}" / label.replace(" ", "_")
#     obj_dir.mkdir(parents=True, exist_ok=True)

#     # Convert to binary: 0 for background, 255 for mask
#     mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
#     mask_image = Image.fromarray((mask * 255).astype(np.uint8))
#     mask_path = obj_dir / f"{frame_path.stem}.png"
#     mask_image.save(mask_path)

#     # print(f"Saved binary mask for {label} (Object ID: {obj_id}) at {mask_path}")
#     return mask_path

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
    decoded_mask = mask_utils.decode({
        'size': rle['size'],
        'counts': bytes(rle['counts']) if isinstance(rle['counts'][0], int) else rle['counts']
    })
    return decoded_mask

# A widget for adding click prompts for object annotations
class ImageAnnotationWidget:
    def __init__(self, coco_dict, image_dir, image_id_to_data, start_frame=0):
        self.image_dir = image_dir
        self.image_id_to_data = image_id_to_data
        self.image_ids = sorted(image_id_to_data.keys())
        self.annotations_by_image = self._group_annotations(coco_dict["annotations"])
        self.category_id_to_name = {cat["id"]: cat["name"] for cat in coco_dict["categories"]}
        self.categories = list(self.category_id_to_name.values())
        self.cat_to_color = self._assign_colors(self.categories)
        self.clicks = {}
        self.current_frame_idx = start_frame
        self.current_xlim = None
        self.current_ylim = None

        # Widgets
        self.category_selector = widgets.Dropdown(options=self.categories, description="Object Type")
        self.object_id_selector = widgets.BoundedIntText(value=1, min=1, max=100, step=1, description="Object #")
        self.prev_button = widgets.Button(description="Previous Frame")
        self.next_button = widgets.Button(description="Next Frame")
        self.output = widgets.Output()

        self.fig, self.ax = plt.subplots(figsize=(9, 7))
        self._connect_events()

        self._display_ui()

    def _group_annotations(self, annotations):
        grouped = {}
        for ann in annotations:
            grouped.setdefault(ann["image_id"], []).append(ann)
        return grouped

    def _assign_colors(self, categories):
        cmap = plt.get_cmap("Set1")
        num_colors = cmap.N
        return {name: cmap(i % num_colors) for i, name in enumerate(sorted(categories))}

    def _connect_events(self):
        self.fig.canvas.mpl_connect('button_press_event', self._on_click())
        self.fig.canvas.mpl_connect('key_press_event', self._on_key())
        self.prev_button.on_click(lambda b: self.update_frame(-1))
        self.next_button.on_click(lambda b: self.update_frame(+1))

    def _display_ui(self):
        controls = widgets.HBox([self.prev_button, self.next_button, self.category_selector, self.object_id_selector])
        display(widgets.VBox([controls, self.output]))
        self.plot_frame()

    def plot_frame(self):
        image_id = self.image_ids[self.current_frame_idx]
        image_info = self.image_id_to_data[image_id]
        image_path = os.path.join(self.image_dir, image_info["file_name"])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        anns = self.annotations_by_image.get(image_id, [])

        self.output.clear_output(wait=True)
        with self.output:
            self.ax.clear()
            self.ax.imshow(image)
            self.ax.set_title(f"{image_info['file_name']}")

            # Draw annotations
            for ann in anns:
                cat_id = ann["category_id"]
                cat_name = self.category_id_to_name[cat_id]
                color = self.cat_to_color[cat_name]

                x, y, w, h = ann["bbox"]
                self.ax.text(x, y - 20, cat_name, color=color, fontsize=10, weight='bold')

                if "segmentation" in ann:
                    rle = ann["segmentation"]
                    mask = decode_rle(rle)
                    self.ax.imshow(np.ma.masked_where(mask == 0, mask),
                                   cmap=mcolors.ListedColormap([color]), alpha=0.2)

            # Draw clicks
            frame_clicks = self.clicks.get(image_id, {})
            for cat, objs in frame_clicks.items():
                for obj_id, data in objs.items():
                    pos = np.array(data.get("pos", []))
                    neg = np.array(data.get("neg", []))
                    if len(pos):
                        self.ax.scatter(pos[:, 0], pos[:, 1], c="green", marker="o")
                        for (x, y) in pos:
                            self.ax.text(x + 5, y, f"{cat} #{obj_id}", color="green", fontsize=10)
                    if len(neg):
                        self.ax.scatter(neg[:, 0], neg[:, 1], c="red", marker="x")
                        for (x, y) in neg:
                            self.ax.text(x + 5, y, f"{cat} #{obj_id}", color="red", fontsize=10)

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

    def _on_click(self):
        def handler(event):
            if not event.inaxes:
                return

            xdata, ydata = round(event.xdata), round(event.ydata)
            image_id = self.image_ids[self.current_frame_idx]
            cat = self.category_selector.value
            obj_id = str(self.object_id_selector.value)

            self.clicks.setdefault(image_id, {}).setdefault(cat, {}).setdefault(obj_id, {"pos": [], "neg": []})
            click_type = "pos" if event.button == 1 else "neg" if event.button == 3 else None
            if click_type is None:
                return

            points = self.clicks[image_id][cat][obj_id][click_type]
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
        for cat, objs in prev_clicks.items():
            for obj_id, data in objs.items():
                curr_clicks.setdefault(cat, {}).setdefault(obj_id, {"pos": [], "neg": []})
                curr_clicks[cat][obj_id]["pos"].extend([pt[:] for pt in data.get("pos", [])])
                curr_clicks[cat][obj_id]["neg"].extend([pt[:] for pt in data.get("neg", [])])


def clean_clicks(clicks):
    """
    Remove empty objects (no pos or neg clicks), and clean up categories and frames
    if they become empty as a result.
    
    Args:
        clicks (dict): Nested dict {frame_id: {category: {object_id: {"pos": [...], "neg": [...]}}}}

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

# A function to track masks over multiple frames
def track_masks(
    cur_frame_idx,
    predictor,
    inference_state,
    max_frame2propagate=None,
):
    # Dictionary to save the generated masks for each frame
    video_segments = {}
    num_frames_to_track = max_frame2propagate - cur_frame_idx + 1  # +1 to include that frame

    print(f"Cur Frame: {cur_frame_idx}")
    # Propage the video and save the masks
    for (
        out_frame_idx,
        out_obj_ids,
        out_mask_logits,
    ) in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=cur_frame_idx,
        reverse=False,
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
        self.image_id_to_filename = {img["id"]: img["file_name"] for img in self.coco["images"]}
        self.categories = {cat["id"]: cat["name"] for cat in self.coco["categories"]}
        self.cat_name_to_id = {v: k for k, v in self.categories.items()}

        self.annotations_by_image = {}
        self.track_to_category = {}
        for ann in self.coco["annotations"]:
            # Store original annotations
            self.annotations_by_image.setdefault(ann["image_id"], []).append(ann)
            # Create mapping of track_id to category_name
            track_id = ann["attributes"]["track_id"]
            category_name = self.categories[ann["category_id"]]
            self.track_to_category[track_id] = category_name

        # Create dropdown options in "Object track_id: category_name" format
        self.track_options = [f"Object {track_id}: {cat_name}" 
                            for track_id, cat_name in sorted(self.track_to_category.items())]
        
        if self.track_options:
            self.active_track_id = int(self.track_options[0].split(':')[0].split()[-1])
            self.active_category_id = next(cat_id for cat_id, name 
                                        in self.categories.items() 
                                        if name == self.track_to_category[self.active_track_id])
        
        self.image_ids = sorted(self.image_id_to_filename.keys())
        self.current_index = start_frame
        self.mode = "draw"
        self.brush_size = 10

        self.mask_history = {}  # Dictionary to hold history for each image_id
        self.last_click_pos = None
        self.zoom_mode = False  # Add this to track if we're waiting for a zoom click
        self.img = None  # Initialize img attribute
        self.fig, self.ax = plt.subplots(figsize=(9, 7))

        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

        self._setup_ui()
        self._update_canvas()

        # Return the widget for display
        display(self.ui)  # Add this line

    def _setup_ui(self):
        self.output = Output()

        self.prev_btn = Button(description="Previous")
        self.next_btn = Button(description="Next")
        self.save_btn = Button(description="Save JSON")
        self.undo_btn = Button(description="Undo")
        self.smooth_btn = Button(description="Smooth Mask")
        self.zoom_in_btn = Button(description="Zoom In")
        self.zoom_out_btn = Button(description="Zoom Out")
        self.reset_zoom_btn = Button(description="Reset Zoom")

        self.mode_toggle = ToggleButtons(options=["draw", "erase"], value="draw")
        self.brush_slider = IntSlider(description="Brush Size", min=1, max=50, value=self.brush_size)

        # Track-based dropdown
        self.object_dropdown = Dropdown(
            options=self.track_options,
            value=self.track_options[0] if self.track_options else None,
            description="Track"
        )

        self.brightness_slider = FloatSlider(description="Brightness", min=0.0, max=2.0, value=1.0, step=0.01)
        self.contrast_slider = FloatSlider(description="Contrast", min=0.0, max=2.0, value=1.0, step=0.01)

        self.prev_btn.on_click(lambda _: self._change_frame(-1))
        self.next_btn.on_click(lambda _: self._change_frame(1))
        self.save_btn.on_click(lambda _: self._save_json())
        self.undo_btn.on_click(lambda _: self._undo_last_action())
        self.smooth_btn.on_click(lambda _: self._smooth_mask())
        self.zoom_in_btn.on_click(self._zoom_in)
        self.zoom_out_btn.on_click(self._zoom_out)
        self.reset_zoom_btn.on_click(self._reset_zoom)

        self.mode_toggle.observe(self._update_mode, names="value")
        self.object_dropdown.observe(self._update_object, names="value")
        self.brush_slider.observe(self._update_brush, names="value")

        self.brightness_slider.observe(self._update_canvas_from_slider, names="value")
        self.contrast_slider.observe(self._update_canvas_from_slider, names="value")

        controls = VBox([
            HBox([self.prev_btn, self.next_btn, self.save_btn, self.undo_btn, self.smooth_btn]),
            HBox([Label("Mode:"), self.mode_toggle, Label("   "), self.zoom_in_btn, self.zoom_out_btn, self.reset_zoom_btn]),
            HBox([Label("Object:"), self.object_dropdown]),
            HBox([self.brush_slider, self.brightness_slider, self.contrast_slider]),
        ])

        self.output = Output()
        self.ui = VBox([controls, self.output])

        #display(VBox([controls, self.output]))

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

    def _update_brush(self, change):
        self.brush_size = change["new"]
    
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

    def _zoom_out(self, b):
        self.zoom_mode = "out"
        print("Click where you want to zoom out")

    def _reset_zoom(self, b):
        # Reset view to full image
        self.ax.set_xlim(0, self.img.shape[1])
        self.ax.set_ylim(self.img.shape[0], 0)
        self.fig.canvas.draw_idle()
    
    def _on_click(self, event):
        if event.inaxes == self.ax:
            self.last_click_pos = (event.xdata, event.ydata)
            if self.zoom_mode == "in":
                self._perform_zoom(zoom_in=True)
                self.zoom_mode = False
            elif self.zoom_mode == "out":
                self._perform_zoom(zoom_in=False)
                self.zoom_mode = False    
    
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
        
        # If zooming out would exceed image size, reset to full view
        if not zoom_in and (new_width > img_width or new_height > img_height):
            self._reset_zoom(None)
            return
            
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

        self._update_annotations(image_id)

    def _update_annotations(self, image_id):
        # Extract updated binary mask for the current category
        category_mask = (self.mask == self.active_category_id)
        anns = self.annotations_by_image.setdefault(image_id, [])
        existing = next((a for a in anns 
                        if a["category_id"] == self.active_category_id 
                        and a["attributes"]["track_id"] == self.active_track_id), None)

        if np.any(category_mask):  # If the category still has pixels
            encoded_rle, area, bbox = sam_mask_to_uncompressed_rle(category_mask, is_binary=True)

            if existing:
                existing["segmentation"] = encoded_rle
                existing["area"] = int(area) #int(np.sum(category_mask))
                existing["bbox"] = bbox #mask_utils.toBbox(encoded_rle).tolist()
            else:
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
        else:
            # Erase the annotation completely
            if existing:
                anns.remove(existing)
                self.coco["annotations"].remove(existing)

        masked_display = np.ma.masked_where(self.mask == 0, self.mask * 40)
        self.img_plot.set_data(masked_display)
        self.img_plot.figure.canvas.draw_idle()

    def _smooth_mask(self):
        image_id = self.image_ids[self.current_index]

        # Save current state before smoothing — this makes undo work!
        anns = self.annotations_by_image.setdefault(image_id, [])
        previous_mask = self.mask.copy()
        previous_anns = deepcopy(anns)
        self.mask_history.setdefault(image_id, []).append((previous_mask, previous_anns))

        binary_mask = (self.mask == self.active_category_id).astype(np.uint8) * 255

        # Morphological closing to smooth edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
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
