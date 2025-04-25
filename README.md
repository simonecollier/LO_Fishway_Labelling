# Lake Ontario Fishway Labelling Tools

This repository contains notebooks to create mask annotations for the Lake Ontario Fishway Dataset. 

 * [LO_propagate_masks.ipynb](LO_propagate_masks.ipynb) is meant for creating masks from click prompts and proagating using [SAMv2](https://github.com/facebookresearch/sam2).
 * [LO_bbox2mask.ipynb](LO_bbox2mask.ipynb) is meant for generating masks from bounding boxes and then refining with click prompts using [SAMv2](https://github.com/facebookresearch/sam2). This notebook also contains older code and methods. It is no longer kept up to date.
 * [labelling_functions.py](labelling_functions.py) stores a lot of the functions that the these notebooks rely on.

## Getting Set-Up

1. **Create conda environment (python 3.11)**

2. **Install Packages:**

```
pip install "numpy<1.26.4"
pip install matplotlib torch torchvision pycocotools opencv-python PIL ipywidgets ipyevents tqmd
pip install 'git+https://github.com/facebookresearch/sam2.git'
conda install -c conda-forge ipympl
```

There may be some other libraries that need to be installed or updated. Make sure to install that older version of numpy before installing the other libraries. Make sure you have ffmpeg (software) installed as this is how we will create the images from the mp4 videos.

3. **Download SAMv2 Checkpoint:**

Download the model checkpoint from here: [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt). This is the most up to date and largest pretrained version of SAMv2 as of now but you can check the [SAMv2 repo](https://github.com/facebookresearch/sam2) for different versions if you like. 

## How [LO_propagate_masks.ipynb](LO_propagate_masks.ipynb) Works

The `ImageAnnotationWidget` allows you to create positive and negative click prompts on any number of frames.

![Image of unmasked fish with click prompts to indicate where the fish is.](demo_images/unmasked_clicked.png)


