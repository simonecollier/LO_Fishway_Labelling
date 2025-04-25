# SAMv2 Setup

1. Create conda environment (python 3.11)

2. Install:

```
pip install "numpy<1.26.4"
pip install matplotlib torch torchvision pycocotools opencv-python PIL ipywidgets ipyevents tqmd
pip install 'git+https://github.com/facebookresearch/sam2.git'
conda install -c conda-forge ipympl
```

3. Ensure you have:
```
sudo apt update
sudo apt install -y libopenh264-6
```

4. Once ready, launch the gradio app:
```
python samv2_autolabeller_gradio_app.py
```