# Installation Instructions and Labelling Guideline

This is a guide for installing all the necessay software and setting up your python environment to run the notebook used to edit generated masks. If you run into any problems during this process feel free to reach out and I can troubleshoot with you! There is also information about how to use the MaskEditor widget and some guidelines for how to label the data.

## Installation and Environment Set Up

### Download and Install Visual Studio Code

Download the appropirate version of VS Code for your operating system from [here](https://code.visualstudio.com/download) and follow the installation instructions. 

### Download and Install Python

Download the latest version of Python for your operating system from [here](https://www.python.org/downloads/) and follow the installation instructions. 

### Create a Python Virtual Environment

1. Create a folder from which you will work on all the labelling tasks. You should place both the labelling_functions.py file and the LO_edit_masks.ipynb file in this directory. 
2. Open the VS Code app and then click on the folder symbol that says "Open..." beside it. Open the folder that you just created above.
3. Create a virtual environment.
4. Open the Command Palette (Shift + Command/Control + P)
5. Start typing the "Python: Create Environment" command to search, and then select the command.
6. The command presents a list of environment types, Venv or Conda. Select Venv.
7. The command then presents a list of interpreters (Python versions) that can be used for your project. Select the interpreter you installed above.
8. After selecting the interpreter, a notification will show the progress of the environment creation and the environment folder (/.venv) will appear in your workspace.
9. Ensure your new environment is selected by using the "Python: Select Interpreter" command from the Command Palette.

### Install Required Libraries

You can open a new terminal window in VS Code by selecting Terminal > New Terminal. Ensure that your virtual environment is activated in the terminal window by double checking ".venv" appears in the command line. Copy and paste the lines below into your terminal window and press enter to install the libraries.

```
pip install numpy matplotlib pycocotools opencv-python pillow IPython ipywidgets ipykernel ipympl tqdm
jupyter nbextension enable --py widgetsnbextension
```

### Open the Notebook

Restart VS Code and open to the folder you created for the project as you did before. Open the IPython Notebook called LO_edit_masks.ipynb. In the righthand corner of the notebook it will say "Select Kernel". Click on this and then select ".venv" from the list that appears. This is the virtual environment that we created and installed all the libraries to. You should now be able to run the notebook without problems. 

## Labelling Guidelines

This is what the widget for editing the masks looks like:

![Image of the MaskEditor widget.](demo_images/MaskEditorWidget.png)

The mask of the fish is plotted transparently over the fish. Notice how some of the fin of the fish are not included in the mask. This is a very common occurance with these masks that are auto-generated. The masks needs to be edited to include all the parts of the fish and ensure that no part of the background is included accidentally.

I will list the function of each of the buttons, drop down menus, and sliders in the widget.

* The `Previous` and `Next` buttons flip between the frames of the video. 
* The Mode buttons `Draw` and `Erase` will allow you to either add to the mask or remove from the mask.
* The Drawing Tool dropdown menu allows you to swtich between drawing methods. The lasso tool is a click and drag drawing tool that will add/erase all the area inside the lasso selection to the mask. The polygon tool allows you to click to draw the vertices of a polygon around the area you want to add/remove to the mask. Try out both of the tools to see which you prefer!
* The `Smooth Mask` button will get rid of any interior holes in the mask and remove any floating points. It will also smooth out some rough edges.
* The `Undo` button undoes the last edit made to the mask currently being displayed. This includes undoing additions and deletions to the mask as well as smoothing the mask. 
* The `Zoom In` and `Zoom Out` buttons allow you to zoom into/out of an area of the image. To zoom, click the `Zoom In` button and then click where you want to zoom in to on the image. The `Reset Zoom` button resets the zoom to the original image size.
* The `Save JSON` button saves all the masks of the video (edited and unedited) to a new JSON file called "edited_instances_default.json".
* The `Brightness` and `Contrast` sliders change the brightness and contrast of the image to help you see the fish when labelling. This doesn't change the appearance of the saved images.
* The Track drowpdown menu allows you to choose which object you want to edit the mask for. In most cases there is only one object so you can ignore this menu.

When editing the masks, make sure that you are including all the parts of the fish that are visible. Sometimes it can help to have the video open beside you so you can get more context on whether something you think you see is actually part of the fish. There are many times when a fin of the fish is in a shadowy area. It is important that we include the fins or even parts of the fins if we can see them. If you are completely unable to see a part of the fish because it is too dark or turbid then do not label it. I will add more examples images and details here...

