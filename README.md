This very small package is meant for notebook interactions using CaImAn and fastplotlib.

These notebooks were created for analyzing miniscope and slice data.

You'll notice these notebooks look a lot like the CaImAn demos and that is because a good bit is from them. However, I have modified various points in the code to automate things as needed. I focused on a limited parameter space to get ROI out of the calcium imaging data. Despite worrying about less parameters, you still have access to all parameters provided by caiman in a very clear way so that they can be adjusted as needed.

These code come from the decode_lab_code package that I was working on for A. Hernan lab, but I decided to make a more general package that does not have dependencies drawn from within the decode_lab_code package. In other words, I want all dependencies to be from more resolved and recognized packages and so these notebooks will define any and all functions within them if they're needed.

Here is the organization of the folders:
calcium_imaging
    |
    |__CNMFE
    |   |
    |   |__CNMFE.ipynb: Code to run CNMFE. Entirely dependent on CaImAn and fastplotlib. 
    |   |__VizResults.ipynb: Code to visualize the results of CNMFE, inspired by mesmerize-viz.
    |   |
    |   |__Refit
    |       |__RefitCNMFE.ipynb: Refit is not supported by CNMFE, but this might be useful for typical CNMF on 2P data
    |   
    |__MoviMods
    |   |__avi_to_tif.ipynb: Code to run convert .avi files to .tif files and combine into a single file. Not really necessary.
    |   |__TrimMovie.ipynb: Code to trim your the spatial inclusion of your FOV (sometimes removing deadspace around GRIN helps CNMFE) and spatially downsampling your film to improve performance

Please contact John Stout at john.j.stout.jr@gmail.com for questions.