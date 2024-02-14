# Jupyter-notebooks to analyze and visualize miniscope data with CaImAn and fastplotlib
Heavily inspired by mesmerize-viz and doesn't come close in comparison to visuals. Here, I focused on very specific visualizations to help curate CaImAn defined components and reduce the load of parameter setting. The output of these notebooks (CNMFE and VizResults) are various files with unique ids to reference to as well as .csv files containing calcium imaging data and ROI masks/spatial footprint data.

Follow these instructions to use:

1) Download CaImAn using their instructions (also written below): https://github.com/flatironinstitute/CaImAn

        conda install -n base -c conda-forge mamba   # install mamba in base environment
        mamba create -n caiman -c conda-forge caiman # install caiman
        conda activate caiman  # activate virtual environment
        caimanmanager install # just their demo stuff (very useful for troubleshooting)
        
3) Download fastplotlib using their instructions (also written below): https://github.com/fastplotlib/fastplotlib
   
        pip install "fastplotlib[notebook]" # make sure that caiman is activated!! (this should follow the steps above in the same terminal)

4) Download this code (requires git language: https://git-scm.com/downloads)

        git clone https://github.com/JohnStout/calcium_imaging

If you are stuck on mamba, try this (recommendation: https://github.com/nel-lab/mesmerize-core):

        conda clean -a # clean up conda in case it takes a while to install mamba

These notebooks were created for analyzing miniscope and slice data.

You'll notice these notebooks look a lot like the CaImAn demos and that is because a good bit is from them. However, I have modified various points in the code to automate things as needed. I focused on a limited parameter space to get ROI out of the calcium imaging data. Despite worrying about less parameters, you still have access to all parameters provided by caiman in a very clear way so that they can be adjusted as needed.

These code come from the decode_lab_code package that I was working on for A. Hernan lab, but I decided to make a more general package that does not have dependencies drawn from within the decode_lab_code package. In other words, I want all dependencies to be from more resolved and recognized packages and so these notebooks will define any and all functions within them if they're needed.

Here is the organization of the folders:

    code
    ├── cnmfe
    │   ├── CNMFE.ipynb: code to analyze miniscope data with CNMFE. Includes interactive plots by caiman and fastplotlib.
    │   └── VizResults.ipynb: code to visualize results from CNMFE. Includes a ton of interactive plots!
    └── movie_mod
        ├── TrimMovie.ipynb: used to trim the FOV (useful if you have deadspace around a GRIN lens) or spatially downsample your data.
        └── avi_to_tif.ipynb: used to convert avi movies to tif files for TrimMovie

Please contact John Stout at john.j.stout.jr@gmail.com for questions.
