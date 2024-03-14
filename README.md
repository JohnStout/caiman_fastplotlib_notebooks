# Jupyter-notebooks to analyze and visualize miniscope data with CaImAn and fastplotlib
Heavily inspired by mesmerize-viz and doesn't come close in comparison to visuals. Here, I focused on very specific visualizations to help curate CaImAn defined components and reduce the load of parameter setting. The output of these notebooks (CNMFE and VizResults) are various files with unique ids to reference to as well as .csv files containing calcium imaging data and ROI masks/spatial footprint data. 

Follow these instructions to use:

        git clone https://github.com/JohnStout/caiman_fastplotlib_notebooks
        cd caiman_fastplotlib_notebooks
        conda env create -n <env_name> -f environment.yml

If the instructions above do not work, go the long route:

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

If you are trying to update a local copy of the code to the updated version:

        git pull https://github.com/JohnStout/calcium_imaging

If the code isn't actually updating from the repo:

        git reset --hard
        git pull https://github.com/JohnStout/calcium_imaging

These notebooks were created for analyzing miniscope and slice data.

You'll notice these notebooks look a lot like the CaImAn demos and that is because a good bit is from them. However, I have modified various points in the code to automate things as needed. I focused on a limited parameter space to get ROI out of the calcium imaging data. Despite worrying about less parameters, you still have access to all parameters provided by caiman in a very clear way so that they can be adjusted as needed.

These code come from the decode_lab_code package that I was working on for A. Hernan lab, but I decided to make a more general package that does not have dependencies drawn from within the decode_lab_code package. In other words, I want all dependencies to be from more resolved and recognized packages and so these notebooks will define any and all functions within them if they're needed.

Here is the organization of the folders:

    code
    ├── miniscope_cnmfe
    │   ├── CNMFE.ipynb: code to analyze miniscope data with CNMFE. Includes interactive plots by caiman and fastplotlib. 
    │   └── VizResults_miniscope.ipynb: code to visualize results from CNMFE. Includes a ton of interactive plots!
    ├── slice_cnmfe: Code to analyze slice imaging data
    │   ├── CNMFE_4D.ipynb: Code to analyze 4D calcium imaging data (structural + functional image recorded at the same time)
    │   ├── CNMFE_multi_plane.ipynb: Code to analyze calcium_imaging data with a structural image that was taken separately
    │   ├── CNMFE_single_plane.ipynb: Code to analyze calcium_imaging data without a structural image (just functional data)
    │   ├── CNMFE_singlecell_body.ipynb: Code to analyze a single cells, cell body activity with CNMFE
    │   ├── CNMFE_singlecell_processes.ipynb: Code to extract a single cells dendrite or axonal activity. This code requires work **
    │   ├── VizResults_slice_4D.ipynb: code to visualize results from two simultaneously recorded data types (structural and functional image)
    │   ├── VizResults_slice_single_plane.ipynb: Code to visualize CNMFE results from a single recording type (Functional)
    │   └── VizResults_slice_multi_plane.ipynb: Code to visualize CNMFE results from a single recording type (Functional) and overlaying single images.
    ├── utils: 
    │   ├── converters.py: functions that support various conversions of data. The most useful which is converters.convert_to_tif which accepts .wmv, .avi, or .mp4 and converts to .tif
    │   ├── plot_tools.py: An object (play_cnmf_movie) that creates fastplotlib plots for one-two liner visualizations. There are also some plotting functions that could be useful here.
    │   ├── Process_slice_images_fiji.ipynb: Code that will help preprocess slice imaging data (trimming and downsampling), prior to photobleach correction in Fiji.
    │   ├── Convert_wmv.ipynb: Converts .wmv files for the user.
    │   └── Consolidating_plots.ipynb: Code that uses the converters.py tools for quick visualization examples
    └── movie_mod (should be merged with utils)
        ├── TrimMovie.ipynb: used to trim the FOV (useful if you have deadspace around a GRIN lens) or spatially downsample your data.
        └── avi_to_tif.ipynb: used to convert avi movies to tif files for TrimMovie

Miniscope code draws on paths formed using D. Aharoni software for miniscope https://github.com/Aharoni-Lab/Miniscope-DAQ-QT-Software.

Please contact John Stout at john.j.stout.jr@gmail.com for questions.
