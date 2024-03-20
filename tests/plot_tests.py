# Import packages
import os
import sys
root_dir = os.path.split(os.getcwd())[0]# get root
utils_dir = os.path.join(root_dir,'code','utils') # get utils folder path
sys.path.append(utils_dir) # add it to system path (not ideal) - doing this to reduce pip installs for local lab usage
import plot_tools

#---------------##

# if you have suite2p or caiman results, you can define your directories here

# path to your plane output from suite2p
suite2p_path = r"/Users/js0403/2p/A04/suite2p/plane0"
movie_path = r"/Users/js0403/2p/A04/img.tif"
caiman_path=None

##--------------##

# Here, we will test the play_movie function, notice the use of show_movie

def testA(movie_path):
    return plot_tools.play_movie(movie_path)

def testB(movie_path):
    mov=plot_tools.play_movie(movie_path,show_movie=False)
    return mov.show()

##--------------##

# Here, we can test all suite2p fastplotlib playthroughs

# testing accepting/rejected components with two gridplots plots
def s2ptest1(suite2p_path):
    obj = plot_tools.play_suite2p_movie(suite2p_path=suite2p_path)
    mov = obj.play_movie_plot_mask(components_type='both',plot_type='double')
    return mov

def s2ptest2(suite2p_path):
    # testing accepting/rejected components with one gridplots plots
    obj = plot_tools.play_suite2p_movie(suite2p_path=suite2p_path)
    iw_movie = obj.play_movie_plot_mask(components_type='both',plot_type='single')
    return iw_movie

def s2ptest3(suite2p_path):
    # testing accepting components
    obj = plot_tools.play_suite2p_movie(suite2p_path=suite2p_path)
    iw_movie = obj.play_movie_plot_mask(components_type='accepted')
    return iw_movie

def s2ptest4(suite2p_path):
    # testing rejected components
    obj = plot_tools.play_suite2p_movie(suite2p_path=suite2p_path)
    iw_movie = obj.play_movie_plot_mask(components_type='rejected')
    return iw_movie

def s2ptest5(suite2p_path):
    obj = plot_tools.play_suite2p_movie(suite2p_path=suite2p_path)
    iw_movie = obj.plot_gaussFiltMovie_plot_mask()
    return iw_movie

def s2ptest6(suite2p_path):
    obj = plot_tools.play_suite2p_movie(suite2p_path=suite2p_path)
    iw_movie = obj.plot_gaussFiltMovie_plot_mask(components_type='accepted')
    return iw_movie

def s2ptest7(suite2p_path):
    obj = plot_tools.play_suite2p_movie(suite2p_path=suite2p_path)
    iw_movie = obj.plot_gaussFiltMovie_plot_mask(components_type='rejected')
    return iw_movie

def s2ptest8(suite2p_path):
    obj = plot_tools.play_suite2p_movie(suite2p_path=suite2p_path)
    iw_movie = obj.play_spatial_footprint(components_type='accepted')
    return iw_movie

def s2ptest9(suite2p_path):
    obj = plot_tools.play_suite2p_movie(suite2p_path=suite2p_path)
    iw_movie = obj.play_spatial_footprint(components_type='rejected')
    return iw_movie






