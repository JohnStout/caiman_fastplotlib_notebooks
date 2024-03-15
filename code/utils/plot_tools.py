'''
This code was written to facilitate visualizations using CaImAn and fastplotlib. 

It is specifically designed to work after you have run and saved the results of CNMF.

John Stout 3/15/2024

'''

# Import packages
import fastplotlib as fpl
import caiman as cm
from caiman.utils.visualization import get_contours
import tifffile
import numpy as np
from ipywidgets import IntSlider, VBox
from caiman.motion_correction import high_pass_filter_space
from scipy.stats import linregress

# a class to handle the cnmf-specific movies using fastplotlib
class play_cnmf_movie():

    def __init__(self, images = None, cnmf_object = None):
        '''
        Args:
            >>> images: 3D numpy array of your dataset
            >>> cnmf_object: cnmf object as a result of CNMF.fit
        '''
        # define object attributes
        self.images = images
        self.cnmf_object = cnmf_object

        if cnmf_object.estimates.idx_components is None and cnmf_object.estimates.idx_components_bad is None:
            assert KeyError("You must have run: nmfe_model.estimates.evaluate_components prior to running this code")

        # get coordinates
        self.d1, self.d2 = np.shape(images[0,:,:])
        self.coors = get_contours(self.cnmf_object.estimates.A, (self.d1, self.d2), thr=0.99)
        self.idx_accepted = self.cnmf_object.estimates.idx_components # redundant line to backtrack on
        self.idx_rejected = self.cnmf_object.estimates.idx_components_bad

    def __drawroi__(self,fpl_widget_gridplot,idx_components,color):

        for i in idx_components:

            # get ROIs and center of mass
            roi = self.coors[i].get('coordinates')
            CoM = self.coors[i].get('CoM')

            fpl_widget_gridplot.add_scatter(np.array((roi.T[0,:],roi.T[1,:])).T,colors=color,alpha=10)

    def __help__(self):

        print("Try the following")
        print("self = play_cnmf_movie(images=images,cnmf_object=cnmfe_model) # instantiate object")
        print("self.play_movie_draw_roi()                           # plot all components")
        print("self.play_movie_draw_roi(plot_type = 'double')       # plots two separate fpl plots")
        print("self.play_movie_draw_roi(components_type='accepted') # only plots accepted")
        print("self.play_movie_draw_roi(components_type='rejected') # only plots rejected")

    def play_movie_draw_roi(self, components_type: str = 'both', plot_type: str = 'single', cmap='gray'):
        '''
        Args:
            >>> components_type: Tells the code which components to plot.
                    Allows the following as input: 
                            'accepted', 'rejected', 'both'
            >>> plot_type: 'single' plots a single fastplotlib plot. 'double' plots two plots
                    **Note: you can only set this to double if components_type is 'both'
        
        Optional Args:
            >>> cmap: default gray, accepts any python colormap

        '''

        if 'both' not in components_type and plot_type is 'double':
            plot_type = 'single'

        # create an image with ROI overlapped
        if 'single' in plot_type:
            iw_mov_roi = fpl.ImageWidget(
                data=self.images,#mask_array 
                slider_dims=["t"],
                cmap=cmap
            )   
        elif 'double' in plot_type:        
            iw_mov_roi = fpl.ImageWidget(
                data=[self.images,self.images.copy()],#mask_array 
                slider_dims=["t"],
                cmap=cmap
            )   

        if 'both' in components_type:
            if 'single' in plot_type:
                self.__drawroi__(fpl_widget_gridplot=iw_mov_roi.gridplot[0,0],idx_components=self.idx_accepted,color='y')
                self.__drawroi__(fpl_widget_gridplot=iw_mov_roi.gridplot[0,0],idx_components=self.idx_rejected,color='r')

            elif 'double' in plot_type:
                self.__drawroi__(fpl_widget_gridplot=iw_mov_roi.gridplot[0,0],idx_components=self.idx_accepted,color='y')
                self.__drawroi__(fpl_widget_gridplot=iw_mov_roi.gridplot[0,1],idx_components=self.idx_rejected,color='r')

        # Ignore the plot_type variable for this
        if 'accepted' in components_type:
            self.__drawroi__(fpl_widget_gridplot=iw_mov_roi.gridplot[0,0],idx_components=self.idx_accepted,color='y')
        if 'rejected' in components_type:
            self.__drawroi__(fpl_widget_gridplot=iw_mov_roi.gridplot[0,0],idx_components=self.idx_rejected,color='r')

        return iw_mov_roi

    def play_gSig_draw_roi(self, components_type: str = 'both', plot_type: str = 'single', cmap='gray'):
        """
        This function creates the gSig filtered image with components overlaid
        """

        if 'both' not in components_type and plot_type is 'double':
            plot_type = 'single'

        # create a slider for gSig_filt
        slider_gsig_filt = IntSlider(value=self.cnmf_object.params.init['gSig'][0], min=1, max=33, step=1,  description="gSig_filt")

        def apply_filter(frame):
            # read slider value
            gSig_filt = (slider_gsig_filt.value, slider_gsig_filt.value)
            
            # apply filter
            return high_pass_filter_space(frame, gSig_filt)

        # we can use frame_apply feature of `ImageWidget` to apply 
        # the filter before displaying frames
        funcs = {
            # data_index: function
            0: apply_filter  # filter shown on right plot, index 1
        }

        # input movie will be shown on left, filtered on right
        if 'single' in plot_type:
            iw_gs = fpl.ImageWidget(
                data=self.images,
                frame_apply=funcs,
                names=[str("gSig of " + str(self.cnmf_object.params.init['gSig'][0]) + " for CNMFE")],
                #grid_plot_kwargs={"size": (1200, 600)},
                cmap=cmap
            )
        elif 'double' in plot_type:
            iw_gs = fpl.ImageWidget(
                data=[self.images,self.images.copy()],
                frame_apply={0: apply_filter, 1: apply_filter},
                names=[str("gSig of " + str(self.cnmf_object.params.init['gSig'][0]) + " for CNMFE"), str("gSig of " + str(self.cnmf_object.params.init['gSig'][0]) + " for CNMFE")],
                #grid_plot_kwargs={"size": (1200, 600)},
                cmap=cmap
            )           

        def force_update(*args):
            # kinda hacky but forces the images to update 
            # when the gSig_filt slider is moved
            iw_gs.current_index = iw_gs.current_index
            iw_gs.reset_vmin_vmax()

        # reset min/max
        iw_gs.reset_vmin_vmax()

        # slide updater
        slider_gsig_filt.observe(force_update, "value")

        if 'both' in components_type:
            if 'single' in plot_type:
                self.__drawroi__(fpl_widget_gridplot=iw_gs.gridplot[0,0],idx_components=self.idx_accepted,color='y')
                self.__drawroi__(fpl_widget_gridplot=iw_gs.gridplot[0,0],idx_components=self.idx_rejected,color='r')
            elif 'double' in plot_type:
                self.__drawroi__(fpl_widget_gridplot=iw_gs.gridplot[0,0],idx_components=self.idx_accepted,color='y')
                self.__drawroi__(fpl_widget_gridplot=iw_gs.gridplot[0,1],idx_components=self.idx_rejected,color='r')

        # Ignore the plot_type variable for this
        if 'accepted' in components_type:
            self.__drawroi__(fpl_widget_gridplot=iw_gs.gridplot[0,0],idx_components=self.idx_accepted,color='y')
        if 'rejected' in components_type:
            self.__drawroi__(fpl_widget_gridplot=iw_gs.gridplot[0,0],idx_components=self.idx_rejected,color='r')

        return iw_gs

    def play_spatial_footprint(self, components_type: str = 'both', cmap='gray'):
        '''
        Args:
            >>> components_type: Tells the code which components to plot.
                    Allows the following as input: 
                            'accepted', 'rejected', 'both'
            >>> plot_type: 'single' plots a single fastplotlib plot. 'double' plots two plots
                    **Note: you can only set this to double if components_type is 'both'
        
        Optional Args:
            >>> cmap: default gray, accepts any python colormap

        '''

        # a video of your footprints
        footprints = list()

        # get accepted and rejected footprints
        if 'accepted' in components_type:
            for i in self.idx_accepted:
                # get spatial footprints
                footprints.append(np.reshape(self.cnmf_object.estimates.A[:, i].toarray(), (self.d1,self.d2), order='F'))
        elif 'rejected' in components_type:
            for i in self.idx_rejected:
                # get spatial footprints
                footprints.append(np.reshape(self.cnmf_object.estimates.A[:, i].toarray(), (self.d1,self.d2), order='F'))
        elif 'both' in components_type:
            for i in range(self.cnmf_object.estimates.A.shape[1]):
                # get spatial footprints
                footprints.append(np.reshape(self.cnmf_object.estimates.A[:, i].toarray(), (self.d1,self.d2), order='F'))

        # convert to numpy
        footprints = np.array(footprints)

        # plot
        iw_footprint = fpl.ImageWidget(
            data=footprints,#mask_array 
            slider_dims=["t"],
            names="Footprints (t = components)",
            cmap="gray"
        )   

        # plot roi 
        if 'accepted' in components_type:
            self.__drawroi__(fpl_widget_gridplot=iw_footprint.gridplot[0,0],idx_components=self.idx_accepted,color='y')
        elif 'rejected' in components_type:
            self.__drawroi__(fpl_widget_gridplot=iw_footprint.gridplot[0,0],idx_components=self.idx_rejected,color='r')
        elif 'both' in components_type:
                self.__drawroi__(fpl_widget_gridplot=iw_footprint.gridplot[0,0],idx_components=self.idx_accepted,color='y')
                self.__drawroi__(fpl_widget_gridplot=iw_footprint.gridplot[0,0],idx_components=self.idx_rejected,color='r')           

        return iw_footprint

def play_movie(images, cmap = 'gray'):
    '''
    Args:
        >>> images: a 3D array (t,row,col) numpy array

    Optional Args:
        >>> cmap: accepts any python acceptable colormap
    '''

    # create an image with ROI overlapped
    iw_cnmf = fpl.ImageWidget(
        data=images,#mask_array 
        slider_dims=["t"],
        cmap=cmap
    )
    
    return iw_cnmf

def scat_data(x,y):

    # polyfit
    b, a = np.polyfit(x, y, deg=1)
    
    # regression line
    reg_line = a + b * x
    
    # regression statistics
    print(linregress(x, y))

    return reg_line, b, a

    