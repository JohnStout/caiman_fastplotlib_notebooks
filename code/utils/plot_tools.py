'''
This code was written to facilitate visualizations using CaImAn and fastplotlib. 

It is specifically designed to work after you have run and saved the results of CNMF.

John Stout 3/15/2024

'''

# Import packages
import fastplotlib as fpl
import tifffile
import numpy as np
from ipywidgets import IntSlider, VBox
from scipy.stats import linregress
import cv2
import os

try:
    import caiman as cm
    from caiman.utils.visualization import get_contours
    print("caiman package loaded...")
except:
    print("CaImAn package not detected")

try:
    import suite2p
    from pathlib import Path
    print("suite2p package loaded...")
except:
    print("Suite2p not detected...")

# a class to handle the cnmf-specific movies using fastplotlib
class play_caiman_movie():

    def __init__(self, images = None, cnmf_object = None):
        '''
        Args:
            >>> images: 3D numpy array of your dataset
            >>> cnmf_object: cnmf object as a result of CNMF.fit

        Returns:
            >>> self.idx_accepted
            >>> self.idx_rejected
            >>> self.coors
            >>> self.mask_array
            >>> self.mask
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

        # get footprint matrices (a list)
        component_footprint = [] # the actual cell 
        component_contour = [] # the area around the cell
        for i in self.idx_accepted:
            component_contour.append(self.coors[i].get('coordinates')) # get the component
            component_footprint.append(np.reshape(self.cnmf_object.estimates.A[:, i].toarray(), (self.d1,self.d2), order='F')) # reshape to footprint
        
        # create our mask
        self.mask_array = np.array(component_footprint) # generate our mask_array
        self.mask = np.max(self.mask_array,axis=0) # generate our mask          

    def __help__(self):

        print("Try the following")
        print("self = play_cnmf_movie(images=images,cnmf_object=cnmfe_model) # instantiate object")
        print("self.play_movie_draw_roi()                           # plot all components")
        print("self.play_movie_draw_roi(plot_type = 'double')       # plots two separate fpl plots")
        print("self.play_movie_draw_roi(components_type='accepted') # only plots accepted")
        print("self.play_movie_draw_roi(components_type='rejected') # only plots rejected")

    def __drawroi__(self,fpl_widget_gridplot,idx_components,color):

        for i in idx_components:

            # get ROIs and center of mass
            roi = self.coors[i].get('coordinates')
            CoM = self.coors[i].get('CoM')

            fpl_widget_gridplot.add_scatter(np.array((roi.T[0,:],roi.T[1,:])).T,colors=color,alpha=10)

    def play_movie_draw_roi(self, components_type: str = 'both', plot_type: str = 'single', cmap='gray', show_movie=True):
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

        if show_movie:
            return iw_mov_roi.show()
        else:
            return iw_mov_roi

    def play_gSig_draw_roi(self, components_type: str = 'both', plot_type: str = 'single', cmap='gray', show_movie=True):
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
            return gauss_filt(frame, gSig_filt)

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

        if show_movie:
            return iw_gs.show()
        else:
            return iw_gs

    def play_spatial_footprint(self, components_type: str = 'both', cmap='gray', show_movie=True):
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

        if show_movie:
            return iw_footprint.show()
        else:
            return iw_footprint

# suite2p plotting
class play_suite2p_movie():

    def __init__(self, suite2p_path: str):
        '''
        Args:
            >>> suite2p_path: path to suite2p output folder
        '''
        
        # list directory content
        listed_data = os.listdir(suite2p_path)

        # loads in your data and automatically assigns their inputs to a variable name
        var_names =[]; data_dict = dict()
        for i in listed_data:
            if '.npy' in i:
                if 'ops' in i:
                    globals()[i.split('.')[0]]=np.load(os.path.join(suite2p_path,i),allow_pickle=True).item()
                    pass
                else:
                    globals()[i.split('.')[0]]=np.load(os.path.join(suite2p_path,i),allow_pickle=True)
                var_names.append([i.split('.')[0]])
        
        for v in var_names:
            setattr(self,v[0],eval(v[0]))

        # define data_path
        if len(self.ops['data_path'])>1 or len(self.ops['tiff_list'])>1:
            print("This code does not support multi-file registration")
        else:
            data_path = os.path.join(self.ops['data_path'][0],self.ops['tiff_list'][0])
            print(data_path)

        # prep stuff
        output_op_file = np.load(Path(self.ops['save_path']).joinpath('ops.npy'), allow_pickle=True).item()
        output_op_file.keys() == self.ops.keys()
        stats_file = Path(self.ops['save_path']).joinpath('stat.npy')
        self.iscell = np.load(Path(self.ops['save_path']).joinpath('iscell.npy'), allow_pickle=True)[:, 0].astype(bool)
        self.stats = np.load(stats_file, allow_pickle=True)

        # get image object
        self.im = suite2p.ROI.stats_dicts_to_3d_array(self.stats, Ly=self.ops['Ly'], Lx=self.ops['Lx'], label_id=True)
        self.im[self.im == 0] = np.nan

        # cell rois
        self.cell_roi = np.nanmax(self.im[self.iscell], axis=0)
        self.cell_roi[~np.isnan(self.cell_roi)]=20    

        # get rejected components as well
        self.cell_roi_rejected = np.nanmax(self.im[self.iscell==False], axis=0)
        self.cell_roi_rejected[~np.isnan(self.cell_roi)]=20   

        # load in the movie        
        self.images = tifffile.memmap(data_path)
  
    def __help__(self):

        print("Try the following")
        print("self = play_suite2p_movie(suite2p_path = r'/your/path/to/plane0')")
        print("mov = self.play_movie_plot_mask(); mov.show() # plot all components")
        print("mov = self.play_movie_plot_mask(components_type='accepted'); mov.show() # plot only accepted components")
        print("mov = self.plot_gaussFiltMovie_plot_mask(); mov # plot gaussian filtered components")

    def play_movie_plot_mask(self, components_type: str = 'both', plot_type='double', show_movie=True):
        
        '''
        Args:
            >>> components_type: whether to plot the accepted, rejected, or all components
        
        Conditional Args:
            >>> plot_type: only responds if you have selected components_type = 'both'
        
            
        Returns:
            iw_movie: fastplotlib object to plot your data

        '''

        if 'accepted' in components_type:
            masks = self.cell_roi
            rgb_idx = 2 # blue color
        elif 'rejected' in components_type:
            masks = self.cell_roi_rejected
            rgb_idx = 0 # red color
        elif 'both' in components_type:
            masks_accepted = self.cell_roi # accepted mask
            masks_rejected = self.cell_roi_rejected # rejected mask
            rgb_accepted_idx = 2 # color code for blue
            rgb_rejected_idx = 0 # color code for red

        if 'both' in components_type:

            if 'double' in plot_type:

                # fastplotlib movie
                iw_movie = fpl.ImageWidget(
                    data=[self.images,self.images.copy()],
                    names=['Accepted','Rejected'],
                    slider_dims=["t"],
                    cmap="gray"
                )

                # add a column to overlay functional activity on structural video
                struct_rgba = np.zeros((self.images.shape[1], self.images.shape[2], 4), dtype=np.float32)
                struct_rgba[:, :, rgb_accepted_idx] = masks_accepted
                struct_rgba[..., -1] = 10
                iw_movie.gridplot[0,0].add_image(struct_rgba, name='struct')
                iw_movie.gridplot[0,0]
                iw_movie.gridplot[0,0]["struct"].data[..., -1] = 0.3

                # add a column to overlay functional activity on structural video
                struct_rgba = np.zeros((self.images.shape[1], self.images.shape[2], 4), dtype=np.float32)
                struct_rgba[:, :, rgb_rejected_idx] = masks_rejected
                struct_rgba[..., -1] = 10
                iw_movie.gridplot[0,1].add_image(struct_rgba, name='struct')
                iw_movie.gridplot[0,1]
                iw_movie.gridplot[0,1]["struct"].data[..., -1] = 0.3

            else:

                # fastplotlib movie
                iw_movie = fpl.ImageWidget(
                    data=self.images,
                    names=['Blue=Good | Red=Bad'],
                    slider_dims=["t"],
                    cmap="gray"
                )

                # add a column to overlay functional activity on structural video
                struct_rgba = np.zeros((self.images.shape[1], self.images.shape[2], 4), dtype=np.float32)
                struct_rgba[:, :, rgb_accepted_idx] = masks_accepted
                struct_rgba[..., -1] = 20
                iw_movie.gridplot[0,0].add_image(struct_rgba, name = 'struct')
                iw_movie.gridplot[0,0]
                iw_movie.gridplot[0,0]["struct"].data[..., -1] = 0.3

                # add a column to overlay functional activity on structural video
                struct_rgba = np.zeros((self.images.shape[1], self.images.shape[2], 4), dtype=np.float32)
                struct_rgba[:, :, rgb_rejected_idx] = masks_rejected
                struct_rgba[..., -1] = 10
                iw_movie.gridplot[0,0].add_image(struct_rgba, name = 'struct2')
                iw_movie.gridplot[0,0]
                iw_movie.gridplot[0,0]["struct2"].data[..., -1] = 0.3

        else:

            # fastplotlib movie
            iw_movie = fpl.ImageWidget(
                data=self.images,
                slider_dims=["t"],
                cmap="gray"
            )

            # add a column to overlay functional activity on structural video
            struct_rgba = np.zeros((self.images.shape[1], self.images.shape[2], 4), dtype=np.float32)
            struct_rgba[:, :, rgb_idx] = masks
            struct_rgba[..., -1] = 10
            iw_movie.gridplot[0,0].add_image(struct_rgba, name='struct')
            iw_movie.gridplot[0,0]
            iw_movie.gridplot[0,0]["struct"].data[..., -1] = 0.3

        if show_movie:
            return iw_movie.show()
        else:
            return iw_movie

    def plot_gaussFiltMovie_plot_mask(self, components_type: str = 'both', plot_type = 'double', show_movie=True):
        '''
        Args:
            >>> components_type: whether to plot the accepted, rejected, or all components
        
        Conditional Args:
            >>> plot_type: only responds if you have selected components_type = 'both'
            
        Returns:
            iw_movie: fastplotlib object to plot your data
            slider_gsig_filt: slider object
        
        Try:
            VBox([iw_movie.show(),slider_gsig_filt])

        '''

        if 'accepted' in components_type:
            masks = self.cell_roi
            rgb_idx = 2 # blue color
        elif 'rejected' in components_type:
            masks = self.cell_roi_rejected
            rgb_idx = 0 # red color
        elif 'both' in components_type:
            masks_accepted = self.cell_roi # accepted mask
            masks_rejected = self.cell_roi_rejected # rejected mask
            rgb_accepted_idx = 2 # color code for blue
            rgb_rejected_idx = 0 # color code for red

        # create a slider for gSig_filt
        slider_gsig_filt = IntSlider(value=3, min=1, max=33, step=1,  description="gSig_filt")

        def apply_filter(frame):
            
            # read slider value
            gSig_filt = (slider_gsig_filt.value, slider_gsig_filt.value)
            
            # apply filter
            return gauss_filt(frame,gSig_filt)

        if 'both' in components_type:

            if 'double' in plot_type:

                # we can use frame_apply feature of `ImageWidget` to apply 
                # the filter before displaying frames
                funcs = {
                    # data_index: function
                    0: apply_filter,
                    1: apply_filter  # filter shown on right plot, index 1
                }

                # fastplotlib movie
                iw_movie = fpl.ImageWidget(
                    data=[self.images,self.images.copy()],
                    names=['Accepted','Rejected'],
                    frame_apply=funcs,
                    slider_dims=["t"],
                    cmap="gray"
                )

                def force_update(*args):
                    # kinda hacky but forces the images to update 
                    # when the gSig_filt slider is moved
                    iw_movie.current_index = iw_movie.current_index
                    iw_movie.reset_vmin_vmax()

                # reset min/max
                iw_movie.reset_vmin_vmax()                

                # add a column to overlay functional activity on structural video
                struct_rgba = np.zeros((self.images.shape[1], self.images.shape[2], 4), dtype=np.float32)
                struct_rgba[:, :, rgb_accepted_idx] = masks_accepted
                struct_rgba[..., -1] = 10
                iw_movie.gridplot[0,0].add_image(struct_rgba, name='struct')
                iw_movie.gridplot[0,0]
                iw_movie.gridplot[0,0]["struct"].data[..., -1] = .3

                # add a column to overlay functional activity on structural video
                struct_rgba = np.zeros((self.images.shape[1], self.images.shape[2], 4), dtype=np.float32)
                struct_rgba[:, :, rgb_rejected_idx] = masks_rejected
                struct_rgba[..., -1] = 10
                iw_movie.gridplot[0,1].add_image(struct_rgba, name='struct')
                iw_movie.gridplot[0,1]
                iw_movie.gridplot[0,1]["struct"].data[..., -1] = .3

                slider_gsig_filt.observe(force_update, "value")
                #VBox([iw_movie.show(), slider_gsig_filt])   

            else:

                # we can use frame_apply feature of `ImageWidget` to apply 
                # the filter before displaying frames
                funcs = {
                    # data_index: function
                    0: apply_filter,  # filter shown on right plot, index 1
                }

                # fastplotlib movie
                iw_movie = fpl.ImageWidget(
                    data=self.images,
                    frame_apply=funcs,
                    names=['Blue=Good | Red=Bad'],
                    slider_dims=["t"],
                    cmap="gray"
                )

                def force_update(*args):
                    # kinda hacky but forces the images to update 
                    # when the gSig_filt slider is moved
                    iw_movie.current_index = iw_movie.current_index
                    iw_movie.reset_vmin_vmax()

                # reset min/max
                iw_movie.reset_vmin_vmax()

                # add a column to overlay functional activity on structural video
                struct_rgba = np.zeros((self.images.shape[1], self.images.shape[2], 4), dtype=np.float32)
                struct_rgba[:, :, rgb_accepted_idx] = masks_accepted
                struct_rgba[..., -1] = 20
                iw_movie.gridplot[0,0].add_image(struct_rgba, name = 'struct')
                iw_movie.gridplot[0,0]
                iw_movie.gridplot[0,0]["struct"].data[..., -1] = .3

                # add a column to overlay functional activity on structural video
                struct_rgba = np.zeros((self.images.shape[1], self.images.shape[2], 4), dtype=np.float32)
                struct_rgba[:, :, rgb_rejected_idx] = masks_rejected
                struct_rgba[..., -1] = 10
                iw_movie.gridplot[0,0].add_image(struct_rgba, name = 'struct2')
                iw_movie.gridplot[0,0]
                iw_movie.gridplot[0,0]["struct2"].data[..., -1] = .3

                # slide updater
                slider_gsig_filt.observe(force_update, "value")
                #VBox([iw_movie.show(), slider_gsig_filt])                         

        else:

            # we can use frame_apply feature of `ImageWidget` to apply 
            # the filter before displaying frames
            funcs = {
                # data_index: function
                0: apply_filter  # filter shown on right plot, index 1
            }

            # fastplotlib movie
            iw_movie = fpl.ImageWidget(
                data=self.images,
                frame_apply=funcs,
                slider_dims=["t"],
                cmap="gray"
            )

            def force_update(*args):
                # kinda hacky but forces the images to update 
                # when the gSig_filt slider is moved
                iw_movie.current_index = iw_movie.current_index
                iw_movie.reset_vmin_vmax()

            # reset min/max
            iw_movie.reset_vmin_vmax()

            # add a column to overlay functional activity on structural video
            struct_rgba = np.zeros((self.images.shape[1], self.images.shape[2], 4), dtype=np.float32)
            struct_rgba[:, :, rgb_idx] = masks
            struct_rgba[..., -1] = 10
            iw_movie.gridplot[0,0].add_image(struct_rgba, name='struct')
            iw_movie.gridplot[0,0]
            iw_movie.gridplot[0,0]["struct"].data[..., -1] = .3

            # slide updater
            slider_gsig_filt.observe(force_update, "value")
            #VBox([iw_movie.show(), slider_gsig_filt])     

        if show_movie:
            return VBox([iw_movie.show(), slider_gsig_filt]) 
        else:
            return iw_movie, slider_gsig_filt

    def play_spatial_footprint(self, components_type: str = 'both', show_movie=True):

        '''
        plays the spatial footprint of your data with whether the data was an accepted (blue) or rejected (red) component

        Args:
            >>> components_type: 'both', 'accepted', or 'rejected'
        
        '''

        if 'accepted' in components_type:
            mask_list = self.im[self.iscell]
            rgb_idx = 2 # blue color
        elif 'rejected' in components_type:
            mask_list = self.im[self.iscell==False]
            rgb_idx = 0 # red color
        elif 'both' in components_type:
            mask_list=self.im
            rgb_accepted_idx = 2 # color code for blue
            rgb_rejected_idx = 0 # color code for red

        # replace nan with 0
        for i in range(len(mask_list)):
            mask_list[i][np.where(np.isnan(mask_list[i]))]=0 

        # fastplotlib movie
        if 'both' in components_type:

            iw_movie = fpl.ImageWidget(
                data=mask_list,
                names=[components_type+" footprints (t = components)"],
                slider_dims=["t"],
                cmap="gray"
            )     

            for i in range(len(mask_list)):
                idx_plot = np.where(mask_list[i]>0) # find values > 0 (contours)
                np_plot = np.vstack((idx_plot[1],idx_plot[0])).T.astype(float) # convert to float and stack data for plotting
                if self.iscell[i]==True:
                    iw_movie.gridplot[0,0].add_scatter(np_plot,colors='b',alpha=0.3)  
                else:
                    iw_movie.gridplot[0,0].add_scatter(np_plot,colors='r',alpha=0.3)  

        else:

            iw_movie = fpl.ImageWidget(
                data=mask_list,
                names=[components_type+" footprints (t = components)"],
                slider_dims=["t"],
                cmap="gray"
            )     

            for i in mask_list:
                idx_plot = np.where(i>0)
                np_plot = np.vstack((idx_plot[1],idx_plot[0])).T.astype(float)
                if 'rejected' in components_type:
                    iw_movie.gridplot[0,0].add_scatter(np_plot,colors='r',alpha=1)
                else:
                    iw_movie.gridplot[0,0].add_scatter(np_plot,colors='b',alpha=1)

        if show_movie:
            return iw_movie.show()
        else:
            return iw_movie

# A simpler class to more broadly plot components
class play_movie_roi():
    def __init__(self, movie_path: str, mask: list):
        '''
        This code is agnostic to accepted or rejected components.

        You will need a mask (np.ndarray) to represent your accepted or rejected components.

        Args:
            >>> movie_path: path to your .tif file
            >>> mask: a matrix with nan or 0 values occupying elements without components, and 1.0s as your components

        '''
    
        # memory map read the frames from your movie
        self.images = tifffile.memmap(movie_path)

        # if you've provided a 3D array with separate masks per component
        if len(mask.shape) > 2:
            self.mask = np.nanmax(mask, axis=0)
            self.mask_list = mask
        else:
            self.mask = mask

    def play_movie_plot_mask(self, color='b', show_movie=True):
        '''
        Args:
            color: accepts 'r','g', or 'b'
            show_movie: default to True. If set to False, you just have to .show() the result
        '''

        if 'b' in color:
            rgb_idx = 2 # blue color
        elif 'r' in color:
            rgb_idx = 0
        elif 'g' in color:
            rgb_idx = 1
            
        # fastplotlib movie
        iw_movie = fpl.ImageWidget(
            data=self.images,
            slider_dims=["t"],
            cmap="gray"
        )

        # add a column to overlay functional activity on structural video
        struct_rgba = np.zeros((self.images.shape[1], self.images.shape[2], 4), dtype=np.float32)
        struct_rgba[:, :, rgb_idx] = self.mask
        struct_rgba[..., -1] = 10
        iw_movie.gridplot[0,0].add_image(struct_rgba, name='struct')
        iw_movie.gridplot[0,0]
        iw_movie.gridplot[0,0]["struct"].data[..., -1] = .3
        
        if show_movie:
            return iw_movie.show()
        else:
            return iw_movie
    
# add this to utils
def gauss_filt(frame, gSig_filt=(7,7)):
    '''
    Smooth your image

    Args:
        >>> frame: 2D image of your movie (an example frame)
        >>> gSig_filt: tuple of smoothing factor (e.g. gSig_filt=(5,5))

    Returns:
        ex: example frame smoothed over
    
    This code was taken from caiman high_pass_filter_space

    '''
    ksize = tuple([(3 * i) // 2 * 2 + 1 for i in gSig_filt])
    ker = cv2.getGaussianKernel(ksize[0], gSig_filt[0])
    ker2D = ker.dot(ker.T)
    nz = np.nonzero(ker2D >= ker2D[:, 0].max())
    zz = np.nonzero(ker2D < ker2D[:, 0].max())
    ker2D[nz] -= ker2D[nz].mean()
    ker2D[zz] = 0
    smoothed_frame = cv2.filter2D(np.array(frame, dtype=np.float32),-1, ker2D, borderType=cv2.BORDER_REFLECT)
    return smoothed_frame

def play_movie(movie_path=None,images=None, cmap = 'gray', show_movie=True):
    '''
    Plays the recorded movie as a fastplotlib object. 

    Recommended to define your movie path, rather than feeding the images to the function.

    movie_path option triggers a memory mapped reading of the data to support visualization of files that cannot fit into RAM

    Args (only define movie_path OR images)
        >>> movie_path: path to movie that you want to watch
        >>> images: a 3D array (t,row,col) numpy array

    Optional Args:
        >>> cmap: accepts any python acceptable colormap
    '''

    if 'movie_path' is not None and 'images' is not None:
        images = tifffile.memmap(movie_path)
    elif 'movie_path' is not None and 'images' is None:
        images = tifffile.memmap(movie_path)

    # create an image with ROI overlapped
    iw_cnmf = fpl.ImageWidget(
        data=images,#mask_array 
        slider_dims=["t"],
        cmap=cmap
    )
    
    if show_movie:
        return iw_cnmf.show()
    else:
        return iw_cnmf

def scat_data(x,y):

    # polyfit
    b, a = np.polyfit(x, y, deg=1)
    
    # regression line
    reg_line = a + b * x
    
    # regression statistics
    print(linregress(x, y))

    return reg_line, b, a

    