import numpy as np

# ############################################################
# ND Histogram Broken out of Numpy, and then HEAVILY MODIFIED 
# ############################################################
def histogram_nd(input_data, input_bin_edges, weights=None, density=False,
                 hist_dtype=None, return_bin_inds=False):
    """
    Compute the multidimensional histogram of some data.

    Always expecting bin edge arrays -- will never give "range"
    Can be variable type (floats and ints living together)
    Will never normalize

    Inputs:
    =============
    input_data: list of Numpy arrays containing data
       --> Each array corresponds to a histogram dimension
    bin_edges: list of numpy arrays containing histogram bin edges
       --> Each array corresponds to a histogram dimension
       --> Should be in same order as input_data

    Returns:
    ===================
    hist, bin_edges, [optional] multi_inds


    To Do:
    =======================
       1. More options for specifying bins
          --> In-line with Numpy's histogramdd()
       2. Include all options given by Numpy's histogramdd()
          --> density, weights, etc.
    """
    # ==============================================
    # Setup
    # ==============================================
    # Check output hist dtype preference
    if hist_dtype is None:
        # No Preference -- go with default -- int32
        hist_dtype = np.int32

    # ---------------------------
    # Check Format of Input Data
    # ---------------------------

    # Input Data
    # ---------------
    # Is input_data a list of arrays?
    if isinstance(input_data, list):
        # Elements are numpy arrays
        for input_list_element in input_data:
            if not isinstance(input_list_element, np.ndarray):
                raise TypeError('Expecting List Contents to be Numpy Arrays')

    # Is input_data Just One Array?
    if isinstance(input_data, np.ndarray):
        # Make sure it has at least 2 dimensions
        # IF not, add shallow 1st dimension that allows proper looping
        input_data = np.atleast_2d(input_data)

    # Bin Edges
    # -------------
    # Is input_bin_edges Just One Array?
    if isinstance(input_bin_edges, np.ndarray):
        # Make sure it has at least 2 dimensions
        # IF not, add shallow 1st dimension that allows proper looping
        input_bin_edges = np.atleast_2d(input_bin_edges)

    # ---------------------------------------------------
    # Consistency Checks
    # ---------------------------------------------------
    # Make sure all dims have same number of data points
    # Use length of first array as standard
    first_array_length = len(input_data[0])
    for input_data_array in input_data:
        # Every subsequent array should have same length as the first array
        if input_data_array.size != first_array_length:
            raise AttributeError("All input arrays must have equal length!")

    # --------------------------------------------------------------------
    # Binning Calculations
    # --------------------------------------------------------------------
    # Number of Data Points
    num_data_points = first_array_length
    # Number of Dimensions
    num_dims = len(input_data)
    # Array for Number of Bins in Each Dimension
    nbins_per_dim = np.empty(num_dims, np.int32)
    # List for Real Bin Edges
    bin_edges = []
    # Loop Over the Histogram's Axes
    for dim_idx in np.arange(num_dims):

        # Make Final Bin Edges List -- Match dtype of Input Data
        # -------------------------------------------------------
        # Extract dtype of this dimension's input data
        dim_data_dtype = input_data[dim_idx].dtype
        # Input Bin Edges
        dim_input_bin_edges = input_bin_edges[dim_idx]
        # Make Certain of ndarray, convert to dtype of dimension's input data
        dim_bin_edges = np.asarray(dim_input_bin_edges, 
                                   dtype=dim_data_dtype).reshape(-1)
        # Append to Bin Edges List
        bin_edges.append(dim_bin_edges)

        # Update Number of Bins in each Dimension
        # ---------------------------------------
        # Reason for +1 ==> np.searchsorted() will implicitly add a bin on the end!
        nbins_per_dim[dim_idx] = np.int32(bin_edges[dim_idx].size + 1)  

    # ==============================================
    # Histogram Processes
    # ==============================================
    """
    Step 1: (a) Calculate Offset (i.e. Depth, in N-Dimensional Bins, 
                into Final Histogram) of Each Axis
                 --> Using np.prod() product of sizes of all axes with 
                     higher dimension than Current Axis
            (b) Calculate 1D Bin Inds, Separately Along Each Axis
            (c) Multiply by Offset
            (d) Add Cumulatively into Multi-Dim Bin Inds Array
    """

    # Calculate Offsets for Each Axis
    # -------------------------------
    # List to Store Offsets
    dim_offsets_list = [None]*num_dims
    # Loop Over Histogram's Axes
    for dim_idx in np.arange(num_dims):
        # Subs for Accessing Sizes of Higher-Dimension Axes 
        higher_dim_axis_inds = slice((dim_idx+1), None,None)
        # Calculate Product of Higher-Dimension Axis Sizes
        current_axis_offset = nbins_per_dim[higher_dim_axis_inds].prod(
            dtype=hist_dtype) # prod()=1 when ndim==1 (perfect!)
        # Put into List
        dim_offsets_list[dim_idx] = current_axis_offset

    # Build Multi-Dimensional Bin Inds
    # ---------------------------------
    # Create Array for Multi-Dimensional Bin Inds
    data_bins_multi_dim = np.zeros(num_data_points, dtype=hist_dtype)
    # Loop Over Histogram's Axes
    for dim_idx in np.arange(num_dims):
        # Extract Axis Offset from List
        current_axis_offset = dim_offsets_list[dim_idx]
        # Create 1D Bin Inds, Multiply By Offset, 
        # Add to Multi-Dimensional Bin Inds
        data_bins_multi_dim += current_axis_offset * np.searchsorted(
            bin_edges[dim_idx], input_data[dim_idx], 
            'right').astype(hist_dtype)  

    """
    Step 2: (a) Calculate Total Number of Bins
            (b) Count Repetitions in Multi-Dimensional Bin Inds 
                with numpy.bincount
                --> i.e. make a 1D histogram
            (c) Reshape on the fly to (nearly) correct dimensionality
            (d) Apply hist_core_slice to truncate outliers
    """

    # Calculate Total Number of Bins (from num along each axis)
    total_num_bins = nbins_per_dim.prod()
    # List of Slice objects -- Will Truncate Outliers
    hist_core_slice = num_dims*[slice(1,-1,None)]
    # Make 1D Histogram with numpy.bincount (padding to reach total_num_bins),
    #    Cast to Desired Data Type, Reshape into Multi-Dimensional Histogram, 
    #    Apply Slice Object List, Copy to make array contiguous in memory
    hist = np.bincount(
        data_bins_multi_dim, 
        minlength=total_num_bins, 
        weights=weights
        ).reshape(nbins_per_dim)[hist_core_slice].astype(hist_dtype)
    
    # =========================================================
    # Return (Histogram, Bin Edges, [Maybe] Data Bin Inds)
    # =========================================================
    if return_bin_inds:
        return hist, bin_edges, data_bins_multi_dim
    else:
        return hist, bin_edges
