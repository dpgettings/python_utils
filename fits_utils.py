# =============================
# Setup
# =============================
import numpy as np
# Recarray util function hidden deep within Numpy
from numpy.lib.recfunctions import stack_arrays as np_stack_arrays
# FITS library import
try:
    from astropy.io import fits
except:
    try:
        import pyfits as fits
    except:
        raise ImportError('Requires a FITS Module!')

# -------------------------------------
# Numpy -> FITS Type Conversions
# -------------------------------------
type_dict = {}
type_dict[np.bool_] = "L"
type_dict[np.int8] = "B"
type_dict[np.uint8] = "B"
type_dict[np.int16] = "I"
type_dict[np.uint16] = "I"
type_dict[np.int32] = "J"
type_dict[np.uint32] = "J"
type_dict[np.int64] = "K"
type_dict[np.uint64] = "K"
type_dict[np.float32] = "E"
type_dict[np.float64] = "D"
type_dict[np.str] = "A"
type_dict[np.string_] = "A"
type_dict[str] = "A"

# --------------------------
# Misc
# --------------------------
# String type def for convenience
STRING_TYPE = np.dtype(np.str).type


# #############################################################
# Convert a Dictionary of Numpy Arrays to a FITS BinTableHDU
# #############################################################
def data_dict_to_hdu(column_data):
    """
    Takes a dictionary of Numpy arrays 
    Makes a FITS BinTableHDU 
       --> Each array becomes a column
       --> The colname is taken from the array's dictionary key
    """

    # ===============================
    # Step 1: Get List of Columns
    # ===============================
    # Empty List for Columns
    FITS_Column_list = []

    # +++++++++++++++++++++
    # Loop Through Columns
    # +++++++++++++++++++++
    for col_name in column_data:

        # ---------------------------
        # Step 1a: Get FITS Datatype
        # ---------------------------

        # Numpy type
        # ---------------
        col_numpy_dtype = column_data[col_name].dtype
        # Numpy Base Type
        col_numpy_base_type = col_numpy_dtype.type

        # FITS type
        # ---------------
        # Look up in type_dict{} using Numpy type
        col_fits_dtype = type_dict[col_numpy_base_type] 
        # For Strings, Add Size Information
        if col_numpy_base_type is STRING_TYPE:
            # Find num chars in string
            numpy_string_size = numpy_dtype.itemsize
            # Add Size info
            col_fits_dtype = str(numpy_string_size) + col_fits_dtype
        
        # ------------------------------
        # Step 1b: Make FITS Column Object
        # ------------------------------
        # Make Column Object
        col_Column_obj = fits.Column(name=col_name, format=col_fits_dtype, 
                                     array=column_data[col_name]) 

        # -----------------------------------
        # Step 1c: Add FITS Column to List
        # -----------------------------------
        # Append Column to List
        FITS_Column_list.append(col_Column_obj)

    # =====================================
    # Step 2: Make New BinTableHDU
    # =====================================
    # Make HDU
    output_hdu = fits.new_table(fits.ColDefs(FITS_Column_list))
    # Verify HDU
    output_hdu.verify('silentfix')

    # Return
    return output_hdu


# ###########################################################################
# Merge (any number of) Line-Matched FITS Tables into a New BinTableHDU
# ###########################################################################
def merge_table_hdus(*input_table_hdus):
    """
    The FITS table analogue of numpy.hstack()

    Takes 2 (or more) FITS table HDUs
    Assumes input tables are row-matched
    Merges input table HDUs into single BinTableHDU
    
    Note: Column name collision will probably cause a crash
    """
    ## Steps:
    ##   1. Create empty list for Column objects
    ##   2. Loop over tables, over table.columns, append each column to list
    ##   3. Input list into fits.ColDefs to make new table
    ##   4. Return new table
    
    # Make New Column List
    # --------------------
    # New list for Column objects
    new_Column_list = []
    # Loop through input tables
    for this_input_table in input_table_hdus:

        # Loop through Column objects of this table
        for col in this_input_table.columns:

            # Append Column object to list
            new_Column_list.append(col)

    # Turn Column List into new BinTableHDU
    # -------------------------------------
    # Make List into New fits.ColDefs Object
    new_ColDefs = fits.ColDefs(new_Column_list)
    # Make New BinTableHDU from ColDefs Object
    new_table = fits.new_table(new_ColDefs)

    # Return Merged Table
    # -------------------
    return new_table

# ##########################################################
# Append (any number of) FITS Tables into a New BinTableHDU 
# (should have the same fields, formats, etc.)
# ##########################################################
def append_table_hdus(*input_table_hdus):
    """
    The FITS table analogue of numpy.vstack()

    Takes 2 (or more) FITS table HDUs
    Assumes input tables have same columns
    Merges input table HDUs into single BinTableHDU
    
    Note: Missing columns are likely not handled gracefully
    """

    # ------------------------------------
    # Read in, store table data to combine
    # ------------------------------------
    # List for input data recarrays
    rec_list = []
    # Initialize header var to None
    new_header = None

    # ++++++++++++++++++++++++++++++
    # Loop through input table HDUs
    # ++++++++++++++++++++++++++++++
    for this_input_hdu in input_table_hdus:
        # Extract numpy.recarray from HDU
        this_input_recarray = this_input_hdu.data.base

        # Store Data and Header
        # ---------------------
        # Append numpy.recarray to list
        rec_list.append(this_input_recarray)
        # Extract Header
        if new_header is None:
            new_header = this_input_hdu.header

    # ---------------------------
    # Create, Return Output HDU
    # ---------------------------

    # Make New BinTableHDU
    # --------------------
    # Make new recarray from input recarrays
    new_array = np_stack_arrays(rec_list, usemask=False, asrecarray=True, 
                                autoconvert=True)
    # Make new BinTableHDU directly from new recarray
    new_hdu = fits.BinTableHDU(data=new_array, header=new_header)

    # Return New HDU
    # --------------
    return new_hdu
