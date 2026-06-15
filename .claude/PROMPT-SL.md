I need a streamlit app to demo workflow

Step 1: Load a spectral dataset (e.g., from a DFS2 file) and display its metadata and geometry.

A plot button should be available here, using mikeio.Datset[item].isel(time=TIMESTEP).plot() to visualize the data for a specific timestep.

The timesteps should be loaded and a slider bar should be created to allow the user to select a specific timestep. The default timestep is the last timestep.
There should be three dropdowns to specify the SURFACE ELEVATION, U and V vectors. The dropdowns should be populated with the items from the dataset. The default selection for the dropdowns should be the first item in the dataset. The SURFACE ELEVATION dropdown is also used for the plot above.

STEP 2: Preprocessing

Here the user specificies the
- timestep range and interval for the analysis (e.g., from timestep 0 to 100 with an interval of 10)
- Area bounds for the analysis (e.g., lat_min, lat_max, lon_min, lon_max)

STEP 3: Specify Directional Wave Analysis setup

The user should be able to specify the following parameters for the directional wave analysis:
- Number of directions (e.g., 36)
- Window length (e.g., 256)
- Overlap (e.g., 50%)
- Window type
- Frequency range (e.g., 0.05 to 0.5 Hz)

A button should say RUN

STEP 4: Display results:

After the analysis is run, the results should be displayed in a new section. A plot is drawn when a button is clicked. 
- Input: spectra from STEP 3:
- Pre-processing and inputs: select frequency range, direction range.
- Output: processed spectra is converted to paramters, and the spatial plot of Hm0 is plottedd