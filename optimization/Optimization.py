import numpy as np
import os
import optuna 
from optuna.samplers import TPESampler
from physicool.config import ConfigFileParser
from physicool.optimization import PhysiCellBlackBox
from pathlib import Path
import datetime
import scipy.io
import alphashape
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
import matplotlib




def process_mat_files(directory, maxtime, timestep):
    """
    Processes .mat simulation files located in a directory, extracting position, type, 
    and velocity data for different cell structures over time.

    Args:
        directory (str): Directory containing the .mat simulation files.
        maxtime (float): Maximum simulation time.
        timestep (float): Time interval between simulation frames.

    Returns:
        dict: Dictionary containing matrices of ID, type, positions (x, y, z), 
              velocities, and categorized structures (cell, nucleus, cytoplasm, membrane),
              as well as cell and nucleus diameters in x and y directions.
    """
    # Initialize data containers
    id_data, type_data = [], []
    posx_data, posy_data, posz_data = [], [], []
    velx_data, vely_data, velz_data = [], [], []

    posx_cell_data, posy_cell_data, posz_cell_data = [], [], []
    posx_nuc_data, posy_nuc_data, posz_nuc_data = [], [], []
    posx_cyto_data, posy_cyto_data, posz_cyto_data = [], [], []
    posx_memb_data, posy_memb_data, posz_memb_data = [], [], []

    celldiam_x_list, celldiam_y_list = [], []
    nucdiam_x_list, nucdiam_y_list = [], []

    maxiteration = int(maxtime // timestep)

    for i in range(maxiteration + 1):
        file_name = f"output{i:08d}_cells.mat"
        file_path = os.path.join(directory, file_name)

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file was not found: {file_path}")

        mat_data = scipy.io.loadmat(file_path)

        if 'cells' not in mat_data:
            raise ValueError(f"The variable 'cells' is not present in {file_path}")

        cells = mat_data['cells']

        if cells.shape[0] < 6:
            raise ValueError(f"The format of the 'cells' array is not valid in {file_path}")

        id_data.append(cells[0, :])
        type_data.append(cells[5, :])
        posx_data.append(cells[1, :])
        posy_data.append(cells[2, :])
        posz_data.append(cells[3, :])

        # Initialize temporary lists for this timestep
        px_cell, py_cell, pz_cell = [], [], []
        px_nuc, py_nuc, pz_nuc = [], [], []
        px_cyto, py_cyto, pz_cyto = [], [], []
        px_memb, py_memb, pz_memb = [], [], []

        for j in range(cells.shape[1]):
            type_val = cells[5, j]
            px, py, pz = cells[1, j], cells[2, j], cells[3, j]

            if type_val == 0:  # Cytoplasm
                px_cyto.append(px)
                py_cyto.append(py)
                pz_cyto.append(pz)
            elif type_val == 1:  # Nucleus
                px_nuc.append(px)
                py_nuc.append(py)
                pz_nuc.append(pz)
            elif type_val == 2:  # Membrane
                px_memb.append(px)
                py_memb.append(py)
                pz_memb.append(pz)

            if type_val in [0, 1, 2]:
                px_cell.append(px)
                py_cell.append(py)
                pz_cell.append(pz)

        posx_cell_data.append(px_cell)
        posy_cell_data.append(py_cell)
        posz_cell_data.append(pz_cell)

        posx_nuc_data.append(px_nuc)
        posy_nuc_data.append(py_nuc)
        posz_nuc_data.append(pz_nuc)

        posx_cyto_data.append(px_cyto)
        posy_cyto_data.append(py_cyto)
        posz_cyto_data.append(pz_cyto)

        posx_memb_data.append(px_memb)
        posy_memb_data.append(py_memb)
        posz_memb_data.append(pz_memb)

        if i == 0:
            velx_data.append(np.zeros_like(cells[1, :]))
            vely_data.append(np.zeros_like(cells[2, :]))
            velz_data.append(np.zeros_like(cells[3, :]))
        else:
            vx = (posx_data[i] - posx_data[i - 1]) / timestep
            vy = (posy_data[i] - posy_data[i - 1]) / timestep
            vz = (posz_data[i] - posz_data[i - 1]) / timestep
            velx_data.append(vx)
            vely_data.append(vy)
            velz_data.append(vz)

        # Calculate cell and nucleus diameters for this timestep
        if px_cell:
            celldiam_x_list.append(np.max(px_cell) - np.min(px_cell))
            celldiam_y_list.append(np.max(py_cell) - np.min(py_cell))
        else:
            celldiam_x_list.append(0.0)
            celldiam_y_list.append(0.0)

        if px_nuc:
            nucdiam_x_list.append(np.max(px_nuc) - np.min(px_nuc))
            nucdiam_y_list.append(np.max(py_nuc) - np.min(py_nuc))
        else:
            nucdiam_x_list.append(0.0)
            nucdiam_y_list.append(0.0)

    if len(velx_data) > 1:
        velx_data[0] = velx_data[1]
        vely_data[0] = vely_data[1]
        velz_data[0] = velz_data[1]

    id_matrix       = np.array(id_data).T
    type_matrix     = np.array(type_data).T
    posx_matrix     = np.array(posx_data).T
    posy_matrix     = np.array(posy_data).T
    posz_matrix     = np.array(posz_data).T
    velx_matrix     = np.array(velx_data).T
    vely_matrix     = np.array(vely_data).T
    velz_matrix     = np.array(velz_data).T

    posx_cell_matrix = np.array(posx_cell_data).T
    posy_cell_matrix = np.array(posy_cell_data).T
    posz_cell_matrix = np.array(posz_cell_data).T

    posx_nuc_matrix = np.array(posx_nuc_data).T
    posy_nuc_matrix = np.array(posy_nuc_data).T
    posz_nuc_matrix = np.array(posz_nuc_data).T

    posx_cyto_matrix = np.array(posx_cyto_data).T
    posy_cyto_matrix = np.array(posy_cyto_data).T
    posz_cyto_matrix = np.array(posz_cyto_data).T

    posx_memb_matrix = np.array(posx_memb_data).T
    posy_memb_matrix = np.array(posy_memb_data).T
    posz_memb_matrix = np.array(posz_memb_data).T

    return {
        'id': id_matrix,
        'type': type_matrix,
        'posx': posx_matrix,
        'posy': posy_matrix,
        'posz': posz_matrix,
        'posx_cell': posx_cell_matrix,
        'posy_cell': posy_cell_matrix,
        'posz_cell': posz_cell_matrix,
        'posx_nuc': posx_nuc_matrix,
        'posy_nuc': posy_nuc_matrix,
        'posz_nuc': posz_nuc_matrix,
        'posx_cyto': posx_cyto_matrix,
        'posy_cyto': posy_cyto_matrix,
        'posz_cyto': posz_cyto_matrix,
        'posx_memb': posx_memb_matrix,
        'posy_memb': posy_memb_matrix,
        'posz_memb': posz_memb_matrix,
        'velx': velx_matrix,
        'vely': vely_matrix,
        'velz': velz_matrix,
        'cellDiam_x': np.array(celldiam_x_list),
        'cellDiam_y': np.array(celldiam_y_list),
        'nucDiam_x': np.array(nucdiam_x_list),
        'nucDiam_y': np.array(nucdiam_y_list)
    }

def calibration_vel1(posx_cell, posy_cell, x_lower, x_upper, y0, yf,timestep):
    """
    This function calculates the average velocity of the first particle
    that is positioned with X between x_lower and x_upper, 
    measuring its travel time from Y0 to Yf.
    
    Args:
        posx_cell (np.array): X positions of particles in the simulation.
        posy_cell (np.array): Y positions of particles in the simulation.
        x_lower (float): Lower boundary for the X position of particles to consider.
        x_upper (float): Upper boundary for the X position of particles to consider.
        y0 (float): Starting Y position to measure travel time from.
        yf (float): Ending Y position to measure travel time to.
    
    Returns:
        float: Vsim - The average velocity calculated as (actual_yf - actual_y0)/(end_timestep - start_timestep).
               Returns None if no qualifying particle is found.
    """
    
    num_particles, num_timesteps = posx_cell.shape

    
    # Initialize variables to track the first particle meeting criteria
    start_timestep = None
    end_timestep = None
    tracked_particle_idx = None
    actual_y0 = None
    actual_yf = None
    
    # Find the first cell particle with X position in the specified range that passes through Y0
    for t in range(num_timesteps): # Iterate over timesteps
        if start_timestep is not None: # If we already found a starting timestep, break out of the loop
            print("Particle already found")
            break
            
        for p in range(num_particles): # Iterate over particles
            # Check if this particle's X position is within the specified range
            if (x_lower <= posx_cell[p, t]) and (posx_cell[p, t] <= x_upper):
                # Check if this particle has passed or is at Y0
                if posy_cell[p, t] >= y0:
                    # print(f"Particle {p} at timestep {t} meets criteria: X in [{x_lower}, {x_upper}], Y >= {y0}.")
                    start_timestep = t
                    tracked_particle_idx = p
                    actual_y0 = posy_cell[p, t]
                    break
    
    # If no particle was found that meets the criteria
    if start_timestep is None:
        print("No particle found in the specified X range that passes through Y0.")
        return None
    
    # Track this particle for the remaining timesteps to find when/if it reaches Yf
    max_y_position = actual_y0
    
    for t in range(start_timestep, num_timesteps):
        current_y = posy_cell[tracked_particle_idx, t]
        
        # Update the maximum y position reached
        if current_y > max_y_position:
            max_y_position = current_y
        
        # Check if the particle has reached or passed Yf
        if current_y >= yf:
            end_timestep = t
            actual_yf = current_y
            break
    
    # If the particle never reached Yf, use the furthest position
    if end_timestep is None:
        end_timestep = num_timesteps - 1
        actual_yf = max_y_position
    
    # Calculate travel time (in timesteps)
    travel_time = end_timestep - start_timestep
    
    # Avoid division by zero
    if travel_time == 0:
        print("Travel time is zero, cannot calculate velocity.")
        return None
    
    # Calculate average velocity: distance/time
    vsim = (actual_yf - actual_y0) / (travel_time*timestep)
    
    return vsim


def membrane_rupture_detector(posx, posy, alpha, trial, ruta_calculos, ruta_resultados, backup_path):
    """
    Detects the rupture of the membrane by counting the number of 
    areas enclosing the points, given a certain alpha.
    
    
    Args:
        posx (list or np.array): List or array with the X coordinates of the points.
        posy (list or np.array): List or array with the Y coordinates of the points.
        alpha (float): Value of alpha to generate the Alpha Shape.

    Returns:
        sensor (int): takes 1 if the membrane is broken, 0 otherwise.
    """
    # Convert the input vectors into a list of points (x, y)
    points = np.column_stack((posx, posy))
    
    # Generate the alpha form with the parameter alpha
    alpha_shape = alphashape.alphashape(points, alpha)
    
    # Detect the number of areas enclosed by the alpha shape
    if isinstance(alpha_shape, Polygon):
        num_areas = 1
        shapes = [alpha_shape]  #  Conversion to list for graphing
        sensor = 0
    elif isinstance(alpha_shape, MultiPolygon):
        num_areas = len(alpha_shape.geoms)
        shapes = list(alpha_shape.geoms)
        sensor = 1
    else:
        num_areas = 0
        shapes = []
        sensor = 1

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(posx, posy, color='red', label='Points')

    # Get a colormap
    cmap = matplotlib.colormaps.get_cmap('tab10').resampled(num_areas if num_areas > 0 else 1)


    for i, shape in enumerate(shapes):
        x, y = shape.exterior.xy
        ax.fill(x, y, alpha=0.5, facecolor=cmap(i), edgecolor='black', linewidth=2, label=f'Area {i+1}')

    ax.set_title(f"Alpha Shape with alpha={alpha} (Enclosed Areas: {num_areas})")
    ax.legend()
    


    # Save figure
    fig_name = f"Trial{trial:05d}.png"
    plt.savefig(fig_name, dpi=300)
    plt.close(fig)  # Close to avoid showing in interactive environments


    return num_areas, sensor


def erf_logistic(x, alpha):
    """
    Calculates the logistic-based error for a single input difference using a logistic function.

    The logistic function used is: L(x) = 1 / (1 + exp(-alpha * x)).
    This function returns the squared difference between 1 and L(x) as a measure of error.

    Args:
        x (float): Difference between the expected and known value.
        alpha (float): Parameter that controls the steepness of the logistic function.

    Returns:
        error (float): Logistic-based error value.
    """
    error = 1 / (1 + np.exp(-alpha * x))
    return error

def erf_sigmoid_corrected(x, alpha):
    """
    Calculates the corresponding error function, taking into account that erf(x=0) = 0 and erf(x=xsat) = 1.

    This function applies a sigmoid correction to the input value and returns the result.
    The sigmoid correction is defined as: erf_sig = 1 / (1 + exp(-x)).

    Args:
        x (float): Input value to be corrected.

    Returns:
        erf_sig (float): Corrected error function value.
    """
    error_logistic  = erf_logistic(x, alpha)
    error_0         = erf_logistic(0, alpha)
    erf_sig_cor     = 2* (error_logistic - error_0)  
    return erf_sig_cor

def compute_alpha(xsat):
    """
    Computes the alpha parameter to saturate the logistic function
    for a logistic function so that L(x0) ≈ y_target.

    Args:
        xsat (float): The x-value at which the logistic function should be close to y_target.

    Returns:
        alpha (float): Value of the alpha parameter.
    """
    y_sat = 0.99  # Default target value for saturation
    if not (0 < y_sat <= 1):
        raise ValueError("y_target must be between 0 and 1")
    
    alpha = - (1 / xsat) * np.log((1 / y_sat) - 1)
    return alpha

def physicool_execution(params):
    """
    Executes PhysiCell with the given params.

    Args:
        params (dict): A dictionary with the optimized parameters:
            Adh_Nuc (float): Adhesion strength for nucleus cells.
            Adh_Cyto (float): Adhesion strength for cytoplasm cells.
            Adh_Memb (float): Adhesion strength for membrane cells.
            Rep_Nuc (float): Repulsion strength for nucleus cells.
            Rep_Cyto (float): Repulsion strength for cytoplasm cells.
            Rep_Memb (float): Repulsion strength for membrane cells.
            F_loc (float): Cell's locomotive force.
            Viscosity (float): Internal viscosity of the cell.
    """


    #...physics-based parameters
    adh_nuc     = params['Adh_Nuc']
    adh_cyto    = params['Adh_Cyto']
    adh_memb    = params['Adh_Memb']
    rep_nuc     = params['Rep_Nuc']
    rep_cyto    = params['Rep_Cyto']
    rep_memb    = params['Rep_Memb']
    f_loc       = params['F_loc']
    viscosity   = params['Viscosity']


    # First, we load the seed.
    # Parse the data from the config file
    file_path="config/PhysiCell_settings.xml"
    xml_data = ConfigFileParser(file_path)

    # Read and change out <user_parameters> data
    user_data = xml_data.read_user_params()
    user_data[6].value = f_loc
    user_data[7].value = viscosity
    xml_data.write_user_params(user_data)

    # Read and change the cell parameters for the "Nucleo" cell definition
    cell_data_nucleo = xml_data.read_cell_data("Nucleo")
    cell_data_nucleo.mechanics.cell_cell_adhesion_strength=adh_nuc
    cell_data_nucleo.mechanics.cell_cell_repulsion_strength=rep_nuc
    xml_data.write_mechanics_params(name="Nucleo", mechanics=cell_data_nucleo.mechanics)

    # Read and change the cell parameters for the "Cyto" cell definition
    cell_data_cyto = xml_data.read_cell_data("Cyto")
    cell_data_cyto.mechanics.cell_cell_adhesion_strength=adh_cyto
    cell_data_cyto.mechanics.cell_cell_repulsion_strength=rep_cyto

    xml_data.write_mechanics_params(name="Cyto", mechanics=cell_data_cyto.mechanics)

    # Read and change the cell parameters for the "Membrana" cell definition
    cell_data_membrana = xml_data.read_cell_data("Membrana")
    cell_data_membrana.mechanics.cell_cell_adhesion_strength=adh_memb
    cell_data_membrana.mechanics.cell_cell_repulsion_strength=rep_memb
    xml_data.write_mechanics_params(name="Membrana", mechanics=cell_data_membrana.mechanics)




    current_time = datetime.datetime.now()
    print("Changed variables")
    print(current_time)

    # Execute the simulation
    print("Starting simulation")
    os.system("make")
    my_model = PhysiCellBlackBox(project_name="project")
    my_model.run(number_of_replicates=1, keep_files=True)

    current_time = datetime.datetime.now()
    print("Simulation completed")
    print(current_time)


# Initial values we want to force
initial_values = {"Adh_Nuc": 0.0007474654998173752, "Rep_Nuc": 0.00983395239960124, "Adh_Cyto": 0.0009644339417029876, "Rep_Cyto": 0.0016736529895809982, "Adh_Memb": 0.0005207245769788185, "Rep_Memb": 0.004085228218690753, "F_loc": 0.0015298641536427562, "Viscosity": 0.009819015910208007}

class CustomSampler(TPESampler):
    def __init__(self, initial_values, **kwargs):
        super().__init__(multivariate=True, **kwargs)  # We activate TPE multivariate
        self.initial_values = initial_values
        self.first_trial_done = False

    def sample_relative(self, study, trial, search_space):
        """For the first trial, we return all the initial values as a dictionary.
           Then we use the normal multivariate distribution.
        """
        
        # First trial -> return initial values as a set
        if not self.first_trial_done:
            params = {}
            for p in search_space.keys():
                if p in self.initial_values:
                    params[p] = self.initial_values[p]
            return params  # This function must return a dict[param -> value]

        # Next trials -> normal multivariate TPE
        return super().sample_relative(study, trial, search_space)

    def sample_independent(self, study, trial, param_name, param_distribution):
        """Only used if sample_relative does not cover some parameter."""
        if not self.first_trial_done and param_name in self.initial_values:
            return self.initial_values[param_name]
        
        return super().sample_independent(study, trial, param_name, param_distribution)

    def after_trial(self, study, trial, state, values):
        """Mark that the first trial has been completed."""
        if not self.first_trial_done:
            self.first_trial_done = True
        return super().after_trial(study, trial, state, values)

# We create a custom sampler based on TPE
sampler = CustomSampler(initial_values)

def objective(trial):
    Adh_Cyto    = trial.suggest_float('Adh_Cyto',   1e-6, 1e-2)
    Adh_Memb    = trial.suggest_float('Adh_Memb',   1e-6, 1e-2)
    Adh_Nuc     = trial.suggest_float('Adh_Nuc' ,   1e-6, 1e-2)
    Rep_Cyto    = trial.suggest_float('Rep_Cyto',   1e-6, 1e-1)
    Rep_Memb    = trial.suggest_float('Rep_Memb',   1e-6, 1e-1)
    Rep_Nuc     = trial.suggest_float('Rep_Nuc' ,   1e-6, 1e-1)
    F_loc       = trial.suggest_float('F_loc',      1e-4, 1e-1)
    Viscosity   = trial.suggest_float('Viscosity',  1e-4, 1e-1)

    if any(p < 0 for p in [Adh_Cyto, Adh_Memb, Adh_Nuc, Rep_Cyto, Rep_Memb, Rep_Nuc, F_loc, Viscosity]):
        raise optuna.TrialPruned()

    params = {'Adh_Nuc': Adh_Nuc,
              'Adh_Cyto': Adh_Cyto,
              'Adh_Memb': Adh_Memb,
              'Rep_Nuc': Rep_Nuc,
              'Rep_Cyto': Rep_Cyto,
              'Rep_Memb': Rep_Memb,  
              'F_loc': F_loc,
              'Viscosity': Viscosity, 
    }


    Diam_cell_res = np.array([
        [20.0988, 16.1788, 15.6336, 12.1500],
        [21.9822, 22.3909, 15.8229, 13.2562]
    ])
    Diam_nuc_res = np.array([
        [11.7628, 10.3318, 9.3685, 8.0992],
        [11.3844, 11.8409, 9.6095, 8.1208]
    ])
    vel_res = np.array([
        [10.0423, 21.7071, 23.3510, 21.8782],
        [16.2839, 16.9286, 16.8110, 14.6977]
    ])
    xsup_res = np.array([
        [1, 2, 3, 4],
        [1, 2, 3, 4]
    ])
    xinf_res = np.array([
        [-1, -2, -3, -4],
        [-1, -2, -3, -4]
    ])

    type_cell = 1 # 0=CAR-T; 1=T-CELL
    size_duct = 2 # 2, 4, 6, 8

    if (type_cell == 0):
        c = 0
    elif (type_cell == 1):
        c = 1


    if (size_duct == 2):
        t = 0
    elif (size_duct == 4):
        t = 1
    elif (size_duct == 6):
        t = 2
    elif (size_duct == 8):
        t = 3

    if trial.number == 0:
        if c==0:
            print(f"Simulating CAR-T in microchannel of {size_duct} um")
        elif c==1:
            print(f"Simulating T-CELL in microchannel of {size_duct} um")   
    
    print(f"*************************************** EVALUATING TRIAL #{trial.number} ***************************************")

    # We define the expected values for the simulation
    Diam_cell_obj = Diam_cell_res[c, t]
    Diam_nuc_obj = Diam_nuc_res[c, t]
    vel_exp = vel_res[c, t]
    xsup = xsup_res[c, t]
    xinf = xinf_res[c, t]
    burst_limit = 1000
    y0 = 8
    yf = 68

    physicool_execution(params)

    directory = "temp"
    maxtime = 3
    timestep = 0.01

    result = process_mat_files(directory, maxtime, timestep)

    # We extract the data from the result dictionary
    posx_nuc = result['posx_nuc']
    posy_nuc = result['posy_nuc']
    posx_cyto = result['posx_cyto']
    posy_cyto = result['posy_cyto']
    posx_memb = result['posx_memb']
    posy_memb = result['posy_memb']
    posx_cell = result['posx_cell']
    posy_cell = result['posy_cell']
    Diam_nuc_exp = result['nucDiam_y']
    Diam_cell_exp = result['cellDiam_y']

    Diam_cell_exp_final = Diam_cell_exp[-1]
    Diam_nuc_exp_final = Diam_nuc_exp[-1]

    # We adjust the positions to prevent particles from escaping the microchannel

    mask_nuc_rows       = (posx_nuc[:, -1]  >= xinf) & (posx_nuc[:, -1]  <= xsup)
    mask_cyto_rows      = (posx_cyto[:, -1] >= xinf) & (posx_cyto[:, -1] <= xsup)
    mask_memb_rows      = (posx_memb[:, -1] >= xinf) & (posx_memb[:, -1] <= xsup)
    mask_cell_rows      = (posx_cell[:, -1] >= xinf) & (posx_cell[:, -1] <= xsup)

    posx_memb_corrected = posx_memb[mask_memb_rows, :]
    posy_memb_corrected = posy_memb[mask_memb_rows, :]
    posx_cell_corrected = posx_cell[mask_cell_rows, :]

    # Detection of membrane rupture errors; we only use the last column (last time step).
    posx_memb_last = posx_memb_corrected[:, -1]
    posy_memb_last = posy_memb_corrected[:, -1] 
    alpha = 10
    ruta_calculos = Path(f'/home/user/PhysiCell/')
    ruta_resultados = Path(f'/home/user/Resultados/Imgs')
    ruta_respaldo_resultados = Path(f'/home/user/Resultados/Imgs')



    Vsim= calibration_vel1(posx_cell, posy_cell,xinf, xsup, y0, yf, timestep)

    if Vsim is None:
        print("No particle found that meets the criteria.")
        loss = 1
        return loss


    num_areas,broken_memb_sensor = membrane_rupture_detector(posx_memb_last, posy_memb_last, alpha, trial.number, ruta_calculos, ruta_resultados, ruta_respaldo_resultados) #broken_memb_sensor (int): takes 1 if the membrane is broken, 0 otherwise.
    print("Number of areas: " + str(num_areas))
   
   # We check to see if the membrane is broken, and if it is, we end the trial with an infinite error.
    if broken_memb_sensor == 1:
        print("The membrane is broken")
    else:  
        print("The membrane is not broken")

    if broken_memb_sensor == 1:
        loss = 1
        return loss


    # We calculate the errors of the simulation
    vel_dif = abs(Vsim - vel_exp)
    alpha_vel = compute_alpha(10)
    vel_error = erf_sigmoid_corrected(vel_dif, alpha_vel)
    print("Vsim: " + str(Vsim) + " Vel_exp: " + str(vel_exp) + " difference: " + str(abs(Vsim - vel_exp)))

    nuc_dif = abs(Diam_nuc_obj-Diam_nuc_exp_final)
    alpha_nuc = compute_alpha(8)
    nucdiam_error = erf_sigmoid_corrected(nuc_dif, alpha_nuc)
    print("Nucleus diameter: " + str(Diam_nuc_exp_final) + " Target nucleus diameter: " + str(Diam_nuc_obj) + " difference: " + str(abs(Diam_nuc_exp_final - Diam_nuc_obj)))
    

    cell_dif = abs(Diam_cell_obj-Diam_cell_exp_final)
    alpha_cell = compute_alpha(9)
    celldiam_error = erf_sigmoid_corrected(cell_dif, alpha_cell)
    print("Cell diameter: " + str(Diam_cell_exp_final) + " Target cell diameter: " + str(Diam_cell_obj) + " difference: " + str(abs(Diam_cell_exp_final - Diam_cell_obj)))

    # Burst detection, takes exponential error of the number of particles out of the microchannel or inf if the limit is exceeded.
    bursted_particles = posx_cell.shape[0] - posx_cell_corrected.shape[0]
    if bursted_particles>burst_limit:
        loss = 1
        return loss
    print("Number of particles: " + str(posx_cell.shape[0]) + " Corrected particles: " + str(posx_cell_corrected.shape[0]))
    alpha_burst = compute_alpha(burst_limit)
    burst_error = erf_sigmoid_corrected(bursted_particles, alpha_burst) 
    print("Bursted particles: " + str(bursted_particles) + " Burst Error: " + str(burst_error))
    

    print("NucDiam Error:  " + str(nucdiam_error))
    print("CellDiam Error: " + str(celldiam_error))
    print("Burst Error:    " + str(burst_error))
    print("Vel Error:      " + str(vel_error))

    loss = (nucdiam_error + celldiam_error + burst_error + vel_error)/4
    print(f"TOTAL ERROR:    " + str(loss))       

    return loss


# Callback to stop the study if Err<Err_max
def max_err_callback(study, trial):
    Err_max = 0.05 # We define the max error 
    if study.best_value < Err_max:
        print(f"Stopping: an error lower than Err_max was reached: ({study.best_value:.5f} < {Err_max})")
        study.stop()

study_name = "FullError_T2"
db_filename = f"{study_name}.sqlite3"
storage = f"sqlite:///{db_filename}"


study = optuna.create_study(study_name = study_name, storage=storage, direction="minimize",sampler=sampler)
study.optimize(objective, n_trials=500, callbacks=[max_err_callback])

