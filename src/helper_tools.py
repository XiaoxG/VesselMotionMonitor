"""
Helper Tools for Vessel Motion Monitoring System (VMMS)

This module provides utility functions for vessel motion calculations and configuration management
as part of the Vessel Intelligence Operation System (VIOS).

Main Functions:
- clip_draft(): Adjusts mesh based on vessel draft parameters
- rao_transfer_function(): Computes complex-valued RAO transfer matrix
- rao(): Calculates Response Amplitude Operator
- validate_config(): Validates vessel configuration with default values
- save_vessel_config(): Saves vessel configuration to YAML file
- configure_vessel(): Fetches and configures vessel parameters from MongoDB

Classes:
- VesselConfigError: Custom exception for configuration errors

Project: Vessel Intelligence Operation System (VIOS) - 
         Vessel Motion Monitoring System (VMMS)
Developer: Dr. GUO, XIAOXIAN @ SJTU/SKLOE
Contact: xiaoxguo@sjtu.edu.cn
Date: 2025-01-07

Copyright (c) 2024 Shanghai Jiao Tong University
All rights reserved.
"""

import numpy as np
from capytaine import Axis
import xarray as xr

import logging

from pymongo import MongoClient, errors as mongo_errors
from contextlib import closing
import configparser
import traceback

import numpy as np
import xarray as xr
import json

import yaml

import os


def clip_draft(mesh, dft_aft, dft_fwd, LPP = 226.9):
    rot_angle = np.arctan((dft_fwd - dft_aft) / LPP)
    rot_axis = Axis(vector=[0, 1, 0], point=[LPP/2, 0, 0])
    mesh = mesh.rotated(rot_axis, angle=-rot_angle)
    tomovez = 19.0 - (dft_fwd + dft_aft)/2
    mesh = mesh.translated([0, 0, tomovez])
    return mesh


def rao_transfer_function(dataset, dissipation=None, stiffness=None, damping=None):
    """Complex-valued matrix used for the computation of the RAO.

    Parameters
    ----------
    dataset: xarray Dataset
        The hydrodynamical dataset.
        This function supposes that variables named 'inertia_matrix' and 'hydrostatic_stiffness' are in the dataset.
        Other variables can be computed by Capytaine, by those two should be manually added to the dataset.
    dissipation: array, optional
        An optional dissipation matrix (e.g. Power Take Off) to be included in the transfer function.
        Default: none.
    stiffness: array, optional
        An optional stiffness matrix (e.g. mooring stiffness) to be included in the transfer function.
        Default: none.

    Returns
    -------
    xarray DataArray
        The matrix as an array depending of omega and the degrees of freedom.
    """

    if not hasattr(dataset, 'inertia_matrix'):
        raise AttributeError('Computing the impedance matrix requires an `inertia_matrix` matrix to be defined in the hydrodynamical dataset')

    if not hasattr(dataset, 'hydrostatic_stiffness'):
        raise AttributeError('Computing the impedance matrix requires an `hydrostatic_stiffness` matrix to be defined in the hydrodynamical dataset')

    if 'encounter_omega' in dataset.coords:
        omega = dataset.coords['encounter_omega']
    else:
        omega = dataset.coords['omega']

    # ASSEMBLE MATRICES
    if damping is not None:
        add_damping = 2 * damping * ((dataset['inertia_matrix']+ dataset['added_mass'])*dataset['hydrostatic_stiffness'])**0.5
        H = (-omega**2*(dataset['inertia_matrix'] + dataset['added_mass'])
         - 1j*omega*(dataset['radiation_damping']+ add_damping.fillna(0))
         + dataset['hydrostatic_stiffness'])
    else:
        H = (-omega**2*(dataset['inertia_matrix'] + dataset['added_mass'])
         - 1j*omega*dataset['radiation_damping']
         + dataset['hydrostatic_stiffness'])

    if dissipation is not None:
        H = H - 1j*omega*dissipation

    if stiffness is not None:
        H = H + stiffness

    return H


def rao(dataset, wave_direction=None, dissipation=None, stiffness=None, damping=None):
    """Response Amplitude Operator.

    Parameters
    ----------
    dataset: xarray Dataset
        The hydrodynamical dataset.
        This function supposes that variables named 'inertia_matrix' and 'hydrostatic_stiffness' are in the dataset.
        Other variables can be computed by Capytaine, by those two should be manually added to the dataset.
    wave_direction: float, optional
        Select a wave directions for the computation. (Not recommended, kept for legacy.)
        Default: all wave directions in the dataset.
    dissipation: array, optional
        An optional dissipation matrix (e.g. Power Take Off) to be included in the RAO.
        Default: none.
    stiffness: array, optional
        An optional stiffness matrix (e.g. mooring stiffness) to be included in the RAO.
        Default: none.

    Returns
    -------
    xarray DataArray
        The RAO as an array depending of omega and the degree of freedom.
    """

    # ASSEMBLE MATRICES
    H = rao_transfer_function(dataset, dissipation, stiffness, damping)
    fex = dataset.excitation_force

    # SOLVE LINEAR SYSTEMS
    # Match dimensions of the arrays to be sure to solve the right systems.
    H, fex = xr.broadcast(H, fex, exclude=["radiating_dof", "influenced_dof"])
    H = H.transpose(..., 'radiating_dof', 'influenced_dof')
    fex = fex.transpose(...,  'influenced_dof')

    if wave_direction is not None:  # Legacy behavior for backward compatibility
        H = H.sel(wave_direction=wave_direction)
        fex = fex.sel(wave_direction=wave_direction)

    # Solve and add coordinates
    rao_dims = [d for d in H.dims if d != 'influenced_dof']
    rao_coords = {c: H.coords[c] for c in H.coords if c != 'influenced_dof'}
    rao = xr.DataArray(np.linalg.solve(H.values, fex.values[..., np.newaxis])[..., 0], coords=rao_coords, dims=rao_dims)

    return rao


def validate_config(vessel_config, config_parser):
    """
    Validate the vessel configuration and set default values for missing fields from config.ini.
    
    Args:
        vessel_config (dict): Vessel configuration dictionary
        config_parser (ConfigParser): Configuration parser object containing default values
        
    Returns:
        dict: Validated vessel configuration with default values if needed
    """
    logger = logging.getLogger('getDailyVesselconfig')
    
    # Get default values from config file
    try:
        defaults = {
            'LCG': config_parser.getfloat('VesselDefaults', 'LCG'),
            'TCG': config_parser.getfloat('VesselDefaults', 'TCG'),
            'VCG': config_parser.getfloat('VesselDefaults', 'VCG'),
            'Draft_aft': config_parser.getfloat('VesselDefaults', 'Draft_aft'),
            'Draft_forward': config_parser.getfloat('VesselDefaults', 'Draft_forward'),
            'Rxx': config_parser.getfloat('VesselDefaults', 'Rxx'),
            'Ryy': config_parser.getfloat('VesselDefaults', 'Ryy'),
            'Rzz': config_parser.getfloat('VesselDefaults', 'Rzz'),
            'LBP': config_parser.getfloat('VesselDefaults', 'LBP'),
            'Speed': config_parser.getfloat('VesselDefaults', 'Speed'),
            'warning_roll': config_parser.getfloat('VesselDefaults', 'warning_roll'),
            'warning_pitch': config_parser.getfloat('VesselDefaults', 'warning_pitch'),
            'warning_acc_x': config_parser.getfloat('VesselDefaults', 'warning_acc_x'),
            'warning_acc_y': config_parser.getfloat('VesselDefaults', 'warning_acc_y'),
            'warning_acc_z': config_parser.getfloat('VesselDefaults', 'warning_acc_z')
        }
    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        logger.error(f"Error reading default values from config file: {str(e)}")
        # Fallback defaults if config file is missing values
        defaults = {
            'LCG': 105.78, 'TCG': 0.0, 'VCG': 11.6,
            'Draft_aft': 12.411, 'Draft_forward': 11.758,
            'Rxx': 10.02, 'Ryy': 57.36, 'Rzz': 58.0,
            'LBP': 198.3, 'Speed': 13.5,
            'warning_roll': 10.25, 'warning_pitch': 5.00,
            'warning_acc_x': 0.25, 'warning_acc_y': 0.25,
            'warning_acc_z': 0.25
        }
        logger.warning("Using hardcoded default values due to missing config entries")
    
    # Check for missing fields and set defaults
    missing_fields = []
    for field, default_value in defaults.items():
        if field not in vessel_config:
            missing_fields.append(field)
            vessel_config[field] = default_value
            logger.warning(f"Missing {field} in vessel configuration. Using default value: {default_value}")
    
    if missing_fields:
        logger.warning(f"Using default values for missing fields: {', '.join(missing_fields)}")
    
    # Validate numerical values
    for field, value in vessel_config.items():
        if field in defaults:
            try:
                vessel_config[field] = float(value)
            except (ValueError, TypeError):
                default_value = defaults[field]
                logger.warning(f"Invalid value for {field}: {value}. Using default value: {default_value}")
                vessel_config[field] = default_value
    
    # Additional validation for specific fields
    if vessel_config['Draft_aft'] <= 0 or vessel_config['Draft_forward'] <= 0:
        logger.warning("Draft values must be positive. Setting to default values.")
        vessel_config['Draft_aft'] = defaults['Draft_aft']
        vessel_config['Draft_forward'] = defaults['Draft_forward']
    
    if vessel_config['LBP'] <= 0:
        logger.warning("LBP must be positive. Setting to default value.")
        vessel_config['LBP'] = defaults['LBP']
    
    # Ensure gyration radius ratios are between 0 and 1
    for radius in ['Rxx', 'Ryy', 'Rzz']:
        if not 0 < vessel_config[radius] < 200:
            logger.warning(f"{radius} must be between 0 and 200. Setting to default value.")
            vessel_config[radius] = defaults[radius]
    
    # Validate warning thresholds are positive
    for warning_param in ['warning_roll', 'warning_pitch', 'warning_acc_x', 'warning_acc_y', 'warning_acc_z']:
        if warning_param in vessel_config and vessel_config[warning_param] <= 0:
            logger.warning(f"{warning_param} must be positive. Setting to default value.")
            vessel_config[warning_param] = defaults[warning_param]
    
    if vessel_config['Speed'] <= 0:
        logger.warning("Speed must be positive. Setting to default value.")
        vessel_config['Speed'] = defaults['Speed']
    
    return vessel_config

class VesselConfigError(Exception):
    """Custom exception for vessel configuration related errors."""
    pass


def save_vessel_config(vessel_config: dict, logger: logging.Logger, base_dir) -> None:
    """
    Save validated vessel configuration to a YAML file in the local directory.
    
    Args:
        vessel_config (dict): Validated vessel configuration dictionary
        logger (logging.Logger): Logger instance
    """
    try:

        # Transform vessel config to match config.yaml format
        config_data = {
            'vessel': {
                'M.V': vessel_config.get('M.V', 'Unknown'),
                'LOA': vessel_config.get('LOA', 201.8),
                'LBP': vessel_config.get('LBP', 198.3),
                'MLD_Breadth': float(vessel_config.get('MLD_Breadth', 33.26)),
                'Draft_forward': vessel_config['Draft_forward'],
                'Draft_aft': vessel_config['Draft_aft'],
                'Displacement': vessel_config.get('Displacement', 47867),
                'GM0': vessel_config.get('GM0', 2.5),
                'GM1': vessel_config.get('GM1', 2.32),
                'VCG': vessel_config['VCG'],
                'LCG': vessel_config['LCG'],
                'TCG': vessel_config['TCG'],
                'RXX': vessel_config['Rxx'],
                'RYY': vessel_config['Ryy'],
                'RZZ': vessel_config['Rzz'],
                'Speed': vessel_config['Speed'],
            },
            'Voyage': {
                'No': vessel_config.get('Voyage', 'Unknown')
            },
            'Monitor': {
                'x': vessel_config['Monitor_x'],
                'y': vessel_config['Monitor_y'],
                'z': vessel_config['Monitor_z']
            },
            'Warning': {
                'Roll': vessel_config['warning_roll'],
                'Pitch': vessel_config['warning_pitch'],
                'acc_x': vessel_config['warning_acc_x'],
                'acc_y': vessel_config['warning_acc_y'],
                'acc_z': vessel_config['warning_acc_z']
            },
            'Mission': {
                'Start': vessel_config.get('Time_Start', '00:00'),
                'End': vessel_config.get('Time_End', '23:59'),
                'Name': vessel_config.get('Name', 'Unknown'),
                'Report_address': vessel_config.get('Report_address', 'Unknown'),
            }
            
        }
        # Generate filename with timestamp
        yaml_dir = os.path.join(base_dir, 'data', 'hydrostatic')
        filename = os.path.join(yaml_dir, 'vessel_config_fromMongoDB.yaml')
        
        # Save to YAML file
        with open(filename, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            
        logger.info(f"Vessel configuration saved to {filename}")
        
    except Exception as e:
        logger.error(f"Error saving vessel configuration to YAML: {str(e)}")
        raise

def configure_vessel(config, logger):
    """
    Fetch and configure vessel parameters from MongoDB.
    """
    try:
        connection_string = config['DBinfo']['connection_string']
        db_name = config['DBinfo']['db_config']
        collection_name = config['DBinfo']['collection_config']

        logger.info("Attempting to connect to MongoDB: %s/%s", db_name, collection_name)

        with closing(MongoClient(connection_string, serverSelectionTimeoutMS=5000)) as client:
            client.server_info()
            db = client[db_name]
            collection = db[collection_name]
            
            vessel_doc = collection.find_one({'status': 1})
            if not vessel_doc:
                logger.warning("No vessel configuration found with status=1")
                vessel_config = {}
            else:
                vessel_config = json.loads(vessel_doc['config'])['vessel']

                # 将关键参数转换为float类型
                float_params = [
                    'LCG', 'TCG', 'VCG', 'Draft_aft', 'Draft_forward', 'LBP',
                    'Rxx', 'Ryy', 'Rzz', 'MLD_Breadth', 'Displacement', 'LOA'
                ]
                
                for param in float_params:
                    if param in vessel_config:
                        vessel_config[param] = float(vessel_config[param])

                monitor_x = vessel_config['LCG']
                monitor_y = vessel_config['TCG']
                monitor_z = vessel_config['VCG']

                vessel_config['Voyage'] = json.loads(vessel_doc['config']).get('Voyage', {}).get('No', 'Unknown')
                
                vessel_config['Monitor_x'] = json.loads(vessel_doc['config']).get('Monitor', {}).get('x', {})
                vessel_config['Monitor_y'] = json.loads(vessel_doc['config']).get('Monitor', {}).get('y', {})
                vessel_config['Monitor_z'] = json.loads(vessel_doc['config']).get('Monitor', {}).get('z', {})

                # Check and replace Monitor_x if necessary
                if vessel_config['Monitor_x'] == {} or vessel_config['Monitor_x'] == 0.0:
                    vessel_config['Monitor_x'] = monitor_x

                # Check and replace Monitor_y if necessary
                if vessel_config['Monitor_y'] == {} or vessel_config['Monitor_y'] == 0.0:
                    vessel_config['Monitor_y'] = monitor_y

                # Check and replace Monitor_z if necessary
                if vessel_config['Monitor_z'] == {} or vessel_config['Monitor_z'] == 0.0:
                    vessel_config['Monitor_z'] = monitor_z
                
                # Get warning parameters from MongoDB config, validate_config will handle defaults if not found
                warning_config = json.loads(vessel_doc['config']).get('Warning', {})
                vessel_config['warning_roll'] = warning_config.get('Roll')
                vessel_config['warning_pitch'] = warning_config.get('Pitch')
                vessel_config['warning_acc_x'] = warning_config.get('acc_x')
                vessel_config['warning_acc_y'] = warning_config.get('acc_y')
                vessel_config['warning_acc_z'] = warning_config.get('acc_z')
            
            # Validate configuration with defaults from config file
            vessel_config = validate_config(vessel_config, config)
            
            logger.info("Successfully configured vessel parameters")
            
            # Extract configuration values
            COG = np.array((
                vessel_config['LCG'],
                vessel_config['TCG'],
                vessel_config['VCG'] - vessel_config['Draft_aft']
            ))
            
            vessel_inertia_unit = np.diag([
                1, 1, 1,
                1 * vessel_config['Rxx']**2,
                1 * vessel_config['Ryy']**2,
                1 * vessel_config['Rzz']**2
            ])

            return (
                COG,
                vessel_inertia_unit,
                vessel_config  # Return the full vessel config for later use
            )

    except mongo_errors.PyMongoError as e:
        logger.error("MongoDB connection error: %s", str(e))
        raise
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in vessel configuration: %s", str(e))
        raise VesselConfigError("Invalid JSON format in vessel configuration")
    except KeyError as e:
        logger.error("Missing configuration key: %s", str(e))
        raise VesselConfigError(f"Missing configuration key: {str(e)}")
    except Exception as e:
        logger.error(f"Error in configure_vessel: {str(e)}")
        logger.debug(traceback.format_exc())
        raise


