"""
Vessel Motion Monitoring System (VMMS) - Daily Vessel Configuration Processor
Part of Vessel Intelligence Operation System (VIOS)

This module handles daily vessel configuration processing and hydrodynamic calculations.

Key Components:
- HydroConfig: Configuration dataclass for hydrodynamic calculations
  - period_min/max: Wave period range
  - period_num: Number of periods to calculate
  - wave_dir_seg: Wave direction segment size
  - sea_rho: Sea water density
  - mesh_path: Path to vessel mesh file

- monitor_performance: Decorator for performance monitoring
  - Tracks execution time and memory usage
  - Logs performance metrics

Main Functions:
- clip_draft(): Adjusts mesh based on vessel draft
- rao(): Calculates Response Amplitude Operator
- configure_vessel(): Fetches vessel configuration
- save_vessel_config(): Saves configuration to YAML

Usage Example:
    config = HydroConfig(
        period_min=3.0,
        period_max=30.0,
        period_num=28,
        wave_dir_seg=0.25,
        sea_rho=1025.0,
        mesh_path='meshes/vessel.dat'
    )
    processor = VesselConfigProcessor(config)
    processor.process_daily_config()

Project: Vessel Intelligence Operation System (VIOS)
         Vessel Motion Monitoring System (VMMS)
Developer: Dr. GUO, XIAOXIAN @ SJTU/SKLOE
Contact: xiaoxguo@sjtu.edu.cn
Date: 2025-01-07

Copyright (c) 2024 Shanghai Jiao Tong University
All rights reserved.
"""

# %%
import os
import logging

from logger_config import setup_logger
from pymongo import errors as mongo_errors
from datetime import date
import configparser
import traceback

import capytaine as cpt
import numpy as np
import xarray as xr
from helper_tools import clip_draft, rao, save_vessel_config, configure_vessel
import meshmagick.mesh
from meshmagick.mmio import write_mesh
import pickle
import os

import glob
from typing import Tuple, Optional, Any, List
from pathlib import Path

import yaml
from dataclasses import dataclass

import psutil
from time import perf_counter
from functools import wraps
from multiprocessing import Process, freeze_support

@dataclass
class HydroConfig:
    """Hydro calculation configuration parameters"""
    period_min: float
    period_max: float
    period_num: int
    wave_dir_seg: float
    sea_rho: float
    mesh_path: str


def monitor_performance(func):
    """性能监控装饰器"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = perf_counter()
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        try:
            result = func(self, *args, **kwargs)    
            end_time = perf_counter()
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024
            
            self.logger.info(
                f"Performance metrics for {func.__name__}:\n"
                f"  - Execution time: {end_time - start_time:.2f} seconds\n"
                f"  - Memory usage: {memory_after - memory_before:.2f} MB"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
            
    return wrapper

class VesselCalculator:
    """Handles vessel calculations and dataset management"""
    
    def __init__(self, config: configparser.ConfigParser, logger: logging.Logger, base_dir: str):
        self.config = config
        self.logger = logger
        self.today = date.today()
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, 'data')
        self.damping_matrix = np.diag(np.array([0.0, 0.0, 0.0, 0.07, 0.0, 0.0]))
        
        # 添加路径属性
        self.paths = {
            'log': os.path.join(base_dir, 'log', 'getDailyVesselconfig.log'),
            'config': os.path.join(base_dir, 'config', 'config.ini'),
            'yaml': os.path.join(base_dir, 'data', 'hydrostatic', 'vessel_config_fromMongoDB.yaml')
        }
        
        # 确保必要目录存在
        self._ensure_directories()

    def _ensure_directories(self):
        """确保所有必要的目录结构存在"""
        required_dirs = [
            os.path.join(self.base_dir, 'log'),
            os.path.join(self.data_dir, 'mesh'),
            os.path.join(self.data_dir, 'hydro'),
            os.path.join(self.data_dir, 'hydrostatic'),
            os.path.join(self.base_dir, 'config')
        ]
        
        for directory in required_dirs:
            os.makedirs(directory, exist_ok=True)

    def get_hydro_config(self) -> HydroConfig:
        """获取水动力计算配置参数，包含默认值"""
        try:
            # 设置默认mesh文件路径
            default_mesh_path = os.path.join(self.base_dir, 'data', 'mesh', 'WAMIT_HULL_62K_BASE.GDF')
            
            # 从配置文件获取mesh_dir
            mesh_path = self.config.get('Hydro', 'mesh_dir', fallback=None)
            
            if mesh_path:
                # 如果配置文件指定了路径，确保它是绝对路径
                if not os.path.isabs(mesh_path):
                    mesh_path = os.path.join(self.base_dir, mesh_path)
            else:
                # 如果未指定，使用默认路径
                mesh_path = default_mesh_path
                
            # 验证mesh文件存在性
            if not os.path.exists(mesh_path):
                self.logger.warning(f"Mesh file not found at: {mesh_path}")
                
                # 定义搜索路径列表
                search_paths = [
                    os.path.join(self.base_dir, 'data', 'mesh'),
                    os.path.join(self.base_dir, 'mesh'),
                    self.base_dir
                ]
                
                # 搜索mesh文件
                for search_dir in search_paths:
                    potential_path = os.path.join(search_dir, 'WAMIT_HULL_62K_BASE.GDF')
                    if os.path.exists(potential_path):
                        mesh_path = potential_path
                        self.logger.info(f"Found mesh file at: {mesh_path}")
                        break
                else:
                    raise FileNotFoundError("Mesh file not found in any expected location")

            # 构建并返回配置
            return HydroConfig(
                period_min=self.config.getfloat('Hydro', 'period_min', fallback=2.0),
                period_max=self.config.getfloat('Hydro', 'period_max', fallback=50.0),
                period_num=self.config.getint('Hydro', 'period_num', fallback=20),
                wave_dir_seg=self.config.getfloat('Hydro', 'wave_dir_seg', fallback=0.25),
                sea_rho=self.config.getfloat('Hydro', 'sea_rho', fallback=1025.0),
                mesh_path=mesh_path
            )
            
        except Exception as e:
            self.logger.error(f"Error in hydro configuration: {str(e)}")
            raise

    def process_vessel_config(self):
        """处理船舶配置的主要流程"""
        try:
            self.logger.info("Starting vessel configuration processing")
            cpt.set_logging('INFO')
            
            # 获取配置
            if not self.config.has_section('Hydro'):
                self.logger.warning("Missing 'Hydro' section in config.ini. Using default values.")
                self.config.add_section('Hydro')
            
            hydro_params = self.get_hydro_config()  # 使用类方法而不是独立函数
            COG, vessel_inertia_unit, vessel_config = configure_vessel(self.config, self.logger)
            
            save_vessel_config(vessel_config, self.logger, self.base_dir)
            
            # 检查数据集和变化
            self._process_calculations(vessel_config, hydro_params, COG, vessel_inertia_unit)
            
            self.logger.info("Process completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in vessel configuration processing: {str(e)}")
            raise

    def _process_calculations(self, vessel_config, hydro_params, COG, vessel_inertia_unit):
        """处理计算逻辑"""
        datasets_exist, _ = self.check_datasets_exist()
        
        if not datasets_exist:
            self.logger.info("Required datasets not found. Performing full calculation.")
            self.perform_full_calculation(vessel_config, hydro_params, COG, vessel_inertia_unit)
            return
            
        # 检查变化
        drafts_changed = self.check_draft_changes(
            {
                'Draft_aft': vessel_config['Draft_aft'],
                'Draft_forward': vessel_config['Draft_forward']
            },
            self.paths['yaml']
        )
        
        params_changed = self.check_parameter_changes(vessel_config, self.paths['yaml'])
        
        if drafts_changed:
            self.logger.info("Significant draft changes detected. Performing full recalculation.")
            self.perform_full_calculation(vessel_config, hydro_params, COG, vessel_inertia_unit)
        elif params_changed:
            self.logger.info("Only vessel parameters changed. Attempting to update existing calculations.")
            self.update_existing_calculation(vessel_config, hydro_params, COG, vessel_inertia_unit)
        else:
            self.logger.info("No significant changes detected. Skipping calculations.")

    def find_latest_rao_dataset(self) -> Optional[str]:
        """Find the most recent RAO dataset file."""
        try:
            # Get list of all RAO dataset files
            rao_files = glob.glob(os.path.join(self.data_dir, 'hydro', 'rao_dataset_*.pkl'))
            
            if not rao_files:
                self.logger.info("No existing RAO dataset files found")
                return None
                
            # Sort files by modification time, most recent first
            latest_file = max(rao_files, key=os.path.getmtime)
            self.logger.info(f"Found latest RAO dataset: {latest_file}")
            
            return latest_file
            
        except Exception as e:
            self.logger.error(f"Error finding latest RAO dataset: {str(e)}")
            return None

    def check_datasets_exist(self) -> Tuple[bool, Optional[str]]:
        """Check if required datasets exist and find the latest RAO dataset."""
        try:
            hydro_dataset_path = os.path.join(self.data_dir, 'hydro', 'hydro_dataset.pkl')
            hydro_exists = os.path.exists(hydro_dataset_path)
            
            # Find latest RAO dataset
            latest_rao = self.find_latest_rao_dataset()
            
            datasets_exist = hydro_exists and latest_rao is not None
            if datasets_exist:
                self.logger.info(f"Found required datasets - Hydro: {hydro_dataset_path}, RAO: {latest_rao}")
            else:
                missing = []
                if not hydro_exists:
                    missing.append("hydro_dataset")
                if latest_rao is None:
                    missing.append("rao_dataset")
                self.logger.warning(f"Missing datasets: {', '.join(missing)}")
                
            return datasets_exist, latest_rao
            
        except Exception as e:
            self.logger.error(f"Error checking datasets: {str(e)}")
            return False, None

    def save_dataset(self, body, dataset, rao_dataset):
        """Save RAO and hydro datasets"""
        try:
            # Use dictionary to simplify data and path mapping
            data_map = {
                'rao': {
                    'path': os.path.join(self.data_dir, 'hydro', f'rao_dataset_{self.today.strftime("%Y%m%d")}.pkl'),
                    'data': rao_dataset
                },
                'hydro': {
                    'path': os.path.join(self.data_dir, 'hydro', 'hydro_dataset.pkl'),
                    'data': (body, dataset)
                }
            }
            
            for dataset_type, info in data_map.items():
                self.logger.info(f"Saving {dataset_type} dataset to {info['path']}")
                with open(info['path'], 'wb') as f:
                    pickle.dump(info['data'], f)
                    
        except Exception as e:
            self.logger.error(f"Error saving datasets: {str(e)}")
            raise

    def save_mesh(self, body):
        """Save mesh to GDF and MAR files"""
        try:
            date_str = self.today.strftime("%Y%m%d")
            mesh_base_path = os.path.join(self.data_dir, 'mesh', f'Mesh_{date_str}')
            mm_mesh = meshmagick.mesh.Mesh(body.mesh.vertices, body.mesh.faces, name=body.mesh.name)
            
            for ext in ['gdf', 'mar']:
                mesh_path = f"{mesh_base_path}.{ext}"
                self.logger.info(f"Saved {ext.upper()} mesh file to {mesh_path} [test only no real write]")
                # write_mesh(mesh_path, mm_mesh.vertices, mm_mesh.faces, ext)
                # self.logger.info(f"Saved {ext.upper()} mesh file to {mesh_path}")
                
        except Exception as e:
            self.logger.error(f"Error saving mesh: {str(e)}", exc_info=True)
            raise

    def solve_bem(self, body, period, wave_dir, vessel_spd, sea_rho, dof=['Surge', 'Sway', 'Heave', 'Roll', 'Pitch', 'Yaw']):
        """Solve Boundary Element Method equations."""
        try:
            vessel_spd_mps = vessel_spd * 0.5144
            
            # 简化 test_matrix 创建逻辑
            base_coords = {
                'period': period,
                'wave_direction': wave_dir,
                'water_depth': [np.inf],
                'rho': sea_rho,
                'forward_speed': vessel_spd_mps
            }
            
            if dof == 'all':
                base_coords['radiating_dof'] = list(body.dofs)
            elif isinstance(dof, list):
                if not all(d in ['Surge', 'Sway', 'Heave', 'Roll', 'Pitch', 'Yaw'] for d in dof):
                    raise ValueError(f"Invalid dof values in list: {dof}")
                base_coords['radiating_dof'] = dof
            elif dof in ['Surge', 'Sway', 'Heave', 'Roll', 'Pitch', 'Yaw']:
                base_coords['radiating_dof'] = [dof]
            else:
                raise ValueError(f"Invalid dof value: {dof}")

            test_matrix = xr.Dataset(coords=base_coords)
            results = cpt.BEMSolver().fill_dataset(test_matrix, body, n_jobs=2)
            return results
            
        except Exception as e:
            self.logger.error(f"Error in BEM solver: {str(e)}", exc_info=True)
            raise

    def compute_rao(self, dataset, body):
        """
        Compute Response Amplitude Operators.
        
        Args:
            dataset: Input dataset
            body: Floating body object
            
        Returns:
            Dataset: RAO dataset
        """
        try:
            self.logger.info("Computing RAO with damping matrix")
            
            add_damping = body.add_dofs_labels_to_matrix(self.damping_matrix)
            rao_dataset = rao(dataset, damping=add_damping)
            return rao_dataset
            
        except Exception as e:
            self.logger.error("Error computing RAO: %s", str(e))
            self.logger.debug(traceback.format_exc())
            raise

    def load_and_prepare_mesh(self, dft_aft: float, dft_fwd: float, lpp: float, mesh_path: str, period_min: float = 2) -> Tuple[Any, Any]:
        """
        Load and prepare the vessel mesh with draft clipping.
        
        Args:
            dft_aft: Aft draft
            dft_fwd: Forward draft
            lpp: Length between perpendiculars
            mesh_path: Path to mesh file
            period_min: Minimum period. Defaults to 2
            
        Returns:
            tuple: (mesh, lid_mesh)
        """
        try:
            if not os.path.exists(mesh_path):
                raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
                
            self.logger.info("Loading mesh file: %s", mesh_path)
            mesh = cpt.load_mesh(mesh_path, file_format='gdf')
            
            self.logger.info("Clipping mesh with drafts - aft: %.2f, forward: %.2f", dft_aft, dft_fwd)
            mesh = clip_draft(mesh, dft_aft, dft_fwd, LPP=lpp)
            
            self.logger.info("Generating lid mesh")
            lid_mesh = mesh.generate_lid(z=mesh.lowest_lid_position(omega_max=2 * np.pi / period_min))
            
            return mesh, lid_mesh
            
        except Exception as e:
            self.logger.error("Error in mesh processing: %s", str(e))
            self.logger.debug(traceback.format_exc())
            raise

    def create_floating_body(self, mesh, lid_mesh, COG, vessel_inertia_unit, sea_rho):
        """
        Create a floating body object with specified parameters and save hydrostatics.
        
        Args:
            mesh: Vessel mesh
            lid_mesh: Lid mesh
            COG: Center of gravity coordinates
            vessel_inertia_unit: Vessel inertia matrix
            sea_rho: Sea water density
            
        Returns:
            FloatingBody: Configured floating body object
        """
        try:
            self.logger.info("Creating floating body with COG: %s", COG)
            body = cpt.FloatingBody(
                mesh=mesh,
                lid_mesh=lid_mesh,
                dofs=cpt.rigid_body_dofs(rotation_center=COG),
                center_of_mass=COG
            ).immersed_part()
            
            self.logger.info("Computing vessel inertia and hydrostatic stiffness")
            body.mass = body.disp_mass(rho=sea_rho)
            vessel_inertia = body.disp_mass(rho=sea_rho) * vessel_inertia_unit
            body.inertia_matrix = body.add_dofs_labels_to_matrix(vessel_inertia)
            body.hydrostatic_stiffness = body.compute_hydrostatic_stiffness(rho=sea_rho)
            
            # Compute hydrostatics
            self.logger.info("Computing hydrostatics")
            hydrostatics = body.compute_hydrostatics(rho=sea_rho)

            # Save hydrostatics to file
            self._save_hydrostatics_report(hydrostatics)
            
            return body
            
        except Exception as e:
            self.logger.error("Error creating floating body: %s", str(e))
            self.logger.debug(traceback.format_exc())
            raise

    def _save_hydrostatics_report(self, hydrostatics):
        """Optimize hydrostatics report saving logic"""
        try:
            hydrostatics_dir = os.path.join(self.data_dir, 'hydrostatic')
            hydrostatics_files = [f for f in glob.glob(os.path.join(hydrostatics_dir, 'hydrostatics_*.txt'))]
            
            total_size = sum(Path(f).stat().st_size for f in hydrostatics_files)
            
            FILE_SIZE_THRESHOLD = 300 * 1024 * 1024  # 300MB
            if total_size > FILE_SIZE_THRESHOLD:
                self._cleanup_old_files(hydrostatics_files)

            filename = os.path.join(hydrostatics_dir, f'hydrostatics_{self.today.strftime("%Y%m%d")}.txt')
            self._write_hydrostatics_report(filename, hydrostatics)
            
        except Exception as e:
            self.logger.error(f"Failed to save hydrostatics report: {str(e)}")
            raise

    def _cleanup_old_files(self, files: List[str], keep_count: int = 10) -> None:
        """Helper method to clean up old files"""
        try:
            sorted_files = sorted(files, key=os.path.getctime)
            for f in sorted_files[:-keep_count]:
                try:
                    os.remove(f)
                    self.logger.info(f"Removed old file: {f}")
                except OSError as e:
                    self.logger.warning(f"Failed to delete file {f}: {e}")
        except Exception as e:
            self.logger.error(f"Error during old files cleanup: {e}")

    def update_vessel_calculations(self, body, dataset, COG, vessel_inertia_unit, sea_rho):
        """Update vessel calculations with new parameters"""
        try:
            self.logger.info("Updating vessel calculations with new parameters")
            
            body.center_of_mass = COG
            self.logger.info(f"Updated center of mass to {COG}")
            
            vessel_inertia = body.disp_mass(rho=sea_rho) * vessel_inertia_unit
            body.inertia_matrix = body.add_dofs_labels_to_matrix(vessel_inertia)
            self.logger.info("Updated inertia matrix")
            
            body.hydrostatic_stiffness = body.compute_hydrostatic_stiffness(rho=sea_rho)
            self.logger.info("Updated hydrostatic stiffness")
            
            add_damping = body.add_dofs_labels_to_matrix(self.damping_matrix)
            rao_dataset = rao(dataset, damping=add_damping)
            self.logger.info("Computed new RAO")
            
            return body, dataset, rao_dataset
            
        except Exception as e:
            self.logger.error(f"Error updating vessel calculations: {str(e)}")
            raise

    def check_parameter_changes(self, current_params: dict, yaml_path: str) -> bool:
        """Check if vessel parameters have changed significantly"""
        try:
            if not os.path.exists(yaml_path):
                self.logger.info(f"No previous configuration file found at {yaml_path}")
                return True
                
            with open(yaml_path, 'r', encoding='utf-8') as f:
                previous_config = yaml.safe_load(f)
                
            params_to_check = {
                'LCG': previous_config['vessel']['LCG'],
                'TCG': previous_config['vessel']['TCG'],
                'VCG': previous_config['vessel']['VCG'],
                'RXX': previous_config['vessel']['RXX'],
                'RYY': previous_config['vessel']['RYY'],
                'RZZ': previous_config['vessel']['RZZ']
            }
            
            current_values = {
                'LCG': current_params['LCG'],
                'TCG': current_params['TCG'],
                'VCG': current_params['VCG'],
                'RXX': current_params['Rxx'],
                'RYY': current_params['Ryy'],
                'RZZ': current_params['Rzz']
            }
            
            changes = {}
            significant_change = False
            
            for param, prev_value in params_to_check.items():
                curr_value = current_values[param]
                if prev_value != 0:
                    change_percent = abs(curr_value - prev_value) / prev_value * 100
                    changes[param] = change_percent
                    if change_percent > 5:
                        significant_change = True
                elif curr_value != 0:
                    significant_change = True
                    changes[param] = float('inf')
                else:
                    changes[param] = 0
            
            if significant_change:
                self.logger.info("Significant parameter changes detected:")
                for param, change in changes.items():
                    self.logger.info(f"{param}: {change:.2f}% change (Previous: {params_to_check[param]:.2f}, "
                                  f"Current: {current_values[param]:.2f})")
            else:
                self.logger.info("No significant parameter changes detected")
                
            return significant_change
            
        except Exception as e:
            self.logger.error(f"Error checking parameter changes: {str(e)}")
            return True

    def perform_full_calculation(self, vessel_config, hydro_params, COG, vessel_inertia_unit):
        """Execute complete vessel calculation process"""
        try:
            with self._calculation_context(vessel_config, hydro_params, COG, vessel_inertia_unit) as calc_results:
                self._save_calculation_results(*calc_results, vessel_config)
        except Exception as e:
            self.logger.error(f"Error during full calculation process: {e}")
            raise

    def _calculation_context(self, vessel_config, hydro_params, COG, vessel_inertia_unit):
        """创建计算上下文管理器"""
        class CalculationContext:
            def __init__(self, outer):
                self.outer = outer
                
            def __enter__(self):
                # 处理网格
                mesh, lid_mesh = self.outer.load_and_prepare_mesh(
                    vessel_config['Draft_aft'],
                    vessel_config['Draft_forward'],
                    vessel_config['LBP'],
                    hydro_params.mesh_path
                )
                
                # 创建浮体
                body = self.outer.create_floating_body(
                    mesh, lid_mesh, COG, vessel_inertia_unit, hydro_params.sea_rho
                )
                
                # 保存网格文件
                self.outer.save_mesh(body)
                
                # 准备计算参数
                period = np.linspace(hydro_params.period_min, hydro_params.period_max, hydro_params.period_num)
                wave_dir = np.arange(0, 1.25, hydro_params.wave_dir_seg) * np.pi
                
                # 求解 BEM
                dataset = self.outer.solve_bem(body, period, wave_dir, 0.0, hydro_params.sea_rho)
                
                # 计算 RAO
                rao_dataset = self.outer.compute_rao(dataset, body)
                
                return body, dataset, rao_dataset
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
                
        return CalculationContext(self)

    def update_existing_calculation(self, vessel_config, hydro_params, COG, vessel_inertia_unit):
        """Update existing calculations with new parameters"""
        try:
            hydro_dataset_path = os.path.join(self.data_dir, 'hydro', 'hydro_dataset.pkl')
            
            # Load existing datasets
            self.logger.info("Loading existing datasets")
            with open(hydro_dataset_path, 'rb') as f:
                body, dataset = pickle.load(f)
            
            # Update calculations
            body, dataset, rao_dataset = self.update_vessel_calculations(
                body, dataset, COG, vessel_inertia_unit, hydro_params.sea_rho
            )
            
            # Save updated results
            self.save_dataset(body, dataset, rao_dataset)
            
        except Exception as e:
            self.logger.error(f"Error updating calculations: {str(e)}")
            self.logger.info("Falling back to full recalculation")
            self.perform_full_calculation(vessel_config, hydro_params, COG, vessel_inertia_unit)

    def check_draft_changes(self, current_drafts: dict, yaml_path: str) -> bool:
        """
        Check if draft values have changed significantly (>10%) compared to saved yaml file.
        
        Args:
            current_drafts (dict): Current draft values
            yaml_path (str): Path to yaml file
            
        Returns:
            bool: True if drafts changed significantly, False otherwise
        """
        try:
            if not os.path.exists(yaml_path):
                self.logger.info(f"No previous configuration file found at {yaml_path}")
                return True
                
            with open(yaml_path, 'r', encoding='utf-8') as f:
                previous_config = yaml.safe_load(f)
                
            previous_drafts = {
                'Draft_aft': previous_config['vessel']['Draft_aft'],
                'Draft_forward': previous_config['vessel']['Draft_forward']
            }
            
            changes = {
                key: abs(current_drafts[key] - previous_drafts[key]) / previous_drafts[key] * 100
                for key in ['Draft_aft', 'Draft_forward']
            }
            
            significant_change = any(change > 10 for change in changes.values())
            
            if significant_change:
                self.logger.info("Significant draft changes detected:")
                for key, change in changes.items():
                    self.logger.info(f"{key}: {change:.2f}% change (Previous: {previous_drafts[key]:.2f}, Current: {current_drafts[key]:.2f})")
            else:
                self.logger.info("No significant draft changes detected:")
                for key, change in changes.items():
                    self.logger.info(f"{key}: {change:.2f}% change (Previous: {previous_drafts[key]:.2f}, Current: {current_drafts[key]:.2f})")
                    
            return significant_change
            
        except Exception as e:
            self.logger.error(f"Error checking draft changes: {str(e)}")
            return True  # On error, proceed with execution to be safe

    def _write_hydrostatics_report(self, filename: str, hydrostatics: dict) -> None:
        """
        Write hydrostatics report to a file
        
        Args:
            filename (str): Path to output file
            hydrostatics (dict): Hydrostatics data to write
        """
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            with open(filename, 'w') as f:
                f.write("Vessel Hydrostatics Report\n")
                f.write("=========================\n\n")
                f.write(f"Date: {self.today.strftime('%Y-%m-%d')}\n\n")
                
                for key, value in hydrostatics.items():
                    if isinstance(value, (np.ndarray, list)):
                        f.write(f"{key}:\n")
                        if isinstance(value, np.ndarray) and value.ndim > 1:
                            # Handle 2D arrays (matrices)
                            for row in value:
                                f.write(f"  {row}\n")
                        else:
                            # Handle 1D arrays or lists
                            f.write(f"  {value}\n")
                    else:
                        # Handle scalar values
                        f.write(f"{key}: {value}\n")
                        
                
            self.logger.info(f"Hydrostatics report saved to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error writing hydrostatics report: {str(e)}")
            raise

    def _save_calculation_results(self, body, dataset, rao_dataset, vessel_config):
        """
        保存计算结果
        
        Args:
            body: 浮体对象
            dataset: 计算数据集
            rao_dataset: RAO数据集
            vessel_config: 船舶配置
        """
        try:
            # 保存数据集
            self.save_dataset(body, dataset, rao_dataset)
            
            # 保存网格
            self.save_mesh(body)
            
            # 计算并保存水动力静力学报告
            # hydrostatics = body.compute_hydrostatics(rho=self.config.getfloat('Hydro', 'sea_rho', fallback=1025.0))
            # self._save_hydrostatics_report(hydrostatics)
            
            # 保存船舶配置
            save_vessel_config(vessel_config, self.logger, self.base_dir)
            
            self.logger.info("Successfully saved all calculation results")
            
        except Exception as e:
            self.logger.error(f"Error saving calculation results: {str(e)}")
            raise


def run_with_timeout(timeout=3600):  # 1 hour = 3600 seconds
    """
    Run main function with timeout control
    
    Args:
        timeout (int): Timeout duration in seconds
    """
    process = Process(target=main)
    process.start()
    process.join(timeout)
    
    if process.is_alive():
        process.terminate()
        process.join()
        # Ensure process is terminated
        if process.is_alive():
            process.kill()
        return False
    return True

def main():
    """简化后的主函数"""
    logger = None
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 设置日志
        log_path = os.path.join(base_dir, 'log', 'getDailyVesselconfig.log')
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        logger = setup_logger('getDailyVesselconfig', log_path, 
                            level=logging.INFO, max_bytes=10*1024*1024, backup_count=5)
        
        # 加载配置
        config = configparser.ConfigParser()
        config.read(os.path.join(base_dir, 'config', 'config.ini'))
        
        # 初始化并运行计算器
        calculator = VesselCalculator(config, logger, base_dir)
        calculator.process_vessel_config()
        
    except Exception as e:
        if logger:
            logger.error(f"Error in main execution: {str(e)}")
            logger.debug(traceback.format_exc())
    finally:
        if logger:
            logger.info("Process finished")
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)



# %%
if __name__ == "__main__":
    freeze_support()  # Windows multiprocessing support
    logger = setup_logger('getDailyVesselconfig', 
                         os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                    'log', 'getDailyVesselconfig.log'),
                         level=logging.INFO)
    
    try:
        if not run_with_timeout(3600):  # 1 hour timeout
            logger.error("Program execution timed out (1 hour), forcefully terminated")
        
    except Exception as e:
        logger.error(f"Runtime error occurred: {str(e)}")
        logger.debug(traceback.format_exc())
    finally:
        if logger:
            logger.info("Process finished")
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
