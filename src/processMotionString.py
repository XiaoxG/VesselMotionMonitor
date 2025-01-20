"""
Ship Motion Processing Module
----------------------------
This module processes ship motion data by transforming accelerations and angles 
between different coordinate systems on a vessel.

Main Functions:
- parse_data_fast: Parses raw motion data string
- transform_acceleration_fast: Transforms acceleration from monitor point to target point
- process_motion_data_fast: Main processing pipeline for motion data

Project: Vessel Intelligence Operation System (VIOS) - 
         Vessel Motion Monitoring System (VMMS)
Developer: Dr. GUO, XIAOXIAN @ SJTU/SKLOE
Contact: xiaoxguo@sjtu.edu.cn
Copyright (c) 2024 Shanghai Jiao Tong University
All rights reserved.
Date: 2025-01-07
"""

# %%
import numpy as np
import configparser
import yaml
from typing import Tuple, List, Union

class ShipMotionProcessor:
    def __init__(self, config_ini_path: str, config_yaml_path: str):
        """
        初始化ShipMotionProcessor
        
        Args:
            config_ini_path: INI配置文件路径
            config_yaml_path: YAML配置文件路径
        """
        self._load_config(config_ini_path, config_yaml_path)
        self._initialize_arrays()
        # 从配置文件读取direction（角度）
        try:
            direction_angle = float(self.config['Monitor_point_acc']['direction'])
        except (KeyError, ValueError):
            direction_angle = 0.0
        # 将角度转换为弧度并创建旋转矩阵
        angle_rad = direction_angle * self.DEG_TO_RAD
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)
        # 绕z轴的旋转矩阵
        self.rotation_matrix = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
    def _load_config(self, config_ini_path: str, config_yaml_path: str) -> None:
        """配置加载逻辑分离到单独的方法"""
        self.config = configparser.ConfigParser()
        self.config.read(config_ini_path)
        
        try:
            with open(config_yaml_path, 'r') as f:
                self.yaml_config = yaml.safe_load(f)
                
            # 尝试从yaml配置获取目标点位置
            try:
                target_x = self.yaml_config['Monitor']['x']
                target_y = self.yaml_config['Monitor']['y']
                target_z = self.yaml_config['Monitor']['z']
            except (KeyError, TypeError):
                # 如果yaml中没有，使用ini中的默认值
                target_x = float(self.config['VesselDefaults']['target_pos_x'])
                target_y = float(self.config['VesselDefaults']['target_pos_y'])
                target_z = float(self.config['VesselDefaults']['target_pos_z'])
                
            # 尝试从yaml获取船舶参数
            try:
                lbp = self.yaml_config['vessel']['LBP']
                vcg = self.yaml_config['vessel']['VCG']
                gm1 = self.yaml_config['vessel']['GM1']
            except (KeyError, TypeError):
                # 如果yaml中没有，使用ini中的默认值
                lbp = float(self.config['VesselDefaults']['LBP'])
                vcg = float(self.config['VesselDefaults']['VCG'])
                gm1 = float(self.config['VesselDefaults']['GM1'])  # 注意：使用GM0作为GM1的默认值
                
        except FileNotFoundError:
            # 如果yaml文件不存在，全部使用ini中的默认值
            target_x = float(self.config['VesselDefaults']['target_pos_x'])
            target_y = float(self.config['VesselDefaults']['target_pos_y'])
            target_z = float(self.config['VesselDefaults']['target_pos_z'])
            lbp = float(self.config['VesselDefaults']['LBP'])
            vcg = float(self.config['VesselDefaults']['VCG'])
            gm1 = float(self.config['VesselDefaults']['GM0'])
        
        # 从配置文件获取测量点位置（单位：米）
        self.monitor_pos = np.array([
            float(self.config['Monitor_point_acc']['Monitor_point_x']),
            float(self.config['Monitor_point_acc']['Monitor_point_y']),
            float(self.config['Monitor_point_acc']['Monitor_point_z'])
        ])
        
        # 设置目标点位置
        self.target_pos = np.array([target_x, target_y, target_z])
        
        # 计算旋转中心
        self.rotation_center = np.array([
            lbp / 2,
            0.0,
            vcg + gm1
        ])
        
        # 预计算相对位置向量
        self.r_mt = (self.target_pos - self.rotation_center) - (self.monitor_pos - self.rotation_center)
        
        # 常量
        self.DEG_TO_RAD = np.float32(np.pi / 180.0)
        self.G_TO_MS2 = np.float32(9.807)

    def _initialize_arrays(self) -> None:
        """初始化数组"""
        self.omega = np.zeros(3, dtype=np.float32)
        self.a_m = np.zeros(3, dtype=np.float32)
        self.a_t = np.zeros(3, dtype=np.float32)
        self._temp_cross = np.zeros(3, dtype=np.float32)
        self._omega_cross_r = np.zeros(3, dtype=np.float32)
        self._parsed_values = np.zeros(8, dtype=np.float32)
        self._motion_data = np.zeros((3, 3), dtype=np.float32)
        self._angles = np.zeros(2, dtype=np.float32)

    @staticmethod
    def parse_data_fast(data_str: str) -> tuple:
        """
        快速解析数据字符串
        输入格式: "$cmd,Timestamp,X,AX,GX,Y,AY,GY,Z,AZ,GZ"
        """
        try:
            # 使用split的maxsplit参数限制分割次数
            parts = data_str.split(',', 11)
            # 直接使用列表推导式，避免fromiter的开销
            values = [float(x) for x in parts[2:10]]
            return (values[0], values[3]), np.array([
                [values[1], values[4], values[7]],
                [values[2], values[5], 0.0],
                [values[0], values[3], 0.0]
            ], dtype=np.float32)
        except:
            return (0.0, 0.0), np.zeros((3, 3), dtype=np.float32)

    def transform_acceleration_fast(self, motion_data: np.ndarray) -> np.ndarray:
        """
        优化的加速度转换函数
        motion_data: shape (3, 3), 包含 [acc, ang_vel, angles]
        """
        # 转换加速度和角速度
        self.a_m = motion_data[0] * self.G_TO_MS2
        self.omega = motion_data[1] * self.DEG_TO_RAD
        
        # 使用旋转矩阵转换加速度和角速度
        self.a_m = self.rotation_matrix @ self.a_m
        self.omega = self.rotation_matrix @ self.omega
        
        # 计算叉乘
        self._temp_cross = np.cross(self.omega, self.r_mt)
        self._omega_cross_r = np.cross(self.omega, self._temp_cross)
        self.a_t = self.a_m + self._omega_cross_r
        
        return self.a_t / self.G_TO_MS2

    def process_motion_data_fast(self, data_str: str) -> List[float]:
        """
        优化的数据处理函数，返回最小必要信息
        返回: (roll, pitch, acc_x, acc_y, acc_z)
        """
        angles, motion_data = self.parse_data_fast(data_str)
        # 转换角度
        rotated_angles = self.rotation_matrix @ np.array([angles[0], angles[1], 0], dtype=np.float32)
        target_acc = self.transform_acceleration_fast(motion_data)
        
        # 将numpy数据类型转换为Python原生float
        return [float(rotated_angles[0]), float(rotated_angles[1]), 
                float(target_acc[0]), float(target_acc[1]), float(target_acc[2])]

# 添加一个便于调用的函数
def create_processor(config_ini_path: str, config_yaml_path: str) -> ShipMotionProcessor:
    """
    创建ShipMotionProcessor实例的工厂函数
    """
    return ShipMotionProcessor(config_ini_path, config_yaml_path)

