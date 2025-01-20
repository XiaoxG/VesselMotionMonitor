"""
Vessel Motion Monitoring System (VMMS) - Hourly Statistics Test Module
Part of Vessel Intelligence Operation System (VIOS)

This module provides test cases for hourly vessel motion statistics functionality.

Key Components:
- TestHourlyStatisMain: Test class for statistics validation
  - setUp(): Initializes test environment and data
  - create_test_config(): Creates test configuration files
  - load_test_data(): Loads test data from CSV files
  - test_hourly_statistics(): Tests hourly statistics calculation
  - plot_statistics(): Visualizes statistical results

Main Functions:
- setUp: Prepares test environment with mock MongoDB and config
- load_test_data: Loads sample vessel motion data
- plot_statistics: Creates visualization of statistical results

Usage Example:
    test = TestHourlyStatisMain()
    test.setUp()  # Initialize test environment
    test.test_hourly_statistics()  # Run statistics test
    test.plot_statistics('x')  # Plot results for x-axis motion

Project: Vessel Intelligence Operation System (VIOS)
         Vessel Motion Monitoring System (VMMS)
Developer: SJTU
Contact: xiaoxguo@sjtu.edu.cn
Date: 2025-01-07
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import mongomock
import tempfile
import shutil
import os
import sys
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')
sys.path.append(src_path)

from getHourlyStatis import main

class TestHourlyStatisMain(unittest.TestCase):
    def setUp(self):
        """测试环境设置"""
        # 创建临时配置文件
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = os.path.join(self.temp_dir, 'config')
        self.log_dir = os.path.join(self.temp_dir, 'log')
        os.makedirs(self.config_dir)
        os.makedirs(self.log_dir)
        
        # 创建配置文件
        self.config_path = os.path.join(self.config_dir, 'config.ini')
        self.create_test_config()
        
        # 设置模拟的MongoDB客户端
        self.mongo_client = mongomock.MongoClient()
        # self.db = self.mongo_client['ship_monitor_2024_10']
        
        # 从tests/data/目录读取测试数据
        self.load_test_data()

    def create_test_config(self):
        """创建测试用的配置文件"""
        # 复制config.ini和config.yaml到测试目录
        src_config_dir = os.path.join(project_root, 'tests', 'config')
        ini_src = os.path.join(src_config_dir, 'config.ini')
        yaml_src = os.path.join(src_config_dir, 'config.yaml')
        
        ini_dst = os.path.join(self.config_dir, 'config.ini')
        yaml_dst = os.path.join(self.config_dir, 'config.yaml')
        
        # 复制配置文件
        shutil.copy2(ini_src, ini_dst)
        shutil.copy2(yaml_src, yaml_dst)

    def load_test_data(self):
        """从CSV文件加载测试数据并存入模拟的MongoDB，每个CSV对应一个collection"""
        data_dir = os.path.join(current_dir, 'data')
        
        # 读取所有CSV文件
        for file in os.listdir(data_dir):
            if file.endswith('.csv'):
                # 从文件名解析日期（假设文件名格式为 YYYY-MM-DD.csv）
                date_str = file.replace('.csv', '')
                # 提取年份和月份用于数据库名称
                year_month = '_'.join(date_str.split('-')[:2])  # 获取 YYYY_MM
                db_name = f'ship_monitor_{year_month}'
                
                print(f"Loading test data for {date_str} into {db_name}")  # 添加日志
                
                # 根据年月获取对应的数据库
                self.db = self.mongo_client[db_name]
                
                # collection名称保持不变
                collection_name = f'ship_monitor_{date_str.replace("-", "_")}'
                collection = self.db[collection_name]
                
                # 读取CSV文件
                file_path = os.path.join(data_dir, file)
                df = pd.read_csv(file_path)
                
                # 确保时间列格式正确
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                    df['updated_at'] = df['time']
                else:
                    raise ValueError(f"CSV文件 {file} 缺少 'time' 列")
                
                # 将数据插入到对应的collection
                records = df.to_dict('records')
                collection.insert_many(records)
                
                print(f"Loaded {len(records)} records for {date_str}")  # 添加日志

    @patch('getHourlyStatis.MongoClient')
    def test_main_function(self, mock_mongo):
        """测试指定时间范围内每个小时的数据处理"""
        # 设置MongoDB模拟
        mock_mongo.return_value = self.mongo_client
        
        # 设置测试的时间范围
        start_time = datetime(2024, 10, 30, 5, tzinfo=timezone.utc)
        end_time = start_time + timedelta(hours=24)  # 96小时 = 4天
        current_time = start_time
        
        # 遍历每个小时
        while current_time <= end_time:
            # 获取当前时间对应的数据库和集合
            year_month = current_time.strftime('%Y_%m')
            date_str = current_time.strftime('%Y_%m_%d')
            db_name = f'ship_monitor_{year_month}'
            collection_name = f'ship_monitor_{date_str}'
            
            current_db = self.mongo_client[db_name]
            current_collection = current_db[collection_name]
            
            # 修改环境变量以使用测试配置
            with patch.dict('os.environ', {
                'TEST_CONFIG_PATH': self.config_path,
                'TEST_TIME': current_time.isoformat()
            }):
                # 运行main函数
                main()
                
                # 验证当前小时的结果
                stats_collection = current_db[f'hourly_stats_{year_month}']
                hour_results = list(stats_collection.find({
                    'timestamp': {
                        '$gte': current_time - timedelta(hours=1),
                        '$lt': current_time
                    }
                }))
                
                # 打印处理结果
                print(f"\nProcessing time: {current_time}")
                print(f"Total processed records: {current_collection.count_documents({})}")
                print(f"Current hour statistics count: {len(hour_results)}")
                if hour_results:
                    print("Statistics example:")
                    print(hour_results[0])
                
                # 基本断言
                self.assertTrue(len(hour_results) > 0, f"Time {current_time} should generate statistical results")
            
            # 移动到下一个小时
            current_time += timedelta(hours=1)
        
        self.plot_statistics(['x', 'y'], start_time, end_time)
        
        # 测试完成后再清理数据
        for month in ['10', '11']:
            db_name = f'ship_monitor_2024_{month}'
            self.mongo_client[db_name][f'hourly_stats_2024_{month}'].delete_many({})
        
        # 测试完成后绘制统计图表
        
        # 在测试结束后检查日志文件中的错误
        log_files = [f for f in os.listdir(self.log_dir) if f.endswith('.log')]
        
        for log_file in log_files:
            log_path = os.path.join(self.log_dir, log_file)
            with open(log_path, 'r') as f:
                log_content = f.read()
                
            # 检查是否存在ERROR日志
            if 'ERROR' in log_content:
                error_lines = [line for line in log_content.split('\n') if 'ERROR' in line]
                error_message = f"发现错误日志:\n" + "\n".join(error_lines)
                self.fail(error_message)

    def plot_statistics(self, variables, start_time, end_time):
        """
        为指定变量绘制统计图表
        Args:
            variables: 需要绘制的变量列表 ['x', 'y']
        """

        for var in variables:
            # 准备数据存储
            timestamps = []
            means = []
            mins = []
            maxs = []
            stds = []
            
            # 收集数据
            current_time = start_time
            while current_time <= end_time:
                year_month = current_time.strftime('%Y_%m')
                db_name = f'ship_monitor_{year_month}'
                stats_collection = self.mongo_client[db_name][f'hourly_stats_{year_month}']
                
                # 查询该小时的统计数据
                hour_results = list(stats_collection.find({
                    'timestamp': {
                        '$gte': current_time - timedelta(hours=1),
                        '$lt': current_time
                    }
                }))
                
                if hour_results:
                    for result in hour_results:
                        timestamps.append(current_time)
                        means.append(result.get(f'{var}_mean', 0))
                        mins.append(result.get(f'{var}_min', 0))
                        maxs.append(result.get(f'{var}_max', 0))
                        stds.append(result.get(f'{var}_std', 0))
                
                current_time += timedelta(hours=1)
            
            # 创建图表
            plt.figure(figsize=(12, 6))
            
            # 绘制各个统计指标
            plt.plot(timestamps, means, label='Mean', marker='o', markersize=4)
            plt.plot(timestamps, mins, label='Min', marker='s', markersize=4)
            plt.plot(timestamps, maxs, label='Max', marker='^', markersize=4)
            plt.plot(timestamps, stds, label='Std', marker='d', markersize=4)
            
            # 设置图表格式
            plt.title(f'Hourly Statistics of {var.upper()} Values')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            # 设置x轴时间格式
            plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))
            plt.xticks(rotation=45)
            
            # 自动调整布局
            plt.tight_layout()
            
            # 保存图表
            output_dir = os.path.join(current_dir, 'output')
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f'hourly_statistics_{var}.png'), dpi=300)
            plt.close()

    def tearDown(self):
        """清理测试环境"""
        # 删除临时目录
        shutil.rmtree(self.temp_dir)
        # 清理模拟的MongoDB
        self.mongo_client.drop_database('test_db_2023_12')

if __name__ == '__main__':
    unittest.main()