"""
Vessel Motion Monitoring System (VMMS) - Hourly Forecast Test Module
Part of Vessel Intelligence Operation System (VIOS)

This module provides test cases for hourly vessel motion forecasting functionality.

Key Components:
- TestHourlyForecast: Test class for forecast validation
  - test_forecast_process(): Tests end-to-end forecast workflow
  - plot_forecast_results(): Visualizes forecast vs actual results
  - run_statis_with_timeout(): Runs statistics with timeout protection

Main Functions:
- timeout_handler: Decorator for function timeout control
- run_statis_with_timeout: Wrapper for statistics calculation with timeout

Usage Example:
    test = TestHourlyForecast()
    test.test_forecast_process()  # Run full forecast test
    test.plot_forecast_results(actual, forecast, 'x', timestamp)  # Plot results

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
import logging
import gc
from functools import wraps
import signal


# 获取项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')
sys.path.append(src_path)

# 导入被测试的模块
from getHourlyStatis import main as statis_main
from getHourlyForcast import main as forecast_main

def timeout_handler(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutError(f"Function execution exceeded {seconds} seconds")
            
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wrapper
    return decorator

# 在statis_main调用处添加超时控制
@timeout_handler(300)  # 5分钟超时
def run_statis_with_timeout():
    statis_main()

class TestHourlyForecast(unittest.TestCase):
    def setUp(self):
        """Test environment setup"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = os.path.join(self.temp_dir, 'config')
        self.log_dir = os.path.join(self.temp_dir, 'log')
        
        # 确保所有必要的目录都存在
        for directory in [self.config_dir, self.log_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # 创建配置文件
        self.config_path = os.path.join(self.config_dir, 'config.ini')
        self.create_test_config()
        
        # 设置模拟的MongoDB客户端
        self.mongo_client = mongomock.MongoClient()
        
        # 加载测试数据
        self.load_test_data()
        
        # 清理并创建输出目录
        self.output_dir = os.path.join(current_dir, 'output', 'forecast_plots')
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def create_test_config(self):
        """Create test configuration file"""
        src_config_dir = os.path.join(project_root, 'tests', 'config')
        ini_src = os.path.join(src_config_dir, 'config.ini')
        yaml_src = os.path.join(src_config_dir, 'config.yaml')
        
        ini_dst = os.path.join(self.config_dir, 'config.ini')
        yaml_dst = os.path.join(self.config_dir, 'config.yaml')
        
        shutil.copy2(ini_src, ini_dst)
        shutil.copy2(yaml_src, yaml_dst)

    def load_test_data(self):
        """Load test data"""
        data_dir = os.path.join(current_dir, 'data')
        
        for file in os.listdir(data_dir):
            if file.endswith('.csv'):
                date_str = file.replace('.csv', '')
                year_month = '_'.join(date_str.split('-')[:2])
                db_name = f'ship_monitor_{year_month}'
                
                self.db = self.mongo_client[db_name]
                date_str = date_str.replace('-', '_')
                collection_name = f'ship_monitor_{date_str}'
                collection = self.db[collection_name]
                
                file_path = os.path.join(data_dir, file)
                df = pd.read_csv(file_path)
                
                # 确保数据包含必要的列
                required_columns = ['time', 'x', 'ax', 'y', 'ay', 'az']
                for col in required_columns:
                    if col not in df.columns:
                        df[col] = np.random.normal(0, 1, len(df))
                
                df['time'] = pd.to_datetime(df['time'])
                df['updated_at'] = df['time']
                
                records = df.to_dict('records')
                collection.insert_many(records)
                
                collection.create_index('time')
                collection.create_index('updated_at')

    def plot_forecast_results(self, actual_data, forecast_data, base_column, timestamp):
        """Plot comparison of forecast results
        
        Args:
            actual_data (dict): Dictionary containing actual data
            forecast_data (dict): Dictionary containing forecast data 
            base_column (str): Base column name (x, y, ax, ay, or az)
            timestamp (str): Timestamp
        """
        fig, ax = plt.subplots(figsize=(20, 8))
        
        metrics = ['mean', 'max', 'min']
        colors = {'mean': 'blue', 'max': 'red', 'min': 'green'}
        
        for metric in metrics:
            column = f'{base_column}_{metric}'
            
            # Plot actual values
            ax.plot(actual_data[column].index, actual_data[column].values,
                   label=f'Actual {metric}', color=colors[metric])
            
            # Plot predicted values  
            ax.plot(forecast_data[column].index, forecast_data[column].values,
                   label=f'Predicted {metric}', color=colors[metric], linestyle='--')
        
        ax.set_title(f'{base_column} Forecast Comparison')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value') 
        ax.legend()
        ax.grid(True)
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save the plot
        output_dir = os.path.join(current_dir, 'output', 'forecast_plots')
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f'forecast_{base_column}_{timestamp}.png')
        
        # Add error handling
        try:
            plt.savefig(plot_path)
            print(f"Successfully saved forecast plot to: {plot_path}")
        except Exception as e:
            print(f"Error saving plot: {str(e)}")
        finally:
            plt.close('all')

    @patch('getHourlyStatis.MongoClient')
    @patch('getHourlyForcast.MongoClient')
    def test_forecast_process(self, mock_forecast_mongo, mock_statis_mongo):
        """Test forecast process"""
        mock_statis_mongo.return_value = self.mongo_client
        mock_forecast_mongo.return_value = self.mongo_client
        
        # Add logger
        test_logger = logging.getLogger('test_forecast')
        test_logger.setLevel(logging.ERROR)
        has_error = False
        
        start_time = datetime(2024, 10, 30, 5, tzinfo=timezone.utc)
        end_time = start_time + timedelta(hours=12)
        
        # First step: Generate statistics for all time points
        current_time = start_time
        statis_end_time = end_time + timedelta(hours=24)
        while current_time <= statis_end_time:
            with patch.dict('os.environ', {
                'TEST_CONFIG_PATH': self.config_path,
                'TEST_TIME': current_time.isoformat()
            }):
                mock_logger = MagicMock()
                mock_logger.level = logging.INFO
                
                # 捕获ERROR级别的日志
                error_messages = []
                def mock_error(msg, *args, **kwargs):
                    error_messages.append(msg)
                mock_logger.error = mock_error
                
                with patch('getHourlyStatis.setup_logger', return_value=mock_logger):
                    try:
                        run_statis_with_timeout()
                        if error_messages:
                            has_error = True
                            test_logger.error(f"Error occurred at time point {current_time}: {error_messages}")
                        
                        total_hours = (statis_end_time - start_time).total_seconds() / 3600
                        elapsed_hours = (current_time - start_time).total_seconds() / 3600
                        progress = elapsed_hours / total_hours * 100
                        bar_length = 50
                        filled_length = int(bar_length * progress / 100)
                        bar = '=' * filled_length + '-' * (bar_length - filled_length)
                        print(f'\rProgress: [{bar}] {progress:.1f}% {current_time.strftime("%Y-%m-%d %H:%M:%S")}', end='')
                        
                    except Exception as e:
                        has_error = True
                        test_logger.error(f"Processing failed at time point {current_time}: {str(e)}")
                        
            current_time += timedelta(hours=1)

        # 第二步的预测部分也需要类似的错误处理
        current_time = start_time
        while current_time <= end_time:
            with patch.dict('os.environ', {
                'TEST_CONFIG_PATH': self.config_path,
                'TEST_TIME': current_time.isoformat()
            }):
                try:
                    forecast_main()
                    
                    # 获取预测结果
                    forecast_collection = self.mongo_client['monitor']['task_forecast']
                    forecasts = list(forecast_collection.find({}))
                    
                    if forecasts:
                        # 获取预测时间范围
                        pred_times = pd.to_datetime([doc['ds'] for doc in forecasts])
                        start_pred = min(pred_times)
                        end_pred = max(pred_times)
                        
                        # 获取所有需要查询的年月
                        months = []
                        current = start_pred.replace(day=1, hour=0)
                        while current <= end_pred:
                            months.append(current.strftime('%Y_%m'))
                            # 获取下个月
                            if current.month == 12:
                                current = current.replace(year=current.year + 1, month=1)
                            else:
                                current = current.replace(month=current.month + 1)
                        
                        # 获取所有月份的数据并合并
                        all_actual_data = []
                        for year_month in months:
                            stats_collection = self.mongo_client[f'ship_monitor_{year_month}'][f'hourly_stats_{year_month}']
                            month_data = list(stats_collection.find({
                                'timestamp': {
                                    '$gte': start_pred - timedelta(hours=1),
                                    '$lt': end_pred
                                }
                            }))
                            all_actual_data.extend(month_data)
                        
                        # 转换为DataFrame并排序
                        actual_df = pd.DataFrame(all_actual_data)
                        actual_df['timestamp'] = pd.to_datetime(actual_df['timestamp'])
                        actual_df.sort_values('timestamp', inplace=True)
                        actual_df.set_index('timestamp', inplace=True)
                        
                        # 对每个指标进行预测结果评估
                        for base_column in ['x', 'ax', 'y', 'ay', 'az']:
                            forecast_metrics = {}
                            actual_metrics = {}
                            
                            for metric in ['mean', 'std', 'max', 'min']:
                                column = f'{base_column}_{metric}'
                                
                                # 获取预测数据
                                forecast_metrics[column] = pd.Series(
                                    [doc[f'{column}_pred'] for doc in forecasts],
                                    index=pred_times
                                )
                                
                                # 获取实际数据
                                actual_metrics[column] = actual_df[column]
                            
                            # 绘制对比图
                            self.plot_forecast_results(
                                actual_metrics,
                                forecast_metrics,
                                base_column,
                                current_time.strftime('%Y%m%d_%H')
                            )
                                
                except Exception as e:
                    has_error = True
                    test_logger.error(f"预测时间点 {current_time} 失败: {str(e)}")
                    
            current_time += timedelta(hours=1)
        
        # Check for errors at the end of test
        if has_error:
            self.fail("ERROR level logs were found during testing. Please check test logs for details")

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
        self.mongo_client.drop_database('monitor')
        for month in ['10', '11']:
            self.mongo_client.drop_database(f'ship_monitor_2024_{month}')

if __name__ == '__main__':
    unittest.main()
