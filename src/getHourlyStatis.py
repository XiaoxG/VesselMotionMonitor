"""
Vessel Motion Monitoring System (VMMS) - Hourly Statistics Processor
Part of Vessel Intelligence Operation System (VIOS)

This module processes vessel motion data and calculates hourly statistics. It handles data retrieval
from MongoDB, preprocessing, statistical analysis, and storage of results.

Key Components:
- HourlyStatsProcessor: Main class for processing vessel motion data
  - read_data(): Retrieves raw data from MongoDB
  - process_data(): Handles data preprocessing including resampling and filtering
  - calculate_stats(): Computes statistical measures and wave periods
  - save_stats(): Stores results back to MongoDB
  
- Custom Exception Classes:
  - ConfigError: Handles configuration-related errors
  - DataProcessingError: Handles data processing errors

Usage Example:
    config_path = 'path/to/config.ini'
    logger = setup_logger('getHourlyStatis', 'path/to/log.log')
    processor = HourlyStatsProcessor(config_path, logger)
    processor.process()

Project: Vessel Intelligence Operation System (VIOS)
         Vessel Motion Monitoring System (VMMS)
Developer: Dr. GUO, XIAOXIAN @ SJTU/SKLOE
Contact: xiaoxguo@sjtu.edu.cn
Date: 2025-01-07

Dependencies:
- numpy
- pandas
- scipy
- sklearn
- pymongo

Copyright (c) 2024 Shanghai Jiao Tong University
All rights reserved.
"""

# %%
import logging
from typing import Optional, Dict, List, Any
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import pymongo
from pymongo import MongoClient
from scipy import stats
from scipy.signal import butter, filtfilt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from contextlib import closing
import configparser
import ast
from logger_config import setup_logger
from waveModel.timeseries import TimeSeries

import sys
import time
import os

class ConfigError(Exception):
    """Configuration file related custom exception class
    
    Attributes:
        message (str): Error message
        config_section (str, optional): Configuration file section where the error occurred
        config_key (str, optional): Configuration key that caused the error
        details (str, optional): Detailed error information
    """
    def __init__(self, message: str = "Configuration error", 
                 config_section: str = None, 
                 config_key: str = None,
                 details: str = None):
        self.message = message
        self.config_section = config_section
        self.config_key = config_key
        self.details = details
        super().__init__(self.message)
        
    def __str__(self) -> str:
        error_msg = [f"Configuration error: {self.message}"]
        
        if self.config_section:
            error_msg.append(f"Configuration section: [{self.config_section}]")
        if self.config_key:
            error_msg.append(f"Configuration key: {self.config_key}")
        if self.details:
            error_msg.append(f"Details: {self.details}")
            
        return " | ".join(error_msg)


class DataProcessingError(Exception):
    """Custom exception class for data processing errors
    
    Attributes:
        message (str): Error message
        data_source (str, optional): Data source
        processing_step (str, optional): Processing step
        details (str, optional): Detailed error information
    """
    def __init__(self, message: str = "Data processing error",
                 data_source: str = None,
                 processing_step: str = None, 
                 details: str = None):
        self.message = message
        self.data_source = data_source
        self.processing_step = processing_step
        self.details = details
        super().__init__(self.message)
        
    def __str__(self) -> str:
        error_msg = [f"Data processing error: {self.message}"]
        
        if self.data_source:
            error_msg.append(f"Data source: {self.data_source}")
        if self.processing_step:
            error_msg.append(f"Processing step: {self.processing_step}")
        if self.details:
            error_msg.append(f"Details: {self.details}")
            
        return " | ".join(error_msg)
    
class HourlyStatsProcessor:
    """Hourly statistics data processor
    
    Responsible for reading data from MongoDB, preprocessing, calculating statistics, and saving the results back to the database.
    
    Attributes:
        config (configparser.ConfigParser): Configuration object
        logger (logging.Logger): Logger instance
        connection_string (str): MongoDB connection string
        db_prefix (str): Database prefix
        columns (List[str]): Columns to process
        retry_count (int): Retry count
        retry_delay (int): Retry delay (seconds)
    """
    
    def __init__(self, config_path: str, logger: logging.Logger):
        """Initialize processor
        
        Parameters:
            config_path (str): Path to configuration file
            logger (logging.Logger): Logger instance
        """
        self.logger = logger
        self.config = self._load_config(config_path)
        
        # 添加配置验证
        self._validate_config()
        
        # Load basic parameters from configuration
        self.connection_string = self.config['DBinfo']['connection_string']
        self.db_prefix = self.config['DBinfo']['db_data_prefix']
        self.columns = ast.literal_eval(self.config['Statistic_Analysis']['column_name'])
        self.retry_count = int(self.config['Statistic_Analysis'].get('retry', '3'))
        self.retry_delay = int(self.config['Statistic_Analysis'].get('retry_delay', '300'))
        self.fs = float(self.config['Statistic_Analysis']['sampling_rate'])
        
        # Set database parameters
        self._setup_db_params()
        
        self.data_df = None
        self.processed_df = None
        self.stats_data = None
        
    def _load_config(self, config_path: str) -> configparser.ConfigParser:
        """Load configuration file"""
        try:
            config = configparser.ConfigParser()
            config.read(config_path)
            if not config.sections():
                raise ConfigError("Config file is empty or not found",
                                details=f"Config path: {config_path}")
            return config
        except Exception as e:
            raise ConfigError(f"Configuration error: {str(e)}")
            
    def _setup_db_params(self):
        """Set database parameters"""
        today = get_current_time().date()
        self.db_name = f'{self.db_prefix}_{today.strftime("%Y_%m")}'
        self.source_collection = f'{self.db_prefix}_{today.strftime("%Y_%m_%d")}'
        self.result_collection = f'hourly_stats_{today.strftime("%Y_%m")}'
        
    def read_data(self) -> bool:
        """Read data from MongoDB"""
        try:
            end_time = pd.Timestamp(get_current_time())
            start_time = end_time - pd.Timedelta(hours=1)
            
            self.logger.info(f"Querying data for time range:")
            self.logger.info(f"  Start time: {start_time.isoformat()}")
            self.logger.info(f"  End time:   {end_time.isoformat()}")
            
            # Determine which databases and collections to query
            dbs_and_collections = []
            current_time = start_time
            
            while current_time <= end_time:
                db_suffix = current_time.strftime("%Y_%m")
                coll_suffix = current_time.strftime("%Y_%m_%d")
                
                # Extract prefix from database and collection names
                db_prefix = self.db_name.rsplit('_', 2)[0]
                coll_prefix = self.source_collection.rsplit('_', 3)[0]
                
                db_name_full = f"{db_prefix}_{db_suffix}"
                coll_name_full = f"{coll_prefix}_{coll_suffix}"
                
                if not any(x == (db_name_full, coll_name_full) for x in dbs_and_collections):
                    dbs_and_collections.append((db_name_full, coll_name_full))
                
                # Move to the next day
                current_time += pd.Timedelta(days=1)
            
            # Query data from each database and collection
            all_data = []
            with closing(MongoClient(self.connection_string)) as client:
                for db_name_full, coll_name_full in dbs_and_collections:
                    self.logger.info(f"Querying database collection: {db_name_full}.{coll_name_full}")
                    
                    db = client[db_name_full]
                    if coll_name_full not in db.list_collection_names():
                        self.logger.warning(f"Collection does not exist: {coll_name_full}")
                        continue
                        
                    collection = db[coll_name_full]
                    
                    query = {
                        'time': {
                            '$gte': start_time.to_pydatetime(),
                            '$lt': end_time.to_pydatetime()
                        }
                    }
                    
                    data = list(collection.find(query))
                    if data:
                        # 获取实际数据的时间范围
                        data_times = [d['time'] for d in data]
                        min_time = min(data_times)
                        max_time = max(data_times)
                        self.logger.info(f"Retrieved {len(data)} records from {coll_name_full}:")
                        self.logger.info(f"  First record time: {min_time.isoformat()}")
                        self.logger.info(f"  Last record time:  {max_time.isoformat()}")
                        all_data.extend(data)
            
            if not all_data:
                raise DataProcessingError(
                    f"No data found in specified time range: {start_time} to {end_time}",
                    details=f"Queried collections: {[x[1] for x in dbs_and_collections]}"
                )
                
            # 记录所有获取数据的总体时间范围
            # if all_data:
            #     all_times = [d['time'] for d in all_data]
            #     total_min_time = min(all_times)
            #     total_max_time = max(all_times)
            #     total_records = len(all_data)
            #     self.logger.info(f"Total data summary:")
            #     self.logger.info(f"  Total records: {total_records}")
            #     self.logger.info(f"  Overall time range:")
            #     self.logger.info(f"    Start: {total_min_time.isoformat()}")
            #     self.logger.info(f"    End:   {total_max_time.isoformat()}")
            #     self.logger.info(f"    Duration: {total_max_time - total_min_time}")
            
            # Combine all data into a DataFrame
            df = pd.DataFrame(all_data)
            
            # Automatic numeric column detection
            for col in df.columns:
                try:
                    numeric_series = pd.to_numeric(df[col], errors='coerce')
                    non_null_ratio = numeric_series.notnull().mean()
                    if non_null_ratio > 0.8:
                        df[col] = numeric_series
                        self.logger.info(f"Successfully converted column '{col}' to numeric type (non-null ratio: {non_null_ratio:.2%})")
                except:
                    self.logger.info(f"Unable to convert column '{col}' to numeric type, skipping")
                    continue
            
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
            # 添加数据验证
            if not self._validate_data(df, "read_data"):
                return False
            
            self.data_df = df
            self.logger.info(f"Total records retrieved: {len(df)}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error reading MongoDB data: {str(e)}")
            return False
            
    def process_data(self) -> bool:
        """Preprocess data: resampling, outlier handling, filtering, and mean removal"""
        try:
            if self.data_df is None:
                raise DataProcessingError("No data to process", processing_step="process_data")
            
            # 添加数据完整性检查
            if len(self.data_df) < self.fs * 60:  # 至少需要1分钟的数据
                raise DataProcessingError(
                    "Insufficient data points",
                    processing_step="process_data",
                    details=f"Expected: {self.fs * 60}, Got: {len(self.data_df)}"
                )

            # Get configuration parameters
            method = self.config['Statistic_Analysis']['outliers_method']
            threshold = float(self.config['Statistic_Analysis']['outliers_threshold'])
            lowpass_cutoff = float(self.config['Statistic_Analysis']['lowpassfilter'])
            
            # Calculate actual sampling frequency
            time_diff = self.data_df.index.to_series().diff().mean().total_seconds()
            actual_fs = 1 / time_diff
            
            # Check sampling frequency deviation
            fs_deviation = abs(actual_fs - self.fs) / self.fs * 100
            self.logger.info(f"Configured sampling rate: {self.fs:.2f} Hz")
            self.logger.info(f"Actual sampling rate: {actual_fs:.2f} Hz")
            
            # Data preprocessing
            self.processed_df = self.data_df[self.columns].copy()
            
            # Numeric conversion
            for col in self.columns:
                self.processed_df[col] = pd.to_numeric(self.processed_df[col], errors='coerce')
            
            # Only perform resampling if deviation exceeds threshold
            if fs_deviation > 10:
                self.logger.warning(
                    f"Sampling rate deviation exceeds 10% "
                    f"(Config: {self.fs:.2f} Hz, Actual: {actual_fs:.2f} Hz, "
                    f"Deviation: {fs_deviation:.1f}%). Performing resampling."
                )
                
                # Resampling
                resample_interval = f'{int(1000/self.fs)}ms'
                resampled_data = {}
                for col in self.columns:
                    series = self.processed_df[col]
                    resampled_series = series.resample(resample_interval).mean()
                    resampled_series = resampled_series.interpolate(
                        method='time',
                        limit_direction='both',
                        limit=int(self.fs * 5)
                    )
                    resampled_data[col] = resampled_series
                
                self.processed_df = pd.DataFrame(resampled_data)
                self.logger.info("Resampling completed successfully")
            else:
                self.logger.info(
                    f"Sampling rate deviation within acceptable range ({fs_deviation:.1f}%). "
                    "Skipping resampling."
                )
            
            # Outlier handling and filtering
            for col in self.columns:
                outliers = self._detect_outliers(self.processed_df[col], method, threshold)
                if outliers.any():
                    self.processed_df.loc[outliers, col] = np.nan
                    self.processed_df[col] = self.processed_df[col].interpolate(
                        method='time',
                        limit_direction='both',
                        limit=int(self.fs * 2)
                    )
                
                if lowpass_cutoff > 0:
                    self.processed_df[col] = self._apply_lowpass_filter(
                        self.processed_df[col].values,
                        self.fs,
                        cutoff=lowpass_cutoff
                    )
                
                # Only remove mean from x
                if col == 'x':
                    col_mean = self.processed_df[col].mean()
                    self.processed_df[col] = self.processed_df[col] - col_mean
                    self.logger.info(f"Removed mean {col_mean:.6f} from column {col}")
                # Subtract 1 from all az values
                elif col == 'az':
                    self.processed_df[col] = self.processed_df[col] - 1
                    self.logger.info(f"Subtracted g from all values in column {col}")
            
            if self.processed_df is None or self.processed_df.empty:
                raise DataProcessingError("Preprocessing failed")
            
            # 添加处理后数据验证
            if not self._validate_data(self.processed_df, "process_data"):
                return False
            
            self.logger.info("Data preprocessing completed")

            # 添加处理后的数据质量检查
            for col in self.columns:
                null_ratio = self.processed_df[col].isnull().mean()
                if null_ratio > 0.3:  # 超过30%的缺失值
                    raise DataProcessingError(
                        f"High null ratio in processed data",
                        processing_step="process_data",
                        details=f"Column {col}: {null_ratio:.2%} null values"
                    )

                # 检查异常值比例
                z_scores = np.abs(stats.zscore(self.processed_df[col].dropna()))
                outlier_ratio = (z_scores > 10).mean()
                if outlier_ratio > 0.1:  # 超过10%的极端异常值
                    self.logger.warning(
                        f"High outlier ratio in column {col}: {outlier_ratio:.2%}"
                    )

            return True
            
        except DataProcessingError as e:
            self.logger.error(f"Data processing error: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error in data processing: {str(e)}")
            self.logger.exception("Detailed error trace:")
            return False
            
    def calculate_stats(self) -> bool:
        """Calculate statistics"""
        try:
            if self.processed_df is None:
                raise DataProcessingError(
                    "No processed data available",
                    processing_step="calculate_stats"
                )

            # 添加数据有效性检查
            for column in self.columns:
                valid_data = self.processed_df[column].dropna()
                if len(valid_data) < self.fs * 30:  # 至少需要30秒的有效数据
                    raise DataProcessingError(
                        f"Insufficient valid data points for {column}",
                        processing_step="calculate_stats",
                        details=f"Expected: {self.fs * 30}, Got: {len(valid_data)}"
                    )

            stats_data = {'timestamp': self.processed_df.index[-1]}
            period_range = ast.literal_eval(self.config['Statistic_Analysis']['find_peak_range'])
            trimme_percent = int(self.config['Statistic_Analysis']['trimme_percent'])
            
            for column in self.columns:
                data = self.processed_df[column].dropna().values
                if len(data) == 0:
                    raise DataProcessingError(f"Column {column} has no valid data")
                
                # Remove outliers and record details
                bounds = np.percentile(data, [trimme_percent, 100-trimme_percent])
                trimmed_data = data[(data >= bounds[0]) & (data <= bounds[1])]
                self.logger.info(f"Column {column} trimming details:")
                self.logger.info(f"  - Trim bounds: [{bounds[0]:.4f}, {bounds[1]:.4f}]")
                self.logger.info(f"  - Final data range: [{np.min(trimmed_data):.4f}, {np.max(trimmed_data):.4f}]")
                
                # Basic statistics calculation
                stats_data.update({
                    f'{column}_mean': float(np.mean(trimmed_data)),
                    f'{column}_std': float(np.std(trimmed_data)),
                    f'{column}_min': float(np.min(trimmed_data)),
                    f'{column}_max': float(np.max(trimmed_data))
                })
                
                # Spectrum analysis and zero crossing calculation
                data_std = np.std(trimmed_data)
                threshold = 0.005 if column == 'ax' else 0.01
                peak_period = None
                zero_crossing_period = None

                # Calculate peak period
                if data_std < threshold:
                    peak_period = 2.0
                    self.logger.info(f"Column {column} std ({data_std:.6f}) < {threshold}, forcing peak_period = 2.0s")
                else:
                    spec = self._calculate_spectrum(trimmed_data, fs=self.fs)
                    if spec is not None:
                        peak_period = self._find_peak_period(spec, period_range)
                        if peak_period is not None:
                            self.logger.info(f"Column {column} spectral analysis:")
                            self.logger.info(f"  - Peak period: {peak_period:.5f} s")

                # Calculate zero crossing period
                zero_crossing_period = self._calculate_zero_crossing_period(trimmed_data, fs=self.fs)
                if zero_crossing_period is not None:
                    self.logger.info(f"Column {column} zero crossing analysis:")
                    self.logger.info(f"  - Zero crossing period: {zero_crossing_period:.5f} s")

                # Store individual period measurements
                stats_data[f'{column}_peak_period'] = float(peak_period) if peak_period is not None else 0
                stats_data[f'{column}_zero_crossing_period'] = float(zero_crossing_period) if zero_crossing_period is not None else 0
                
                peak_period = stats_data[f'{column}_peak_period']
                zero_crossing_period = stats_data[f'{column}_zero_crossing_period']

                # Calculate final period using both measurements
                both_in_range = (4 <= peak_period <= 15) and (4 <= zero_crossing_period <= 15)
                
                if both_in_range:
                    diff_ratio = abs(peak_period - zero_crossing_period) / min(peak_period, zero_crossing_period)
                    if diff_ratio <= 0.2:
                        avg_period = (peak_period + zero_crossing_period) / 2
                        self.logger.info(f"Column {column} using average of both periods:")
                    else:
                        avg_period = peak_period
                        self.logger.info(f"Column {column} using peak period:")
                elif 4 <= peak_period <= 15:
                    avg_period = peak_period
                    self.logger.info(f"Column {column} using peak period in valid range:")
                elif 4 <= zero_crossing_period <= 15:
                    avg_period = zero_crossing_period
                    self.logger.info(f"Column {column} using zero crossing period in valid range:")
                else:
                    avg_period = peak_period
                    self.logger.info(f"Column {column} using peak period as fallback:")
                
                if data_std < threshold:
                    avg_period = 2.0
                    self.logger.info(f"Column {column} std ({data_std:.6f}) < {threshold}, forcing period = 2.0s")
                
                stats_data[f'{column}_period'] = float(avg_period)

                self.logger.info(f"  - Period: {avg_period:.2f} s")
            
            # Round all numeric values to 5 decimal places
            self.stats_data = self._round_stats_data(stats_data)
            self.logger.info("Statistics calculation completed")

            # 添加结果验证
            required_stats = [
                'mean', 'std', 'min', 'max', 'peak_period',
                'zero_crossing_period', 'period'
            ]
            
            for column in self.columns:
                missing_stats = []
                for stat in required_stats:
                    key = f"{column}_{stat}"
                    if key not in self.stats_data:
                        missing_stats.append(stat)
                    elif self.stats_data[key] is None:
                        missing_stats.append(stat)
                
                if missing_stats:
                    raise DataProcessingError(
                        f"Missing statistics for column {column}",
                        processing_step="calculate_stats",
                        details=f"Missing: {missing_stats}"
                    )

            return True

        except DataProcessingError as e:
            self.logger.error(f"Statistics calculation error: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error in statistics calculation: {str(e)}")
            self.logger.exception("Detailed error trace:")
            return False
            
    def save_stats(self) -> bool:
        """Save statistics results to database"""
        try:
            if self.stats_data is None:
                raise DataProcessingError(
                    "No statistics data to save",
                    processing_step="save_stats"
                )

            # 验证数据完整性
            timestamp = get_current_time()
            self.stats_data['timestamp'] = timestamp
            if timestamp is None:
                raise DataProcessingError(
                    "Missing timestamp in statistics data",
                    processing_step="save_stats"
                )

            # 检查是否已存在相同时间戳的数据
            with closing(MongoClient(self.connection_string)) as client:
                db = client[self.db_name]
                existing_record = db[self.result_collection].find_one({
                    'timestamp': timestamp
                })
                
                if existing_record:
                    self.logger.warning(
                        f"Record with timestamp {timestamp} already exists. "
                        "Updating existing record..."
                    )
                    db[self.result_collection].replace_one(
                        {'timestamp': timestamp},
                        self.stats_data
                    )
                else:
                    db[self.result_collection].insert_one(self.stats_data)
                    self.logger.info(f"Successfully saved statistics data to {self.db_name}.{self.result_collection}")
                    self.logger.info(f"Timestamp: {timestamp}")

            return True

        except DataProcessingError as e:
            self.logger.error(f"Error saving statistics: {str(e)}")
            return False
        except pymongo.errors.PyMongoError as e:
            self.logger.error(f"MongoDB error: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error saving statistics: {str(e)}")
            self.logger.exception("Detailed error trace:")
            return False
         
    def process(self) -> bool:
        """Execute complete processing workflow"""
        for attempt in range(self.retry_count):
            try:
                self.logger.info(f"Processing attempt {attempt + 1}/{self.retry_count}")
                
                # 添加数据状态检查
                if hasattr(self, 'data_df') and self.data_df is not None:
                    self.logger.info("Clearing previous data state")
                    self.data_df = None
                    self.processed_df = None
                    self.stats_data = None
                
                # 添加具体的错误类型处理
                if not self.read_data():
                    raise DataProcessingError("Failed to read data", processing_step="read_data")
                
                if not self.process_data():
                    raise DataProcessingError("Failed to process data", processing_step="process_data")
                
                if not self.calculate_stats():
                    raise DataProcessingError("Failed to calculate statistics", processing_step="calculate_stats")
                
                if not self.save_stats():
                    raise DataProcessingError("Failed to save statistics", processing_step="save_stats")
                
                self.logger.info("Data processing completed successfully")
                return True
                
            except DataProcessingError as e:
                self.logger.error(f"Processing error in step {e.processing_step}: {str(e)}")
                if attempt < self.retry_count - 1:
                    self.logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
            except MongoClient.errors.ConnectionError as e:
                self.logger.error(f"MongoDB connection error: {str(e)}")
                if attempt < self.retry_count - 1:
                    self.logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
            except Exception as e:
                self.logger.error(f"Unexpected error: {str(e)}")
                self.logger.exception("Detailed error trace:")
                return False
                
        return False

    def _detect_outliers(self, data: pd.Series, method: str, threshold: float = 3.0) -> np.ndarray:
        """Detect outliers"""
        try:
            # Check data standard deviation
            data_std = data.std()
            if data_std < 0.01:
                self.logger.info(f"Small data standard deviation ({data_std:.6f}), using relaxed zscore method")
                outliers = np.abs(stats.zscore(data)) > 10.0
                self.logger.info(f"Found {np.sum(outliers)} outliers using relaxed zscore")
                return outliers
                
            # Record outlier detection method used
            self.logger.info(f"Using outlier detection method: {method}")
            self.logger.info(f"Initial threshold set to: {threshold}")
            
            outlier_methods = {
                'zscore': lambda x, t: np.abs(stats.zscore(x)) > t,
                'robust_zscore': lambda x, t: np.abs(0.6745 * (x - np.median(x)) / 
                                            np.median(np.abs(x - np.median(x)))) > t,
                'iqr': lambda x, t: (x < x.quantile(0.25) - t * (x.quantile(0.75) - x.quantile(0.25))) | 
                            (x > x.quantile(0.75) + t * (x.quantile(0.75) - x.quantile(0.25))),
                'isolation_forest': lambda x, t: IsolationForest(random_state=42, contamination=0.1)
                                            .fit_predict(x.values.reshape(-1, 1)) == -1,
                'lof': lambda x, t: LocalOutlierFactor(n_neighbors=20, contamination=0.1)
                                .fit_predict(x.values.reshape(-1, 1)) == -1
            }
            
            if method not in outlier_methods:
                raise ValueError(f"Invalid method: {method}")
            
            # First outlier detection
            outliers = outlier_methods[method](data, threshold)
            outlier_ratio = np.sum(outliers) / len(data)
            self.logger.info(f"Initial outlier ratio: {outlier_ratio:.2%}")
            
            # If outlier ratio is too high, use zscore method and gradually relax threshold
            if outlier_ratio > 0.02:
                self.logger.info("Outlier ratio > 2%, switching to relaxed zscore method")
                outliers = np.abs(stats.zscore(data)) > 5.0
                outlier_ratio = np.sum(outliers) / len(data)
                self.logger.info(f"Outlier ratio with threshold=5: {outlier_ratio:.2%}")
                
                if outlier_ratio > 0.02:
                    self.logger.info("Outlier ratio still > 2%, further relaxing threshold")
                    outliers = np.abs(stats.zscore(data)) > 10.0
                    outlier_ratio = np.sum(outliers) / len(data)
                    self.logger.info(f"Outlier ratio with threshold=10: {outlier_ratio:.2%}")
                    
                    if outlier_ratio > 0.02:
                        self.logger.info("Outlier ratio still > 2%, disabling outlier detection")
                        return np.zeros(len(data), dtype=bool)
            
            self.logger.info(f"Final outlier count: {np.sum(outliers)}")
            return outliers
            
        except Exception as e:
            self.logger.error(f"Outlier detection error: {str(e)}")
            return np.zeros(len(data), dtype=bool)

    def _apply_lowpass_filter(self, data: np.ndarray, fs: float, 
                            cutoff: float = 2.0, order: int = 6) -> np.ndarray:
        """Apply lowpass filter"""
        try:
            # Check if input data contains NaN
            if np.isnan(data).any():
                self.logger.warning("Input data contains NaN values, interpolating before filtering")
                # Use linear interpolation to fill NaN
                nan_mask = np.isnan(data)
                x = np.arange(len(data))
                data = np.interp(x, x[~nan_mask], data[~nan_mask])

            # Check if data is empty or all zeros
            if len(data) == 0:
                self.logger.error("Input data is empty")
                return data
            if np.all(data == 0):
                self.logger.warning("Input data is all zeros, skipping filtering")
                return data

            # Convert cutoff frequency from rad/s to Hz
            cutoff_hz = cutoff / (2 * np.pi)
            
            nyq = 0.5 * fs  # Calculate Nyquist frequency
            normal_cutoff = cutoff_hz / nyq  # Calculate normalized cutoff frequency
            
            # Check if cutoff frequency is valid
            if normal_cutoff <= 0 or normal_cutoff >= 1:
                self.logger.warning(f"Invalid normalized cutoff frequency ({normal_cutoff}), skipping filtering")
                return data

            self.logger.info(f"Applying lowpass filter - Cutoff frequency: {cutoff_hz:.2f} Hz, Sampling rate: {fs:.2f} Hz")
            
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            filtered_data = filtfilt(b, a, data)
            
            # Check filtered data
            if np.isnan(filtered_data).any():
                self.logger.error("Filtered data contains NaN values, returning original data")
                return data
            
            # Check if filtering causes data to be excessively attenuated
            if np.std(filtered_data) < np.std(data) * 0.1:
                self.logger.warning("Filtering may cause data to be excessively attenuated, returning original data")
                return data
            
            return filtered_data
            
        except Exception as e:
            self.logger.error(f"Filter error: {str(e)}")
            return data

    def _calculate_spectrum(self, data: np.ndarray, fs: float = 20.0) -> Optional[Any]:
        """Calculate spectrum"""
        try:
            T = np.arange(len(data)) / fs
            ts = TimeSeries(data, T)
            return ts.tospecdata(L=3000, method='cov')
        except Exception as e:
            self.logger.error(f"Spectrum calculation error: {str(e)}")
            return None

    def _find_peak_period(self, spec: Any, period_range: List[float] = [2, 30]) -> Optional[float]:
        """Find peak period"""
        try:
            fs_low = 2 * np.pi / period_range[1]
            fs_high = 2 * np.pi / period_range[0]
            
            mask = (spec.args >= fs_low) & (spec.args <= fs_high)
            filtered_data = spec.data[mask]
            filtered_args = spec.args[mask]
            
            if len(filtered_data) == 0:
                raise DataProcessingError("No data in period range")
            
            peak_idx = np.argmax(filtered_data)
            peak_freq = filtered_args[peak_idx]
            peak_period = 2 * np.pi / peak_freq if peak_freq != 0 else np.inf
            
            return np.clip(peak_period, period_range[0], period_range[1])
            
        except Exception as e:
            self.logger.error(f"Peak period detection error: {str(e)}")
            return None

    def _round_stats_data(self, stats_data: Dict[str, Any]) -> Dict[str, Any]:
        """Round statistics data to 5 decimal places"""
        try:
            rounded_data = {}
            for key, value in stats_data.items():
                if key == 'timestamp':
                    rounded_data[key] = value
                elif isinstance(value, (int, float, np.float32, np.float64)):
                    rounded_data[key] = round(float(value), 5)
                else:
                    rounded_data[key] = value
                
            self.logger.info("Successfully rounded statistics data to 5 decimal places")
            return rounded_data
            
        except Exception as e:
            self.logger.error(f"Error rounding statistics data: {str(e)}")
            return stats_data

    def _calculate_zero_crossing_period(self, data: np.ndarray, fs: float) -> Optional[float]:
        """Calculate average zero crossing period of time series"""
        try:
            # Remove mean
            data_zero_mean = data - np.mean(data)
            
            # Find all zero crossings
            zero_crossings = np.where(np.diff(np.signbit(data_zero_mean)))[0]
            
            # Calculate zero crossings
            up_crossings = [i for i in zero_crossings if data_zero_mean[i+1] > 0]
            
            if len(up_crossings) < 2:
                self.logger.warning("Not enough up zero crossings to calculate period")
                return None
            
            # Calculate interval between adjacent zero crossings
            intervals = np.diff(up_crossings)
            
            # Convert interval to seconds
            periods = intervals / fs
            
            # Calculate average period
            mean_period = np.mean(periods)
            
            return float(mean_period)
            
        except Exception as e:
            self.logger.error(f"Error calculating zero crossing period: {str(e)}")
            return None

    def _validate_data(self, data: pd.DataFrame, stage: str) -> bool:
        """验证数据的有效性
        
        Args:
            data: 要验证的数据
            stage: 处理阶段名称
        """
        try:
            if data is None or data.empty:
                raise DataProcessingError(f"Empty data in {stage}")
            
            # 检查必需列
            missing_cols = set(self.columns) - set(data.columns)
            if missing_cols:
                raise DataProcessingError(
                    f"Missing required columns in {stage}",
                    details=f"Missing columns: {missing_cols}"
                )
            
            # 检查数据类型
            non_numeric_cols = [col for col in self.columns 
                              if not np.issubdtype(data[col].dtype, np.number)]
            if non_numeric_cols:
                raise DataProcessingError(
                    f"Non-numeric columns found in {stage}",
                    details=f"Columns: {non_numeric_cols}"
                )
            
            # 检查缺失值比例
            na_ratios = data[self.columns].isna().mean()
            high_na_cols = na_ratios[na_ratios > 0.2].index.tolist()
            if high_na_cols:
                self.logger.warning(
                    f"High missing value ratio in columns: {high_na_cols}\n"
                    f"Missing ratios: {na_ratios[high_na_cols]}"
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Data validation error in {stage}: {str(e)}")
            return False

    def _validate_config(self) -> None:
        """验证配置参数"""
        required_sections = ['DBinfo', 'Statistic_Analysis']
        required_params = {
            'DBinfo': ['connection_string', 'db_data_prefix'],
            'Statistic_Analysis': [
                'column_name', 'sampling_rate', 'outliers_method',
                'outliers_threshold', 'lowpassfilter', 'find_peak_range',
                'trimme_percent'
            ]
        }
        
        # 检查必需的配置部分
        missing_sections = set(required_sections) - set(self.config.sections())
        if missing_sections:
            raise ConfigError(
                "Missing required configuration sections",
                details=f"Missing sections: {missing_sections}"
            )
        
        # 检查每个部分的必需参数
        for section, params in required_params.items():
            missing_params = set(params) - set(self.config[section].keys())
            if missing_params:
                raise ConfigError(
                    f"Missing required parameters in section [{section}]",
                    config_section=section,
                    details=f"Missing parameters: {missing_params}"
                )


def get_current_time():
    """Get current time, supporting test time override"""
    test_time = os.environ.get('TEST_TIME')
    if test_time:
        return datetime.fromisoformat(test_time)
    return datetime.now(timezone.utc)


def main():
    """Main function"""
    logger = None
    handlers = []
    
    try:
        # Set configuration file path
        config_path = os.environ.get('TEST_CONFIG_PATH')
        if not config_path:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.dirname(current_dir)
            config_path = os.path.join(base_dir, 'config', 'config.ini')
            
        # Set up logger
        base_dir = os.path.dirname(os.path.dirname(config_path))
        log_dir = os.path.join(base_dir, 'log')
        log_path = os.path.join(log_dir, 'getHourlyStatis.log')
        os.makedirs(log_dir, exist_ok=True)
        
        logger = setup_logger(
            'getHourlyStatis',
            log_path,
            level=logging.INFO,
            max_bytes=10*1024*1024,
            backup_count=5
        )
        
        # Create processor instance and execute processing
        processor = HourlyStatsProcessor(config_path, logger)
        if not processor.process():
            sys.exit(1)
            
    except Exception as e:
        if logger:
            logger.error(f"Fatal error: {str(e)}")
            logger.exception("Detailed stack trace:")
        sys.exit(1)
        
    finally:
        if logger:
            logger.info("Process ended")
            handlers = logger.handlers[:]
            for handler in handlers:
                handler.close()
                logger.removeHandler(handler)
            logging.shutdown()

if __name__ == "__main__":
    main()

# %%
