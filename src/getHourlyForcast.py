"""
Vessel Intelligence Operation System (VIOS) - Vessel Motion Monitoring System (VMMS)
Time Series Forecasting Module

Main functionalities:
1. Retrieve historical vessel motion data from MongoDB
2. Generate forecasts using Prophet and ARIMA models with configurable weights
3. Combine predictions with stability thresholds for x/y channels
4. Store forecast results back to MongoDB

Key components:
- ForecastConfig: Configuration class for forecast parameters including stability thresholds
- DatabaseConfig: Configuration class for MongoDB connection settings
- MongoDBHandler: Class for handling database operations
- TimeSeriesForecaster: Class for generating forecasts using Prophet and ARIMA models
- Multi-threaded processing for improved performance

Key functions:
- fetch_and_prepare_data: Retrieve and preprocess data
- process_column: Process predictions for individual channels
- combine_all_forecasts: Merge all forecast results
- resample_time_series: Resample time series data
- main: Main function coordinating the entire forecasting process

Developer: Dr. GUO, XIAOXIAN @ SJTU/SKLOE
Contact: xiaoxguo@sjtu.edu.cn
Project: Vessel Intelligence Operation System (VIOS) - Vessel Motion Monitoring System (VMMS)
Date: 2025-01-07

Copyright (c) 2024 Shanghai Jiao Tong University
All rights reserved.
"""

# %%
import logging
from typing import Optional, List
from logger_config import setup_logger, setup_prophet_logger
from pymongo import MongoClient, errors as mongo_errors
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from contextlib import closing
import configparser
from prophet import Prophet
from pmdarima import auto_arima
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

@dataclass
class ForecastConfig:
    """存储预测配置的数据类"""
    used_days: int
    forecast_days: int
    prophet_weight: float
    changepoint_prior_scale: float
    stable_x_threshold: float      # 新增：x通道稳定性阈值
    predict_x_max_threshold: float # 新增：x最大预测阈值
    predict_y_max_threshold: float # 新增：y最大预测阈值
    stable_period: int
    start_period: int
    frequency: str = 'h'
    
@dataclass
class DatabaseConfig:
    """存储数据库配置的数据类"""
    connection_string: str
    db_prefix: str
    db_name: str
    collection_name: str

class MongoDBHandler:
    """处理MongoDB数据库操作的类"""
    
    def __init__(self, connection_string: str, logger: logging.Logger):
        self.connection_string = connection_string
        self.logger = logger
        
    def get_client(self) -> MongoClient:
        """创建MongoDB客户端连接"""
        try:
            return MongoClient(self.connection_string)
        except mongo_errors.ConnectionFailure as e:
            self.logger.error(f"Failed to connect to MongoDB: {e}")
            raise
            
    def read_data(self, db_name: str, collection_name: str, start_time: datetime, end_time: datetime) -> Optional[pd.DataFrame]:
        """
        从MongoDB读取指定时间范围内的数据
        
        Args:
            db_name (str): 数据库名称
            collection_name (str): 集合名称
            start_time (datetime): 数据读取的起始时间
            end_time (datetime): 数据读取的结束时间
            
        Returns:
            Optional[pd.DataFrame]: 读取的数据，如果没有数据返回None
        """
        try:
            with closing(self.get_client()) as client:
                collection = client[db_name][collection_name]
                
                # 确保时间戳有正确的时区信息
                if start_time.tzinfo is None:
                    start_time = start_time.replace(tzinfo=timezone.utc)
                if end_time.tzinfo is None:
                    end_time = end_time.replace(tzinfo=timezone.utc)
                
                # 构建查询条件
                query = {
                    "timestamp": {
                        "$gte": start_time,
                        "$lte": end_time
                    }
                }
                
                # 添加索引以提升查询性能
                collection.create_index([("timestamp", 1)])
                
                # 执行查询
                data = list(collection.find(query))
                
                if not data:
                    self.logger.warning(
                        f"No data found in collection {collection_name} "
                        f"between {start_time} and {end_time}"
                    )
                    return None
                    
                # 转换为DataFrame
                df = pd.DataFrame(data)
                df['ds'] = pd.to_datetime(df['timestamp'])
                
                # 确保时区信息一致
                if df['ds'].dt.tz is None:
                    df['ds'] = df['ds'].dt.tz_localize('UTC')
                
                # 删除多余的timestamp列，因为我们已经有了ds列
                if 'timestamp' in df.columns:
                    df = df.drop('timestamp', axis=1)
                
                # 按时间排序
                df = df.sort_values('ds')
                
                self.logger.info(
                    f"Successfully read {len(df)} records from MongoDB "
                    f"(time range: {start_time} to {end_time})"
                )
                
                # 输出一些基本的数据统计信息
                if not df.empty:
                    self.logger.info(f"Data time range: {df['ds'].min()} to {df['ds'].max()}")
                    
                return df
                
        except mongo_errors.OperationFailure as e:
            self.logger.error(f"MongoDB operation failed: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error reading from MongoDB: {e}")
            raise

    def store_forecast(self, db_name: str, collection_name: str, forecast_data: pd.DataFrame) -> None:
        """将预测结果存储到MongoDB"""
        try:
            with closing(self.get_client()) as client:
                db = client[db_name]
                
                # 如果集合已存在，先删除
                if collection_name in db.list_collection_names():
                    db[collection_name].drop()
                    self.logger.info(f"Existing collection '{collection_name}' dropped")
                
                collection = db[collection_name]
                
                # 确保时区信息正确
                if forecast_data['ds'].dt.tz is None:
                    forecast_data['ds'] = forecast_data['ds'].dt.tz_localize('UTC')
                
                # 批量插入数据
                batch_size = 1000
                records = forecast_data.to_dict("records")
                for i in range(0, len(records), batch_size):
                    batch = records[i:i + batch_size]
                    collection.insert_many(batch)
                
                self.logger.info(f"Successfully stored {len(records)} forecast records")
                
        except Exception as e:
            self.logger.error(f"Error storing forecast in MongoDB: {e}")
            raise

class TimeSeriesForecaster:
    """时间序列预测类，使用动态权重"""
    
    def __init__(self, config: ForecastConfig, db_config: DatabaseConfig, logger: logging.Logger):
        self.config = config
        self.db_config = db_config
        self.logger = logger
        self.mongodb_handler = MongoDBHandler(db_config.connection_string, logger)
        
    def get_prophet_weight(self, column_name: str) -> float:
        """
        根据列名确定Prophet模型的权重
        
        Args:
            column_name (str): 列名
            
        Returns:
            float: Prophet模型的权重
        """
        if '_mean' in column_name:
            return self.config.prophet_weight     # 对均值列使用较高的Prophet权重
        return 1 - self.config.prophet_weight  # 对其他列使用较低的Prophet权重
        
    def prepare_data(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """准备预测数据"""
        try:
            prepared_df = df[['ds', column_name]].copy()
            prepared_df = prepared_df.rename(columns={column_name: 'y'})
            
            # 删除缺失值
            prepared_df = prepared_df.dropna()
            
            # 确保数据按时间排序
            prepared_df = prepared_df.sort_values('ds')
            
            # 去除ds中的时区信息
            prepared_df['ds'] = prepared_df['ds'].dt.tz_localize(None)
            
            return prepared_df
            
        except Exception as e:
            self.logger.error(f"Error in prepare_data for {column_name}: {str(e)}")
            raise
        
    def run_prophet_model(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """运行Prophet模型"""
        try:
            model = Prophet(
                changepoint_prior_scale=0.2,  # 增加以允许更多变化
            )
            
            model.fit(df)
            
            future = model.make_future_dataframe(
                periods=self.config.forecast_days * 24,
                freq=self.config.frequency
            )
            forecast = model.predict(future)
            result = forecast[['ds', 'yhat']]
            return result[result['ds'] > df['ds'].max()]
            
        except Exception as e:
            self.logger.error(f"Prophet model failed: {str(e)}")
            return None
            
    def run_arima_model(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """运行ARIMA模型"""
        try:
            model = auto_arima(
                df['y'].values,
                start_p=15, start_q=3,
                max_p=30, max_q=8,
                d=1, max_d=2,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                max_order=None,
                n_jobs=1
            )
            
            forecast_values = model.predict(n_periods=self.config.forecast_days * 24)
            forecast_dates = pd.date_range(
                start=df['ds'].max() + pd.Timedelta(hours=1),
                periods=self.config.forecast_days * 24,
                freq=self.config.frequency
            )
            
            return pd.DataFrame({
                'ds': forecast_dates,
                'yhat': forecast_values
            })
            
        except Exception as e:
            self.logger.error(f"ARIMA model failed: {str(e)}")
            return None
            
    def combine_forecasts(self, prophet_result: Optional[pd.DataFrame],
                         arima_result: Optional[pd.DataFrame],
                         column_name: str) -> Optional[pd.DataFrame]:
        """
        合并Prophet和ARIMA的预测结果，使用基于列名的动态权重
        
        Args:
            prophet_result: Prophet模型的预测结果
            arima_result: ARIMA模型的预测结果
            column_name: 预测的列名，用于确定权重
        """
        try:
            if prophet_result is not None and arima_result is not None:
                prophet_weight = self.get_prophet_weight(column_name)
                arima_weight = 1 - prophet_weight
                
                merged = pd.merge(
                    prophet_result,
                    arima_result,
                    on='ds',
                    suffixes=('_prophet', '_arima')
                )
                
                self.logger.info(
                    f"Combining forecasts for {column_name} with weights: "
                    f"Prophet={prophet_weight:.2f}, ARIMA={arima_weight:.2f}"
                )
                
                merged['yhat'] = (
                    prophet_weight * merged['yhat_prophet'] +
                    arima_weight * merged['yhat_arima']
                )
                return merged[['ds', 'yhat']]
                
            elif prophet_result is not None:
                self.logger.warning(f"Using only Prophet results for {column_name} due to ARIMA failure")
                return prophet_result
            elif arima_result is not None:
                self.logger.warning(f"Using only ARIMA results for {column_name} due to Prophet failure")
                return arima_result
            else:
                self.logger.error(f"Both Prophet and ARIMA models failed for {column_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error combining forecasts for {column_name}: {str(e)}")
            return None

    def forecast_all_channels(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """执行所有通道的预测"""
        try:
            # 定义需要预测的列
            columns_to_forecast = [
                'y_mean', 'x_std', 'y_std',
                'ax_mean', 'ax_std',
                'ay_mean', 'ay_std',
                'az_mean', 'az_std'
            ]

            # 使用线程池并行处理预测
            all_forecasts = []
            with ThreadPoolExecutor(max_workers=min(len(columns_to_forecast), 4)) as executor:
                future_to_column = {
                    executor.submit(self.process_column, df, column): column
                    for column in columns_to_forecast
                }
                
                for future in as_completed(future_to_column):
                    column = future_to_column[future]
                    try:
                        forecast = future.result()
                        if forecast is not None:
                            all_forecasts.append(forecast)
                            self.logger.info(f"Forecast completed for {column}")
                    except Exception as e:
                        self.logger.error(f"Error processing {column}: {e}")

            return all_forecasts if all_forecasts else None

        except Exception as e:
            self.logger.error(f"Error in forecast_all_channels: {e}")
            return None

    def post_process_forecasts(self, all_forecasts: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """对预测结果进行后处理"""
        try:
            combined_forecast = self.combine_all_forecasts(all_forecasts)
            
            # 设置 x_mean 为 0
            if 'x_mean_pred' not in combined_forecast.columns:
                combined_forecast['x_mean_pred'] = 0
            
            # 确保所有 std 预测值为正
            std_columns = [col for col in combined_forecast.columns if col.endswith('_std_pred')]
            for col in std_columns:
                combined_forecast[col] = combined_forecast[col].abs()
            
            # 添加 max 和 min 预测值
            for channel in ['x', 'y', 'ax', 'ay', 'az']:
                std_col = f'{channel}_std_pred'
                mean_col = f'{channel}_mean_pred'
                if std_col in combined_forecast.columns and mean_col in combined_forecast.columns:
                    max_pred = combined_forecast[mean_col] + 2 * combined_forecast[std_col]
                    min_pred = combined_forecast[mean_col] - 2 * combined_forecast[std_col]
                    
                    # 对x和y的最大最小值进行限制
                    if channel == 'x':
                        max_pred = max_pred.clip(upper=self.config.predict_x_max_threshold)
                        min_pred = min_pred.clip(lower=-self.config.predict_x_max_threshold)
                    elif channel == 'y':
                        max_pred = max_pred.clip(upper=self.config.predict_y_max_threshold)
                        min_pred = min_pred.clip(lower=-self.config.predict_y_max_threshold)
                    
                    combined_forecast[f'{channel}_max_pred'] = max_pred
                    combined_forecast[f'{channel}_min_pred'] = min_pred
            
            return combined_forecast

        except Exception as e:
            self.logger.error(f"Error in post_process_forecasts: {e}")
            return None

    def run_forecast(self) -> bool:
        """运行完整的预测流程"""
        try:
            # 获取数据
            df = self.fetch_and_prepare_data()
            if df is None:
                self.logger.error("No data available for forecasting")
                return False

            # 执行所有通道的预测
            all_forecasts = self.forecast_all_channels(df)
            if not all_forecasts:
                self.logger.warning("No forecasts were generated")
                return False

            # 后处理预测结果
            combined_forecast = self.post_process_forecasts(all_forecasts)
            if combined_forecast is None:
                return False

            # 存储预测结果
            self.mongodb_handler.store_forecast(
                self.db_config.db_name,
                self.db_config.collection_name,
                combined_forecast
            )
            self.logger.info("Forecast process completed successfully")
            return True

        except Exception as e:
            self.logger.critical(f"Critical error in run_forecast: {e}")
            return False

    def process_column(self, df: pd.DataFrame, column: str) -> Optional[pd.DataFrame]:
        """
        处理单个列的预测
        
        Args:
            df: 输入数据框
            column: 要预测的列名
        """
        try:
            self.logger.info(f"Starting forecast for column: {column}")
            
            # 检查数据有效性
            if df['insufficient_data'].iloc[0]:
                # 创建一个包含默认值的预测结果
                current_time = get_current_time()
                forecast_dates = pd.date_range(
                    start=current_time,
                    periods=self.config.forecast_days * 24,
                    freq='h'
                )
                
                # 根据不同列设置不同的默认值
                default_value = 0.1  # 默认值
                
                if '_mean' in column:
                    default_value = 0.0  # 所有_mean默认为0
                elif '_std' in column:
                    if column.startswith(('x', 'y')):
                        default_value = 0.1  # x_std, y_std设为0.1
                    elif column.startswith(('ax', 'ay', 'az')):
                        default_value = 0.001  # ax_std, ay_std, az_std设为0.001
                    
                forecast = pd.DataFrame({
                    'ds': forecast_dates,
                    'yhat': default_value,
                    'column': column
                })
                self.logger.warning(f"Insufficient data for {column}, using default value {default_value}")
                return forecast
                
            # 检查x通道稳定性
            if df['x_stable'].iloc[0]:
                # 使用过去5小时的平均值
                avg_col = f'{column}_5h_avg'
                if avg_col in df.columns:
                    avg_value = df[avg_col].iloc[0]
                    current_time = get_current_time()
                    forecast_dates = pd.date_range(
                        start=current_time,
                        periods=self.config.forecast_days * 24,
                        freq='h'
                    )
                    forecast = pd.DataFrame({
                        'ds': forecast_dates,
                        'yhat': avg_value,
                        'column': column
                    })
                    self.logger.info(f"X channel stable (x_max <= 0.5), using 5-hour average {avg_value:.3f} for {column}")
                    return forecast
            
            # 如果以上条件都不满足，执行正常的预测流程
            prepared_data = self.prepare_data(df, column)
            
            # 运行模型
            prophet_result = self.run_prophet_model(prepared_data)
            arima_result = self.run_arima_model(prepared_data)
            
            # 使用动态权重合并结果
            forecast = self.combine_forecasts(prophet_result, arima_result, column)
            
            if forecast is not None:
                forecast['column'] = column
                self.logger.info(f"Successfully completed forecast for {column}")
                return forecast
            else:
                self.logger.warning(f"No forecast generated for {column}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error processing column {column}: {e}")
            self.logger.exception("Detailed error trace:")
            return None

    def combine_all_forecasts(self, forecasts: List[pd.DataFrame]) -> pd.DataFrame:
        """合并所有预测结果"""
        try:
            combined = forecasts[0][['ds']].copy()
            for forecast in forecasts:
                column_name = forecast['column'].iloc[0]
                combined[f'{column_name}_pred'] = forecast['yhat']
                
            # 添加时区信息
            if combined['ds'].dt.tz is None:
                combined['ds'] = combined['ds'].dt.tz_localize('UTC')
                
            # 计算距离现在的小时数
            combined['hours_from_now'] = (
                (combined['ds'] - get_current_time()) /
                pd.Timedelta(hours=1)
            ).round(2)
            
            return combined
            
        except Exception as e:
            self.logger.error(f"Error combining forecasts: {e}")
            raise

    def fetch_and_prepare_data(self) -> Optional[pd.DataFrame]:
        """从MongoDB获取并准备预测所需的数据"""
        try:
            # 计算时间范围
            end_time = get_current_time()
            start_time = end_time - pd.Timedelta(days=self.config.used_days)
            
            # 构建数据库和集合名称
            current_db_name = f"{self.db_config.db_prefix}_{end_time.strftime('%Y_%m')}"
            current_collection = f"hourly_stats_{end_time.strftime('%Y_%m')}"
            
            # 存储获取的数据框
            dataframes = []
            
            # 如果需要跨月份获取数据
            if start_time.month != end_time.month:
                previous_db_name = f"{self.db_config.db_prefix}_{start_time.strftime('%Y_%m')}"
                previous_collection = f"hourly_stats_{start_time.strftime('%Y_%m')}"
                
                self.logger.info(f"Fetching previous month's data from {previous_db_name}.{previous_collection}")
                
                # 计算上个月的结束时间（当前月的开始）
                month_boundary = end_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                
                df_previous = self.mongodb_handler.read_data(
                    previous_db_name,
                    previous_collection,
                    start_time,
                    month_boundary
                )
                
                if df_previous is not None and not df_previous.empty:
                    dataframes.append(df_previous)
                    self.logger.info(f"Successfully fetched {len(df_previous)} records from previous month")
            
            # 获取当前月份的数据
            self.logger.info(f"Fetching current month's data from {current_db_name}.{current_collection}")
            df_current = self.mongodb_handler.read_data(
                current_db_name,
                current_collection,
                max(start_time, end_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)),
                end_time
            )
            
            if df_current is not None and not df_current.empty:
                dataframes.append(df_current)
                self.logger.info(f"Successfully fetched {len(df_current)} records from current month")
            
            # 如果没有获取到任何数据
            if not dataframes:
                self.logger.warning("No data fetched from any month")
                return None
            
            # 合并数据框
            df = pd.concat(dataframes, ignore_index=True) if len(dataframes) > 1 else dataframes[0]
            
            # 数据清洗和预处理
            self.logger.info("Starting data preprocessing")
            
            # 删除重复数据
            original_len = len(df)
            df = df.drop_duplicates(subset=['ds'])
            if len(df) < original_len:
                self.logger.info(f"Removed {original_len - len(df)} duplicate records")
            
            # 确保时间戳列格式正确
            df['ds'] = pd.to_datetime(df['ds'])
            
            # 按时间排序
            df = df.sort_values('ds')
            
            # 填充缺失值
            df = df.infer_objects(copy=False)
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].interpolate(method='linear', limit_direction='both')
            
            # 执行重采样
            df = self.resample_time_series(df, start_time, end_time)
            
            if df is not None:
                current_time = get_current_time()
                last_12_hours = current_time - pd.Timedelta(hours=self.config.start_period)
                recent_data = df[df['ds'] >= last_12_hours]
                
                if len(recent_data) < self.config.start_period:
                    self.logger.warning("Less than 12 hours of valid data available")
                    df['insufficient_data'] = True
                else:
                    df['insufficient_data'] = False
                    
                # 检查x_max通道数据
                last_5_hours = current_time - pd.Timedelta(hours=self.config.stable_period)
                recent_5h_data = df[df['ds'] >= last_5_hours]
                if len(recent_5h_data) > 0:
                    x_max_avg = recent_5h_data['x_max'].mean()
                    df['x_stable'] = x_max_avg <= self.config.stable_x_threshold
                    if x_max_avg <= self.config.stable_x_threshold:
                        self.logger.info(f"X channel stable: average x_max = {x_max_avg:.3f} degrees in last 5 hours")
                        # 计算所有相关通道的5小时平均值
                        channels = ['x', 'y', 'ax', 'ay', 'az']
                        for channel in channels:
                            for suffix in ['_mean', '_std']:
                                col = f'{channel}{suffix}'
                                if col in recent_5h_data.columns:
                                    df[f'{col}_5h_avg'] = recent_5h_data[col].mean()
                                    self.logger.debug(f"5-hour average for {col}: {df[f'{col}_5h_avg'].iloc[0]:.3f}")
                else:
                    df['x_stable'] = False
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in fetch_and_prepare_data: {str(e)}")
            return None

    def resample_time_series(self, df: pd.DataFrame, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        将时间序列数据重采样为小时间隔，使用指定的时间范围
        
        Args:
            df (pd.DataFrame): 输入数据框，必须包含'ds'列作为时间戳
            start_time (datetime): 重采样的开始时间
            end_time (datetime): 重采样的结束时间
        
        Returns:
            pd.DataFrame: 重采样后的数据框
        """
        try:
            self.logger.info(f"Starting time series resampling from {start_time} to {end_time}")
            
            # 获取数值列
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            self.logger.info(f"Found {len(numeric_columns)} numeric columns to process")
            
            # 初始化结果数据
            resampled_data = {}
            
            # 定义不同类型列的重采样方法
            resample_methods = {
                '_mean': 'mean',
                '_std': 'mean',
                '_min': 'min',
                '_max': 'max',
                '_peak_period': 'mean'  # 周期取平均
            }
            
            # 处理每个数值列
            for col in numeric_columns:
                try:
                    # 创建临时数据框
                    temp_df = pd.DataFrame({'ds': df['ds'], col: df[col]})
                    temp_df.set_index('ds', inplace=True)
                    
                    # 确保时间索引有时区信息
                    if temp_df.index.tz is None:
                        temp_df.index = temp_df.index.tz_localize('UTC')
                    
                    # 确定重采样方法
                    method = 'mean'  # 默认方法
                    for suffix, resample_method in resample_methods.items():
                        if suffix in col:
                            method = resample_method
                            break
                    
                    # 执行重采样
                    self.logger.debug(f"Resampling column {col} using method: {method}")
                    
                    # 先对原始数据进行重采样
                    resampled = getattr(temp_df[col].resample('1h'), method)()
                    
                    # 处理缺失值
                    original_nan_count = resampled.isna().sum()
                    if original_nan_count > 0:
                        self.logger.debug(f"Column {col} has {original_nan_count} NaN values before interpolation")
                    
                    # 使用线性插值填充缺失值
                    resampled = resampled.interpolate(
                        method='time',
                        limit_direction='both',
                        limit=24  # 限制插值范围为24小时
                    )

                    # 存储重采样结果
                    resampled_data[col] = resampled
                    
                    self.logger.info(f"Successfully resampled {col}: original shape {len(df)}, "
                              f"resampled shape {len(resampled)}")
                    
                except Exception as e:
                    self.logger.error(f"Error processing column {col}: {str(e)}")
                    # 继续处理其他列
                    continue
            
            # 创建最终的数据框
            final_df = pd.DataFrame(resampled_data)
            
            # 最终数据质量检查
            nan_check = final_df.isna().sum()
            if nan_check.any():
                self.logger.warning("Found NaN values in final dataset:")
                for col, count in nan_check[nan_check > 0].items():
                    self.logger.warning(f"  {col}: {count} NaN values")
            
            # 基本统计信息
            self.logger.info(f"Resampling completed: {len(df)} original records -> {len(final_df)} hourly records")

            # 将索引复制并重新命名为ds列
            final_df = final_df.reset_index().rename(columns={'index': 'ds'})
            
            return final_df
            
        except Exception as e:
            self.logger.error(f"Error in resample_time_series: {str(e)}")
            raise

def get_current_time():
    """获取当前时间，支持测试时间覆盖"""
    test_time = os.environ.get('TEST_TIME')
    if test_time:
        return datetime.fromisoformat(test_time)
    return datetime.now(timezone.utc)

def main():
    """主函数"""
    logger = None
    handlers = []
    
    try:
        # 配置初始化
        config_path = os.environ.get('TEST_CONFIG_PATH')
        if not config_path:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.dirname(current_dir)
            config_path = os.path.join(base_dir, 'config', 'config.ini')
            
        # 日志设置
        base_dir = os.path.dirname(os.path.dirname(config_path))
        log_dir = os.path.join(base_dir, 'log')
        log_path = os.path.join(log_dir, 'getHourlyForecast.log')
        os.makedirs(log_dir, exist_ok=True)
        
        logger = setup_logger(
            'getHourlyForecast',
            log_path,
            level=logging.INFO,
            max_bytes=10 * 1024 * 1024,
            backup_count=5
        )
        setup_prophet_logger(level=logging.ERROR)
        
        logger.info("Process starting ...")

        # 读取配置
        config = configparser.ConfigParser()
        config.read(config_path)
        
        # 初始化配置对象
        db_config = DatabaseConfig(
            connection_string=config['DBinfo']['connection_string'],
            db_prefix=config['DBinfo']['db_data_prefix'],
            db_name='monitor',
            collection_name='task_forecast'
        )
        
        forecast_config = ForecastConfig(
            used_days=int(config['Forecast']['used_days']),
            forecast_days=int(config['Forecast']['forecast_days']),
            prophet_weight=float(config['Forecast']['prophet_weight']),
            changepoint_prior_scale=float(config['Forecast']['changepoint_prior_scale']),
            stable_x_threshold=float(config['Forecast']['stable_x_threshold']),
            predict_x_max_threshold=float(config['Forecast']['predict_x_max_threshold']),
            predict_y_max_threshold=float(config['Forecast']['predict_y_max_threshold']),
            stable_period=int(config['Forecast']['stable_period']),
            start_period=int(config['Forecast']['start_period'])
        )
        
        # 创建预测器实例并运行
        forecaster = TimeSeriesForecaster(forecast_config, db_config, logger)
        forecaster.run_forecast()
        
    except Exception as e:
        if logger:
            logger.critical(f"Critical error in main process: {e}")
        raise
    finally:
        if logger:
            logger.info("Process ended")
            handlers = logger.handlers[:]
            for handler in handlers:
                handler.close()
                logger.removeHandler(handler)
            logging.shutdown()
# %%
if __name__ == "__main__":
    main()


# %%
