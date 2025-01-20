import unittest
import numpy as np
import os
from src.processMotionString import ShipMotionProcessor

class TestShipMotionProcessor(unittest.TestCase):
    def setUp(self):
        # 创建测试用的配置文件
        self.ini_content = """
[Monitor_point_acc]
Monitor_point_x = 10.0
Monitor_point_y = 0.0
Monitor_point_z = 5.0
direction = 0

[VesselDefaults]
target_pos_x = 15.0
target_pos_y = 0.0
target_pos_z = 8.0
LBP = 100.0
VCG = 10.0
GM0 = 5.0
"""
        self.yaml_content = """
Monitor:
  x: 15.0
  y: 0.0
  z: 8.0
vessel:
  LBP: 100.0
  VCG: 10.0
  GM1: 5.0
"""
        # 写入临时配置文件
        with open('test_config.ini', 'w') as f:
            f.write(self.ini_content)
        with open('test_config.yaml', 'w') as f:
            f.write(self.yaml_content)
            
        self.processor = ShipMotionProcessor('test_config.ini', 'test_config.yaml')

    def tearDown(self):
        # 清理临时文件
        os.remove('test_config.ini')
        os.remove('test_config.yaml')

    def test_initialization(self):
        """测试初始化参数是否正确"""
        np.testing.assert_array_almost_equal(
            self.processor.monitor_pos,
            np.array([10.0, 0.0, 5.0])
        )
        np.testing.assert_array_almost_equal(
            self.processor.target_pos,
            np.array([15.0, 0.0, 8.0])
        )

    def test_parse_data_fast(self):
        """测试数据解析功能"""
        test_data = "$cmd,2024-03-21 10:00:00.000,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"
        angles, motion_matrix = self.processor.parse_data_fast(test_data)
        
        self.assertEqual(len(angles), 2)
        self.assertEqual(motion_matrix.shape, (3, 3))

    def test_process_motion_data_fast(self):
        """测试完整的数据处理流程"""
        test_data = "$cmd,2024-03-21 10:00:00.000,2.5,2.3,0.8,1.2,0.5,0.9,1.1,0.7,0.4"
        result = self.processor.process_motion_data_fast(test_data)
        
        print("\n测试数据处理结果：")
        print(f"输入字符串: {test_data}")
        print(f"输出结果: {result}")
        
        self.assertEqual(len(result), 5)
        self.assertTrue(all(isinstance(x, float) for x in result))
        self.assertTrue(any(x != 0 for x in result), "所有结果都不应该为0")

    def test_invalid_data_handling(self):
        """测试无效数据处理"""
        invalid_data = "invalid,data,string"
        angles, motion_matrix = self.processor.parse_data_fast(invalid_data)
        
        self.assertEqual(angles, (0.0, 0.0))
        np.testing.assert_array_equal(motion_matrix, np.zeros((3, 3)))

if __name__ == '__main__':
    unittest.main() 