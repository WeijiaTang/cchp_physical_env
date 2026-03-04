"""
xlsx文件转为csv文件
源目录: data/raw/load/industrial_park_scidata2023/osf/xlsx/
输出目录: data/raw/load/industrial_park_scidata2023/osf/csv/
"""

import os
import pandas as pd
from pathlib import Path

# 使用相对路径（基于项目根目录）
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # scripts/file-convert -> 项目根目录
XLSX_DIR = PROJECT_ROOT / "data/raw/load/industrial_park_scidata2023/osf/xlsx"
CSV_DIR = PROJECT_ROOT / "data/raw/load/industrial_park_scidata2023/osf/csv"

# 要处理的子目录
SOURCE_FOLDERS = ["Electric power load data", "Weather data"]

for folder in SOURCE_FOLDERS:
    source_path = XLSX_DIR / folder
    
    for root, dirs, files in os.walk(source_path):
        for file in files:
            if file.endswith('.xlsx'):
                excel_path = Path(root) / file
                
                # 计算相对路径，保持目录结构
                rel_path = excel_path.relative_to(XLSX_DIR)
                csv_path = CSV_DIR / rel_path.with_suffix('.csv')
                
                # 创建输出目录
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                
                print(f"Converting {excel_path.relative_to(PROJECT_ROOT)} -> {csv_path.relative_to(PROJECT_ROOT)}")
                try:
                    df = pd.read_excel(excel_path)
                    df.to_csv(csv_path, index=False)
                except Exception as e:
                    print(f"Error converting {file}: {e}")
