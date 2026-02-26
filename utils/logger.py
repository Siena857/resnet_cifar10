import os
import glob
import logging
from utils.config import CONFIG

def init_logger():
    """初始化日志：保存到文件+控制台输出，只保留最新2个日志（学长要求）"""
    # 清理旧日志（只保留最新2个）
    log_files = sorted(glob.glob(os.path.join(CONFIG["path"]["log_path"], "*.log")))
    if len(log_files) > 2:
        for old_log in log_files[:-2]:
            os.remove(old_log)

    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            # 保存到文件
            logging.FileHandler(
                os.path.join(CONFIG["path"]["log_path"], "train.log"),
                encoding="utf-8",
                mode="w"
            ),
            # 输出到控制台
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# 全局日志对象
LOGGER = init_logger()