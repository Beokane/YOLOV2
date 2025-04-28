# config.py
import yaml
import os

def load_config(cfg_path: str) -> dict:
    """
    加载 YAML 格式的配置文件。

    :param cfg_path: 配置文件路径
    :return: 解析后的配置字典
    """
    # root_path = os.path.dirname(os.path.abspath(__file__))
    # print(root_path)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"配置文件未找到: {cfg_path}")

    with open(cfg_path, 'r', encoding='utf-8') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"解析配置文件出错: {e}")

    return config
