import os
import shutil
import requests
from dotenv import load_dotenv

_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_current_dir)
load_dotenv(os.path.join(_src_dir, '.env'))

import yaml
from infra.logging.logger import logger as log

from utils.basic_utils import read_yaml, save_yaml

app_title = "video_mix"

local_audio_tts_providers = ['chatTTS', 'GPTSoVITS', 'CosyVoice']
local_audio_recognition_providers = ['fasterwhisper', 'sensevoice']
local_audio_recognition_fasterwhisper_module_names = ['large-v3', 'large-v2', 'large-v1', 'distil-large-v3',
                                                      'distil-large-v2', 'medium', 'base', 'small', 'tiny']
local_audio_recognition_fasterwhisper_device_types = ['cuda', 'cpu', 'auto']
local_audio_recognition_fasterwhisper_compute_types = ['int8', 'int8_float16', 'float16']

vpc = "/obs"  #vpc储存卷挂载路径
RESOURCE_DIR = "./resource"
FINAL_DIR = "./final"
FONT_DIR = "./font"
OUTPUT_DIR = "./work"

driver_types = {
    "chrome": 'chrome',
    "firefox": 'firefox'}

# 获取当前脚本的绝对路径
script_path = os.path.abspath(__file__)

# print("当前脚本的绝对路径是:", script_path)

# 脚本所在的目录
script_dir = os.path.dirname(script_path)

config_example_file_name = "config.example.yml"
config_file_name = "config.yml"

config_example_file = os.path.join(script_dir, config_example_file_name)
config_file = os.path.join(script_dir, config_file_name)


def load_config():
    # 加载配置文件
    if not os.path.exists(config_file):
        shutil.copy(config_example_file, config_file)
    if os.path.exists(config_file):
        return read_yaml(config_file)
    return None


def test_config(todo_config, *args):
    temp_config = todo_config
    for arg in args:
        if arg not in temp_config:
            temp_config[arg] = {}
        temp_config = temp_config[arg]


def save_config():
    # 保存配置文件
    if os.path.exists(config_file):
        save_yaml(config_file, my_config)


my_config = load_config()
load_dotenv(os.path.join(script_dir, '.env'))
ENV = os.getenv('env')
if not ENV:
    log.warning("未找到 .env 文件，环境变量加载失败")
    ENV = my_config['env']
