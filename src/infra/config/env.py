# infra/config/env.py — 环境变量读取与校验
# 职责:
#   1. 从 .env 文件加载环境变量
#   2. 校验必需变量是否存在 (如 API Key)
#   3. 提供获取特定变量的便捷函数
#   4. 支持多环境 (.env.development, .env.production)
