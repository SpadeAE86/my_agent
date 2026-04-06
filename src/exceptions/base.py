# exceptions/base.py — 基础异常类
class AppException(Exception):
    """所有自定义异常的基类"""
    def __init__(self, message: str, code: str = "UNKNOWN", details: dict = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)
