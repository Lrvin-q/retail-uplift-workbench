"""运行资源由wheel携带；仓库镜像一致性由测试验证。"""
from importlib.resources import files

def resource_text(name: str) -> str:
    return files(__package__).joinpath(name).read_text(encoding="utf-8-sig")
