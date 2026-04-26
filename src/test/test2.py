import numpy
print(numpy.__file__)  # 看实际加载的是哪个路径
print(numpy.__version__)

import site
print("用户级路径:", site.getusersitepackages())
print("是否启用用户级:", site.ENABLE_USER_SITE)