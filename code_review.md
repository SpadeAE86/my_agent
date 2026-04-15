# Code Review: commit `6b41631` — add file registry

> 25 files changed, +1064 / -241

---

## 1. FileRegistry (`infra/storage/file_registry.py`) — ⚠️ 有问题

### ✅ 好的设计
- 短 ID 映射物理路径，对 Agent 友好（减少 token 消耗）
- Redis 分布式锁 + Lua 原子解锁，思路正确
- `resolve_file` 兜底逻辑（绝对路径直接放行）很实用

### 🐛 Bug: Lua 脚本语法错误

```lua
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else        -- ← 这里缺少 `end` 之前要写 `return 0`
    return 0
end
```

等等，再看一遍... 实际上这段 Lua 语法是对的（`if...then...else...end`），没问题。**但有另一个问题**：

### 🐛 Bug: `acquire_write_lock` 传入的是 `local_path`，但内部 `resolve_file` 又解析一次

```python
async def acquire_write_lock(self, local_path: str, timeout: float = 5.0):
    path = self.resolve_file(local_path)  # 如果传的是 file_id，这里解析成路径
    ...

async def write_to_file(self, local_path: str, ...):
    real_path = self.resolve_file(local_path)    # ① 解析一次
    lock_id = await self.acquire_write_lock(real_path)  # ② 把路径传进去又 resolve 一次！
```

`write_to_file` 里先 `resolve_file` 拿到 `real_path`，然后把 `real_path` 传给 `acquire_write_lock`，里面又 `resolve_file(real_path)`。如果 `real_path` 不在注册表里，就会走兜底逻辑（`os.path.isabs`），碰巧能工作，但逻辑不清晰且有隐患。

> **建议**: `acquire_write_lock` / `release_write_lock` / `check_can_read` 内部统一接收物理路径，不要再 resolve。或者明确约定入参是 file_id 还是物理路径。

### ⚠️ 潜在问题: 读操作没有真正的并发安全

`check_can_read` 只是自旋等到没有写锁就返回，但返回之后到实际读取之间，另一个协程可能又拿到写锁了。这是一个 TOCTOU（Time-of-Check-Time-of-Use）问题。在单进程 asyncio 里影响不大，但要注意。

### ⚠️ 内存泄漏风险

`_id_to_path` 和 `_path_to_id` 只增不减，长时间运行的服务会持续累积。建议加个 `unregister` 或 LRU 淘汰。

---

## 2. RedisManager (`infra/cache/redis_client.py`) — ✅ 基本OK

### ✅ 好的
- 懒加载，不连接 Redis 就不会报错
- 同步/异步双客户端

### 🐛 Bug: `aioredis.Redis` 的 `options` 参数

```python
self.client = aioredis.Redis(
    ...
options = {"health_check_interval": 30},  # ← 这个参数不存在！
)
```

`redis.asyncio.Redis` 没有 `options` 关键字参数。`health_check_interval` 应该是直接传的顶层参数：

```python
self.client = aioredis.Redis(
    ...
health_check_interval = 30,
)
```

---

## 3. MySQLManager (`infra/storage/mysql_client.py`) — ✅ 合理

### ✅ 好的
- SQLAlchemy 的 `pool_pre_ping` + `pool_recycle` 组合很专业
- 同步/异步双引擎
- `get_sync_session` / `get_async_session` 适配 FastAPI Depends

### ⚠️ 小问题
- `get_sync_session` 是 generator（`yield`），用作 FastAPI Dependency 没问题，但直接调用会拿到 generator 而不是 session。注释说明一下比较好。

---

## 4. OpenSearchManager (`infra/storage/opensearch_client.py`) — ✅ OK

没有异步客户端，目前只有同步的 `opensearchpy.OpenSearch`。如果未来在 async handler 里用，需要 `asyncio.to_thread` 包一下。

---

## 5. RabbitMQManager (`infra/mq/rabbitmq_client.py`) — ⚠️ 有问题

### 🐛 Bug: `acquire_channel` 的 channel 会在 `async with` 退出后连接被放回池

```python
async def acquire_channel(self):
    pool = await self.get_pool()
    async with pool.acquire() as connection:
        return await connection.channel()  # ← connection 在这行之后就被释放了！
```

`async with pool.acquire()` 退出后 connection 放回池，此时返回的 channel 挂在一个**已归还的连接**上，后续操作会报错或者行为不可预测。

> **建议**: 要么返回 `(connection, channel)` 让调用者管理生命周期，要么改成 context manager。

### ⚠️ `loop` 参数废弃

`aio_pika.pool.Pool(..., loop=loop)` — `loop` 参数在 Python 3.10+ 已废弃，3.12 会警告。直接删掉即可。

---

## 6. Logger (`infra/logging/logger.py`) — ⚠️ 重复初始化

```python
if not ENV:
    # 这个分支里做了一次 logger.remove() + logger.add()
    ...

# 文件末尾又做了一次
logger.remove()
logger.add(...)

setup_logger()  # 又调了一次！
```

`logger.remove()` + `logger.add()` 至少执行了 **2~3 次**。每次 `remove()` 会清空之前的 handler，所以最终只有最后一次生效，但中间的日志可能丢失。

> **建议**: 只保留一处初始化。

### ✅ `time_it` 装饰器

带守护线程的计时装饰器思路不错，能在长耗时函数运行期间定期报告。但只支持同步函数，async 函数需要单独处理。

---

## 7. tool_manager.py — ✅ 好的改动

把 `%s` 格式化改成 f-string，和你项目里 loguru 的使用方式一致。

> ⚠️ 但注意：这次远程改用的是 `logging.getLogger(__name__)`，而你本地已经改成了 `from infra.logging.logger import logger`。**merge 时会冲突**，保留本地的 `infra.logging.logger` 版本即可。

---

## 8. obs_utils.py — ⚠️ 安全问题

### 🚨 硬编码密钥

```python
obs_client = ObsClient(
    access_key_id='UJDPK31ANIBV0XTEUN5N',
    secret_access_key='NhQExxv9PUYsvmvGnVReizRksaiHcJdQ6vMMw19d',
    ...
)
```

AK/SK 不应该提交到 Git。应该从环境变量或 `.env` 读取。

### ✅ 其他
- `diskcache` 结合 OBS 下载的缓存策略合理
- `uuid` 作为临时文件名防并发冲突，好

---

## 9. config.yml — ✅ 清理了大量废弃配置

删除了 callback URL、obs_img_prefix 等旧配置，新增了 log、cache_config、opensearch 的分环境配置。结构更清晰了。

> ⚠️ OpenSearch 密码明文写在 yml 里，生产环境建议走 secrets manager 或环境变量。

---

## 10. 其他变更

| 文件 | 变更 | 评价 |
|---|---|---|
| `file_manager.py` | 删除（被 FileRegistry 替代） | ✅ |
| `file_store.py` | 删除（yaml 工具函数已迁移） | ✅ |
| `read_file.py` / `write_file.py` | 改为使用 `file_registry` | ✅ |
| `environment.yaml` / `pyproject.toml` / `requirements.txt` | 新增依赖管理 | ✅ |
| `playground/` | 从 `utils/` 改名 | ✅ |

---

## 总结

| 优先级 | 问题 | 文件 |
|---|---|---|
| 🔴 必修 | `acquire_channel` 返回的 channel 挂在已归还连接上 | `rabbitmq_client.py` |
| 🔴 必修 | `aioredis.Redis` 的 `options` 参数不存在 | `redis_client.py` |
| 🟡 建议 | FileRegistry 的 resolve 双重调用 | `file_registry.py` |
| 🟡 建议 | Logger 重复初始化 2~3 次 | `logger.py` |
| 🟡 建议 | RabbitMQ Pool 的 `loop=` 参数废弃 | `rabbitmq_client.py` |
| 🟡 建议 | 内存映射只增不减 | `file_registry.py` |
| 🔴 安全 | OBS AK/SK 硬编码在源码里 | `obs_utils.py` |
| 🟡 安全 | 数据库/OpenSearch 密码明文在 yml | `config.yml` |

整体架构方向是对的：**懒加载单例 + 分环境配置 + Redis 分布式锁替代内存锁**。主要需要修的是 RabbitMQ channel 生命周期和 Redis async 参数这两个 bug。
