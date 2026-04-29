import time
import asyncio, json

from exceptions.infra import ServiceException
from nls.token import getToken
import nls
from config.config import MY_CONFIG
import threading
from infra.logging.logger import logger as log
from utils.post_utils import post
import httpx
URL = "wss://nls-gateway-cn-shanghai.aliyuncs.com/ws/v1"
token = None
expire_time = 0

TEXT = '曙光重临，一款治愈系Q萌画风的沉浸式抓宠游戏'


# 以下代码会根据上述TEXT文本反复进行语音合成
class AliTTS:
    def __init__(self, tid, test_file, voice = "zhimao", speed = 0, volume = 80, TOKEN = ""):
        self.__id = tid
        self.__test_file = test_file
        global token, expire_time
        if token is None or time.time() > expire_time:
            log.info(f"token has expired or not initialized")
            token = getToken(MY_CONFIG['audio']['Ali']['access_key_id'],
                                  MY_CONFIG['audio']['Ali']['access_key_secret'])
            expire_time = time.time() + 86400
        self.TOKEN = token
        self.APPKEY = MY_CONFIG['audio']['Ali']['app_key']
        self._error_event = threading.Event()
        self._error_msg = ""
        self.voice = voice
        self.speed = speed
        self.volume = volume

    def start(self, text):
        for i in range(5):
            try:
                self.__text = text
                self.__f = open(self.__test_file, "wb")
                # print(f"{self.__id} start at {time.time()}")
                self.__run()
                if self._error_event.is_set():
                    raise ServiceException(777, f"TTS failed: {self._error_msg}")
                break
            except Exception as e:
                # log.info(f"request too frequent: {e}, retry 1 second later")
                if i == 4:
                    # log.info(f"all retry for {self.__text} failed, give up")
                    raise ServiceException(572, f"all retry for {self.__text} failed, give up: {e}")
                time.sleep(1)

        return self.__test_file

    def on_metainfo(self, message, *args):
        # print("on_metainfo message=>{}".format(message))
        pass

    def on_error(self, message, *args):
        if message:
            body = json.loads(message)
            if "header" in body:
                error_info = body["header"]
                if "status" in error_info and error_info["status"] == "40000005":
                    self._error_event.set()
                if "status_text" in error_info:
                    self._error_msg = error_info["status_text"]
            else:
                self._error_event.set()
                self._error_msg = "unknown error"
        pass
        # print("on_error args=>{}".format(args))

    def on_close(self, *args):
        # print("on_close: args=>{}".format(args))
        try:
            self.__f.close()
        except Exception as e:
            print("close file failed since:", e)

    def on_data(self, data, *args):
        try:
            self.__f.write(data)
        except Exception as e:
            print("write data failed:", e)

    def on_completed(self, message, *args):
        # print("on_completed:args=>{} message=>{}".format(args, message))
        pass


    def __run(self):
        # print("thread:{} start..".format(self.__id))

        tts = nls.NlsSpeechSynthesizer(url=URL,
                                   token=self.TOKEN,
                                   appkey=self.APPKEY,
                                   on_metainfo=self.on_metainfo,
                                   on_data=self.on_data,
                                   on_completed=self.on_completed,
                                   on_error=self.on_error,
                                   on_close=self.on_close,
                                   callback_args=[self.__id])
        r = tts.start(self.__text, voice=self.voice, volume=self.volume, aformat='wav', wait_complete=True, speech_rate=int(self.speed))
        # print("{}: tts done with result:{}".format(self.__id, r))

    async def async_start(self, text):
        """
        异步启动方法。
        使用 asyncio.Future 桥接 SDK 的回调，实现非阻塞等待。
        """
        self.__text = text
        loop = asyncio.get_running_loop()
        # 创建一个 Future 对象，用于在回调中通知主线程任务完成
        done_future = loop.create_future()

        # 定义回调函数，注意：SDK是在子线程运行回调的，所以必须用 call_soon_threadsafe
        def on_complete_bridge(message, *args):
            try:
                # 告诉 asyncio 任务完成了
                loop.call_soon_threadsafe(done_future.set_result, self.__test_file)

            except Exception as e:
                pass

        def on_error_bridge(message, *args):
            error_msg = "Unknown error"
            if message:
                try:
                    body = json.loads(message)
                    error_msg = body.get("header", {}).get("status_text", message)
                except:
                    pass
            # 告诉 asyncio 任务失败了
            loop.call_soon_threadsafe(done_future.set_exception, ServiceException(777, f"TTS Error: {error_msg}"))

        def on_data_bridge(data, *args):
            try:
                if self.__f:
                    self.__f.write(data)
            except Exception as e:
                print(f"Write failed: {e}")

        # 打开文件
        self.__f = open(self.__test_file, "wb")

        # 初始化 SDK
        tts = nls.NlsSpeechSynthesizer(
            url=URL,
            token=self.TOKEN,
            appkey=self.APPKEY,
            on_data=on_data_bridge,
            on_completed=on_complete_bridge,
            on_error=on_error_bridge,
            callback_args=[self.__id],
            on_close=self.on_close
        )

        try:
            # 关键点 1: 在线程池中运行 tts.start
            # 即使 wait_complete=False，start() 依然会阻塞等待握手，所以必须放到 to_thread 里
            # wait_complete=False 意味着握手成功后立即释放线程，而不是傻等到音频传完
            await asyncio.to_thread(
                tts.start,
                self.__text,
                voice=self.voice,
                volume=self.volume,
                aformat='wav',
                wait_complete=False,  # 关键点 2: 不等待完成，立即返回
                speech_rate=int(self.speed)
            )

            # 关键点 3: 真正的“等待”在这里，这是异步的，不占用线程
            result_file = await done_future
            self.close_file()
            return result_file

        except Exception as e:
            self.close_file()
            # 如果是 start 阶段就报错（比如握手失败），直接抛出
            raise e

    async def restful_request(self, t):
        host = "https://nls-gateway-cn-shanghai.aliyuncs.com/stream/v1/tts"
        body = {
            "appkey": self.APPKEY,
            "text": t,
            "token": self.TOKEN,
            "format": "wav",
            "volume": self.volume,
            "speech_rate": self.speed,
            "voice": self.voice,
        }
        resp_bytes = None
        retry = 5
        async with httpx.AsyncClient() as client:
            for i in range(retry):
                try:
                    msg = await client.post(host, json=body, headers={},  # 添加 headers 参数
                                            timeout=60.0)
                    content_type = msg.headers.get("Content-Type", "")
                    request_id = msg.headers.get("X-NLS-RequestId", None)
                    if msg.status_code == 200:
                        if content_type == "application/json":
                            log.info(f"第{i+1}次请求失败, request_id: {request_id}, resp: {msg.json()}")
                            raise ServiceException(508, "阿里云TTS服务异常")
                        else:
                            log.info(f"请求成功")
                            resp_bytes = msg.content
                        break
                except Exception as e:
                    log.info(f"请求超时/失败:{e}")
                if i < retry - 1:
                    log.info(f"10秒后重试...")
                    await asyncio.sleep(10)
                else:
                    log.error(f"重试{retry}次后仍失败")
                    raise ServiceException(509, "阿里云TTS服务请求失败")
        # 存到文件
        with open(self.__test_file, "wb") as f:
            f.write(resp_bytes)
        return self.__test_file

async def multiruntest(num=500):
    tasks = []
    for i in range(0, num):
        name = "thread" + str(i)
        t = AliTTS(name, f"test{i}.wav")
        tasks.append(asyncio.to_thread(t.start, TEXT))
    result = await asyncio.gather(*tasks)
    return result


if __name__ == "__main__":
    nls.enableTrace(True)
    t1 = time.time()
    print(f"start time at {t1}")
    result = asyncio.run(multiruntest(3))
    t2 = time.time()
    print(f"end time at {t2}, last time: {t2-t1} s")
    print(result)
