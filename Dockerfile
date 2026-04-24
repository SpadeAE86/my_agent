# 使用自制的python3.12 包含ffmpeg rocketmq的基础镜像
FROM swr.cn-east-3.myhuaweicloud.com/freeuuu/python-agent:1.0


COPY . .


#从requirements.txt里安装依赖
RUN pip install --timeout=600 \
    -r requirements.txt 


WORKDIR /app/src

# 设置系统时区为上海
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 设置 Hugging Face 镜像加速，解决国内网络无法访问 Hugging Face 下载模型的问题
ENV HF_ENDPOINT=https://hf-mirror.com
ENV HF_HUB_ENABLE_HF_TRANSFER=1




# 暴露端口
EXPOSE 5000


# 启动命令
CMD ["uvicorn", "FastAPI_server:app", "--host", "0.0.0.0", "--port", "5000"]