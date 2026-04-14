cd 到 src文件夹
服务后端启动使用 uvicorn FastAPI_server:app --host 0.0.0.0 --port 8004 --reload

linux:
gunicorn FastAPI_server:app \
  -k uvicorn.workers.UvicornWorker \
  -b 0.0.0.0:8004 \
  -w 4
