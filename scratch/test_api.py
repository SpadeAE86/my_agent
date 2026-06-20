import asyncio
import httpx

async def test():
    url = "https://show.bilibili.com/api/ticket/project/infoByDate?id=1001476&date=2026-06-18"
    
    python_headers = {
        "accept": "*/*",
        "content-type": "application/json",
        "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
        "origin": "https://show.bilibili.com",
        "referer": "https://show.bilibili.com/",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "sec-fetch-site": "same-origin",
        "sec-fetch-mode": "cors",
        "sec-fetch-dest": "empty",
    }
    
    async with httpx.AsyncClient() as client:
        r = await client.get(url, headers=python_headers)
        res = r.json()
        print("Success:", res.get("success"), "Code:", res.get("code"), "Msg:", res.get("message"))
        data = res.get("data", {})
        print("Type of data:", type(data))
        if isinstance(data, dict):
            print("Keys of data:", list(data.keys()))
            for k, v in data.items():
                print(f"  {k}: {type(v)}")
                if isinstance(v, list) and len(v) > 0:
                    print(f"    First element type: {type(v[0])}")
                    if isinstance(v[0], dict):
                        print(f"    First element keys: {list(v[0].keys())}")
        elif isinstance(data, list):
            print("Length of list data:", len(data))
            if len(data) > 0:
                print("First element:", data[0])

if __name__ == "__main__":
    asyncio.run(test())
