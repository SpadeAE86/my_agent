from transformers.integrations.flash_attention import get_target_dtype
from urllib3.util.request import body_to_chunks

from utils.post_utils import post, get
import httpx
from bs4 import BeautifulSoup
import asyncio

async def get_text(host, params=None, headers=None):
    async with httpx.AsyncClient() as client:
        msg = await client.get(
            host,
            params=params,
            headers=headers
        )

        return msg
get_target= "https://ecp.sgcc.com.cn/ecp2.0/portal/#/doc/doci-win/2605228014091430_2018060501171107"
headers = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "zh-CN,zh;q=0.9",
    "priority": "u=1, i",
    "sec-ch-ua": "\"Google Chrome\";v=\"141\", \"Not?A_Brand\";v=\"8\", \"Chromium\";v=\"141\"",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "\"macOS\"",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"
}

url_to_crawl = [
    "https://ecp.sgcc.com.cn/ecp2.0/ecpwcmcore//index/searchList",
    "https://www.zjdlzb.com/",
    "http://ec.chng.com.cn",
    "https://www.hascfjtyl.com/",
    "https://ecp.chinasalt.com.cn/"
]

body = {"index": "1", "size": 20, "searchValue": "", "keyWord": "电缆"}

async def try_get(url: str):
    try:
        response = await get_text(url, body, headers=headers)
        if response:
            print(f"{response}")
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                print(f"title: {soup.title.string if soup.title else 'No title found'}")
                print(f"content: {soup.get_text()}")
        else:
            print(f"Failed to crawl {url}, status code: {response.status_code}")
    except Exception as e:
        print(f"Error crawling {url}: {e}")

async def try_post(url: str):
    try:
        result = await post(url, body, headers=headers)
        if result:
            print(f"{result}")
        else:
            print(f"Failed to crawl {url}, status code: {result.status_code}")
    except Exception as e:
        print(f"Error crawling {url}: {e}")

"https://ecp.sgcc.com.cn/ecp2.0/portal/#/doc/{doctype}/{id}_{firstPageMenuId}"
if __name__ == "__main__":

    # tasks = [try_get(url) for url in url_to_crawl[:1]]
    # url = url_to_crawl[0]
    # asyncio.run(try_post(url))
    asyncio.run(try_get(get_target))

