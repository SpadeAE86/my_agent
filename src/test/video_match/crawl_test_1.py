import httpx

url = "https://ecp.sgcc.com.cn/ecp2.0/ecpwcmcore/index/getWinFile"
headers = {
    "accept": "application/json, text/plain, */*",
    "content-type": "application/json",
    "origin": "https://ecp.sgcc.com.cn",
    "referer": "https://ecp.sgcc.com.cn/ecp2.0/portal/",
    "user-agent": "Mozilla/5.0"
}

r = httpx.post(
    url,
    content="2605228014091430",
    headers=headers
)

print(r.text)