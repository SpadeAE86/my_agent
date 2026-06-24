from selenium import webdriver
from torch.xpu import device

options = webdriver.ChromeOptions()
options.add_argument(r"--user-data-dir=C:\temp\selenium_profile_3")
driver = webdriver.Chrome(options=options)
driver.get("https://www.bilibili.com")
session = "1be7b181%2C1797240686%2C34e07%2A62CjDUIrzp5CY_pnOQ8KBOmExYCqoVV9P-hsacAFoneTueaTUNWw_ABlC7KX13TB-C6DoSVkE0RW5JTm5FUTJtMnVUOU9YbEI1NlpMWVNyeUNhczhaMHl3MHRZTHNCSEtuTXZEZWdmTVNLLTBzQ3NtZGdCd00wMmcxVU9mS1h4UEFOb2dRcC1mOXVBIIEC"
account_token = "wGoyVPAAD0eCAA9UMQEAAQANYRs"

device = "pc_web"
ticket_id = 1001476
target_url = rf"https://show.bilibili.com/platform/detail.html?id={ticket_id}&from=pc_ticketlist&msource={device}"
confirm_url = rf"https://mall.bilibili.com/neul-next/ticket/confirmOrder.html?&token={account_token}.&project_id={ticket_id}&noTitleBar=1"
request_token_url = rf"https://show.bilibili.com/api/ticket/order/prepare?project_id=1001476"
driver.add_cookie({
    "name": "SESSDATA",
    "value": session,
    "domain": ".bilibili.com",   # 建议用主域
    "path": "/"
})
# 3. 刷新页面让 cookie 生效
driver.refresh()

driver.get(confirm_url)
input("enter to continue...")
#
# # 2. 添加 cookie
# driver.add_cookie({
#     "name": "_uuid",
#     "value": "B82B99310-10C38-9410A-53A8-937C2A1D556D64351infoc",
#     "domain": ".bilibili.com",   # 建议用主域
#     "path": "/"
# })
#
