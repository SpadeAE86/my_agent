import json, asyncio
from tokenize import cookie_re

from aiohttp import payload_type
import time
from utils.post_utils import get, post

ticket_info_url = "https://mall.bilibili.com/mall-search-items/items_detail/info"
# payload = {
#     "itemsId": items_id,
#     "itemsDetailPageType": 3,
#     "from": "pc_ticketlist"
# }
#
# response_data = await post(ticket_info_url, payload, headers=header, task_id="ticket_info_request", cookies = cookies)
# response = response_data.get("data", {}).get("screenList", [])
origin = time.time() * 1000
person_name = "林诺诚"
telephone = "15618435583"

session = "35ca2942%2C1795675962%2Cc1803%2A52CjCWJ2B1N9IeYqBkfEoFD-ol2h9s02d6UDtXl7CypyM2AEXU3aejtZLbS3wBfRpaWFUSVkV1dmpyTmZiOG1meDBvSEd6aWhwZ0V5SXd3Wmxsa0U4MFd0anFwZ3UtekpfSGJNbUxKd19pM0FTVUJ2eEtmV016Ykk5MEp6bE8wTi0zWEMxZzNvSmxnIIEC"
header = {
    "accept": "*/*",
    "content-type": "application/json",
    "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
    "origin": "https://show.bilibili.com",
    "referer": "https://show.bilibili.com/",
    "user-agent": "Mozilla/5.0 (...)",
    "sec-fetch-site": "same-origin",
    "sec-fetch-mode": "cors",
    "sec-fetch-dest": "empty",
}
cookies = {
    "SESSDATA": session
}
async def get_ticket_info(items_id: int):
    ticket_abstract_url = rf"https://show.bilibili.com/api/ticket/project/getV2?version=134&id={items_id}&project_id={items_id}&requestSource=pc-new"
    print("GET", ticket_abstract_url)
    abstract_data = await get(ticket_abstract_url, headers=header, task_id="ticket_date_request", cookies=cookies)

    abstract_info = abstract_data.get("data", {})
    if not abstract_info:
        print("当前信息:")
        print(json.dumps(abstract_data, indent=4, ensure_ascii=False))
        print("还未开票，未获取到演出信息")
        return []
    abstract = {
        "name": abstract_info["name"],
        "description": abstract_info["description"],
        "sales_dates": abstract_info.get("sales_dates", []),
        "project_label": abstract_info["project_label"]
    }

    print("演出信息:")
    for k, v in abstract.items():
        print(f"{k}: {v}")
    response = {}
    if not abstract["sales_dates"]:
        print("未开售，无法获取token")
        return None
    else:
        for date_item in abstract["sales_dates"][:3]:  # 只获取前三天的票务信息，避免请求过多
            print(f"日期: {date_item}")
            date = date_item["date"]
            ticket_info_bydate_url = rf"https://show.bilibili.com/api/ticket/project/infoByDate?id={items_id}&date={date}"
            screen_list_data = await get(ticket_info_bydate_url, headers=header, task_id="ticket_info_bydate_request", cookies=cookies)
            screen_list = screen_list_data["data"]["screen_list"][0]
            if screen_list["saleFlag"]["number"] == 1:
                return response
            response[date] = screen_list
            print("debug:\n", json.dumps(screen_list, indent=4, ensure_ascii=False))
            await asyncio.sleep(1)

    return response

async def get_token_info(project_id: int, screen_id: int, sku_id: int):
    payload = {
        "project_id": project_id,
        "screen_id": screen_id,
        "count": 1,
        "ignoreRequestLimit": True,
        "pay_money": 3000,
        "order_type": 1,
        "timestamp": 1781680653686,
        "deviceId": "b91800ce78f2a77070a3ab35c8f3227c",
        "buyer": "林诺诚",
        "tel": "15618435583",
        "sku_id": sku_id,
        "clickPosition": "{\"x\":994,\"y\":798,\"origin\":1781680637198,\"now\":1781680653686}",
        "token": "wGoySf4AD0eCAA9TswEAAQANYJ0.",
        "newRisk": True,
        "requestSource": "pc-new"
    }

    request_token_url = rf"https://show.bilibili.com/api/ticket/order/prepare?project_id={project_id}"
    response = await post(request_token_url, payload, headers=header, task_id="token_info_request", cookies=cookies)
    return response

async def get_ticket_status(ticket_id: int, ticket_info):

    ticket_summary = {}
    if not ticket_info:
        print("未提供票务信息，无法获取票务状态")
        return ticket_summary
    for date, screen in ticket_info.items():
        ticket_list = []
        screen_id = screen["id"]
        for ticket in screen["ticket_list"]:
            if ticket["sale_flag"]["number"] == 3:
                print(f"{date} {ticket['desc']} 已售罄")
                continue

            sku_id = ticket["id"]
            token_data = await get_token_info(ticket_id, screen_id, sku_id)
            token = token_data["data"]["token"]
            ticket_status = {
                "screen_id": screen_id,
                "sku_id": sku_id,
                "screen_name": ticket["screen_name"],
                "token": token,
                "price": ticket["price"],
                "desc": ticket["desc"]
            }
            ticket_list.append(ticket_status)
        ticket_summary[date] = ticket_list
    return ticket_summary

async def buy_ticket(date, sku_id, project_id, screen_id, token, price, again=0):
    print(f"准备购买， 票价{price/100}元，日期{date}")

    now = time.time() * 1000
    click_position = {
        "x": 994 + int((now - origin) / 1000) % 20,  # 模拟点击位置的变化
        "y": 798 + int((now - origin) / 1000) % 20,  # 模拟点击位置的变化
        "origin": origin,
        "now": now
    }
    payload = {
        "project_id": project_id,
        "screen_id": screen_id,
        "count": 1,
        "pay_money": price,
        "order_type": 1,
        "timestamp": now,
        "deviceId": "b91800ce78f2a77070a3ab35c8f3227c",
        "buyer": person_name,
        "tel": telephone,
        "sku_id": sku_id,
        "clickPosition": json.dumps(click_position, separators=(",", ":")),
        "token": token,
        "newRisk": True,
        "requestSource": "pc-new"
    }
    if again > 0:
        payload["again"]=1
    buy_ticket_url = rf"https://show.bilibili.com/api/ticket/order/createV2?project_id={project_id}"
    response = await post(buy_ticket_url, payload, headers=header, task_id="buy_ticket_request", cookies=cookies)
    return response

async def buy_first_available_ticket(ticket_summary, ticket_id, dates_to_buy=3, budget_per_ticket=13000):
    buy_count = 0
    if not dates_to_buy:
        return buy_count
    for date, tickets in ticket_summary.items():
        if tickets:
            for ticket in tickets:
                token = ticket["token"]
                if not token:
                    print(f"日期: {date}, 票名: {ticket['desc']}, 票价{ticket['price']/100}, 未获取到token，无法购买")
                    continue
                if ticket["price"] > budget_per_ticket:
                    print(f"日期: {date}, 票名: {ticket['desc']}, 票价{ticket['price']/100}元超过预算，跳过购买")
                    continue
                print(f"日期: {date}, 票名: {ticket['desc']}, 票价{ticket['price']/100}, 查询到有票，准备购买...")
                buy_response = await buy_ticket(date, ticket["sku_id"], ticket_id, ticket["screen_id"], token, ticket["price"], again=buy_count)
                print(f"购买响应: {json.dumps(buy_response, indent=4, ensure_ascii=False)}")
                if buy_response.get("errno") == 0:
                    print(f"购买成功！演出日期: {date}, 票名: {ticket['desc']}, 票价{ticket['price']/100}, 跳过当天剩余票")
                    buy_count += 1
                    dates_to_buy -= 1
                    break
                elif buy_response.get("errno") == 900001:
                    print(f"前方拥堵，请重试")
                elif buy_response.get("errno") == 100048:
                    print(f"有尚未完成订单，点击查看")
                await asyncio.sleep(0.5)  # 模拟用户操作间隔
        if not dates_to_buy:
             break
    return buy_count

async def buy_ticket_main_flow(ticket_id: int):
    ticket_info = await get_ticket_info(ticket_id)
    retry_count = 0
    while not ticket_info and retry_count < 10:
        retry_count += 1
        print(f"未获取到票务信息，可能未开售，1秒后重试 ({retry_count}/10)...")
        await asyncio.sleep(1)
        ticket_info = await get_ticket_info(ticket_id)
        
    if not ticket_info:
        print("已达到最大重试次数（10次），未获取到任何票务信息。任务终止。")
        return 0
        
    summary = await get_ticket_status(ticket_id, ticket_info)
    summary_display = json.dumps(summary, indent=4, ensure_ascii=False)
    print("余票情况:")
    print(summary_display)
    bought_ticket_amounts = await buy_first_available_ticket(summary, ticket_id, dates_to_buy=3)
    return bought_ticket_amounts


test_ticket_id = 1001653
test_screen_id = 1005651
test_sku_id = 876701

if __name__ == '__main__':
    bought_ticket = asyncio.run(buy_ticket_main_flow(test_ticket_id))
    print(f"测试完成, 购买{'成功' if bought_ticket else '失败'}, 共购买到{bought_ticket}张票")

