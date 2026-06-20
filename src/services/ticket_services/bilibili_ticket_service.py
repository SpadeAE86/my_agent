import json
import asyncio
import time
from typing import List, Dict, Any, Optional
from sqlmodel import select
from infra.storage.mysql_connector import mysql_connector
from models.sqlmodel.ticket_buyer import TicketBuyer
from utils.post_utils import get, post
from infra.logging.logger import logger as log

# Bilibili default headers for show api
TICKET_HEADERS = {
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

async def get_show_list(
    page: int = 1,
    pagesize: int = 20,
    area: str = "310000",
    p_type: str = "展览",
    style: int = 2,
    location: str = "121.4768,31.2243",
    start_time: str = "",
    end_time: str = ""
) -> List[Dict[str, Any]]:
    """
    获取漫展列表，并返回更精简的结构，方便前端展示
    """
    url = "https://show.bilibili.com/api/ticket/project/listV2"
    params = {
        "version": "134",
        "page": page,
        "pagesize": pagesize,
        "platform": "web",
        "area": area,
        "p_type": p_type,
        "style": style,
        "start_time": start_time,
        "end_time": end_time,
        "location": location,
    }
    
    response_data = await get(url, params=params, headers=TICKET_HEADERS, task_id="get_show_list")
    if not response_data or response_data.get("code") != 0:
        log.warning(f"Failed to fetch show list: {response_data}")
        return []
        
    result_list = response_data.get("data", {}).get("result", [])
    simplified_list = []
    
    for item in result_list:
        # 1. 票价范围 (price range) - 转换为元
        price_low = item.get("price_low", 0) / 100
        price_high = item.get("price_high", 0) / 100
        price_range = f"{price_low} - {price_high}" if price_low != price_high else f"{price_low}"
        
        # 2. 地点 (location)
        city = item.get("city", "")
        district = item.get("district_name", "")
        venue = item.get("venue_name", "")
        location_desc = f"{city} {district} {venue}".strip()
        
        # 3. 封面图 (cover)
        cover_url = item.get("cover", "")
        if cover_url and cover_url.startswith("//"):
            cover_url = f"https:{cover_url}"
            
        # 4. 票的状态 (ticket status)
        ticket_status = item.get("sale_flag", "")
        countdown = item.get("countdown", "")
        if countdown:
            ticket_status = f"{ticket_status} ({countdown})"
            
        simplified_item = {
            "project_id": item.get("project_id") or item.get("id"),
            "project_name": item.get("project_name", ""),
            "location": location_desc,
            "time_range": item.get("tlabel", ""),
            "start_time": item.get("start_time", ""),
            "end_time": item.get("end_time", ""),
            "price_low": price_low,
            "price_high": price_high,
            "price_range": price_range,
            "description": item.get("sale_point", "") or item.get("words", ""),
            "ticket_status": ticket_status,
            "cover": cover_url
        }
        simplified_list.append(simplified_item)
        
    return simplified_list

async def add_buyer(buyer_name: str, tel: str, sessdata: str, device_id: Optional[str] = None) -> TicketBuyer:
    """
    添加或更新买票人信息到数据库
    """
    async with mysql_connector.session_scope() as session:
        statement = select(TicketBuyer).where(TicketBuyer.buyer_name == buyer_name, TicketBuyer.tel == tel)
        results = await session.execute(statement)
        buyer = results.scalar_one_or_none()
        
        if buyer:
            buyer.sessdata = sessdata
            if device_id:
                buyer.device_id = device_id
        else:
            buyer = TicketBuyer(
                buyer_name=buyer_name,
                tel=tel,
                sessdata=sessdata,
                device_id=device_id or "b91800ce78f2a77070a3ab35c8f3227c"
            )
        session.add(buyer)
        await session.commit()
        await session.refresh(buyer)
        return buyer

async def get_buyer_by_id(buyer_id: int) -> Optional[TicketBuyer]:
    """
    根据 ID 获取买票人信息
    """
    async with mysql_connector.session_scope() as session:
        return await session.get(TicketBuyer, buyer_id)

async def list_buyers() -> List[TicketBuyer]:
    """
    列出所有买票人信息
    """
    async with mysql_connector.session_scope() as session:
        statement = select(TicketBuyer)
        results = await session.execute(statement)
        return list(results.scalars().all())

async def run_buy_flow_with_db_buyer(ticket_id: int, buyer_id: int) -> bool:
    """
    利用数据库中存储的买票人信息运行买票流程，仅保留在后端
    """
    buyer = await get_buyer_by_id(buyer_id)
    if not buyer:
        log.error(f"Buyer with ID {buyer_id} not found in database.")
        return False
        
    cookies = {
        "SESSDATA": buyer.sessdata
    }
    
    # 1. 获取演出基本信息
    ticket_abstract_url = rf"https://show.bilibili.com/api/ticket/project/getV2?version=134&id={ticket_id}&project_id={ticket_id}&requestSource=pc-new"
    abstract_data = await get(ticket_abstract_url, headers=TICKET_HEADERS, task_id="ticket_date_request", cookies=cookies)
    abstract_info = abstract_data.get("data", {})
    if not abstract_info:
        log.warning("还未开票，未获取到演出信息")
        return False
        
    sales_dates = abstract_info.get("sales_dates", [])
    if not sales_dates:
        log.warning("未开售，无法获取 token")
        return False
        
    # 2. 获取前两日的票务场次列表
    ticket_info = {}
    for date_item in sales_dates[:2]:
        date = date_item["date"]
        ticket_info_bydate_url = rf"https://show.bilibili.com/api/ticket/project/infoByDate?id={ticket_id}&date={date}"
        screen_list_data = await get(ticket_info_bydate_url, headers=TICKET_HEADERS, task_id="ticket_info_bydate_request", cookies=cookies)
        if screen_list_data and screen_list_data.get("data"):
            screen_list = screen_list_data["data"].get("screen_list", [])
            ticket_info[date] = screen_list
        await asyncio.sleep(1)
        
    if not ticket_info:
        log.warning("没有可用的票务信息")
        return False
        
    # 3. 过滤并获取下单 token
    ticket_summary = {}
    for date, screen_list in ticket_info.items():
        ticket_list = []
        for screen in screen_list:
            if screen["saleFlag"]["number"] == 3:  # 已售罄
                continue
            screen_id = screen["id"]
            sku_id = screen["ticket_list"][0]["id"]
            price = screen["ticket_list"][0]["price"]
            
            payload = {
                "project_id": ticket_id,
                "screen_id": screen_id,
                "count": 1,
                "ignoreRequestLimit": True,
                "pay_money": price,
                "order_type": 1,
                "timestamp": int(time.time() * 1000),
                "deviceId": buyer.device_id or "b91800ce78f2a77070a3ab35c8f3227c",
                "buyer": buyer.buyer_name,
                "tel": buyer.tel,
                "sku_id": sku_id,
                "clickPosition": "{\"x\":994,\"y\":798,\"origin\":1781680637198,\"now\":1781680653686}",
                "token": "",
                "newRisk": True,
                "requestSource": "pc-new"
            }
            request_token_url = rf"https://show.bilibili.com/api/ticket/order/prepare?project_id={ticket_id}"
            token_data = await post(request_token_url, payload, headers=TICKET_HEADERS, task_id="token_info_request", cookies=cookies)
            
            if token_data and token_data.get("data") and token_data["data"].get("token"):
                token = token_data["data"]["token"]
                ticket_list.append({
                    "screen_id": screen_id,
                    "sku_id": sku_id,
                    "screen_name": screen["ticket_list"][0]["screen_name"],
                    "token": token,
                    "price": price
                })
        ticket_summary[date] = ticket_list
        
    # 4. 尝试自动购买第一个可用的票档
    origin_time = time.time() * 1000
    for date, tickets in ticket_summary.items():
        for ticket in tickets:
            token = ticket["token"]
            if not token:
                continue
                
            log.info(f"日期: {date}, 票名: {ticket['screen_name']}, 查询到有票，准备购买...")
            now = time.time() * 1000
            click_position = {
                "x": 994 + int((now - origin_time) / 1000) % 20,
                "y": 798 + int((now - origin_time) / 1000) % 20,
                "origin": origin_time,
                "now": now
            }
            
            buy_payload = {
                "project_id": ticket_id,
                "screen_id": ticket["screen_id"],
                "count": 1,
                "pay_money": ticket["price"],
                "order_type": 1,
                "timestamp": now,
                "deviceId": buyer.device_id or "b91800ce78f2a77070a3ab35c8f3227c",
                "buyer": buyer.buyer_name,
                "tel": buyer.tel,
                "sku_id": ticket["sku_id"],
                "clickPosition": json.dumps(click_position, separators=(",", ":")),
                "token": token,
                "newRisk": True,
                "requestSource": "pc-new"
            }
            
            buy_ticket_url = rf"https://show.bilibili.com/api/ticket/order/createV2?project_id={ticket_id}"
            buy_response = await post(buy_ticket_url, buy_payload, headers=TICKET_HEADERS, task_id="buy_ticket_request", cookies=cookies)
            
            log.info(f"购买响应: {json.dumps(buy_response, indent=4, ensure_ascii=False)}")
            if buy_response and buy_response.get("errno") == 0:
                log.info(f"购买成功！演出日期: {date}, 票名: {ticket['screen_name']}")
                return True
            elif buy_response and buy_response.get("errno") == 900001:
                log.info("前方拥堵，请重试")
            elif buy_response and buy_response.get("errno") == 100048:
                log.info("有尚未完成订单，点击查看")
                
            await asyncio.sleep(0.5)
            break  # 仅测试首张，避免重复下单
            
    return False

async def get_showcase_main_flow(
    page: int = 1,
    pagesize: int = 20,
    area: str = "310000",
    p_type: str = "展览",
    style: int = 2,
    location: str = "121.4768,31.2243",
    start_time: str = "",
    end_time: str = ""
) -> List[Dict[str, Any]]:
    """
    漫展资讯主流程函数：获取精简的漫展列表并打印，便于未来前端调用展示
    """
    log.info(f"开始获取漫展列表主流程... 参数: page={page}, pagesize={pagesize}, area={area}, p_type={p_type}")
    shows = await get_show_list(
        page=page,
        pagesize=pagesize,
        area=area,
        p_type=p_type,
        style=style,
        location=location,
        start_time=start_time,
        end_time=end_time
    )
    log.info(f"获取完成，共找到 {len(shows)} 条漫展资讯。")
    return shows

if __name__ == '__main__':
    async def run_test():
        # 本地直接测试获取前 3 条漫展资讯并打印
        shows = await get_showcase_main_flow(page=1, pagesize=3)
        print("\n" + "="*30 + " 漫展列表简化数据展示 " + "="*30)
        print(json.dumps(shows, indent=4, ensure_ascii=False))
        print("="*80 + "\n")

    asyncio.run(run_test())

