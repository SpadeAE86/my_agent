import re
from typing import Dict, Any

class CategoryRouter:
    """
    根据标签的中文或英文文本，将其分流路由到不同的语义投影空间。
    支持的空间类别：Character (角色), Hair (发型发色), Clothing (服饰), Scenery (场景背景), General (通用)
    """
    def __init__(self):
        # 常见高频标签直接映射字典（中英双语）
        self.fast_map = {
            # 角色类
            "亚丝娜": "Character",
            "亚斯娜": "Character",
            "闪光亚丝娜": "Character",
            "闪光亚斯娜": "Character",
            "结城明日奈": "Character",
            "桐人": "Character",
            "桐谷和人": "Character",
            "雷电将军": "Character",
            "八重神子": "Character",
            "神里绫华": "Character",
            "影": "Character",
            
            # 发型类
            "双马尾": "Hair",
            "单马尾": "Hair",
            "金发": "Hair",
            "黑发": "Hair",
            "白毛": "Hair",
            "银发": "Hair",
            "短发": "Hair",
            "长发": "Hair",
            
            # 服饰类
            "过膝袜": "Clothing",
            "百褶裙": "Clothing",
            "水手服": "Clothing",
            "裤袜": "Clothing",
            "女仆装": "Clothing",
            "丝袜": "Clothing",
            
            # 场景类
            "蓝天白云": "Scenery",
            "落日余晖": "Scenery",
            "樱花树下": "Scenery",
            "教室内": "Scenery",
            "户外": "Scenery",
            "室内": "Scenery",
        }
        
        # 英文正则规则匹配 (常见 Danbooru 后缀与关键词)
        self.en_rules = [
            (re.compile(r".*_\(sao\)$", re.IGNORECASE), "Character"),
            (re.compile(r".*_\(genshin\)$", re.IGNORECASE), "Character"),
            (re.compile(r".*_\(re:zero\)$", re.IGNORECASE), "Character"),
            (re.compile(r".*(hair|twintails|ponytail|braid|drill|bob_cut|ahoge).*", re.IGNORECASE), "Hair"),
            (re.compile(r".*(skirt|socks|uniform|dress|suit|pantyhose|apron|costume|fuku).*", re.IGNORECASE), "Clothing"),
            (re.compile(r".*(sky|sunset|blossoms|background|scenery|cloud|scenic|street|room|classroom|forest).*", re.IGNORECASE), "Scenery"),
        ]

        # 中文关键词正则匹配规则
        self.cn_rules = [
            (re.compile(r".*(发|马尾|白毛|黑毛|金发|呆毛|卷发)$"), "Hair"),
            (re.compile(r".*(裙|袜|服|装|鞋|衬衫|制服|衣)$"), "Clothing"),
            (re.compile(r".*(天空|落日|夕阳|樱花|教室|背景|场景|房间|街道|树林|草地)$"), "Scenery"),
        ]

    def route(self, tag: str) -> str:
        """
        判断标签文本的分类
        """
        clean_tag = tag.strip().lower()
        if not clean_tag:
            return "General"
            
        # 1. 极速字典匹配
        if tag in self.fast_map:
            return self.fast_map[tag]
        if clean_tag in self.fast_map:
            return self.fast_map[clean_tag]
            
        # 2. 英文正则规则匹配
        for pattern, category in self.en_rules:
            if pattern.match(clean_tag):
                return category
                
        # 3. 中文正则规则匹配
        for pattern, category in self.cn_rules:
            if pattern.match(tag):
                return category
                
        # 4. 默认兜底分类
        return "General"
