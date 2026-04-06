import os
import shutil
import requests
from dotenv import load_dotenv

_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_current_dir)
load_dotenv(os.path.join(_src_dir, '.env'))

import yaml
from utils.log_utils import logger as log

from utils.file_utils import read_yaml, save_yaml

app_title = "video_mix"

local_audio_tts_providers = ['chatTTS', 'GPTSoVITS', 'CosyVoice']
local_audio_recognition_providers = ['fasterwhisper', 'sensevoice']
local_audio_recognition_fasterwhisper_module_names = ['large-v3', 'large-v2', 'large-v1', 'distil-large-v3',
                                                      'distil-large-v2', 'medium', 'base', 'small', 'tiny']
local_audio_recognition_fasterwhisper_device_types = ['cuda', 'cpu', 'auto']
local_audio_recognition_fasterwhisper_compute_types = ['int8', 'int8_float16', 'float16']

vpc = "/obs"  #vpc储存卷挂载路径
RESOURCE_DIR = "./resource"
FINAL_DIR = "./final"
FONT_DIR = "./font"
OUTPUT_DIR = "./work"

driver_types = {
    "chrome": 'chrome',
    "firefox": 'firefox'}

filter_options = ["temperature", "tint", "hue", "saturation", "brightness", "contrast", "sharpness", "gamma", "boxblur",
                  "gblur", "dblur"]

audio_speech_rate = {
    "kenny" : 4.651496641981914,
    "aifei" : 5.165382903618127,
    "aiqian" : 4.294418512571484,
    "maoxiaomei" : 4.947731915548115,
    "siyue" : 5.101272657175458,
    "xiaoxian" : 4.759277536672374,
    "zhimao" : 4.639149256865643,
    "zhixiaomei" : 4.509686234988655,
    "zhixiaoxia" : 4.537818839709418,
    "zhifeng_emo": 4.019146284436771,
    "zhibing_emo": 4.887035715216325,
    "zhimiao_emo": 4.737786722269002,
    "zhimi_emo": 4.583573593523443,
    "zhiyan_emo": 4.577448153779756,
    "zhibei_emo": 4.357534410051467,
    "zhitian_emo": 4.351227712210594
}

male_voice = ["Kenny(温暖男声)", "艾飞(激昂解说男声)"]
female_voice = ["艾倩(资讯女声)", "猫小美(活力女声)", "思悦(温柔女声)", "小仙(亲切女声)", "知猫(普通话女声)", "知小妹(直播数字人)", "知小夏(对话数字人)"]

emotion_option = ["neutral","happy","angry","sad","surprise"]
emotion_voice = {
    "知锋_多情感": "zhifeng_emo",
    "知冰_多情感": "zhibing_emo",
    "知妙_多情感": "zhimiao_emo",
    "知米_多情感": "zhimi_emo",
    "知燕_多情感": "zhiyan_emo",
    "知贝_多情感": "zhibei_emo",
    "知甜_多情感": "zhitian_emo"
}


douyin_site = "https://creator.douyin.com/creator-micro/content/upload"
shipinhao_site = "https://channels.weixin.qq.com/platform/post/create"
kuaishou_site = "https://cp.kuaishou.com/article/publish/video"
xiaohongshu_site = "https://creator.xiaohongshu.com/publish/publish?source=official"
bilibili_site = "https://member.bilibili.com/platform/upload/video/frame"

# 定义请求体的字符串限制 todo: 需要额外封装进某个专门储存字典的文件内
audio_speed = ["normal", "fast", "faster", "fastest", "slow", "slower", "slowest"]
volcano_voice_options = {
    #通用
    "冷酷哥哥": "zh_male_lengkugege_emo_v2_mars_bigtts",
    "甜心小美": "zh_female_tianxinxiaomei_emo_v2_mars_bigtts",
    "高冷御姐": "zh_female_gaolengyujie_emo_v2_mars_bigtts",
    "傲娇霸总": "zh_male_aojiaobazong_emo_v2_mars_bigtts",
    "广州德哥": "zh_male_guangzhoudege_emo_mars_bigtts",
    "京腔侃爷": "zh_male_jingqiangkanye_emo_mars_bigtts",
    "邻居阿姨": "zh_female_linjuayi_emo_v2_mars_bigtts",
    "优柔公子": "zh_male_yourougongzi_emo_v2_mars_bigtts",
    "儒雅男友": "zh_male_ruyayichen_emo_v2_mars_bigtts",
    "俊朗男友": "zh_male_junlangnanyou_emo_v2_mars_bigtts",
    "北京小爷": "zh_male_beijingxiaoye_emo_v2_mars_bigtts",
    "柔美女友": "zh_female_roumeinvyou_emo_v2_mars_bigtts",
    "阳光青年": "zh_male_yangguangqingnian_emo_v2_mars_bigtts",
    "魅力女友": "zh_female_meilinvyou_emo_v2_mars_bigtts",
    "爽快思思": "zh_female_shuangkuaisisi_emo_v2_mars_bigtts",
    #趣味口音
    "沪普男": "zh_male_hupunan_mars_bigtts",
    "粤语小溏": "zh_female_yueyunv_mars_bigtts",
    "鲁班七号": "zh_male_lubanqihao_mars_bigtts",
    "林潇": "zh_female_yangmi_mars_bigtts",
    "玲玲姐姐": "zh_female_linzhiling_mars_bigtts",
    "春日部姐姐": "zh_female_jiyejizi2_mars_bigtts",
    "唐僧": "zh_male_tangseng_mars_bigtts",
    "庄周": "zh_male_zhuangzhou_mars_bigtts",
    "猪八戒": "zh_male_zhubajie_mars_bigtts",
    "感冒电音姐姐": "zh_female_ganmaodianyin_mars_bigtts",
    "直率英子": "zh_female_naying_mars_bigtts",
    "女雷神": "zh_female_leidian_mars_bigtts",
    "豫州子轩": "zh_male_yuzhouzixuan_moon_bigtts",
    "呆萌川妹": "zh_female_daimengchuanmei_moon_bigtts",
    "广西远舟": "zh_male_guangxiyuanzhou_moon_bigtts",
    "双节棍小哥": "zh_male_zhoujielun_emo_v2_mars_bigtts",
    "湾湾小何": "zh_female_wanwanxiaohe_moon_bigtts",
    "湾区大叔": "zh_female_wanqudashu_moon_bigtts",
    "浩宇小哥": "zh_male_haoyuxiaoge_moon_bigtts",
    "妹坨洁儿": "zh_female_meituojieer_moon_bigtts",
    # 角色扮演
    "纯真少女": "ICL_zh_female_chunzhenshaonv_e588402fb8ad_tob",
    "奶气小生": "ICL_zh_male_xiaonaigou_edf58cf28b8b_tob",
    "精灵向导": "ICL_zh_female_jinglingxiangdao_1beb294a9e3e_tob",
    "闷油瓶小哥": "ICL_zh_male_menyoupingxiaoge_ffed9fc2fee7_tob",
    "黯刃秦主": "ICL_zh_male_anrenqinzhu_cd62e63dcdab_tob",
    "霸道总裁": "ICL_zh_male_badaozongcai_v1_tob",
    "妩媚可人": "ICL_zh_female_ganli_v1_tob",
    "邪魅御姐": "ICL_zh_female_xiangliangya_v1_tob",
    "嚣张小哥": "ICL_zh_male_ms_tob",
    "油腻大叔": "ICL_zh_male_you_tob",
    "孤傲公子": "ICL_zh_male_guaogongzi_v1_tob",
    "胡子叔叔": "ICL_zh_male_huzi_v1_tob",
    "性感魅惑": "ICL_zh_female_luoqing_v1_tob",
    "病弱公子": "ICL_zh_male_bingruogongzi_tob",
    "邪魅女王": "ICL_zh_female_bingjiao3_tob",
    "傲慢青年": "ICL_zh_male_aomanqingnian_tob",
    "醋精男生": "ICL_zh_male_cujingnansheng_tob",
    "爽朗少年": "ICL_zh_male_shuanglangshaonian_tob",
    "撒娇男友": "ICL_zh_male_sajiaonanyou_tob",
    "温柔男友": "ICL_zh_male_wenrounanyou_tob",
    "温顺少年": "ICL_zh_male_wenshunshaonian_tob",
    "粘人男友": "ICL_zh_male_naigounanyou_tob",
    "撒娇男生": "ICL_zh_male_sajiaonansheng_tob",
    "活泼男友": "ICL_zh_male_huoponanyou_tob",
    "甜系男友": "ICL_zh_male_tianxinanyou_tob",
    "活力青年": "ICL_zh_male_huoliqingnian_tob",
    "开朗青年": "ICL_zh_male_kailangqingnian_tob",
    "冷漠兄长": "ICL_zh_male_lengmoxiongzhang_tob",
    "天才同桌": "ICL_zh_male_tiancaitongzhuo_tob",
    "翩翩公子": "ICL_zh_male_pianpiangongzi_tob",
    "懵懂青年": "ICL_zh_male_mengdongqingnian_tob",
    "冷脸兄长": "ICL_zh_male_lenglianxiongzhang_tob",
    "病娇少年": "ICL_zh_male_bingjiaoshaonian_tob",
    "病娇男友": "ICL_zh_male_bingjiaonanyou_tob",
    "病弱少年": "ICL_zh_male_bingruoshaonian_tob",
    "意气少年": "ICL_zh_male_yiqishaonian_tob",
    "干净少年": "ICL_zh_male_ganjingshaonian_tob",
    "冷漠男友": "ICL_zh_male_lengmonanyou_tob",
    "精英青年": "ICL_zh_male_jingyingqingnian_tob",
    "热血少年": "ICL_zh_male_rexueshaonian_tob",
    "清爽少年": "ICL_zh_male_qingshuangshaonian_tob",
    "中二青年": "ICL_zh_male_zhongerqingnian_tob",
    "凌云青年": "ICL_zh_male_lingyunqingnian_tob",
    "自负青年": "ICL_zh_male_zifuqingnian_tob",
    "不羁青年": "ICL_zh_male_bujiqingnian_tob",
    "儒雅君子": "ICL_zh_male_ruyajunzi_tob",
    "低音沉郁": "ICL_zh_male_diyinchenyu_tob",
    "冷脸学霸": "ICL_zh_male_lenglianxueba_tob",
    "儒雅总裁": "ICL_zh_male_ruyazongcai_tob",
    "深沉总裁": "ICL_zh_male_shenchenzongcai_tob",
    "小侯爷": "ICL_zh_male_xiaohouye_tob",
    "孤高公子": "ICL_zh_male_gugaogongzi_tob",
    "仗剑君子": "ICL_zh_male_zhangjianjunzi_tob",
    "温润学者": "ICL_zh_male_wenrunxuezhe_tob",
    "亲切青年": "ICL_zh_male_qinqieqingnian_tob",
    "温柔学长": "ICL_zh_male_wenrouxuezhang_tob",
    "高冷总裁": "ICL_zh_male_gaolengzongcai_tob",
    "冷峻高智": "ICL_zh_male_lengjungaozhi_tob",
    "孱弱少爷": "ICL_zh_male_chanruoshaoye_tob",
    "自信青年": "ICL_zh_male_zixinqingnian_tob",
    "青涩青年": "ICL_zh_male_qingseqingnian_tob",
    "学霸同桌": "ICL_zh_male_xuebatongzhuo_tob",
    "冷傲总裁": "ICL_zh_male_lengaozongcai_tob",
    "元气少年": "ICL_zh_male_yuanqishaonian_tob",
    "洒脱青年": "ICL_zh_male_satuoqingnian_tob",
    "直率青年": "ICL_zh_male_zhishuaiqingnian_tob",
    "斯文青年": "ICL_zh_male_siwenqingnian_tob",
    "俊逸公子": "ICL_zh_male_junyigongzi_tob",
    "仗剑侠客": "ICL_zh_male_zhangjianxiake_tob",
    "机甲智能": "ICL_zh_male_jijiaozhineng_tob",
    "奶气萌娃": "zh_male_naiqimengwa_mars_bigtts",
    "婆婆": "zh_female_popo_mars_bigtts",
    "高冷御姐2": "zh_female_gaolengyujie_moon_bigtts",
    "傲娇霸总2": "zh_male_aojiaobazong_moon_bigtts",
    "魅力女友2": "zh_female_meilinvyou_moon_bigtts",
    "深夜播客": "zh_male_shenyeboke_moon_bigtts",
    "柔美女友2": "zh_female_sajiaonvyou_moon_bigtts",
    "撒娇学妹": "zh_female_yuanqinvyou_moon_bigtts",
    "病弱少女": "ICL_zh_female_bingruoshaonv_tob",
    "活泼女孩": "ICL_zh_female_huoponvhai_tob",
    "东方浩然": "zh_male_dongfanghaoran_moon_bigtts",
    "绿茶小哥": "ICL_zh_male_lvchaxiaoge_tob",
    "娇弱萝莉": "ICL_zh_female_jiaoruoluoli_tob",
    "冷淡疏离": "ICL_zh_male_lengdanshuli_tob",
    "憨厚敦实": "ICL_zh_male_hanhoudunshi_tob",
    "活泼刁蛮": "ICL_zh_female_huopodiaoman_tob",
    "固执病娇": "ICL_zh_male_guzhibingjiao_tob",
    "撒娇粘人": "ICL_zh_male_sajiaonianren_tob",
    "傲慢娇声": "ICL_zh_female_aomanjiaosheng_tob",
    "潇洒随性": "ICL_zh_male_xiaosasuixing_tob",
    "诡异神秘": "ICL_zh_male_guiyishenmi_tob",
    "儒雅才俊": "ICL_zh_male_ruyacaijun_tob",
    "正直青年": "ICL_zh_male_zhengzhiqingnian_tob",
    "娇憨女王": "ICL_zh_female_jiaohannvwang_tob",
    "病娇萌妹": "ICL_zh_female_bingjiaomengmei_tob",
    "青涩小生": "ICL_zh_male_qingsenaigou_tob",
    "纯真学弟": "ICL_zh_male_chunzhenxuedi_tob",
    "优柔帮主": "ICL_zh_male_youroubangzhu_tob",
    "优柔公子2": "ICL_zh_male_yourougongzi_tob",
    "调皮公主": "ICL_zh_female_tiaopigongzhu_tob",
    "贴心男友": "ICL_zh_male_tiexinnanyou_tob",
    "少年将军": "ICL_zh_male_shaonianjiangjun_tob",
    "病娇哥哥": "ICL_zh_male_bingjiaogege_tob",
    "学霸男同桌": "ICL_zh_male_xuebanantongzhuo_tob",
    "幽默叔叔": "ICL_zh_male_youmoshushu_tob",
    "假小子": "ICL_zh_female_jiaxiaozi_tob",
    "温柔男同桌": "ICL_zh_male_wenrounantongzhuo_tob",
    "幽默大爷": "ICL_zh_male_youmodaye_tob",
    "枕边低语": "ICL_zh_male_asmryexiu_tob",
    "神秘法师": "ICL_zh_male_shenmifashi_tob",
    "娇喘女声": "zh_female_jiaochuan_mars_bigtts",
    "开朗弟弟": "zh_male_livelybro_mars_bigtts",
    "谄媚女声": "zh_female_flattery_mars_bigtts",
    "冷峻上司": "ICL_zh_male_lengjunshangsi_tob",
    #角色扮演 S2S-SC
    "醋精男友": "ICL_zh_male_cujingnanyou_tob",
    "风发少年": "ICL_zh_male_fengfashaonian_tob",
    "磁性男嗓": "ICL_zh_male_cixingnansang_tob",
    "成熟总裁": "ICL_zh_male_chengshuzongcai_tob",
    "傲娇精英": "ICL_zh_male_aojiaojingying_tob",
    "傲娇公子": "ICL_zh_male_aojiaogongzi_tob",
    "霸道少爷": "ICL_zh_male_badaoshaoye_tob",
    "腹黑公子": "ICL_zh_male_fuheigongzi_tob",
    "暖心学姐": "ICL_zh_female_nuanxinxuejie_tob",
    "可爱女生": "ICL_zh_female_keainvsheng_tob",
    "成熟姐姐": "ICL_zh_female_chengshujiejie_tob",
    "病娇姐姐": "ICL_zh_female_bingjiaojiejie_tob",
    "妩媚御姐": "ICL_zh_female_wumeiyujie_tob",
    "傲娇女友": "ICL_zh_female_aojiaonvyou_tob",
    "贴心女友": "ICL_zh_female_tiexinnvyou_tob",
    "性感御姐": "ICL_zh_female_xingganyujie_tob",
    "病娇弟弟": "ICL_zh_male_bingjiaodidi_tob",
    "傲慢少爷": "ICL_zh_male_aomanshaoye_tob",
    "傲气凌人": "ICL_zh_male_aiqilingren_tob",
    "病娇白莲": "ICL_zh_male_bingjiaobailian_tob",
    #客服
    "理性圆子": "ICL_zh_female_lixingyuanzi_cs_tob",
    "清甜桃桃": "ICL_zh_female_qingtiantaotao_cs_tob",
    "清晰小雪": "ICL_zh_female_qingxixiaoxue_cs_tob",
    "清甜莓莓": "ICL_zh_female_qingtianmeimei_cs_tob",
    "开朗婷婷": "ICL_zh_female_kailangtingting_cs_tob",
    "清新沐沐": "ICL_zh_male_qingxinmumu_cs_tob",
    "爽朗小阳": "ICL_zh_male_shuanglangxiaoyang_cs_tob",
    "清新波波": "ICL_zh_male_qingxinbobo_cs_tob",
    "温婉珊珊": "ICL_zh_female_wenwanshanshan_cs_tob",
    "甜美小雨": "ICL_zh_female_tianmeixiaoyu_cs_tob",
    "热情艾娜": "ICL_zh_female_reqingaina_cs_tob",
    "甜美小橘": "ICL_zh_female_tianmeixiaoju_cs_tob",
    "沉稳明仔": "ICL_zh_male_chenwenmingzai_cs_tob",
    "亲切小卓": "ICL_zh_male_qinqiexiaozhuo_cs_tob",
    "灵动欣欣": "ICL_zh_female_lingdongxinxin_cs_tob",
    "乖巧可儿": "ICL_zh_female_guaiqiaokeer_cs_tob",
    "暖心茜茜": "ICL_zh_female_nuanxinqianqian_cs_tob",
    "软萌团子": "ICL_zh_female_ruanmengtuanzi_cs_tob",
    "阳光洋洋": "ICL_zh_male_yangguangyangyang_cs_tob",
    "软萌糖糖": "ICL_zh_female_ruanmengtangtang_cs_tob",
    "秀丽倩倩": "ICL_zh_female_xiuliqianqian_cs_tob",
    "开心小鸿": "ICL_zh_female_kaixinxiaohong_cs_tob",
    "轻盈朵朵": "ICL_zh_female_qingyingduoduo_cs_tob",
    "暖阳女声": "zh_female_kefunvsheng_mars_bigtts",
    #视频配音
    "悠悠君子": "zh_male_M100_conversation_wvae_bigtts",
    "文静毛毛": "zh_female_maomao_conversation_wvae_bigtts",
    "倾心少女": "ICL_zh_female_qiuling_v1_tob",
    "醇厚低音": "ICL_zh_male_buyan_v1_tob",
    "咆哮小哥": "ICL_zh_male_BV144_paoxiaoge_v1_tob",
    "和蔼奶奶": "ICL_zh_female_heainainai_tob",
    "邻居阿姨2": "ICL_zh_female_linjuayi_tob",
    "温柔小雅": "zh_female_wenrouxiaoya_moon_bigtts",
    "天才童声": "zh_male_tiancaitongsheng_mars_bigtts",
    "猴哥": "zh_male_sunwukong_mars_bigtts",
    "熊二": "zh_male_xionger_mars_bigtts",
    "佩奇猪": "zh_female_peiqi_mars_bigtts",
    "武则天": "zh_female_wuzetian_mars_bigtts",
    "顾姐": "zh_female_gujie_mars_bigtts",
    "樱桃丸子": "zh_female_yingtaowanzi_mars_bigtts",
    "广告解说": "zh_male_chunhui_mars_bigtts",
    "少儿故事": "zh_female_shaoergushi_mars_bigtts",
    "四郎": "zh_male_silang_mars_bigtts",
    "俏皮女声": "zh_female_qiaopinvsheng_mars_bigtts",
    "懒音绵宝": "zh_male_lanxiaoyang_mars_bigtts",
    "亮嗓萌仔": "zh_male_dongmanhaimian_mars_bigtts",
    "磁性解说男声": "zh_male_jieshuonansheng_mars_bigtts",
    "鸡汤妹妹": "zh_female_jitangmeimei_mars_bigtts",
    "贴心女声": "zh_female_tiexinnvsheng_mars_bigtts",
    "萌丫头": "zh_female_mengyatou_mars_bigtts",
    #新增
    "Vivi": "zh_female_vv_mars_bigtts",
    "机灵小伙": "ICL_zh_male_shenmi_v1_tob",
    "温暖少年": "ICL_zh_male_yangyang_v1_tob",
    "开朗轻快": "ICL_zh_male_kailangqingkuai_tob",
    "擎苍": "zh_male_qingcang_mars_bigtts",
    "心灵鸡汤": "zh_female_xinlingjitang_moon_bigtts",
    "甜美悦悦": "zh_female_tianmeiyueyue_moon_bigtts"




}

alivoice_options = {
    "知小白(普通话女声)": "zhixiaobai",
    "知小夏(对话数字人)": "zhixiaoxia",
    "知小妹(直播数字人)": "zhixiaomei",
    "知柜(普通话女声)": "zhigui",
    "知硕(普通话男声)": "zhishuo",
    "艾夏(普通话女声)": "aixia",
    "小云(标准女声)": "xiaoyun",
    "小刚(标准男声)": "xiaogang",
    "若兮(温柔女声)": "ruoxi",
    "思琪(温柔女声)": "siqi",
    "思佳(标准女声)": "sijia",
    "思诚(标准男声)": "sicheng",
    "艾琪(温柔女声)": "aiqi",
    "艾佳(标准女声)": "aijia",
    "艾诚(标准男声)": "aicheng",
    "艾达(标准男声)": "aida",
    "宁儿(标准女声)": "ninger",
    "瑞琳(标准女声)": "ruilin",
    "思悦(温柔女声)": "siyue",
    "艾雅(严厉女声)": "aiya",
    "艾美(甜美女声)": "aimei",
    "艾雨(自然女声)": "aiyu",
    "艾悦(温柔女声)": "aiyue",
    "艾静(严厉女声)": "aijing",
    "小美(甜美女声)": "xiaomei",
    "艾娜(浙普女声)": "aina",
    "依娜(浙普女声)": "yina",
    "思婧(严厉女声)": "sijing",
    "思彤(儿童音)": "sitong",
    "小北(萝莉女声)": "xiaobei",
    "艾彤(儿童音)": "aitong",
    "艾薇(萝莉女声)": "aiwei",
    "艾宝(萝莉女声)": "aibao",
    "知猫(普通话女声)": "zhimao",
    "艾倩(资讯女声)": "aiqian",
    "艾伦(悬疑解说男声)": "ailun",
    "艾飞(激昂解说男声)": "aifei",
    "小仙(亲切女声)": "xiaoxian",
    "猫小美(活力女声)": "maoxiaomei",
    "Kenny(温暖男声)": "kenny",
    "知锋_多情感": "zhifeng_emo",
    "知冰_多情感": "zhibing_emo",
    "知妙_多情感": "zhimiao_emo",
    "知米_多情感": "zhimi_emo",
    "知燕_多情感": "zhiyan_emo",
    "知贝_多情感": "zhibei_emo",
    "知甜_多情感": "zhitian_emo",
    "male": "random_male",
    "female": "random_female"
}

digital_human_platform = [
    "华为云",
    "阿里云",
    "火山引擎",
    "即梦",
    "通译万象"
]

# 获取当前脚本的绝对路径
script_path = os.path.abspath(__file__)

# print("当前脚本的绝对路径是:", script_path)

# 脚本所在的目录
script_dir = os.path.dirname(script_path)

config_example_file_name = "config.example.yml"
config_file_name = "config.yml"

config_example_file = os.path.join(script_dir, config_example_file_name)
config_file = os.path.join(script_dir, config_file_name)


def load_config():
    # 加载配置文件
    if not os.path.exists(config_file):
        shutil.copy(config_example_file, config_file)
    if os.path.exists(config_file):
        return read_yaml(config_file)
    return None


def test_config(todo_config, *args):
    temp_config = todo_config
    for arg in args:
        if arg not in temp_config:
            temp_config[arg] = {}
        temp_config = temp_config[arg]


def save_config():
    # 保存配置文件
    if os.path.exists(config_file):
        save_yaml(config_file, my_config)

    

my_config = load_config()
ENV = my_config['env']
# 调用外部接口并更新 CosyVoice_voice
# CosyVoice_voice = fetch_CosyVoice_voice() or CosyVoice_voice  # 如果外部接口失败，则保留原有数据
