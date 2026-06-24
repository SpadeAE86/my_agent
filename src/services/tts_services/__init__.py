import logging
from config.config import MY_CONFIG
from core.voice.tts.tts_manager import TTSManager
from services.tts_services.volco_tts_service import VolcoTTSEngine
from services.tts_services.volcano_v3_tts_service import VolcanoV3TTSServiceEngine, VolcanoV3BidirectionalServiceEngine
from services.tts_services.qwen_tts_service import QwenTTSEngine
from services.tts_services.dashscope_tts_service import DashScopeServiceEngine, QwenOnlineServiceEngine

log = logging.getLogger("tts_services")

# Instantiate singleton manager
tts_manager = TTSManager()

# Register engines
volcano_engine = VolcoTTSEngine()
volcano_v3_engine = VolcanoV3TTSServiceEngine()
volcano_v3_bidirectional_engine = VolcanoV3BidirectionalServiceEngine()
qwen3_local_engine = QwenTTSEngine()
qwen3_online_engine = QwenOnlineServiceEngine()
dashscope_engine = DashScopeServiceEngine()

tts_manager.register_engine("volcano", volcano_engine)
tts_manager.register_engine("volcano_v3", volcano_v3_engine)
tts_manager.register_engine("volcano_v3_bidirectional", volcano_v3_bidirectional_engine)
tts_manager.register_engine("qwen3_local", qwen3_local_engine)
tts_manager.register_engine("qwen3", qwen3_local_engine)  # Compatibility alias
tts_manager.register_engine("qwen3_online", qwen3_online_engine)
tts_manager.register_engine("dashscope", dashscope_engine)

# Determine default engine from config.yml or environment
provider = MY_CONFIG.get("audio", {}).get("provider", "Volcano")
if isinstance(provider, str):
    provider_lower = provider.lower()
    if "qwen" in provider_lower:
        default_engine = "qwen3"
    else:
        default_engine = "volcano"
else:
    default_engine = "volcano"

try:
    tts_manager.default_engine_name = default_engine
    log.info(f"Initialized TTSManager with default engine: {default_engine} (configured provider: {provider})")
except Exception as e:
    log.error(f"Failed to set default engine: {e}")
