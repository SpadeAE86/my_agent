import os
import numpy as np
import librosa
import torch
from typing import Dict, Any, Optional, Tuple

class VoiceprintValidator:
    """
    音色指纹与音频质量评估工具，用于筛选合格的音频进行音色演进或二次克隆。
    """
    
    @staticmethod
    def calculate_spectral_flatness(fft_vals: np.ndarray, epsilon: float = 1e-10) -> float:
        """
        计算频谱平坦度 (Spectral Flatness Measure, SFM)。
        SFM 越接近 0，表示声音越具有明显的谐波结构（健康的语音）；
        SFM 越接近 1，表示声音越接近白噪声（噪点多，质量退化）。
        """
        power = fft_vals ** 2 + epsilon
        log_power = np.log(power)
        gmean = np.exp(np.mean(log_power))
        amean = np.mean(power)
        return float(gmean / amean)

    @staticmethod
    def analyze_spectral_quality(
        wav: np.ndarray, 
        sr: int = 24000, 
        cutoff_hz: float = 8000.0,
        hfer_limit: float = 0.005,
        sfm_limit: float = 0.060
    ) -> Dict[str, Any]:
        """
        分析音频的频谱特征：
        1. 计算高于 cutoff_hz 的高频能量占比。异常偏高通常指示声码器数字噪声（hiss）。
        2. 计算频谱平坦度 (SFM)。
        """
        # FFT 变换
        fft_vals = np.abs(np.fft.rfft(wav))
        freqs = np.fft.rfftfreq(len(wav), 1/sr)
        
        # 1. 高频能量比率
        total_energy = np.sum(fft_vals ** 2)
        high_freq_energy = np.sum(fft_vals[freqs >= cutoff_hz] ** 2)
        hfe_ratio = float(high_freq_energy / (total_energy + 1e-10))
        
        # 2. 频谱平坦度
        sfm = VoiceprintValidator.calculate_spectral_flatness(fft_vals)
        
        # 门限逻辑
        is_clean_hf = hfe_ratio < hfer_limit
        is_clean_sfm = sfm < sfm_limit
        
        return {
            "hfe_ratio": hfe_ratio,
            "sfm": sfm,
            "is_clean_hf": is_clean_hf,
            "is_clean_sfm": is_clean_sfm,
            "is_spectral_clean": is_clean_hf and is_clean_sfm
        }

    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """计算两个特征向量的余弦相似度"""
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        return float(dot_product / (norm_v1 * norm_v2 + 1e-10))

    @classmethod
    def evaluate_audio_for_evolution(
        cls,
        audio_path: str,
        model: Optional[Any] = None,
        ref_voiceprint: Optional[Any] = None,
        target_sr: int = 24000,
        hfer_limit: float = 0.005,
        sfm_limit: float = 0.060,
        sim_limit: float = 0.965
    ) -> Dict[str, Any]:
        """
        评估音频是否适合进行音色演进或二次克隆。
        
        Args:
            audio_path: 待测音频文件路径 (.wav / .mp3)
            model: 可选。Qwen3TTSModel 实例，若提供则会自动提取音色指纹并计算相似度。
            ref_voiceprint: 可选。标准原版音色指纹 (numpy.ndarray, torch.Tensor, VoiceClonePromptItem 或其 pt 文件路径)。
            target_sr: 目标采样率，默认 24000Hz。
            hfer_limit: 高频能量比率阈值，默认 0.0050。
            sfm_limit: 频谱平坦度阈值，默认 0.0600。
            sim_limit: 余弦相似度阈值，默认 0.9650。
            
        Returns:
            Dict: 包含评估详情、合格判定、指标和文字建议的报告字典。
        """
        report = {
            "audio_file": os.path.basename(audio_path),
            "suitable_for_evolution": False,
            "similarity": None,
            "hfe_ratio": None,
            "sfm": None,
            "warnings": [],
            "advice": ""
        }
        
        # 1. 检查文件并载入音频
        if not os.path.exists(audio_path):
            report["warnings"].append(f"文件不存在: {audio_path}")
            report["advice"] = "请检查音频文件路径。"
            return report
            
        try:
            wav, sr = librosa.load(audio_path, sr=target_sr)
        except Exception as e:
            report["warnings"].append(f"音频读取失败: {e}")
            report["advice"] = "请确保音频是正确的 WAV 或 MP3 格式。"
            return report
            
        # 2. 分析频谱质量
        spec_result = cls.analyze_spectral_quality(
            wav, 
            sr=target_sr, 
            hfer_limit=hfer_limit, 
            sfm_limit=sfm_limit
        )
        report["hfe_ratio"] = spec_result["hfe_ratio"]
        report["sfm"] = spec_result["sfm"]
        
        if not spec_result["is_clean_hf"]:
            report["warnings"].append(
                f"高频能量比率超标 ({spec_result['hfe_ratio']:.6f} > {hfer_limit:.4f})。"
                "这通常意味着音频中含有声码器重构产生的数字白噪或录音环境的高频 Hiss 噪声。"
            )
            
        if not spec_result["is_clean_sfm"]:
            report["warnings"].append(
                f"频谱平坦度异常高 ({spec_result['sfm']:.6f} > {sfm_limit:.4f})。"
                "这表示音频细节被抹平，趋近于白噪声，失去了真实的共振峰和基频细节。"
            )
            
        # 3. 比对音色特征相似度
        if model is not None and ref_voiceprint is not None:
            try:
                # 解析参考指纹
                if isinstance(ref_voiceprint, str):
                    if os.path.exists(ref_voiceprint):
                        ref_prompt_list = torch.load(ref_voiceprint, weights_only=False)
                    else:
                        raise FileNotFoundError(f"参考指纹文件未找到: {ref_voiceprint}")
                else:
                    ref_prompt_list = ref_voiceprint
                
                # 获取参考指纹的 spk_embedding (numpy format)
                if isinstance(ref_prompt_list, (np.ndarray, torch.Tensor)):
                    ref_emb = ref_prompt_list
                elif isinstance(ref_prompt_list, list) and len(ref_prompt_list) > 0:
                    ref_emb = getattr(ref_prompt_list[0], "ref_spk_embedding", ref_prompt_list[0])
                elif isinstance(ref_prompt_list, dict):
                    ref_emb = ref_prompt_list.get("ref_spk_embedding", [None])[0]
                else:
                    ref_emb = getattr(ref_prompt_list, "ref_spk_embedding", None)
                    
                if ref_emb is None:
                    raise ValueError("无法解析参考指纹中的 ref_spk_embedding。")
                
                if isinstance(ref_emb, torch.Tensor):
                    ref_emb = ref_emb.cpu().float().numpy()
                
                # 提取待测音频的指纹
                with torch.no_grad():
                    test_emb = model.model.extract_speaker_embedding(wav, target_sr).cpu().float().numpy()
                
                # 计算余弦相似度
                sim = cls.cosine_similarity(ref_emb, test_emb)
                report["similarity"] = sim
                
                if sim < sim_limit:
                    report["warnings"].append(
                        f"与标准原声相似度偏低 ({sim:.4f} < {sim_limit:.4f})。"
                        "说明音色在流式生成或压缩过程中发生了漂移，偏离了说话人的原始声线。"
                    )
            except Exception as e:
                report["warnings"].append(f"相似度比对失败: {e}")
                
        # 4. 做出最终判定
        hfe_ok = spec_result["is_clean_hf"]
        sfm_ok = spec_result["is_clean_sfm"]
        sim_ok = (report["similarity"] is None) or (report["similarity"] >= sim_limit)
        
        if hfe_ok and sfm_ok and sim_ok:
            report["suitable_for_evolution"] = True
            report["advice"] = "音频声学特征纯净，音色与原声高度一致，非常适合作为音色演进或二次克隆的参考源！"
        else:
            report["suitable_for_evolution"] = False
            report["advice"] = (
                "警告：该音频不建议作为二次克隆的参考源。建议直接复用最初录制的真人原声指纹（.pt 文件），"
                "或重新生成纯净无损的 WAV 格式音频进行提取。"
            )
            
        return report

if __name__ == "__main__":
    import argparse
    import json
    import sys
    
    # Setup path if needed
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    
    # Ensure flash_attn module mapping is done
    try:
        import flash_attn_3
        import flash_attn_interface
        sys.modules['flash_attn'] = flash_attn_3
        sys.modules['flash_attn.flash_attn_interface'] = flash_attn_interface
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Voiceprint Quality & Evolution Suitability Gatekeeper")
    parser.add_argument("--audio", required=True, help="Path to the audio file to evaluate")
    parser.add_argument("--ref-audio", help="Path to the reference/baseline audio file (optional)")
    parser.add_argument("--ref-voiceprint", help="Path to the reference voiceprint .pt file (optional)")
    parser.add_argument("--model-dir", default=r"C:\Users\admin\.cache\modelscope\hub\models\Qwen\Qwen3-TTS-12Hz-1.7B-Base", help="Path to Qwen3-TTS model directory")
    parser.add_argument("--hfer-limit", type=float, default=0.005, help="Limit for high-frequency energy ratio (default: 0.005)")
    parser.add_argument("--sfm-limit", type=float, default=0.060, help="Limit for spectral flatness measure (default: 0.060)")
    parser.add_argument("--sim-limit", type=float, default=0.965, help="Limit for cosine similarity (default: 0.965)")
    parser.add_argument("--json", action="store_true", help="Output result in JSON format")
    
    args = parser.parse_args()
    
    # Check audio file
    if not os.path.exists(args.audio):
        if args.json:
            print(json.dumps({"error": f"Audio file not found: {args.audio}"}))
        else:
            print(f"Error: Audio file not found: {args.audio}")
        sys.exit(1)
        
    model = None
    ref_emb = None
    
    # Load model if similarity comparison is requested
    if args.ref_audio or args.ref_voiceprint:
        try:
            from qwen_tts import Qwen3TTSModel
            if not args.json:
                print("Loading Qwen3 model for similarity comparison...")
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model = Qwen3TTSModel.from_pretrained(
                args.model_dir,
                device_map=device,
                dtype=torch.bfloat16
            )
            if not args.json:
                print("Model loaded.")
        except Exception as e:
            if args.json:
                print(json.dumps({"error": f"Failed to load Qwen3 model: {str(e)}"}))
            else:
                print(f"Error loading Qwen3 model: {e}")
            sys.exit(1)
            
        # Extract reference embedding
        if args.ref_audio:
            if not os.path.exists(args.ref_audio):
                if args.json:
                    print(json.dumps({"error": f"Reference audio not found: {args.ref_audio}"}))
                else:
                    print(f"Error: Reference audio not found: {args.ref_audio}")
                sys.exit(1)
            try:
                if not args.json:
                    print(f"Extracting reference voiceprint from: {args.ref_audio}")
                ref_wav, ref_sr = librosa.load(args.ref_audio, sr=24000)
                with torch.no_grad():
                    ref_emb = model.model.extract_speaker_embedding(ref_wav, 24000).cpu().float().numpy()
            except Exception as e:
                if args.json:
                    print(json.dumps({"error": f"Failed to extract reference embedding: {str(e)}"}))
                else:
                    print(f"Error extracting reference embedding: {e}")
                sys.exit(1)
        elif args.ref_voiceprint:
            ref_emb = args.ref_voiceprint
            
    # Run evaluation
    report = VoiceprintValidator.evaluate_audio_for_evolution(
        audio_path=args.audio,
        model=model,
        ref_voiceprint=ref_emb,
        target_sr=24000,
        hfer_limit=args.hfer_limit,
        sfm_limit=args.sfm_limit,
        sim_limit=args.sim_limit
    )
    
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print("\n" + "="*80)
        print("VOICEPRINT QUALITY & EVOLUTION SUITABILITY REPORT")
        print("="*80)
        print(f"File Name: {report['audio_file']}")
        print(f"High-Freq Energy (>8kHz): {report['hfe_ratio']:.6f} (Limit < {args.hfer_limit:.4f})")
        print(f"Spectral Flatness (SFM):   {report['sfm']:.6f} (Limit < {args.sfm_limit:.4f})")
        if report['similarity'] is not None:
            print(f"Cosine Similarity to Ref:  {report['similarity']:.4f} (Limit >= {args.sim_limit:.4f})")
        
        status_str = "[PASS] APPROVED" if report['suitable_for_evolution'] else "[FAIL] REJECTED"
        print(f"Suitability Status:        {status_str}")
        
        if report['warnings']:
            print("Warnings (警告细节):")
            for w in report['warnings']:
                print(f"  * {w}")
        print(f"Advice (改进建议): {report['advice']}")
        print("="*80)
