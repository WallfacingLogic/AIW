# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════╗
║        AI Text Detector - 文章AI率检测器          ║
║   基于多维统计特征分析文本是否由AI生成              ║
╚══════════════════════════════════════════════════╝

依赖安装:
  pip install jieba numpy

使用方式:
  python ai_detector.py                    # 交互式输入
  python ai_detector.py -f novel.txt       # 从文件读取
  python ai_detector.py -f novel.txt --perplexity  # 启用困惑度分析
"""

import re
import math
import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

try:
    import jieba
    import jieba.posseg as pseg
except ImportError:
    print("❌ 缺少依赖: pip install jieba")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("❌ 缺少依赖: pip install numpy")
    sys.exit(1)


# ============================================================
# 数据结构定义
# ============================================================
@dataclass
class DimensionScore:
    """单个检测维度的得分"""
    name: str            # 维度名称
    score: float         # AI概率得分 0.0(人类) ~ 1.0(AI)
    weight: float        # 权重
    detail: str          # 详细说明
    raw_value: float     # 原始计算值

@dataclass
class DetectionReport:
    """完整检测报告"""
    text_length: int
    sentence_count: int
    word_count: int
    dimensions: List[DimensionScore] = field(default_factory=list)
    final_score: float = 0.0
    
    @property
    def ai_percentage(self) -> float:
        return round(self.final_score * 100, 1)
    
    @property
    def verdict(self) -> str:
        s = self.final_score
        if s < 0.20:
            return "✅ 极大概率是人类写作"
        elif s < 0.40:
            return "🟢 大概率是人类写作（可能经过少量AI辅助润色）"
        elif s < 0.60:
            return "🟡 存在AI生成嫌疑（可能是人机混合写作）"
        elif s < 0.80:
            return "🟠 大概率是AI生成（可能经过人工修改）"
        else:
            return "🔴 极大概率是AI生成"


# ============================================================
# 文本预处理工具
# ============================================================
class TextPreprocessor:
    """文本预处理：分句、分词、清洗"""
    
    # 中文句子结束符
    SENT_ENDERS = re.compile(r'[。！？!?…]+')
    # 段落分隔
    PARA_SPLIT = re.compile(r'\n\s*\n')
    # 对话标记
    DIALOGUE_PATTERN = re.compile(r'[""「『](.*?)[""」』]')
    
    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """将文本切分为句子列表"""
        # 按句子结束符切分，保留非空句子
        raw = TextPreprocessor.SENT_ENDERS.split(text)
        sentences = [s.strip() for s in raw if len(s.strip()) > 1]
        return sentences
    
    @staticmethod
    def split_paragraphs(text: str) -> List[str]:
        """将文本切分为段落列表"""
        paras = TextPreprocessor.PARA_SPLIT.split(text)
        return [p.strip() for p in paras if len(p.strip()) > 5]
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """中文分词"""
        return [w for w in jieba.cut(text) if w.strip() and not w.isspace()]
    
    @staticmethod
    def pos_tag(text: str) -> List[Tuple[str, str]]:
        """词性标注"""
        words = pseg.cut(text)
        return [(w.word, w.flag) for w in words if w.word.strip()]
    
    @staticmethod
    def extract_dialogues(text: str) -> List[str]:
        """提取对话内容"""
        return TextPreprocessor.DIALOGUE_PATTERN.findall(text)
    
    @staticmethod
    def clean_text(text: str) -> str:
        """清洗文本：去除多余空白、特殊字符"""
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()


# ============================================================
# 核心分析引擎
# ============================================================
class TextAnalyzer:
    """多维文本特征分析器"""
    
    def __init__(self, text: str):
        self.raw_text = text
        self.text = TextPreprocessor.clean_text(text)
        self.sentences = TextPreprocessor.split_sentences(self.text)
        self.paragraphs = TextPreprocessor.split_paragraphs(self.text)
        self.words = TextPreprocessor.tokenize(self.text)
        self.word_counter = Counter(self.words)
        self.dialogues = TextPreprocessor.extract_dialogues(self.text)
    
    # ----------------------------------------------------------
    # 1. 突发性分析 (Burstiness)
    # ----------------------------------------------------------
    def analyze_burstiness(self) -> DimensionScore:
        """
        检测句子长度的变化程度。
        AI文本：句子长度均匀 → 突发性低
        人类文本：长短句交替 → 突发性高
        """
        if len(self.sentences) < 3:
            return DimensionScore("突发性(Burstiness)", 0.5, 0.20,
                                  "句子数过少，无法分析", 0)
        
        sent_lengths = [len(TextPreprocessor.tokenize(s)) for s in self.sentences]
        mean_len = np.mean(sent_lengths)
        std_len = np.std(sent_lengths)
        
        # 变异系数 (Coefficient of Variation)
        cv = std_len / mean_len if mean_len > 0 else 0
        
        # 计算相邻句子长度差异的方差（捕捉节奏变化）
        diffs = np.diff(sent_lengths)
        diff_variance = np.var(diffs) if len(diffs) > 1 else 0
        
        # 综合突发性得分
        burstiness = cv * 0.6 + min(diff_variance / 100, 1.0) * 0.4
        
        # 映射到AI概率：突发性越低 → AI概率越高
        # 经验阈值：CV < 0.3 偏向AI，CV > 0.7 偏向人类
        ai_score = np.clip(1.0 - (burstiness - 0.2) / 0.8, 0.0, 1.0)
        
        detail = (f"句长均值={mean_len:.1f}, 标准差={std_len:.1f}, "
                  f"变异系数CV={cv:.3f}, 突发性指数={burstiness:.3f}")
        
        return DimensionScore("突发性(Burstiness)", round(ai_score, 3), 
                              0.20, detail, burstiness)
    
    # ----------------------------------------------------------
    # 2. 词汇多样性分析 (Lexical Diversity)
    # ----------------------------------------------------------
    def analyze_lexical_diversity(self) -> DimensionScore:
        """
        检测用词的丰富程度和独特性。
        AI文本：用词偏"安全"、重复率高、罕用词少
        人类文本：用词丰富、个性化强、罕用词多
        """
        if len(self.words) < 10:
            return DimensionScore("词汇多样性", 0.5, 0.20,
                                  "词数过少，无法分析", 0)
        
        total_words = len(self.words)
        unique_words = len(self.word_counter)
        
        # (1) Type-Token Ratio (TTR)
        ttr = unique_words / total_words
        
        # (2) Hapax Legomena Ratio（只出现一次的词占比）
        hapax = sum(1 for c in self.word_counter.values() if c == 1)
        hapax_ratio = hapax / unique_words if unique_words > 0 else 0
        
        # (3) Top词集中度（前20高频词占总词数比例）
        top20_count = sum(c for _, c in self.word_counter.most_common(20))
        top20_ratio = top20_count / total_words
        
        # (4) 平均词频
        avg_freq = total_words / unique_words if unique_words > 0 else 0
        
        # 综合词汇多样性得分
        # TTR高 + Hapax高 + Top20占比低 → 人类特征
        diversity = (ttr * 0.3 + hapax_ratio * 0.3 + 
                     (1 - top20_ratio) * 0.2 + 
                     min(1.0 / avg_freq, 1.0) * 0.2)
        
        # 映射到AI概率
        ai_score = np.clip(1.0 - (diversity - 0.15) / 0.65, 0.0, 1.0)
        
        detail = (f"TTR={ttr:.3f}, 单次词比={hapax_ratio:.3f}, "
                  f"Top20集中度={top20_ratio:.3f}, 平均词频={avg_freq:.2f}")
        
        return DimensionScore("词汇多样性(Lexical Diversity)", 
                              round(ai_score, 3), 0.20, detail, diversity)
    
    # ----------------------------------------------------------
    # 3. 重复模式检测 (Repetition Patterns)
    # ----------------------------------------------------------
    def analyze_repetition(self) -> DimensionScore:
        """
        检测文本中的重复短语和模板化表达。
        AI文本：高频使用相同的过渡短语和句式模板
        人类文本：表达变化多，重复少
        """
        if len(self.words) < 20:
            return DimensionScore("重复模式", 0.5, 0.15,
                                  "文本过短，无法分析", 0)
        
        # (1) Bigram / Trigram 重复率
        bigrams = [f"{self.words[i]} {self.words[i+1]}" 
                   for i in range(len(self.words)-1)]
        trigrams = [f"{self.words[i]} {self.words[i+1]} {self.words[i+2]}" 
                    for i in range(len(self.words)-2)]
        
        bigram_counter = Counter(bigrams)
        trigram_counter = Counter(trigrams)
        
        # 重复的bigram（出现>2次）
        repeated_bigrams = sum(1 for c in bigram_counter.values() if c > 2)
        repeated_trigrams = sum(1 for c in trigram_counter.values() if c > 2)
        
        bigram_repeat_ratio = repeated_bigrams / len(bigram_counter) if bigram_counter else 0
        trigram_repeat_ratio = repeated_trigrams / len(trigram_counter) if trigram_counter else 0
        
        # (2) AI常见模板短语检测
        ai_phrases = [
            "值得注意的是", "总而言之", "综上所述", "不可否认",
            "与此同时", "在此基础上", "从某种意义上来说", "众所周知",
            "毫无疑问", "事实上", "换言之", "具体来说",
            "需要指出的是", "由此可见", "总的来说", "归根结底",
            "在这个背景下", "从本质上看", "深入分析", "全面分析",
            "至关重要", "不言而喻", "一脉相承", "与时俱进",
            "首先.*其次.*最后", "不仅.*而且", "虽然.*但是",
        ]
        
        phrase_hits = 0
        for phrase in ai_phrases:
            if re.search(phrase, self.text):
                phrase_hits += 1
        
        phrase_density = phrase_hits / max(len(self.sentences), 1)
        
        # (3) 句首重复模式（AI喜欢用相同的句式开头）
        sent_starts = []
        for s in self.sentences:
            words_in_sent = TextPreprocessor.tokenize(s)
            if len(words_in_sent) >= 2:
                sent_starts.append(f"{words_in_sent[0]}{words_in_sent[1]}")
        
        start_counter = Counter(sent_starts)
        most_common_start_count = start_counter.most_common(1)[0][1] if start_counter else 0
        start_repetition = most_common_start_count / len(self.sentences) if self.sentences else 0
        
        # 综合重复得分
        repetition = (bigram_repeat_ratio * 0.25 + 
                      trigram_repeat_ratio * 0.25 + 
                      min(phrase_density, 1.0) * 0.30 +
                      start_repetition * 0.20)
        
        ai_score = np.clip(repetition * 2.5, 0.0, 1.0)
        
        detail = (f"Bigram重复率={bigram_repeat_ratio:.3f}, "
                  f"Trigram重复率={trigram_repeat_ratio:.3f}, "
                  f"模板短语命中={phrase_hits}个, "
                  f"句首重复率={start_repetition:.3f}")
        
        return DimensionScore("重复模式(Repetition)", 
                              round(ai_score, 3), 0.15, detail, repetition)
    
    # ----------------------------------------------------------
    # 4. 句式结构均匀度 (Sentence Uniformity)
    # ----------------------------------------------------------
    def analyze_sentence_uniformity(self) -> DimensionScore:
        """
        检测句式结构的变化程度。
        AI文本：句式结构单一（主谓宾为主），标点使用规整
        人类文本：句式多变（倒装、省略、感叹、反问等）
        """
        if len(self.sentences) < 5:
            return DimensionScore("句式均匀度", 0.5, 0.15,
                                  "句子数过少，无法分析", 0)
        
        # (1) 标点符号多样性
        punctuation_pattern = re.compile(r'[，。！？；：、…—""''「」（）\.!\?,;:]')
        puncts = punctuation_pattern.findall(self.text)
        punct_counter = Counter(puncts)
        punct_diversity = len(punct_counter) / max(len(set(puncts)), 1)
        
        # 感叹号和问号的使用比例（人类更常用）
        exclam_ratio = (punct_counter.get('！', 0) + punct_counter.get('!', 0)) / max(len(puncts), 1)
        question_ratio = (punct_counter.get('？', 0) + punct_counter.get('?', 0)) / max(len(puncts), 1)
        emotional_punct = exclam_ratio + question_ratio
        
        # (2) 句子结构类型检测
        structure_types = []
        for sent in self.sentences:
            if re.search(r'[？?]', sent) or sent.strip().endswith('吗') or sent.strip().endswith('呢'):
                structure_types.append('question')
            elif re.search(r'[！!]', sent):
                structure_types.append('exclamation')
            elif len(sent) < 6:
                structure_types.append('short_fragment')
            elif '……' in sent or '...' in sent:
                structure_types.append('ellipsis')
            elif re.match(r'^[但可是而且虽然而]', sent):
                structure_types.append('conjunction_start')
            else:
                structure_types.append('standard')
        
        structure_counter = Counter(structure_types)
        # 结构熵：熵越高 → 句式越多样 → 越像人类
        total_structs = len(structure_types)
        entropy = 0
        for count in structure_counter.values():
            p = count / total_structs
            if p > 0:
                entropy -= p * math.log2(p)
        
        max_entropy = math.log2(len(structure_counter)) if len(structure_counter) > 1 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.5
        
        # (3) "standard"句式占比（AI倾向于全是标准句式）
        standard_ratio = structure_counter.get('standard', 0) / total_structs
        
        # 综合
        uniformity = (standard_ratio * 0.4 + 
                      (1 - normalized_entropy) * 0.3 + 
                      (1 - min(emotional_punct * 10, 1.0)) * 0.3)
        
        ai_score = np.clip(uniformity, 0.0, 1.0)
        
        detail = (f"结构熵={entropy:.3f}(归一化={normalized_entropy:.3f}), "
                  f"标准句式占比={standard_ratio:.3f}, "
                  f"情感标点密度={emotional_punct:.4f}, "
                  f"句式类型={dict(structure_counter)}")
        
        return DimensionScore("句式均匀度(Sentence Uniformity)", 
                              round(ai_score, 3), 0.15, detail, uniformity)
    
    # ----------------------------------------------------------
    # 5. 情感波动分析 (Emotional Volatility)
    # ----------------------------------------------------------
    def analyze_emotional_volatility(self) -> DimensionScore:
        """
        检测文本情感的变化和波动。
        AI文本：情感平稳，缺少剧烈波动
        人类文本：情感起伏大，有突然的情绪变化
        """
        if len(self.sentences) < 5:
            return DimensionScore("情感波动", 0.5, 0.10,
                                  "句子数过少，无法分析", 0)
        
        # 简易情感词典（正面/负面/强烈情感词）
        positive_words = set([
            '好', '美', '棒', '赞', '爱', '喜欢', '快乐', '幸福', '开心',
            '优秀', '精彩', '完美', '温暖', '感动', '欣慰', '甜蜜', '灿烂',
        ])
        negative_words = set([
            '坏', '恶', '恨', '怕', '痛', '苦', '悲伤', '愤怒', '绝望',
            '恐怖', '凄惨', '痛苦', '悲伤', '忧郁', '凄凉', '残酷', '黑暗',
        ])
        intensifiers = set([
            '非常', '极其', '特别', '十分', '太', '真', '好', '超',
            '万分', '无比', '格外', '异常', '简直', '竟然', '居然',
        ])
        
        # 逐句情感打分
        sent_emotions = []
        for sent in self.sentences:
            words_in_sent = set(TextPreprocessor.tokenize(sent))
            pos = len(words_in_sent & positive_words)
            neg = len(words_in_sent & negative_words)
            intensity = len(words_in_sent & intensifiers)
            
            # 情感值：正-负，强度加权
            emotion_val = (pos - neg) * (1 + intensity * 0.5)
            sent_emotions.append(emotion_val)
        
        if len(sent_emotions) < 2:
            return DimensionScore("情感波动", 0.5, 0.10,
                                  "有效情感数据不足", 0)
        
        emotions = np.array(sent_emotions)
        
        # (1) 情感标准差（波动大小）
        emotion_std = np.std(emotions)
        
        # (2) 情感变化频率（相邻句子情感符号变化的比例）
        sign_changes = sum(1 for i in range(1, len(emotions)) 
                          if emotions[i] * emotions[i-1] < 0)
        change_freq = sign_changes / (len(emotions) - 1)
        
        # (3) 情感极值比例
        max_abs = np.max(np.abs(emotions))
        extreme_ratio = np.sum(np.abs(emotions) > max_abs * 0.7) / len(emotions)
        
        # 综合情感波动得分
        volatility = (min(emotion_std, 3.0) / 3.0 * 0.4 + 
                      change_freq * 0.3 + 
                      extreme_ratio * 0.3)
        
        # 波动低 → AI概率高
        ai_score = np.clip(1.0 - volatility * 2.0, 0.0, 1.0)
        
        detail = (f"情感标准差={emotion_std:.3f}, "
                  f"情感转折频率={change_freq:.3f}, "
                  f"极值句比例={extreme_ratio:.3f}")
        
        return DimensionScore("情感波动(Emotional Volatility)", 
                              round(ai_score, 3), 0.10, detail, volatility)
    
    # ----------------------------------------------------------
    # 6. 段落结构分析 (Paragraph Structure)
    # ----------------------------------------------------------
    def analyze_paragraph_structure(self) -> DimensionScore:
        """
        检测段落之间的长度和结构变化。
        AI文本：段落长度均匀，结构对称
        人类文本：段落长短不一，有长有短
        """
        if len(self.paragraphs) < 3:
            return DimensionScore("段落结构", 0.5, 0.10,
                                  "段落数过少，无法分析", 0)
        
        para_lengths = [len(TextPreprocessor.tokenize(p)) for p in self.paragraphs]
        mean_len = np.mean(para_lengths)
        std_len = np.std(para_lengths)
        cv = std_len / mean_len if mean_len > 0 else 0
        
        # 段落句子数的方差
        para_sent_counts = [len(TextPreprocessor.split_sentences(p)) 
                           for p in self.paragraphs]
        sent_count_cv = (np.std(para_sent_counts) / np.mean(para_sent_counts) 
                        if np.mean(para_sent_counts) > 0 else 0)
        
        uniformity = (1 - min(cv, 1.0)) * 0.5 + (1 - min(sent_count_cv, 1.0)) * 0.5
        ai_score = np.clip(uniformity, 0.0, 1.0)
        
        detail = (f"段落长度CV={cv:.3f}, "
                  f"段落句数CV={sent_count_cv:.3f}, "
                  f"段落数={len(self.paragraphs)}")
        
        return DimensionScore("段落结构(Paragraph Structure)", 
                              round(ai_score, 3), 0.10, detail, uniformity)
    
    # ----------------------------------------------------------
    # 7. 困惑度估算 (Perplexity Estimation) — 可选
    # ----------------------------------------------------------
    def analyze_perplexity(self) -> Optional[DimensionScore]:
        """
        使用语言模型估算文本困惑度。
        需要 transformers 和 torch。
        困惑度低 → 文本太"顺滑" → AI嫌疑大
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError:
            return None
        
        print("  ⏳ 正在加载语言模型计算困惑度（首次运行需下载模型）...")
        
        try:
            # 使用较小的中文GPT模型
            model_name = "uer/gpt2-chinese-cluecorpussmall"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model.eval()
            
            # 取前2000字计算（避免过长）
            sample = self.text[:2000]
            inputs = tokenizer(sample, return_tensors="pt", 
                             truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                perplexity = math.exp(loss.item())
            
            # 映射到AI概率
            # 经验值：AI文本困惑度通常 < 50, 人类文本 > 80
            ai_score = np.clip(1.0 - (perplexity - 30) / 100, 0.0, 1.0)
            
            detail = f"Perplexity={perplexity:.2f} (模型: {model_name})"
            
            return DimensionScore("困惑度(Perplexity)", 
                                 round(ai_score, 3), 0.20, detail, perplexity)
        
        except Exception as e:
            print(f"  ⚠️ 困惑度计算失败: {e}")
            return None
    
    # ----------------------------------------------------------
    # 8. 对话风格分析 (Dialogue Style) — 小说专用
    # ----------------------------------------------------------
    def analyze_dialogue_style(self) -> Optional[DimensionScore]:
        """
        分析小说中不同角色对话的风格差异。
        AI文本：不同角色说话方式趋同
        人类文本：不同角色有独特的口癖和语气
        """
        if len(self.dialogues) < 5:
            return None
        
        # 提取每段对话的特征词
        dialogue_features = []
        for d in self.dialogues:
            words = TextPreprocessor.tokenize(d)
            dialogue_features.append(set(words))
        
        if len(dialogue_features) < 2:
            return None
        
        # 计算对话之间的Jaccard相似度
        similarities = []
        for i in range(len(dialogue_features)):
            for j in range(i+1, min(i+5, len(dialogue_features))):
                a, b = dialogue_features[i], dialogue_features[j]
                if len(a | b) > 0:
                    jaccard = len(a & b) / len(a | b)
                    similarities.append(jaccard)
        
        avg_similarity = np.mean(similarities) if similarities else 0.5
        
        # 相似度高 → 对话风格趋同 → AI嫌疑
        ai_score = np.clip(avg_similarity * 3, 0.0, 1.0)
        
        detail = (f"对话段数={len(self.dialogues)}, "
                  f"平均Jaccard相似度={avg_similarity:.3f}")
        
        return DimensionScore("对话风格(Dialogue Style)", 
                              round(ai_score, 3), 0.10, detail, avg_similarity)
    
    # ----------------------------------------------------------
    # 汇总所有分析
    # ----------------------------------------------------------
    def run_full_analysis(self, use_perplexity: bool = False) -> DetectionReport:
        """运行所有维度的分析并生成报告"""
        report = DetectionReport(
            text_length=len(self.text),
            sentence_count=len(self.sentences),
            word_count=len(self.words)
        )
        
        # 运行各维度分析
        analyzers = [
            self.analyze_burstiness,
            self.analyze_lexical_diversity,
            self.analyze_repetition,
            self.analyze_sentence_uniformity,
            self.analyze_emotional_volatility,
            self.analyze_paragraph_structure,
        ]
        
        for analyzer in analyzers:
            result = analyzer()
            report.dimensions.append(result)
        
        # 小说对话分析（可选）
        dialogue_result = self.analyze_dialogue_style()
        if dialogue_result:
            report.dimensions.append(dialogue_result)
        
        # 困惑度分析（可选）
        if use_perplexity:
            perplexity_result = self.analyze_perplexity()
            if perplexity_result:
                report.dimensions.append(perplexity_result)
        
        # 计算加权总分
        report.final_score = ScoringEngine.calculate(report.dimensions)
        
        return report


# ============================================================
# 评分引擎
# ============================================================
class ScoringEngine:
    """加权评分计算器"""
    
    @staticmethod
    def calculate(dimensions: List[DimensionScore]) -> float:
        """根据各维度得分和权重计算最终AI概率"""
        total_weight = sum(d.weight for d in dimensions)
        if total_weight == 0:
            return 0.5
        
        weighted_sum = sum(d.score * d.weight for d in dimensions)
        normalized = weighted_sum / total_weight
        
        # 非线性映射：让极端值更突出
        # 使用sigmoid-like变换
        if normalized > 0.5:
            adjusted = 0.5 + 0.5 * ((normalized - 0.5) / 0.5) ** 0.8
        else:
            adjusted = 0.5 - 0.5 * ((0.5 - normalized) / 0.5) ** 0.8
        
        return round(adjusted, 4)


# ============================================================
# 报告渲染器
# ============================================================
class ReportRenderer:
    """将检测结果渲染为可读报告"""
    
    @staticmethod
    def _get_score_indicator(score: float) -> str:
        """根据得分返回对应的指示符"""
        if score < 0.20:
            return "🟢"
        elif score < 0.40:
            return "🔵"
        elif score < 0.60:
            return "🟡"
        elif score < 0.80:
            return "🟠"
        else:
            return "🔴"
    
    @staticmethod
    def _create_progress_bar(value: float, length: int = 25) -> str:
        """创建渐变色进度条"""
        filled = int(value * length)
        empty = length - filled
        
        # 使用不同的字符表示填充程度
        if filled == 0:
            bar = "○" * length
        elif filled == length:
            bar = "●" * length
        else:
            bar = "●" * filled + "○" * empty
        
        return bar
    
    @staticmethod
    def render(report: DetectionReport) -> str:
        lines = []
        width = 70
        
        # 标题区域
        lines.append("")
        lines.append("┌" + "─" * width + "┐")
        lines.append("│" + "📊 AI文本检测报告".center(width) + "│")
        lines.append("├" + "─" * width + "┤")
        
        # 基础统计信息
        lines.append("│" + "".ljust(width) + "│")
        lines.append("│" + "📋 基础信息".ljust(width+1) + "│")
        lines.append("│  " + ("·" * (width-2)).ljust(width) + "│")
        lines.append(f"│  ✍️  文本长度: {report.text_length:>8,} 字符".ljust(width+1) + "│")
        lines.append(f"│  📝 句子数量: {report.sentence_count:>8} 句".ljust(width+1) + "│")
        lines.append(f"│  📚 分词数量: {report.word_count:>8,} 词".ljust(width+1) + "│")
        lines.append("│" + "".ljust(width) + "│")
        
        # 维度分析区域
        lines.append("├" + "─" * width + "┤")
        lines.append("│" + "🔍 多维度分析结果".ljust(width+1) + "│")
        lines.append("├" + "═" * width + "┤")
        
        for idx, dim in enumerate(report.dimensions, 1):
            indicator = ReportRenderer._get_score_indicator(dim.score)
            bar = ReportRenderer._create_progress_bar(dim.score, 30)
            pct = f"{dim.score * 100:5.1f}%"
            
            # 维度标题行
            title_line = f"  {idx}. {indicator} {dim.name}"
            lines.append("│" + title_line.ljust(width) + "│")
            
            # 进度条行
            bar_line = f"     [{bar}] {pct}"
            lines.append("│" + bar_line.ljust(width) + "│")
            
            # 详细信息（分行显示）
            detail_parts = dim.detail.split(", ")
            for part in detail_parts[:3]:  # 最多显示3个关键指标
                detail_line = f"       • {part}"
                lines.append("│" + detail_line.ljust(width) + "│")
            
            lines.append("│" + "".ljust(width) + "│")
        
        # 最终结果区域
        lines.append("├" + "═" * width + "┤")
        lines.append("│" + "".ljust(width) + "│")
        
        # 最终得分
        final_indicator = ReportRenderer._get_score_indicator(report.final_score)
        final_bar = ReportRenderer._create_progress_bar(report.final_score, 40)
        
        lines.append("│" + f"  {final_indicator} 综合AI概率".ljust(width) + "│")
        lines.append("│" + f"     [{final_bar}]".ljust(width) + "│")
        lines.append("│" + f"      {report.ai_percentage:>6.1f}%".ljust(width) + "│")
        lines.append("│" + "".ljust(width) + "│")
        
        # 判定结果
        verdict_line = f"  🎯 判定: {report.verdict}"
        lines.append("│" + verdict_line.ljust(width) + "│")
        lines.append("│" + "".ljust(width) + "│")
        lines.append("└" + "─" * width + "┘")
        
        # 声明和说明
        lines.append("")
        lines.append("💡 使用说明:")
        lines.append("   🟢 0-20%  : 极大概率是人类写作")
        lines.append("   🔵 20-40% : 大概率人类写作（可能有AI辅助）")
        lines.append("   🟡 40-60% : 存在AI生成嫌疑（人机混合）")
        lines.append("   🟠 60-80% : 大概率AI生成（可能经人工修改）")
        lines.append("   🔴 80-100%: 极大概率是AI生成")
        lines.append("")
        lines.append("⚠️  免责声明:")
        lines.append("   本检测基于统计特征分析，结果仅供参考。")
        lines.append("   风格规整的人类作者可能被误判，经过人工润色的AI文本可能漏判。")
        lines.append("")
        
        return "\n".join(lines)


# ============================================================
# 主程序入口
# ============================================================
def read_text_from_file(filepath: str) -> str:
    """从文件读取文本"""
    encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16']
    for enc in encodings:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"无法读取文件 {filepath}，请检查文件编码")


def interactive_input() -> str:
    """交互式输入文本"""
    print("\n┌" + "─" * 58 + "┐")
    print("│" + "📝 请输入待检测的文章内容".center(56) + "│")
    print("│" + "".ljust(58) + "│")
    print("│  💡 提示：输入完毕后，按两次回车键结束输入".ljust(59) + "│")
    print("└" + "─" * 58 + "┘")
    print()
    
    lines = []
    empty_count = 0
    while True:
        try:
            line = input()
            if line == "":
                empty_count += 1
                if empty_count >= 2:
                    break
            else:
                empty_count = 0
            lines.append(line)
        except EOFError:
            break
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="📊 文章AI率检测器 - 基于多维统计特征分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python ai_detector.py                       # 交互式输入
  python ai_detector.py -f novel.txt          # 检测文件
  python ai_detector.py -f novel.txt -p       # 启用困惑度分析
  python ai_detector.py -f novel.txt -n 500   # 只分析前500字
        """
    )
    parser.add_argument('-f', '--file', type=str, 
                       help='待检测的文本文件路径')
    parser.add_argument('-p', '--perplexity', action='store_true',
                       help='启用困惑度分析（需要transformers和torch）')
    parser.add_argument('-n', '--max-chars', type=int, default=0,
                       help='最大分析字符数（0=不限制）')
    
    args = parser.parse_args()
    
    # Banner
    print("""
╔══════════════════════════════════════════════════╗
║                                                  ║
║         🤖 AI文本检测器 v1.0                    ║
║     AI Text Detector - Statistical Analysis      ║
║                                                  ║
║   基于多维统计特征分析文本是否由AI生成            ║
║                                                  ║
╚══════════════════════════════════════════════════╝
    """)
    
    # 读取文本
    if args.file:
        print(f"\n📂 正在读取文件: {args.file}")
        text = read_text_from_file(args.file)
    else:
        text = interactive_input()
    
    if not text or len(text.strip()) < 50:
        print("\n❌ 文本过短（至少需要50个字符），无法进行有效分析。")
        sys.exit(1)
    
    # 截断处理
    if args.max_chars > 0 and len(text) > args.max_chars:
        text = text[:args.max_chars]
        print(f"✂️  已截取前 {args.max_chars} 字进行分析\n")
    
    print(f"✅ 已加载文本: {len(text)} 字符")
    print("🔄 正在进行多维度分析...\n")
    
    # 运行分析
    analyzer = TextAnalyzer(text)
    report = analyzer.run_full_analysis(use_perplexity=args.perplexity)
    
    # 渲染并输出报告
    output = ReportRenderer.render(report)
    print(output)
    
    # 保存报告到文件（如果从文件读取）
    if args.file:
        report_file = args.file + ".ai_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"\n💾 报告已保存至: {report_file}")


if __name__ == "__main__":
    main()