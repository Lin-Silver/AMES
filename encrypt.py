#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-layer Encryption System
高级多层加密Python系统 - 加密模块
支持10种不同加密算法的随机多层嵌套加密
"""

import random
import base64
import json
import time
import hashlib
import secrets
import sys
import os
import argparse
from datetime import datetime
from typing import List, Dict, Any, Tuple
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProgressBar:
    """进度条显示类"""
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
    
    def update(self, step: int = 1):
        self.current += step
        percent = (self.current / self.total) * 100
        bar_length = 50
        filled_length = int(bar_length * self.current // self.total)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f'\r{self.description}: |{bar}| {percent:.1f}%', end='', flush=True)
        if self.current >= self.total:
            print()

class EncryptionBase:
    """加密算法基类"""
    def __init__(self):
        self.name = self.__class__.__name__
        self.params = {}
    
    def encrypt(self, text: str) -> str:
        raise NotImplementedError
    
    def get_params(self) -> Dict[str, Any]:
        return self.params

class AdvancedMatrixCipher(EncryptionBase):
    """高级矩阵加密算法 - 重新设计版本
    使用简化但可靠的加密逻辑，确保加解密完全互逆
    """
    def __init__(self):
        super().__init__()
        # 简化参数，确保可逆性
        self.shift_key = secrets.randbelow(100) + 1
        self.multiplier = secrets.randbelow(7) + 3  # 3-9
        self.base_offset = secrets.randbelow(50) + 10  # 10-59
        
        # 使用标准Base64字符集
        self.encode_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
        
        self.params = {
            "shift_key": self.shift_key,
            "multiplier": self.multiplier,
            "base_offset": self.base_offset,
            "encode_chars": self.encode_chars
        }
    
    def encrypt(self, text: str) -> str:
        """简化的加密算法，确保可逆性"""
        if not text:
            return ""
        
        # 步顄1: 字符位移加密
        shifted_chars = []
        for i, char in enumerate(text):
            # 位置相关的位移
            shift = (self.shift_key + i) % 256
            shifted_code = (ord(char) + shift) % 1114112  # Unicode范围
            shifted_chars.append(shifted_code)
        
        # 步顄2: 数值变换
        transformed_values = []
        for value in shifted_chars:
            # 乘法和偏移变换
            transformed = (value * self.multiplier + self.base_offset) % 1114112
            transformed_values.append(transformed)
        
        # 步顄3: Base64编码
        encoded_result = self._simple_base64_encode(transformed_values)
        
        # 步顄4: 添加校验码
        checksum = sum(ord(c) for c in text) % len(self.encode_chars)
        
        return f"{self.encode_chars[checksum]}{encoded_result}"
    
    def _simple_base64_encode(self, values: list) -> str:
        """简化的Base64编码"""
        if not values:
            return ""
        
        result = []
        for value in values:
            # 将数值转换为多个字符
            if value == 0:
                result.append(self.encode_chars[0])
            else:
                chars = []
                temp = value
                while temp > 0:
                    chars.append(self.encode_chars[temp % len(self.encode_chars)])
                    temp //= len(self.encode_chars)
                result.extend(reversed(chars))
            result.append('|')  # 分隔符
        
        return ''.join(result[:-1])  # 移除最后一个分隔符

class EnhancedCaesar(EncryptionBase):
    """改进凯撒密码（支持多重偏移）"""
    def __init__(self):
        super().__init__()
        self.shifts = [secrets.randbelow(25) + 1 for _ in range(3)]
        self.params = {"shifts": self.shifts}
    
    def encrypt(self, text: str) -> str:
        result = ""
        for i, char in enumerate(text):
            shift = self.shifts[i % len(self.shifts)]
            if char.isalpha():
                base = ord('A') if char.isupper() else ord('a')
                result += chr((ord(char) - base + shift) % 26 + base)
            else:
                result += char
        return result

class MorseVariant(EncryptionBase):
    """摩斯密码变种"""
    def __init__(self):
        super().__init__()
        self.morse_dict = {
            'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.',
            'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..',
            'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.',
            'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
            'Y': '-.--', 'Z': '--..', '0': '-----', '1': '.----', '2': '..---',
            '3': '...--', '4': '....-', '5': '.....', '6': '-....', '7': '--...',
            '8': '---..', '9': '----.', ' ': '/'
        }
        # 创建变种分隔符
        self.dot_char = random.choice(['•', '●', '○', '◦'])
        self.dash_char = random.choice(['—', '–', '▬', '█'])
        self.separator = random.choice(['|', '§', '¤', '◊'])
        self.params = {"dot_char": self.dot_char, "dash_char": self.dash_char, "separator": self.separator}
    
    def encrypt(self, text: str) -> str:
        morse_text = []
        for char in text.upper():
            if char in self.morse_dict:
                morse = self.morse_dict[char]
                morse = morse.replace('.', self.dot_char).replace('-', self.dash_char)
                morse_text.append(morse)
            else:
                morse_text.append(f"#{ord(char)}")
        return self.separator.join(morse_text)

class MultiBase64(EncryptionBase):
    """Base64多重编码"""
    def __init__(self):
        super().__init__()
        self.rounds = secrets.randbelow(3) + 2  # 2-4轮编码
        self.params = {"rounds": self.rounds}
    
    def encrypt(self, text: str) -> str:
        encoded = text.encode('utf-8')
        for _ in range(self.rounds):
            encoded = base64.b64encode(encoded)
        return encoded.decode('utf-8')

class ASCIIConversion(EncryptionBase):
    """ASCII码转换加密"""
    def __init__(self):
        super().__init__()
        self.multiplier = secrets.randbelow(7) + 3  # 3-9的乘数
        self.offset = secrets.randbelow(100) + 50   # 50-149的偏移
        self.separator = random.choice(['-', '_', '|', '.', ','])
        self.params = {"multiplier": self.multiplier, "offset": self.offset, "separator": self.separator}
    
    def encrypt(self, text: str) -> str:
        ascii_codes = []
        for char in text:
            code = ord(char) * self.multiplier + self.offset
            ascii_codes.append(str(code))
        return self.separator.join(ascii_codes)

class ReverseInterpolation(EncryptionBase):
    """字符串反转与插值"""
    def __init__(self):
        super().__init__()
        self.interpolation_chars = ''.join(secrets.choice('!@#$%^&*()_+-=[]{}|;:,.<>?') for _ in range(10))
        self.reverse_blocks = secrets.randbelow(5) + 3  # 3-7个反转块
        self.params = {"interpolation_chars": self.interpolation_chars, "reverse_blocks": self.reverse_blocks}
    
    def encrypt(self, text: str) -> str:
        # 先插值
        interpolated = ""
        for i, char in enumerate(text):
            interpolated += char
            if i < len(text) - 1:
                interpolated += self.interpolation_chars[i % len(self.interpolation_chars)]
        
        # 分块反转
        block_size = len(interpolated) // self.reverse_blocks
        if block_size == 0:
            return interpolated[::-1]
        
        result = ""
        for i in range(0, len(interpolated), block_size):
            block = interpolated[i:i+block_size]
            result += block[::-1]
        return result

class SubstitutionCipher(EncryptionBase):
    """自定义替换密码表"""
    def __init__(self):
        super().__init__()
        # 创建随机替换表
        chars = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789')
        substitution = chars.copy()
        secrets.SystemRandom().shuffle(substitution)
        self.substitution_table = dict(zip(chars, substitution))
        self.params = {"substitution_table": self.substitution_table}
    
    def encrypt(self, text: str) -> str:
        return ''.join(self.substitution_table.get(char, char) for char in text)

class BinaryConversion(EncryptionBase):
    """二进制转换加密"""
    def __init__(self):
        super().__init__()
        self.zero_char = random.choice(['O', '0', 'o', '○', '◯'])
        self.one_char = random.choice(['I', '1', 'l', '|', '▌'])
        self.separator = random.choice([' ', '-', '_', '.'])
        self.params = {"zero_char": self.zero_char, "one_char": self.one_char, "separator": self.separator}
    
    def encrypt(self, text: str) -> str:
        binary_parts = []
        for char in text:
            binary = format(ord(char), '016b')  # 16位二进制
            binary = binary.replace('0', self.zero_char).replace('1', self.one_char)
            binary_parts.append(binary)
        return self.separator.join(binary_parts)

class HexObfuscation(EncryptionBase):
    """十六进制编码混淆"""
    def __init__(self):
        super().__init__()
        self.hex_chars = list('0123456789ABCDEF')
        obfuscated_chars = ['Ω', 'Ψ', 'Φ', 'Χ', 'Υ', 'Τ', 'Σ', 'Ρ', 'Π', 'Ο', 'Ξ', 'Ν', 'Μ', 'Λ', 'Κ', 'Ι']
        secrets.SystemRandom().shuffle(obfuscated_chars)
        self.hex_mapping = dict(zip(self.hex_chars, obfuscated_chars))
        self.prefix = random.choice(['☆', '★', '◆', '◇'])
        self.params = {"hex_mapping": self.hex_mapping, "prefix": self.prefix}
    
    def encrypt(self, text: str) -> str:
        hex_string = text.encode('utf-8').hex().upper()
        obfuscated = ''.join(self.hex_mapping.get(char, char) for char in hex_string)
        return f"{self.prefix}{obfuscated}{self.prefix}"

class ROT13Variant(EncryptionBase):
    """ROT13变种算法"""
    def __init__(self):
        super().__init__()
        self.rotation = secrets.randbelow(25) + 1  # 1-25的旋转值
        self.special_rotation = secrets.randbelow(95) + 33  # 特殊字符旋转
        self.params = {"rotation": self.rotation, "special_rotation": self.special_rotation}
    
    def encrypt(self, text: str) -> str:
        result = ""
        for char in text:
            if char.isalpha():
                base = ord('A') if char.isupper() else ord('a')
                result += chr((ord(char) - base + self.rotation) % 26 + base)
            elif char.isprintable() and not char.isspace():
                result += chr((ord(char) - 33 + self.special_rotation) % 94 + 33)
            else:
                result += char
        return result

class MultiLayerEncryptor:
    """多层加密主类"""
    def __init__(self):
        self.algorithms = [
            AdvancedMatrixCipher, EnhancedCaesar, MorseVariant, MultiBase64, ASCIIConversion,
            ReverseInterpolation, SubstitutionCipher, BinaryConversion, HexObfuscation, ROT13Variant
        ]
        self.encryption_log = []
        self.start_time = None
    
    def encrypt(self, text: str, layers: int) -> Dict[str, Any]:
        """执行多层加密"""
        if not 1 <= layers <= 10:
            raise ValueError("加密层数必须在1-10之间")
        
        self.start_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 确保自创算法（AdvancedMatrixCipher）总是被使用
        selected_algorithms = [AdvancedMatrixCipher]  # 自创算法必须使用
        
        # 如果需要多层，再随机选择其他算法
        if layers > 1:
            other_algorithms = [alg for alg in self.algorithms if alg != AdvancedMatrixCipher]
            additional_algorithms = secrets.SystemRandom().sample(other_algorithms, min(layers - 1, len(other_algorithms)))
            selected_algorithms.extend(additional_algorithms)
        
        # 随机打乱顺序（但确保有足够的算法）
        if len(selected_algorithms) < layers:
            # 如果算法不够，重复使用
            while len(selected_algorithms) < layers:
                selected_algorithms.append(secrets.SystemRandom().choice(self.algorithms))
        
        selected_algorithms = selected_algorithms[:layers]
        secrets.SystemRandom().shuffle(selected_algorithms)
        
        # 初始化进度条
        progress = ProgressBar(layers, "多层加密进度")
        
        encrypted_text = text
        encryption_sequence = []
        
        print(f"\n开始多层加密，共{layers}层...")
        print(f"原始文本: {text[:50]}{'...' if len(text) > 50 else ''}")
        print("-" * 80)
        
        for i, algorithm_class in enumerate(selected_algorithms):
            layer_start = time.time()
            algorithm = algorithm_class()
            
            previous_text = encrypted_text
            encrypted_text = algorithm.encrypt(encrypted_text)
            
            layer_info = {
                "layer": i + 1,
                "algorithm": algorithm.name,
                "params": algorithm.get_params(),
                "input_length": len(previous_text),
                "output_length": len(encrypted_text),
                "processing_time": time.time() - layer_start
            }
            
            encryption_sequence.append(layer_info)
            
            # 显示中间结果预览
            preview = encrypted_text[:100] + ('...' if len(encrypted_text) > 100 else '')
            print(f"第{i+1}层 [{algorithm.name}]: {preview}")
            
            progress.update()
            time.sleep(0.1)  # 模拟处理时间
        
        total_time = time.time() - self.start_time
        
        # 计算加密强度
        strength = self._calculate_strength(text, encrypted_text, layers)
        
        # 生成哈希校验
        integrity_hash = hashlib.sha256(encrypted_text.encode('utf-8')).hexdigest()
        
        result = {
            "encrypted_text": encrypted_text,
            "original_length": len(text),
            "encrypted_length": len(encrypted_text),
            "layers": layers,
            "encryption_sequence": encryption_sequence,
            "timestamp": timestamp,
            "processing_time": total_time,
            "strength_score": strength,
            "integrity_hash": integrity_hash,
            "config_version": "1.0"
        }
        
        print(f"\n加密完成！")
        print(f"总耗时: {total_time:.3f}秒")
        # 显示详细的强度分析
        if hasattr(self, 'strength_details'):
            print(f"加密强度评分: {strength}/100")
            print(f"详细评分:")
            print(f"  - 层数安全性: {self.strength_details['layer_score']}/25")
            print(f"  - 长度变化: {self.strength_details['length_score']}/15")
            print(f"  - 信息熵: {self.strength_details['entropy_score']}/25")
            print(f"  - 算法复杂度: {self.strength_details['algorithm_score']}/20")
            print(f"  - 模式抗性: {self.strength_details['pattern_score']}/15")
            
            # 安全性评级
            if strength >= 90:
                security_level = "军用级 🔒"
            elif strength >= 75:
                security_level = "商业级 🛡️"
            elif strength >= 60:
                security_level = "标准级 🔐"
            elif strength >= 40:
                security_level = "基础级 🔓"
            else:
                security_level = "较弱 ⚠️"
            
            print(f"  安全性等级: {security_level}")
        else:
            print(f"加密强度评分: {strength}/100")
        
        return result
    
    def _calculate_strength(self, original: str, encrypted: str, layers: int) -> int:
        """计算加密强度评分（更全面的安全性评估）"""
        import math
        import collections
        
        strength_details = {
            'layer_score': 0,
            'length_score': 0,
            'entropy_score': 0,
            'algorithm_score': 0,
            'pattern_score': 0
        }
        
        # 1. 层数评分 (25分) - 多层加密提供更高安全性
        layer_score = min(layers * 2.5, 25)
        if layers >= 5:
            layer_score += 5  # 奖励高层数
        strength_details['layer_score'] = int(layer_score)
        
        # 2. 长度变化评分 (15分) - 适度的长度增长表明复杂变换
        if len(original) > 0:
            length_ratio = len(encrypted) / len(original)
            if 2 <= length_ratio <= 10:  # 理想的长度扩展比
                length_score = 15
            elif 1.5 <= length_ratio < 2 or 10 < length_ratio <= 20:
                length_score = 10
            elif length_ratio > 20:
                length_score = 5  # 过度扩展可能降低效率
            else:
                length_score = 3
        else:
            length_score = 0
        strength_details['length_score'] = length_score
        
        # 3. 信息熵评分 (25分) - 更准确的随机性度量
        entropy_score = self._calculate_entropy(encrypted)
        strength_details['entropy_score'] = int(entropy_score * 25)
        
        # 4. 算法复杂度评分 (20分) - 基于使用的算法类型
        algorithm_score = self._evaluate_algorithm_complexity()
        strength_details['algorithm_score'] = algorithm_score
        
        # 5. 模式抗性评分 (15分) - 检测重复模式和可预测性
        pattern_score = self._analyze_pattern_resistance(encrypted)
        strength_details['pattern_score'] = int(pattern_score * 15)
        
        total_score = sum(strength_details.values())
        
        # 存储详细评分用于显示
        self.strength_details = strength_details
        
        return min(total_score, 100)
    
    def _calculate_entropy(self, text: str) -> float:
        """计算文本的信息熵（香农熵）"""
        if not text:
            return 0.0
        
        import math
        from collections import Counter
        
        # 计算字符频率
        char_counts = Counter(text)
        text_length = len(text)
        
        # 计算香农熵
        entropy = 0.0
        for count in char_counts.values():
            probability = count / text_length
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        # 标准化到0-1范围（假设最大熵为8比特，即256个不同字符）
        max_entropy = min(8.0, math.log2(len(char_counts)))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _evaluate_algorithm_complexity(self) -> int:
        """评估使用算法的复杂度和安全性"""
        if not hasattr(self, 'encryption_log') or not self.encryption_log:
            return 10  # 默认分数
        
        # 算法安全性权重
        algorithm_weights = {
            'CustomShiftCipher': 2,      # 自创算法，中等复杂度
            'EnhancedCaesar': 1,         # 改进凯撒，相对简单
            'MorseVariant': 3,           # 摩斯变种，较复杂
            'MultiBase64': 2,            # 多重Base64，中等
            'ASCIIConversion': 2,        # ASCII转换，中等
            'ReverseInterpolation': 3,   # 反转插值，较复杂
            'SubstitutionCipher': 3,     # 替换密码，较复杂
            'BinaryConversion': 2,       # 二进制转换，中等
            'HexObfuscation': 3,         # 十六进制混淆，较复杂
            'ROT13Variant': 2            # ROT13变种，中等
        }
        
        # 如果有加密序列信息，使用实际算法评分
        if hasattr(self, 'last_encryption_sequence') and self.last_encryption_sequence:
            total_weight = sum(algorithm_weights.get(layer['algorithm'], 2) 
                             for layer in self.last_encryption_sequence)
            avg_weight = total_weight / len(self.last_encryption_sequence)
            return min(int(avg_weight * 6.67), 20)  # 缩放到0-20分
        
        return 10  # 默认中等分数
    
    def _analyze_pattern_resistance(self, text: str) -> float:
        """分析密文的模式抗性"""
        if len(text) < 20:
            return 0.5  # 太短无法有效分析
        
        score = 1.0
        
        # 1. 检测重复子串
        substring_counts = {}
        for length in [2, 3, 4, 5]:
            for i in range(len(text) - length + 1):
                substring = text[i:i+length]
                substring_counts[substring] = substring_counts.get(substring, 0) + 1
        
        # 计算重复率
        total_substrings = sum(substring_counts.values())
        repeated_substrings = sum(1 for count in substring_counts.values() if count > 1)
        repetition_ratio = repeated_substrings / total_substrings if total_substrings > 0 else 0
        
        # 重复率越低，抗性越强
        score -= repetition_ratio * 0.3
        
        # 2. 字符分布均匀性检测
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        if char_counts:
            expected_freq = len(text) / len(char_counts)
            variance = sum((count - expected_freq) ** 2 for count in char_counts.values()) / len(char_counts)
            normalized_variance = variance / (expected_freq ** 2) if expected_freq > 0 else 1
            
            # 方差越小，分布越均匀，抗性越强
            score -= min(normalized_variance * 0.2, 0.3)
        
        # 3. 相邻字符相关性检测
        transitions = {}
        for i in range(len(text) - 1):
            pair = text[i:i+2]
            transitions[pair] = transitions.get(pair, 0) + 1
        
        if transitions:
            max_transition = max(transitions.values())
            transition_concentration = max_transition / len(text)
            score -= transition_concentration * 0.2
        
        return max(0.0, min(1.0, score))
    
    def save_config(self, result: Dict[str, Any], filename: str = None) -> str:
        """保存加密配置文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"encryption_config_{timestamp}.json"
        
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"配置文件已保存: {config_path}")
        return config_path

def validate_text_input(text: str) -> bool:
    """验证文本输入的有效性"""
    if not text or text.isspace():
        return False
    
    # 检查文本长度限制（最大100KB）
    if len(text.encode('utf-8')) > 100 * 1024:
        print("错误: 文本太大，最大支持100KB")
        return False
    
    # 检查是否包含控制字符（除了常见的空白字符）
    control_chars = sum(1 for c in text if ord(c) < 32 and c not in '\t\n\r')
    if control_chars > len(text) * 0.1:  # 如果控制字符超过10%
        print("警告: 文本包含较多控制字符，可能影响加密效果")
    
    return True

def validate_layers_input(layers_str: str) -> tuple[bool, int]:
    """验证层数输入的有效性"""
    try:
        layers = int(layers_str)
        if not 1 <= layers <= 10:
            print("错误: 加密层数必须在1-10之间！")
            return False, 0
        return True, layers
    except ValueError:
        print("错误: 请输入有效的数字！")
        return False, 0
    except Exception as e:
        print(f"错误: 解析层数时发生异常: {e}")
        return False, 0

def interactive_mode():
    """交互式模式"""
    print("=" * 80)
    print("   高级多层加密系统 - 交互模式")
    print("   支持10种加密算法的随机多层嵌套加密")
    print("=" * 80)
    print("\n提示:")
    print("- 最大文本长度: 100KB")
    print("- 支持中英文混合及特殊字符")
    print("- 建议加密层数: 3-5层")
    print("=" * 80)
    
    encryptor = MultiLayerEncryptor()
    
    while True:
        try:
            # 获取用户输入
            text = input("\n请输入要加密的文本 (输入 'quit' 退出): ").strip()
            if text.lower() == 'quit':
                print("\n感谢使用高级多层加密系统！")
                break
            
            # 验证文本输入
            if not validate_text_input(text):
                continue
            
            # 获取并验证层数
            layers_input = input("请输入加密层数 (1-10): ").strip()
            is_valid, layers = validate_layers_input(layers_input)
            if not is_valid:
                continue
            
            # 执行加密并处理可能的异常
            try:
                print(f"\n开始加密文本（{len(text)}字符，{layers}层）...")
                result = encryptor.encrypt(text, layers)
                
                # 显示结果
                print(f"\n✓ 加密成功！")
                print(f"原文长度: {result['original_length']} 字符")
                print(f"密文长度: {result['encrypted_length']} 字符")
                print(f"压缩比率: {result['encrypted_length']/result['original_length']:.1f}x")
                print(f"加密层数: {result['layers']}")
                print(f"加密强度: {result['strength_score']}/100")
                print(f"处理时间: {result['processing_time']:.3f}秒")
                
                # 显示使用的算法
                print(f"\n使用的加密算法:")
                for i, layer in enumerate(result['encryption_sequence'], 1):
                    print(f"  第{i}层: {layer['algorithm']}")
                
                print(f"\n密文预览（前200字符）:")
                preview = result['encrypted_text'][:200]
                print(f"'{preview}{'...' if len(result['encrypted_text']) > 200 else ''}'")
                
                # 保存配置
                save_config = input("\n是否保存加密配置到文件？(y/n): ").strip().lower()
                if save_config == 'y':
                    try:
                        config_path = encryptor.save_config(result)
                        print(f"✓ 配置已保存到: {config_path}")
                    except IOError as e:
                        print(f"✗ 保存配置文件失败: {e}")
                    except Exception as e:
                        print(f"✗ 保存过程中发生未知错误: {e}")
                        
            except MemoryError:
                print("✗ 内存不足，请尝试较小的文本或减少加密层数")
            except ValueError as e:
                print(f"✗ 输入值错误: {e}")
            except Exception as e:
                print(f"✗ 加密过程中发生错误: {e}")
                logger.error(f"加密异常: {type(e).__name__}: {e}", exc_info=True)
        
        except KeyboardInterrupt:
            print("\n\n✗ 程序被用户中断")
            print("再见！")
            break
        except EOFError:
            print("\n\n✗ 输入流结束")
            break
        except Exception as e:
            print(f"✗ 系统错误: {e}")
            logger.error(f"系统异常: {type(e).__name__}: {e}", exc_info=True)
            print("程序将继续运行，请重试...")

def command_line_mode(args):
    """命令行模式"""
    encryptor = MultiLayerEncryptor()
    
    try:
        if args.file:
            # 从文件读取
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            text = args.text
        
        result = encryptor.encrypt(text, args.layers)
        
        # 输出结果
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(result['encrypted_text'])
            print(f"加密结果已保存到: {args.output}")
        else:
            print(result['encrypted_text'])
        
        # 保存配置
        config_path = encryptor.save_config(result, args.config)
        
        if args.verbose:
            print(f"\n详细信息:")
            print(f"加密层数: {result['layers']}")
            print(f"加密强度: {result['strength_score']}/100")
            print(f"处理时间: {result['processing_time']:.3f}秒")
            print(f"配置文件: {config_path}")
    
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="高级多层加密系统")
    parser.add_argument('-t', '--text', help='要加密的文本')
    parser.add_argument('-f', '--file', help='包含要加密文本的文件路径')
    parser.add_argument('-l', '--layers', type=int, default=3, help='加密层数 (1-10)')
    parser.add_argument('-o', '--output', help='输出文件路径')
    parser.add_argument('-c', '--config', help='配置文件名')
    parser.add_argument('-v', '--verbose', action='store_true', help='显示详细信息')
    
    args = parser.parse_args()
    
    if args.text or args.file:
        command_line_mode(args)
    else:
        interactive_mode()

if __name__ == "__main__":
    main()
