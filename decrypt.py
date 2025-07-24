#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-layer Decryption System
高级多层加密Python系统 - 解密模块
支持配置文件解密、暴力破解和智能识别
"""

import json
import base64
import time
import hashlib
import sys
import os
import argparse
import itertools
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

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

class DecryptionBase:
    """解密算法基类"""
    def __init__(self, params: Dict[str, Any] = None):
        self.name = self.__class__.__name__
        self.params = params or {}
    
    def decrypt(self, text: str) -> str:
        raise NotImplementedError

class AdvancedMatrixDecryptor(DecryptionBase):
    """高级矩阵加密算法解密 - 重新设计版本"""
    def decrypt(self, text: str) -> str:
        if not text or len(text) < 2:
            return ""
        
        try:
            # 获取参数
            shift_key = self.params.get("shift_key", 1)
            multiplier = self.params.get("multiplier", 3)
            base_offset = self.params.get("base_offset", 10)
            encode_chars = self.params.get("encode_chars", 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/')
            
            # 移除校验码
            checksum_char = text[0]
            encoded_data = text[1:]
            
            # 步骤1: Base64解码
            decoded_values = self._simple_base64_decode(encoded_data, encode_chars)
            if not decoded_values:
                return ""
            
            # 步骤2: 逆数值变换 (完全逆向加密的步骤2)
            # 加密: (value * multiplier + base_offset) % 1114112
            # 解密: (transformed_value - base_offset) * mod_inverse(multiplier) % 1114112
            shifted_chars = []
            
            # 计算乘法逆元素
            inv_multiplier = self._mod_inverse(multiplier, 1114112)
            if inv_multiplier is None:
                # 如果没有逆元，尝试简单除法
                for transformed_value in decoded_values:
                    temp_value = (transformed_value - base_offset + 1114112) % 1114112
                    original_value = temp_value // multiplier
                    shifted_chars.append(original_value)
            else:
                for transformed_value in decoded_values:
                    # 先减去偏移，确保结果为正
                    temp_value = (transformed_value - base_offset + 1114112) % 1114112
                    # 再乘以逆元
                    original_value = (temp_value * inv_multiplier) % 1114112
                    shifted_chars.append(original_value)
            
            # 步骤3: 逆位移 (完全逆向加密的步骤1)
            # 加密: (ord(char) + shift) % 1114112
            # 解密: (shifted_code - shift) % 1114112
            result_chars = []
            for i, shifted_code in enumerate(shifted_chars):
                shift = (shift_key + i) % 256
                # 确保正确的模运算
                original_code = (shifted_code - shift + 1114112) % 1114112
                try:
                    if 0 <= original_code <= 1114111:  # 检查Unicode范围
                        result_chars.append(chr(original_code))
                    else:
                        result_chars.append('?')
                except (ValueError, OverflowError):
                    result_chars.append('?')
            
            return ''.join(result_chars)
            
        except Exception as e:
            return text  # 如果解密失败，返回原文
    
    def _mod_inverse(self, a: int, m: int) -> int:
        """计算模逆元素（使用扩展欧几里得算法）"""
        def extended_gcd(a, b):
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd, x, y
        
        gcd, x, _ = extended_gcd(a % m, m)
        if gcd != 1:
            return None  # 逆元不存在
        return (x % m + m) % m
    
    def _simple_base64_decode(self, encoded_text: str, encode_chars: str) -> list:
        """简化的Base64解码（与新的编码完全匹配）"""
        if not encoded_text:
            return []
        
        try:
            # 构建解码映射
            decode_map = {char: i for i, char in enumerate(encode_chars)}
            char_count = len(encode_chars)
            
            # 按分隔符分割
            parts = encoded_text.split('|')
            result = []
            
            for part in parts:
                if not part:
                    continue
                
                # 将每个部分转换回数值（限制在合理范围内）
                value = 0
                for char in part:
                    if char in decode_map:
                        value = value * char_count + decode_map[char]
                        # 限制值的大小以防止溢出
                        if value > 1114111:  # Unicode最大值
                            value = value % 1114112
                    else:
                        value = 0
                        break
                result.append(value)
            
            return result
            
        except Exception as e:
            return []
    
    def _calculate_inverse_matrix(self, matrix):
        """计算逆矩阵（简化版，仅支持2x2和3x3）"""
        try:
            size = len(matrix)
            if size == 2:
                det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
                if abs(det) < 0.001:
                    return None
                return [[matrix[1][1]/det, -matrix[0][1]/det],
                       [-matrix[1][0]/det, matrix[0][0]/det]]
            elif size == 3:
                # 简化的3x3逆矩阵计算
                det = (matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
                      matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
                      matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]))
                if abs(det) < 0.001:
                    return None
                # 返回简化的逆矩阵（近似值）
                return [[1/matrix[0][0] if matrix[0][0] != 0 else 1, 0, 0],
                       [0, 1/matrix[1][1] if matrix[1][1] != 0 else 1, 0],
                       [0, 0, 1/matrix[2][2] if matrix[2][2] != 0 else 1]]
            else:
                # 对于其他大小，返回对角逆矩阵
                inverse = [[0] * size for _ in range(size)]
                for i in range(size):
                    inverse[i][i] = 1/matrix[i][i] if matrix[i][i] != 0 else 1
                return inverse
        except:
            return None
    
    def _matrix_multiply(self, matrix, vector):
        """矩阵与向量相乘（纯 Python 实现）"""
        result = []
        for i in range(len(matrix)):
            row_sum = 0
            for j in range(len(vector)):
                row_sum += matrix[i][j] * vector[j]
            result.append(int(row_sum) % 256)
        return result
    
    def _custom_base64_decode(self, encoded_text: str, encode_chars: str) -> list:
        """自定义Base64解码（完全逆向编码过程）"""
        if not encoded_text:
            return []
        
        try:
            # 构建解码映射
            decode_map = {char: i for i, char in enumerate(encode_chars)}
            char_count = len(encode_chars)
            
            # 逆向工程：根据编码策略重建原始数据
            # 编码策略：每3个字节 -> 1-3个字符
            result = []
            i = 0
            
            while i < len(encoded_text):
                # 预测下一组的大小（通过模式匹配）
                remaining = len(encoded_text) - i
                
                if remaining >= 3:
                    # 尝试3字符解码为3字节
                    chars = [encoded_text[i], encoded_text[i+1], encoded_text[i+2]]
                    indices = [decode_map.get(c, 0) for c in chars]
                    
                    # 重建 combined 值（逆向计算）
                    # 原编码: combined = (group[0] << 16) | (group[1] << 8) | group[2]
                    # 输出: [combined % char_count, (combined >> 6) % char_count, (combined >> 12) % char_count]
                    
                    # 使用中国剩余定理重建 combined
                    # 这是一个近似解，因为原始编码有信息损失
                    val1, val2, val3 = indices[0], indices[1], indices[2]
                    
                    # 尝试重建原始 combined
                    # combined % char_count = val1
                    # (combined >> 6) % char_count = val2  
                    # (combined >> 12) % char_count = val3
                    
                    # 简化重建（近似）
                    combined = val1 + (val2 << 6) + (val3 << 12)
                    
                    # 提取原始3个字节
                    byte1 = (combined >> 16) & 0xFF
                    byte2 = (combined >> 8) & 0xFF  
                    byte3 = combined & 0xFF
                    
                    result.extend([byte1, byte2, byte3])
                    i += 3
                    
                elif remaining >= 2:
                    # 2字符解码为2字节
                    chars = [encoded_text[i], encoded_text[i+1]]
                    indices = [decode_map.get(c, 0) for c in chars]
                    
                    # 重建 combined
                    combined = indices[0] + (indices[1] << 6)
                    
                    byte1 = (combined >> 8) & 0xFF
                    byte2 = combined & 0xFF
                    
                    result.extend([byte1, byte2])
                    i += 2
                    
                else:
                    # 1字符解码为1字节
                    char = encoded_text[i]
                    value = decode_map.get(char, 0)
                    result.append(value)
                    i += 1
            
            # 移除末尾的零字节
            while result and result[-1] == 0:
                result.pop()
                
            return result
            
        except Exception:
            return []

class CustomShiftDecryptor(DecryptionBase):
    """自创字符位移混淆算法解密"""
    def decrypt(self, text: str) -> str:
        shift_pattern = self.params.get("shift_pattern", [])
        result = ""
        for i, char in enumerate(text):
            shift = shift_pattern[i % len(shift_pattern)]
            if char.isalpha():
                base = ord('A') if char.isupper() else ord('a')
                result += chr((ord(char) - base - shift) % 26 + base)
            else:
                result += chr((ord(char) - shift) % 1114112)
        return result

class EnhancedCaesarDecryptor(DecryptionBase):
    """改进凯撒密码解密"""
    def decrypt(self, text: str) -> str:
        shifts = self.params.get("shifts", [])
        result = ""
        for i, char in enumerate(text):
            shift = shifts[i % len(shifts)]
            if char.isalpha():
                base = ord('A') if char.isupper() else ord('a')
                result += chr((ord(char) - base - shift) % 26 + base)
            else:
                result += char
        return result

class MorseVariantDecryptor(DecryptionBase):
    """摩斯密码变种解密"""
    def decrypt(self, text: str) -> str:
        reverse_morse = {
            '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E', '..-.': 'F',
            '--.': 'G', '....': 'H', '..': 'I', '.---': 'J', '-.-': 'K', '.-..': 'L',
            '--': 'M', '-.': 'N', '---': 'O', '.--.': 'P', '--.-': 'Q', '.-.': 'R',
            '...': 'S', '-': 'T', '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X',
            '-.--': 'Y', '--..': 'Z', '-----': '0', '.----': '1', '..---': '2',
            '...--': '3', '....-': '4', '.....': '5', '-....': '6', '--...': '7',
            '---..': '8', '----.': '9', '/': ' '
        }
        
        dot_char = self.params.get("dot_char", "•")
        dash_char = self.params.get("dash_char", "—")
        separator = self.params.get("separator", "|")
        
        # 替换回标准摩斯符号
        text = text.replace(dot_char, '.').replace(dash_char, '-')
        parts = text.split(separator)
        
        result = ""
        for part in parts:
            if part.startswith('#') and part[1:].isdigit():
                result += chr(int(part[1:]))
            elif part in reverse_morse:
                result += reverse_morse[part]
            else:
                result += '?'
        
        return result

class MultiBase64Decryptor(DecryptionBase):
    """Base64多重解码"""
    def decrypt(self, text: str) -> str:
        rounds = self.params.get("rounds", 2)
        decoded = text.encode('utf-8')
        for _ in range(rounds):
            decoded = base64.b64decode(decoded)
        return decoded.decode('utf-8')

class ASCIIConversionDecryptor(DecryptionBase):
    """ASCII码转换解密"""
    def decrypt(self, text: str) -> str:
        multiplier = self.params.get("multiplier", 3)
        offset = self.params.get("offset", 50)
        separator = self.params.get("separator", "-")
        
        codes = text.split(separator)
        result = ""
        for code_str in codes:
            try:
                code = int(code_str)
                original_code = (code - offset) // multiplier
                result += chr(original_code)
            except (ValueError, OverflowError):
                result += '?'
        return result

class ReverseInterpolationDecryptor(DecryptionBase):
    """字符串反转与插值解密"""
    def decrypt(self, text: str) -> str:
        interpolation_chars = self.params.get("interpolation_chars", "")
        reverse_blocks = self.params.get("reverse_blocks", 3)
        
        # 恢复分块反转
        block_size = len(text) // reverse_blocks
        if block_size == 0:
            reversed_text = text[::-1]
        else:
            reversed_text = ""
            for i in range(0, len(text), block_size):
                block = text[i:i+block_size]
                reversed_text += block[::-1]
        
        # 移除插值字符
        result = ""
        interpolation_set = set(interpolation_chars)
        for i, char in enumerate(reversed_text):
            if i % 2 == 0:  # 原始字符位置
                result += char
            # 跳过插值字符
        
        return result

class SubstitutionDecryptor(DecryptionBase):
    """自定义替换密码表解密"""
    def decrypt(self, text: str) -> str:
        substitution_table = self.params.get("substitution_table", {})
        # 创建反向映射
        reverse_table = {v: k for k, v in substitution_table.items()}
        return ''.join(reverse_table.get(char, char) for char in text)

class BinaryConversionDecryptor(DecryptionBase):
    """二进制转换解密"""
    def decrypt(self, text: str) -> str:
        zero_char = self.params.get("zero_char", "O")
        one_char = self.params.get("one_char", "I")
        separator = self.params.get("separator", " ")
        
        if not text or separator not in text:
            return text
        
        try:
            binary_parts = text.split(separator)
            result = ""
            for binary_str in binary_parts:
                if not binary_str:  # 跳过空字符串
                    continue
                # 转换回标准二进制
                binary = binary_str.replace(zero_char, '0').replace(one_char, '1')
                if len(binary) == 16:  # 确保是16位二进制
                    char_code = int(binary, 2)
                    if 0 <= char_code <= 1114111:  # Unicode范围检查
                        result += chr(char_code)
                    else:
                        result += '?'
                else:
                    result += '?'
            return result
        except Exception:
            return text

class HexObfuscationDecryptor(DecryptionBase):
    """十六进制编码混淆解密"""
    def decrypt(self, text: str) -> str:
        hex_mapping = self.params.get("hex_mapping", {})
        prefix = self.params.get("prefix", "☆")
        
        # 移除前缀后缀
        if text.startswith(prefix) and text.endswith(prefix):
            text = text[len(prefix):-len(prefix)]
        
        # 创建反向映射
        reverse_mapping = {v: k for k, v in hex_mapping.items()}
        hex_string = ''.join(reverse_mapping.get(char, char) for char in text)
        
        try:
            return bytes.fromhex(hex_string).decode('utf-8')
        except (ValueError, UnicodeDecodeError):
            return text

class ROT13VariantDecryptor(DecryptionBase):
    """ROT13变种算法解密"""
    def decrypt(self, text: str) -> str:
        rotation = self.params.get("rotation", 13)
        special_rotation = self.params.get("special_rotation", 47)
        
        result = ""
        for char in text:
            if char.isalpha():
                base = ord('A') if char.isupper() else ord('a')
                # 确保正确的模运算：加26确保结果为正
                result += chr((ord(char) - base - rotation + 26) % 26 + base)
            elif char.isprintable() and not char.isspace():
                # 确保正确的模运算：加94确保结果为正
                result += chr((ord(char) - 33 - special_rotation + 94) % 94 + 33)
            else:
                result += char
        return result

class MultiLayerDecryptor:
    """多层解密主类"""
    def __init__(self):
        self.decryptor_map = {
            'AdvancedMatrixCipher': AdvancedMatrixDecryptor,
            'EnhancedCaesar': EnhancedCaesarDecryptor,
            'MorseVariant': MorseVariantDecryptor,
            'MultiBase64': MultiBase64Decryptor,
            'ASCIIConversion': ASCIIConversionDecryptor,
            'ReverseInterpolation': ReverseInterpolationDecryptor,
            'SubstitutionCipher': SubstitutionDecryptor,
            'BinaryConversion': BinaryConversionDecryptor,
            'HexObfuscation': HexObfuscationDecryptor,
            'ROT13Variant': ROT13VariantDecryptor
        }
    
    def decrypt_with_config(self, config_path: str) -> Dict[str, Any]:
        """使用配置文件解密"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        except json.JSONDecodeError:
            raise ValueError(f"配置文件格式错误: {config_path}")
        
        encrypted_text = config['encrypted_text']
        encryption_sequence = config['encryption_sequence']
        
        # 验证完整性
        integrity_hash = hashlib.sha256(encrypted_text.encode('utf-8')).hexdigest()
        if integrity_hash != config.get('integrity_hash'):
            print("警告: 密文完整性校验失败，可能已被篡改!")
        
        print(f"开始解密，共{len(encryption_sequence)}层...")
        progress = ProgressBar(len(encryption_sequence), "多层解密进度")
        
        decrypted_text = encrypted_text
        
        # 逆序解密
        for i, layer_info in enumerate(reversed(encryption_sequence)):
            algorithm_name = layer_info['algorithm']
            params = layer_info['params']
            
            if algorithm_name not in self.decryptor_map:
                raise ValueError(f"未知的解密算法: {algorithm_name}")
            
            decryptor_class = self.decryptor_map[algorithm_name]
            decryptor = decryptor_class(params)
            
            previous_text = decrypted_text
            decrypted_text = decryptor.decrypt(decrypted_text)
            
            # 显示中间结果
            preview = decrypted_text[:100] + ('...' if len(decrypted_text) > 100 else '')
            layer_num = len(encryption_sequence) - i
            print(f"解密第{layer_num}层 [{algorithm_name}]: {preview}")
            
            progress.update()
        
        return {
            "decrypted_text": decrypted_text,
            "layers_processed": len(encryption_sequence),
            "original_config": config
        }
    
    def brute_force_decrypt(self, encrypted_text: str, max_layers: int = 3) -> List[Dict[str, Any]]:
        """暴力破解解密（简化版）"""
        print(f"开始暴力破解模式，最大尝试{max_layers}层...")
        results = []
        
        # 尝试单层解密
        for name, decryptor_class in self.decryptor_map.items():
            try:
                # 使用默认参数尝试
                decryptor = decryptor_class({})
                result = decryptor.decrypt(encrypted_text)
                if self._is_readable_text(result):
                    results.append({
                        "decrypted_text": result,
                        "algorithm": name,
                        "confidence": self._calculate_confidence(result)
                    })
            except Exception:
                continue
        
        return sorted(results, key=lambda x: x['confidence'], reverse=True)
    
    def _is_readable_text(self, text: str) -> bool:
        """判断文本是否可读"""
        if not text:
            return False
        
        printable_ratio = sum(1 for c in text if c.isprintable()) / len(text)
        return printable_ratio > 0.7
    
    def _calculate_confidence(self, text: str) -> float:
        """计算解密结果置信度"""
        score = 0.0
        
        # 可打印字符比例
        printable_ratio = sum(1 for c in text if c.isprintable()) / len(text)
        score += printable_ratio * 30
        
        # 字母比例
        alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text)
        score += alpha_ratio * 25
        
        # 空格和标点符号比例
        space_punct_ratio = sum(1 for c in text if c.isspace() or c in '.,!?;:') / len(text)
        score += min(space_punct_ratio * 20, 15)
        
        # 长度合理性
        if 10 <= len(text) <= 1000:
            score += 15
        elif len(text) < 10:
            score -= 10
        
        # 常见词汇检测（简化）
        common_words = ['the', 'and', 'is', 'to', 'a', 'in', 'that', 'have', 'for', 'not', 'with', 'you', 'this', 'be', 'it', 'on', 'as', 'are', 'was', 'but', 'or', 'an', 'if', 'of', 'at', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'beside', 'beyond', 'within', 'without', 'upon', 'against', 'across', 'around', 'toward', 'towards', 'underneath', 'beneath', 'over', 'under']
        text_lower = text.lower()
        word_matches = sum(1 for word in common_words if word in text_lower)
        score += min(word_matches, 15)
        
        return min(score, 100.0)

def validate_config_file(file_path: str) -> bool:
    """验证配置文件的有效性"""
    if not file_path:
        print("错误: 配置文件路径为空")
        return False
    
    if not os.path.exists(file_path):
        print(f"错误: 配置文件不存在: {file_path}")
        return False
    
    if not file_path.lower().endswith('.json'):
        print("警告: 配置文件扩展名不是.json，可能不是有效的配置文件")
    
    # 检查文件大小（配置文件不应该太大）
    try:
        file_size = os.path.getsize(file_path)
        if file_size > 10 * 1024 * 1024:  # 10MB限制
            print("错误: 配置文件过大（>10MB），可能不是有效的配置文件")
            return False
        if file_size == 0:
            print("错误: 配置文件为空")
            return False
    except OSError as e:
        print(f"错误: 无法读取配置文件信息: {e}")
        return False
    
    return True

def validate_encrypted_text(text: str) -> bool:
    """验证加密文本的有效性"""
    if not text or text.isspace():
        print("错误: 加密文本为空")
        return False
    
    # 检查文本长度
    if len(text.encode('utf-8')) > 50 * 1024 * 1024:  # 50MB限制
        print("错误: 加密文本过大（>50MB）")
        return False
    
    if len(text) < 10:
        print("警告: 加密文本太短，可能不是有效的密文")
    
    return True

def interactive_mode():
    """交互式解密模式"""
    print("=" * 80)
    print("   高级多层解密系统 - 交互模式")
    print("=" * 80)
    print("\n功能说明:")
    print("1. 配置文件解密 - 使用加密时生成的JSON配置文件")
    print("2. 暴力破解模式 - 当配置文件丢失时的智能解密尝试")
    print("3. 支持多种算法的自动识别和解密")
    print("=" * 80)
    
    decryptor = MultiLayerDecryptor()
    
    while True:
        try:
            print("\n解密选项:")
            print("1. 使用配置文件解密")
            print("2. 暴力破解模式")
            print("3. 退出")
            
            choice = input("请选择 (1-3): ").strip()
            
            if choice == '3':
                print("\n感谢使用高级多层解密系统！")
                break
            elif choice == '1':
                config_path = input("请输入配置文件路径: ").strip()
                
                # 自动去除双引号
                if config_path.startswith('"') and config_path.endswith('"'):
                    config_path = config_path[1:-1]
                elif config_path.startswith("'") and config_path.endswith("'"):
                    config_path = config_path[1:-1]
                
                if not config_path:
                    print("✗ 配置文件路径不能为空")
                    continue
                
                if not os.path.exists(config_path):
                    print(f"✗ 配置文件不存在: {config_path}")
                    continue
                
                try:
                    print("\n正在读取配置文件并解密...")
                    result = decryptor.decrypt_with_config(config_path)
                    
                    print(f"\n✓ 解密成功！")
                    print(f"解密层数: {result['layers_processed']}")
                    print(f"原文长度: {len(result['decrypted_text'])} 字符")
                    
                    # 显示解密算法序列
                    print(f"\n解密算法序列（逆序）:")
                    encryption_sequence = result['original_config']['encryption_sequence']
                    for i, layer in enumerate(reversed(encryption_sequence), 1):
                        print(f"  步骤{i}: 解密 {layer['algorithm']}")
                    
                    print(f"\n解密结果:")
                    print(f"'{result['decrypted_text']}'")
                    
                    # 询问是否保存
                    save_choice = input("\n是否保存解密结果到文件？(y/n): ").strip().lower()
                    if save_choice == 'y':
                        try:
                            output_file = f"decrypted_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                            with open(output_file, 'w', encoding='utf-8') as f:
                                f.write(result['decrypted_text'])
                            print(f"✓ 解密结果已保存到: {output_file}")
                        except IOError as e:
                            print(f"✗ 保存失败: {e}")
                
                except FileNotFoundError as e:
                    print(f"✗ {e}")
                except ValueError as e:
                    print(f"✗ {e}")
                except json.JSONDecodeError:
                    print("✗ 配置文件格式错误，请检查文件内容")
                except MemoryError:
                    print("✗ 内存不足，无法处理此解密任务")
                except Exception as e:
                    print(f"✗ 解密过程中发生错误: {e}")
                    logger.error(f"解密异常: {type(e).__name__}: {e}", exc_info=True)
                
            elif choice == '2':
                encrypted_text = input("请输入加密文本: ").strip()
                
                # 验证加密文本
                if not validate_encrypted_text(encrypted_text):
                    continue
                
                max_layers_input = input("最大尝试层数 (1-5, 默认3): ").strip()
                try:
                    max_layers = int(max_layers_input) if max_layers_input.isdigit() else 3
                    if not 1 <= max_layers <= 5:
                        print("警告: 层数超出范围，使用默认值3")
                        max_layers = 3
                except ValueError:
                    print("警告: 无效输入，使用默认层数3")
                    max_layers = 3
                
                try:
                    print(f"\n开始暴力破解（最多{max_layers}层）...")
                    print("注意: 此过程可能需要较长时间")
                    
                    results = decryptor.brute_force_decrypt(encrypted_text, max_layers)
                    
                    if results:
                        print(f"\n✓ 找到 {len(results)} 个可能的解密结果:")
                        print("-" * 60)
                        for i, result in enumerate(results[:5], 1):  # 显示前5个结果
                            confidence_bar = "█" * int(result['confidence'] / 10) + "░" * (10 - int(result['confidence'] / 10))
                            print(f"{i}. 算法: {result['algorithm']}")
                            print(f"   置信度: {result['confidence']:.1f}/100 [{confidence_bar}]")
                            preview = result['decrypted_text'][:100]
                            print(f"   结果: '{preview}{'...' if len(result['decrypted_text']) > 100 else ''}'")
                            print("-" * 60)
                        
                        # 询问是否保存最佳结果
                        if results[0]['confidence'] > 50:
                            save_best = input("\n是否保存最佳结果？(y/n): ").strip().lower()
                            if save_best == 'y':
                                try:
                                    output_file = f"brute_force_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                                    with open(output_file, 'w', encoding='utf-8') as f:
                                        f.write(f"算法: {results[0]['algorithm']}\n")
                                        f.write(f"置信度: {results[0]['confidence']:.1f}\n\n")
                                        f.write(results[0]['decrypted_text'])
                                    print(f"✓ 最佳结果已保存到: {output_file}")
                                except IOError as e:
                                    print(f"✗ 保存失败: {e}")
                    else:
                        print("✗ 未找到有效的解密结果")
                        print("建议: 检查输入文本是否正确，或尝试增加最大层数")
                        
                except MemoryError:
                    print("✗ 内存不足，请尝试较短的文本或减少最大层数")
                except Exception as e:
                    print(f"✗ 暴力破解过程中发生错误: {e}")
                    logger.error(f"暴力破解异常: {type(e).__name__}: {e}", exc_info=True)
            else:
                print("✗ 无效选择，请输入1、2或3！")
        
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
    decryptor = MultiLayerDecryptor()
    
    try:
        if args.config:
            result = decryptor.decrypt_with_config(args.config)
            decrypted_text = result['decrypted_text']
        elif args.brute_force:
            if args.file:
                with open(args.file, 'r', encoding='utf-8') as f:
                    encrypted_text = f.read()
            else:
                encrypted_text = args.text
            
            results = decryptor.brute_force_decrypt(encrypted_text, args.max_layers)
            if results:
                decrypted_text = results[0]['decrypted_text']
                if args.verbose:
                    print(f"最佳匹配算法: {results[0]['algorithm']}")
                    print(f"置信度: {results[0]['confidence']:.1f}")
            else:
                print("暴力破解失败，未找到有效结果")
                return
        else:
            print("请指定配置文件 (-c) 或启用暴力破解模式 (-b)")
            return
        
        # 输出结果
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(decrypted_text)
            print(f"解密结果已保存到: {args.output}")
        else:
            print(decrypted_text)
    
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="高级多层解密系统")
    parser.add_argument('-c', '--config', help='加密配置文件路径')
    parser.add_argument('-b', '--brute-force', action='store_true', help='启用暴力破解模式')
    parser.add_argument('-t', '--text', help='要解密的文本')
    parser.add_argument('-f', '--file', help='包含加密文本的文件路径')
    parser.add_argument('-o', '--output', help='输出文件路径')
    parser.add_argument('-m', '--max-layers', type=int, default=3, help='暴力破解最大层数')
    parser.add_argument('-v', '--verbose', action='store_true', help='显示详细信息')
    
    args = parser.parse_args()
    
    if args.config or args.brute_force:
        command_line_mode(args)
    else:
        interactive_mode()

if __name__ == "__main__":
    main()
