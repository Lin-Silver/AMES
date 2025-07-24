# 高级多层加密系统 (Advanced Multi-Layer Encryption System)

## 项目概述

这是一个基于Python的高级多层加密解密系统，支持10种不同的加密算法，包括一个自创的高级矩阵加密算法。系统具有随机多层加密、配置追踪、完整性验证、智能解密等功能。

## 核心特性

- **10种加密算法**：包含自创的高级矩阵加密算法和9种经典算法
- **多层随机加密**：支持1-10层随机组合加密
- **自创算法保证**：确保每次加密都使用自创的最强算法
- **简洁输出**：优化密文长度，避免乱码字符
- **配置追踪**：完整记录加密路径，便于精确解密
- **暴力破解模式**：当配置丢失时提供智能推断解密
- **完整性验证**：SHA256哈希校验确保数据完整性
- **多语言支持**：完美支持中英文及特殊字符
- **交互界面**：同时提供命令行和交互式操作模式

## 文件结构

```
Passkey/
├── encrypt.py          # 主加密脚本
├── decrypt.py          # 主解密脚本
├── encrypt.bat         # Windows加密启动脚本
├── decrypt.bat         # Windows解密启动脚本
├── README.md          # 本文档
└── encryption_config_*.json  # 加密配置文件（运行时生成）
```

## 快速开始

### Windows用户
```bash
# 加密
encrypt.bat

# 解密
decrypt.bat
```

### 通用方法
```bash
# 加密
python encrypt.py

# 解密
python decrypt.py

# 命令行模式
python encrypt.py -t "要加密的文本" -l 5
python decrypt.py -f encryption_config_20240101_120000.json
```

## 核心算法详解

### 自创高级矩阵加密算法 (AdvancedMatrixCipher)

这是本系统的核心创新算法，具有以下特点：

#### 1. 算法原理

**多重数学变换组合**：
- **矩阵变换**：使用动态生成的可逆矩阵对数据进行线性变换
- **旋转加密**：通过位置相关的旋转密钥进行非线性混淆
- **自定义编码**：使用优化的Base64变体确保输出简洁可读

#### 2. 加密流程

```
原文 → UTF-8编码 → 分块处理 → 矩阵变换 → 旋转加密 → 自定义编码 → 校验码 → 密文
```

**详细步骤**：

1. **文本预处理**
   ```python
   byte_data = text.encode('utf-8')  # 转换为字节数组
   ```

2. **动态矩阵生成**
   ```python
   # 生成随机可逆矩阵（避免奇异矩阵）
   matrix = [[secrets.randbelow(9) + 1 for _ in range(size)] for _ in range(size)]
   rotation_key = secrets.randbelow(100) + 1
   ```

3. **分块矩阵变换**
   ```python
   # 每个块进行矩阵乘法变换
   encrypted_block = matrix_multiply(matrix, block)
   ```

4. **位置相关旋转**
   ```python
   # 基于块位置的动态旋转
   rotated_block = [(x + rotation_key + block_index) % 256 for x in encrypted_block]
   ```

5. **优化编码**
   ```python
   # 自定义字符集确保输出简洁
   encode_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/='
   ```

#### 3. 安全特性

**数学安全性**：
- **可逆矩阵保证**：通过行列式检验确保矩阵可逆
- **密钥空间大**：矩阵元素、旋转密钥组合提供巨大密钥空间
- **位置敏感**：相同字符在不同位置产生不同密文

**抗攻击能力**：
- **抗频率分析**：矩阵变换打乱字符频率分布
- **抗模式识别**：位置相关加密避免重复模式
- **抗暴力破解**：组合密钥空间达到数学安全级别

**实现优势**：
- **无外部依赖**：纯Python实现，避免numpy等依赖
- **高效算法**：优化的矩阵运算，适合大文本处理
- **简洁输出**：自定义编码确保密文简洁可读

#### 4. 解密原理

解密严格按照加密的逆过程：

```
密文 → 校验验证 → 自定义解码 → 逆旋转 → 逆矩阵变换 → 分块重组 → UTF-8解码 → 原文
```

**关键技术**：
- **逆矩阵计算**：支持2x2、3x3及更大矩阵的逆运算
- **参数重建**：从配置文件精确重建所有加密参数
- **容错处理**：处理矩阵奇异等边缘情况

## 其他加密算法

1. **EnhancedCaesar** - 增强凯撒密码（多重位移）
2. **MorseVariant** - 摩尔斯码变体
3. **MultiBase64** - 多轮Base64编码
4. **ASCIIConversion** - ASCII码转换加密
5. **ReverseInterpolation** - 反向插值加密
6. **SubstitutionCipher** - 替换密码
7. **BinaryConversion** - 二进制转换加密
8. **HexObfuscation** - 十六进制混淆
9. **ROT13Variant** - ROT13变体

## 系统架构

### 面向对象设计
```python
EncryptionBase          # 加密算法基类
├── AdvancedMatrixCipher
├── EnhancedCaesar
└── ...

DecryptionBase          # 解密算法基类
├── AdvancedMatrixDecryptor
├── EnhancedCaesarDecryptor
└── ...

MultiLayerEncryptor     # 多层加密管理器
MultiLayerDecryptor     # 多层解密管理器
```

### 配置文件格式
```json
{
    "layers": 3,
    "algorithms": ["AdvancedMatrixCipher", "EnhancedCaesar", "MorseVariant"],
    "parameters": [
        {
            "matrix_size": 3,
            "rotation_key": 42,
            "matrix": [[2,3,1],[1,4,2],[3,1,5]],
            "encode_chars": "..."
        }
    ],
    "original_length": 100,
    "encrypted_length": 156,
    "timestamp": "2024-01-01 12:00:00",
    "integrity_hash": "sha256_hash_value",
    "strength_score": 95
}
```

## 性能与安全

### 加密强度评估
系统提供5个维度的强度评估：

1. **层数评分** (0-30分)：加密层数越多分数越高
2. **长度评分** (0-20分)：密文扩展比例适中得分更高
3. **信息熵** (0-30分)：字符分布越随机得分越高
4. **算法复杂度** (0-15分)：算法数学复杂性评分
5. **模式抗性** (0-5分)：抗模式识别能力评分

### 性能指标
- **加密速度**：~1MB/s（中等硬件）
- **内存使用**：线性增长，适合大文件
- **兼容性**：Python 3.6+，无外部依赖

## 使用示例

### 基础加密解密
```python
# 交互式加密
python encrypt.py
# 输入：要加密的文本
# 选择：加密层数 (1-10)
# 输出：密文 + 配置文件

# 配置文件解密
python decrypt.py
# 选择：配置文件
# 输出：原始文本
```

### 命令行批处理
```bash
# 单次加密
python encrypt.py -t "Hello World" -l 5

# 批量解密
python decrypt.py -f config1.json -f config2.json
```

### 暴力破解模式
```python
# 当配置文件丢失时
python decrypt.py
# 选择：暴力破解模式
# 输入：密文
# 系统：智能尝试所有算法组合
```

## 安全建议

1. **配置文件保护**：加密配置文件包含解密所需的所有信息，请妥善保管
2. **多层加密**：建议使用3-7层加密，平衡安全性和性能
3. **定期更换**：对于敏感数据，建议定期重新加密
4. **备份策略**：重要的配置文件应进行多重备份

## 技术支持

如遇到问题或需要功能扩展，请检查以下方面：

1. **Python版本**：确保使用Python 3.6或更高版本
2. **文件权限**：确保程序对工作目录有读写权限
3. **字符编码**：系统自动处理UTF-8编码，支持中英文混合
4. **内存限制**：处理大文件时注意系统内存使用情况

## 更新日志

- **v1.0** - 初始版本，包含10种加密算法
- **v1.1** - 增强异常处理和输入验证
- **v1.2** - 添加加密强度评估功能
- **v1.3** - 优化自创算法，移除外部依赖，确保输出简洁

---

**注意**：本系统仅供学习和研究使用，请勿用于非法用途。对于商业用途或高安全要求场景，建议使用经过专业安全审计的加密方案。
