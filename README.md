# Advanced Multi-Layer Encryption System

## Project Overview

This is a Python-based advanced multi-layer encryption and decryption system that supports 10 different encryption algorithms, including a self-designed Advanced Matrix Encryption algorithm. The system features random multi-layer encryption, configuration tracking, integrity verification, intelligent decryption, and more.

## Core Features

* **10 Encryption Algorithms**: Includes a custom Advanced Matrix Encryption algorithm and 9 classic algorithms
* **Multi-Layer Random Encryption**: Supports 1–10 layers of randomly combined encryption
* **Custom Algorithm Guarantee**: Ensures each encryption uses the strongest custom algorithm
* **Clean Output**: Optimized ciphertext length to avoid garbled characters
* **Configuration Tracking**: Fully records encryption path for precise decryption
* **Brute-Force Mode**: Intelligent decryption inference when configuration is lost
* **Integrity Verification**: SHA256 hash check to ensure data integrity
* **Multilingual Support**: Fully supports Chinese, English, and special characters
* **Interactive Interface**: Offers both command-line and interactive modes

## File Structure

```
Passkey/
├── encrypt.py                 # Main encryption script
├── decrypt.py                 # Main decryption script
├── encrypt.bat                # Windows encryption startup script
├── decrypt.bat                # Windows decryption startup script
├── README.md                  # This documentation
└── encryption_config_*.json   # Encryption config files (generated at runtime)
```

## Quick Start

### For Windows Users

```bash
# Encryption
encrypt.bat

# Decryption
decrypt.bat
```

### General Method

```bash
# Encryption
python encrypt.py

# Decryption
python decrypt.py

# Command-line mode
python encrypt.py -t "Text to encrypt" -l 5
python decrypt.py -f encryption_config_20240101_120000.json
```

## Core Algorithm Details

### Custom Advanced Matrix Encryption Algorithm (AdvancedMatrixCipher)

This is the system's core innovative algorithm, featuring the following:

#### 1. Algorithm Principle

**Combination of Multiple Mathematical Transformations**:

* **Matrix Transformation**: Linearly transforms data using dynamically generated invertible matrices
* **Rotational Encryption**: Applies nonlinear obfuscation using position-based rotation keys
* **Custom Encoding**: Uses an optimized Base64 variant to ensure readable output

#### 2. Encryption Process

```
Plaintext → UTF-8 Encoding → Chunk Processing → Matrix Transformation → Rotational Encryption → Custom Encoding → Checksum → Ciphertext
```

**Detailed Steps**:

1. **Text Preprocessing**

   ```python
   byte_data = text.encode('utf-8')
   ```

2. **Dynamic Matrix Generation**

   ```python
   matrix = [[secrets.randbelow(9) + 1 for _ in range(size)] for _ in range(size)]
   rotation_key = secrets.randbelow(100) + 1
   ```

3. **Chunk Matrix Transformation**

   ```python
   encrypted_block = matrix_multiply(matrix, block)
   ```

4. **Position-Based Rotation**

   ```python
   rotated_block = [(x + rotation_key + block_index) % 256 for x in encrypted_block]
   ```

5. **Optimized Encoding**

   ```python
   encode_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/='
   ```

#### 3. Security Features

**Mathematical Security**:

* **Invertible Matrix Assurance**: Validated using determinant checks
* **Large Key Space**: Matrix elements and rotation keys provide massive key combinations
* **Position Sensitivity**: Same character produces different ciphertext in different positions

**Resistance Against Attacks**:

* **Anti-Frequency Analysis**: Matrix transformation scrambles character frequency distribution
* **Anti-Pattern Recognition**: Position-based encryption avoids repetitive patterns
* **Anti-Brute Force**: Key space combination reaches mathematical security levels

**Implementation Advantages**:

* **No External Dependencies**: Pure Python implementation, avoids numpy, etc.
* **Efficient Algorithms**: Optimized matrix operations suitable for large text
* **Readable Output**: Custom encoding ensures clean, readable ciphertext

#### 4. Decryption Principle

Decryption strictly follows the reverse encryption process:

```
Ciphertext → Integrity Check → Custom Decoding → Inverse Rotation → Inverse Matrix Transformation → Reassembly → UTF-8 Decoding → Plaintext
```

**Key Techniques**:

* **Inverse Matrix Calculation**: Supports 2x2, 3x3, and larger matrices
* **Parameter Reconstruction**: Rebuilds all encryption parameters from the config file
* **Fault Tolerance**: Handles edge cases like singular matrices

## Other Encryption Algorithms

1. **EnhancedCaesar** – Enhanced Caesar cipher (multi-shift)
2. **MorseVariant** – Morse code variant
3. **MultiBase64** – Multi-round Base64 encoding
4. **ASCIIConversion** – ASCII code conversion encryption
5. **ReverseInterpolation** – Reverse interpolation encryption
6. **SubstitutionCipher** – Substitution cipher
7. **BinaryConversion** – Binary conversion encryption
8. **HexObfuscation** – Hexadecimal obfuscation
9. **ROT13Variant** – ROT13 variant

## System Architecture

### Object-Oriented Design

```python
EncryptionBase          # Base class for encryption algorithms
├── AdvancedMatrixCipher
├── EnhancedCaesar
└── ...

DecryptionBase          # Base class for decryption algorithms
├── AdvancedMatrixDecryptor
├── EnhancedCaesarDecryptor
└── ...

MultiLayerEncryptor     # Multi-layer encryption manager
MultiLayerDecryptor     # Multi-layer decryption manager
```

### Configuration File Format

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

## Performance and Security

### Encryption Strength Assessment

The system evaluates encryption strength across 5 dimensions:

1. **Layer Score** (0–30): More layers, higher score
2. **Length Score** (0–20): Balanced ciphertext expansion gets higher score
3. **Entropy** (0–30): More randomness, higher score
4. **Algorithm Complexity** (0–15): Based on mathematical complexity
5. **Pattern Resistance** (0–5): Resistance to pattern recognition

### Performance Metrics

* **Encryption Speed**: \~1MB/s (on average hardware)
* **Memory Usage**: Linear growth, suitable for large files
* **Compatibility**: Python 3.6+, no external dependencies

## Usage Examples

### Basic Encryption/Decryption

```python
# Interactive encryption
python encrypt.py
# Input: text to encrypt
# Select: number of layers (1–10)
# Output: ciphertext + config file

# Config file decryption
python decrypt.py
# Select: config file
# Output: original text
```

### Command-Line Batch Processing

```bash
# Single encryption
python encrypt.py -t "Hello World" -l 5

# Batch decryption
python decrypt.py -f config1.json -f config2.json
```

### Brute-Force Mode

```python
# When config file is lost
python decrypt.py
# Select: brute-force mode
# Input: ciphertext
# System: intelligently tries all algorithm combinations
```

## Security Recommendations

1. **Protect Config Files**: They contain all decryption information
2. **Use Multi-Layer Encryption**: 3–7 layers recommended for balance
3. **Periodic Rotation**: Re-encrypt sensitive data periodically
4. **Backup Strategy**: Keep multiple backups of important config files

## Technical Support

For issues or feature requests, please check the following:

1. **Python Version**: Ensure Python 3.6 or higher
2. **File Permissions**: Ensure read/write access to the working directory
3. **Character Encoding**: UTF-8 auto-handled; supports mixed Chinese and English
4. **Memory Limits**: Monitor memory when handling large files

---

**Note**: This system is for educational and research purposes only. Do not use for illegal activities. For commercial or high-security use cases, professional-grade encryption with security audits is recommended.
