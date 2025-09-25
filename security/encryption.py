#!/usr/bin/env python3
"""
Data Encryption and Security Hardening Module
Provides field-level encryption for sensitive data and security utilities
"""

import os
import base64
import json
import hashlib
import secrets
from typing import Any, Dict, Optional, Union
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import sqlite3
from datetime import datetime

class DataEncryption:
    """Handle encryption and decryption of sensitive data"""

    def __init__(self, encryption_key: Optional[bytes] = None):
        """Initialize encryption with key"""
        if encryption_key:
            self.key = encryption_key
        else:
            # Generate or load encryption key
            self.key = self._get_or_create_encryption_key()

        self.fernet = Fernet(self.key)
        self.aes_gcm = AESGCM(self.key[:32])  # Use first 32 bytes for AES-GCM

    def _get_or_create_encryption_key(self) -> bytes:
        """Get encryption key from environment or create new one"""
        # Try to get key from environment variable
        env_key = os.environ.get('AI_SKILL_PLANNER_ENCRYPTION_KEY')
        if env_key:
            try:
                return base64.urlsafe_b64decode(env_key.encode())
            except Exception:
                pass

        # Generate new key and save to file (for development)
        key_file = os.path.join(os.path.dirname(__file__), '.encryption_key')

        if os.path.exists(key_file):
            try:
                with open(key_file, 'rb') as f:
                    return f.read()
            except Exception:
                pass

        # Create new key
        key = Fernet.generate_key()

        try:
            with open(key_file, 'wb') as f:
                f.write(key)
            # Make file readable only by owner
            os.chmod(key_file, 0o600)
        except Exception:
            pass

        return key

    def encrypt_string(self, plaintext: str) -> str:
        """Encrypt a string and return base64 encoded result"""
        if not plaintext:
            return plaintext

        try:
            encrypted_data = self.fernet.encrypt(plaintext.encode('utf-8'))
            return base64.urlsafe_b64encode(encrypted_data).decode('utf-8')
        except Exception as e:
            raise ValueError(f"Encryption failed: {e}")

    def decrypt_string(self, encrypted_text: str) -> str:
        """Decrypt a base64 encoded encrypted string"""
        if not encrypted_text:
            return encrypted_text

        try:
            encrypted_data = base64.urlsafe_b64decode(encrypted_text.encode('utf-8'))
            decrypted_data = self.fernet.decrypt(encrypted_data)
            return decrypted_data.decode('utf-8')
        except Exception as e:
            raise ValueError(f"Decryption failed: {e}")

    def encrypt_json(self, data: Dict[str, Any]) -> str:
        """Encrypt JSON data"""
        try:
            json_string = json.dumps(data, separators=(',', ':'))
            return self.encrypt_string(json_string)
        except Exception as e:
            raise ValueError(f"JSON encryption failed: {e}")

    def decrypt_json(self, encrypted_text: str) -> Dict[str, Any]:
        """Decrypt JSON data"""
        try:
            json_string = self.decrypt_string(encrypted_text)
            return json.loads(json_string)
        except Exception as e:
            raise ValueError(f"JSON decryption failed: {e}")

    def encrypt_sensitive_fields(self, data: Dict[str, Any], sensitive_fields: list) -> Dict[str, Any]:
        """Encrypt specified sensitive fields in a dictionary"""
        result = data.copy()

        for field in sensitive_fields:
            if field in result and result[field] is not None:
                result[field] = self.encrypt_string(str(result[field]))

        return result

    def decrypt_sensitive_fields(self, data: Dict[str, Any], sensitive_fields: list) -> Dict[str, Any]:
        """Decrypt specified sensitive fields in a dictionary"""
        result = data.copy()

        for field in sensitive_fields:
            if field in result and result[field] is not None:
                try:
                    result[field] = self.decrypt_string(result[field])
                except Exception:
                    # If decryption fails, assume data is not encrypted
                    pass

        return result

    def hash_sensitive_data(self, data: str, salt: Optional[bytes] = None) -> tuple:
        """Hash sensitive data with salt for secure storage"""
        if salt is None:
            salt = secrets.token_bytes(32)

        # Use PBKDF2 with SHA-256
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )

        key = kdf.derive(data.encode('utf-8'))
        hash_value = base64.urlsafe_b64encode(key).decode('utf-8')
        salt_value = base64.urlsafe_b64encode(salt).decode('utf-8')

        return hash_value, salt_value

    def verify_hash(self, data: str, hash_value: str, salt_value: str) -> bool:
        """Verify data against stored hash"""
        try:
            salt = base64.urlsafe_b64decode(salt_value.encode('utf-8'))
            new_hash, _ = self.hash_sensitive_data(data, salt)
            return new_hash == hash_value
        except Exception:
            return False

class SecureDatabase:
    """Database wrapper with automatic encryption for sensitive fields"""

    def __init__(self, db_path: str, encryption_key: Optional[bytes] = None):
        self.db_path = db_path
        self.encryptor = DataEncryption(encryption_key)

        # Define sensitive fields that should be encrypted
        self.sensitive_fields = {
            'users': ['email', 'full_name'],
            'people': ['email', 'personal_info'],
            'evidence': ['description', 'personal_notes'],
            'audit_logs': ['details']
        }

    def get_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def insert_with_encryption(self, table: str, data: Dict[str, Any]) -> bool:
        """Insert data with automatic encryption of sensitive fields"""
        try:
            if table in self.sensitive_fields:
                data = self.encryptor.encrypt_sensitive_fields(
                    data, self.sensitive_fields[table]
                )

            # Add encryption metadata
            data['encrypted_at'] = datetime.utcnow().isoformat()
            data['encryption_version'] = '1.0'

            with self.get_connection() as conn:
                cursor = conn.cursor()

                columns = ', '.join(data.keys())
                placeholders = ', '.join(['?' for _ in data])
                query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"

                cursor.execute(query, list(data.values()))
                conn.commit()
                return True

        except Exception as e:
            print(f"Encrypted insert failed: {e}")
            return False

    def select_with_decryption(self, table: str, where_clause: str = None, params: tuple = None) -> list:
        """Select data with automatic decryption of sensitive fields"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                query = f"SELECT * FROM {table}"
                if where_clause:
                    query += f" WHERE {where_clause}"

                cursor.execute(query, params or ())
                rows = cursor.fetchall()

                # Decrypt sensitive fields
                if table in self.sensitive_fields:
                    decrypted_rows = []
                    for row in rows:
                        row_dict = dict(row)
                        row_dict = self.encryptor.decrypt_sensitive_fields(
                            row_dict, self.sensitive_fields[table]
                        )
                        decrypted_rows.append(row_dict)
                    return decrypted_rows

                return [dict(row) for row in rows]

        except Exception as e:
            print(f"Encrypted select failed: {e}")
            return []

class SecurityHardening:
    """Security hardening utilities and checks"""

    def __init__(self):
        self.encryption = DataEncryption()

    def validate_password_strength(self, password: str) -> Dict[str, Any]:
        """Validate password strength and return score"""
        score = 0
        issues = []

        # Length check
        if len(password) >= 12:
            score += 2
        elif len(password) >= 8:
            score += 1
        else:
            issues.append("Password should be at least 8 characters long")

        # Character variety checks
        if any(c.islower() for c in password):
            score += 1
        else:
            issues.append("Password should contain lowercase letters")

        if any(c.isupper() for c in password):
            score += 1
        else:
            issues.append("Password should contain uppercase letters")

        if any(c.isdigit() for c in password):
            score += 1
        else:
            issues.append("Password should contain numbers")

        if any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            score += 2
        else:
            issues.append("Password should contain special characters")

        # Common password check (basic)
        common_passwords = [
            'password', '123456', '123456789', 'qwerty', 'abc123',
            'password123', 'admin', 'letmein', 'welcome', 'monkey'
        ]

        if password.lower() in common_passwords:
            score = max(0, score - 3)
            issues.append("Password is too common")

        # Calculate strength level
        if score >= 6:
            strength = "strong"
        elif score >= 4:
            strength = "medium"
        elif score >= 2:
            strength = "weak"
        else:
            strength = "very_weak"

        return {
            'score': score,
            'strength': strength,
            'issues': issues,
            'is_acceptable': score >= 4
        }

    def sanitize_input(self, input_string: str, max_length: int = 1000) -> str:
        """Sanitize user input to prevent injection attacks"""
        if not input_string:
            return ""

        # Limit length
        sanitized = input_string[:max_length]

        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')

        # Basic HTML escape for XSS prevention
        sanitized = sanitized.replace('<', '&lt;').replace('>', '&gt;')
        sanitized = sanitized.replace('"', '&quot;').replace("'", '&#x27;')
        sanitized = sanitized.replace('&', '&amp;')

        return sanitized

    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure random token"""
        return secrets.token_urlsafe(length)

    def check_sql_injection_patterns(self, query: str) -> bool:
        """Basic SQL injection pattern detection"""
        dangerous_patterns = [
            r'(\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b)',
            r'(--|\#|\/\*)',
            r'(\bor\b.*=.*=)',
            r'(\band\b.*=.*=)',
            r'([\'\"];\s*(union|select|insert|update|delete))',
        ]

        import re
        query_lower = query.lower()

        for pattern in dangerous_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return True

        return False

    def log_security_event(self, event_type: str, details: Dict[str, Any], severity: str = "INFO"):
        """Log security-related events"""
        try:
            from security.auth import AuthManager
            auth_manager = AuthManager()

            auth_manager.log_audit_event(
                user_id="system",
                action=f"security_{event_type}",
                resource="security",
                details={
                    "severity": severity,
                    "event_details": details,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        except Exception:
            # Fallback to file logging
            import logging
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)
            logger.info(f"Security event [{severity}] {event_type}: {details}")

    def perform_security_scan(self) -> Dict[str, Any]:
        """Perform basic security checks on the system"""
        results = {
            "scan_time": datetime.utcnow().isoformat(),
            "checks": [],
            "overall_score": 0,
            "recommendations": []
        }

        # Check 1: Encryption key security
        try:
            key_file = os.path.join(os.path.dirname(__file__), '.encryption_key')
            if os.path.exists(key_file):
                stat_info = os.stat(key_file)
                is_secure = stat_info.st_mode & 0o077 == 0  # Only owner can read/write

                results["checks"].append({
                    "name": "Encryption Key Security",
                    "status": "PASS" if is_secure else "WARN",
                    "details": f"File permissions: {oct(stat_info.st_mode)[-3:]}"
                })

                if is_secure:
                    results["overall_score"] += 2
                else:
                    results["recommendations"].append("Set encryption key file permissions to 600")
            else:
                results["checks"].append({
                    "name": "Encryption Key Security",
                    "status": "INFO",
                    "details": "Key loaded from environment or generated in memory"
                })
                results["overall_score"] += 1
        except Exception as e:
            results["checks"].append({
                "name": "Encryption Key Security",
                "status": "ERROR",
                "details": f"Could not check key security: {e}"
            })

        # Check 2: Database file permissions
        try:
            from database.init_db import get_db_path
            db_path = get_db_path()

            if os.path.exists(db_path):
                stat_info = os.stat(db_path)
                is_secure = stat_info.st_mode & 0o077 == 0

                results["checks"].append({
                    "name": "Database File Security",
                    "status": "PASS" if is_secure else "WARN",
                    "details": f"File permissions: {oct(stat_info.st_mode)[-3:]}"
                })

                if is_secure:
                    results["overall_score"] += 1
                else:
                    results["recommendations"].append("Set database file permissions to 600")
        except Exception as e:
            results["checks"].append({
                "name": "Database File Security",
                "status": "ERROR",
                "details": f"Could not check database security: {e}"
            })

        # Check 3: Environment security
        env_check = {
            "name": "Environment Security",
            "status": "PASS",
            "details": []
        }

        if os.environ.get('AI_SKILL_PLANNER_ENCRYPTION_KEY'):
            env_check["details"].append("Encryption key loaded from environment ✓")
            results["overall_score"] += 1
        else:
            env_check["details"].append("Encryption key not in environment (using file/generated)")
            env_check["status"] = "INFO"

        if os.environ.get('JWT_SECRET_KEY'):
            env_check["details"].append("JWT secret loaded from environment ✓")
            results["overall_score"] += 1
        else:
            env_check["details"].append("JWT secret not in environment (using default)")
            env_check["status"] = "WARN"
            results["recommendations"].append("Set JWT_SECRET_KEY environment variable")

        results["checks"].append(env_check)

        # Calculate overall security level
        max_score = 5
        score_percentage = (results["overall_score"] / max_score) * 100

        if score_percentage >= 80:
            results["security_level"] = "GOOD"
        elif score_percentage >= 60:
            results["security_level"] = "MODERATE"
        else:
            results["security_level"] = "NEEDS_IMPROVEMENT"

        return results

# Global instances for easy access
default_encryption = DataEncryption()
security_hardening = SecurityHardening()