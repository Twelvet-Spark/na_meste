"""Формирование RTSP URL для Dahua NVR.

Стандартный формат Dahua:
    rtsp://USERNAME:PASSWORD@HOST:554/cam/realmonitor?channel=N&subtype=M

    channel — канал камеры на NVR (нумерация с 1)
    subtype=0 — основной поток (main)
    subtype=1 — дополнительный поток (sub)
"""
import ipaddress
import re
from urllib.parse import quote


def normalize_host(value: str) -> str:
    """Accept a bare DNS name/IP only; keep canonical unbracketed storage."""
    value = value.strip()
    if not value or any(character.isspace() or ord(character) < 32 for character in value):
        raise ValueError("Укажите IP или имя устройства без адреса протокола, пути и порта")
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    try:
        return ipaddress.ip_address(value).compressed.lower()
    except ValueError:
        if ":" in value or any(character in value for character in r"/\@?#%[]"):
            raise ValueError("Укажите IP или имя устройства без адреса протокола, пути и порта") from None
    try:
        host = value.rstrip(".").encode("idna").decode("ascii").lower()
    except UnicodeError:
        raise ValueError("Некорректное имя устройства") from None
    labels = host.split(".")
    if (len(host) > 253 or any(not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
                               for label in labels)):
        raise ValueError("Некорректное имя устройства")
    return host


def url_host(value: str) -> str:
    host = normalize_host(value)
    return f"[{host}]" if ":" in host else host

def subtype_code(stream_type: str) -> int:
    """'main' → 0, 'sub' → 1."""
    return 0 if stream_type == "main" else 1


def build_rtsp_url(
    nvr_host: str,
    rtsp_port: int,
    username: str,
    password: str,
    channel: int,
    stream_type: str,
) -> str:
    """Собирает RTSP URL из параметров подключения.

    Логин и пароль URL-кодируются: спецсимволы (@ : # ! % и т.п.)
    не должны ломать URL.
    """
    auth = f"{quote(str(username), safe='')}:{quote(str(password), safe='')}"
    return (
        f"rtsp://{auth}@{url_host(nvr_host)}:{rtsp_port}"
        f"/cam/realmonitor?channel={channel}&subtype={subtype_code(stream_type)}"
    )


_CREDENTIALS_RE = re.compile(r"(//[^:/@\s]+:)[^@\s]*(@)")


def mask_url(text: str) -> str:
    """Заменяет пароль в RTSP URL на ***. Применяется ко всему, что идёт в логи."""
    return _CREDENTIALS_RE.sub(r"\1***\2", text)
