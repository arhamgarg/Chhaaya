"""Detect the WhatsApp photo formats accepted by the document providers."""


def image_media_type(image: bytes) -> str:
    if image.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    raise ValueError("unsupported document image format: expected JPEG or PNG")
