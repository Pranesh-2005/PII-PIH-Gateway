"""Mapping vault: AES-GCM encrypted, session-keyed, TTL-bounded, in-memory.

The mapping is the re-identification key for the whole note. Holding it in a plain dict is
the obvious mistake -- a heap dump, a log line or a stray ``repr()`` re-identifies the
patient. So it is encrypted the moment de-identification finishes and decrypted only for the
duration of one rehydrate call.

The key comes from ``PHI_VAULT_KEY`` (base64, 32 bytes). If unset, an ephemeral key is
generated per process: mappings then die with the process, which is the safe failure -- a
lost mapping means a failed rehydration, not a leaked one.
"""

from __future__ import annotations

import base64
import os
import secrets
import time
from dataclasses import dataclass

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from ..types import Mapping

_NONCE_BYTES = 12


class MappingExpired(KeyError):
    """The session's mapping passed its TTL. Rehydration is impossible; say so loudly
    rather than returning half-rehydrated text."""


def _load_key(env_var: str) -> bytes:
    raw = os.environ.get(env_var)
    if not raw:
        return AESGCM.generate_key(bit_length=256)
    try:
        key = base64.b64decode(raw, validate=True)
    except Exception:
        key = raw.encode("utf-8")
    if len(key) not in (16, 24, 32):
        # Stretch/shrink deterministically rather than refusing to start; a short dev key
        # should not be the reason the demo cannot run.
        import hashlib

        key = hashlib.sha256(key).digest()
    return key


@dataclass
class _Record:
    ciphertext: bytes
    nonce: bytes
    expires_at: float


class MappingStore:
    """In-memory encrypted store. One process, one key, TTL per record.

    ponytail: in-memory dict is the whole store. Swap for Redis when more than one
    gateway process needs to share sessions -- the interface stays put/get/delete.
    """

    def __init__(self, ttl_seconds: int = 3600, key_env: str = "PHI_VAULT_KEY") -> None:
        self.ttl_seconds = ttl_seconds
        self._key = _load_key(key_env)
        self._aead = AESGCM(self._key)
        self._records: dict[str, _Record] = {}

    def new_session_id(self) -> str:
        return secrets.token_urlsafe(24)

    def put(self, mapping: Mapping, session_id: str | None = None) -> str:
        sid = session_id or self.new_session_id()
        nonce = os.urandom(_NONCE_BYTES)
        # Session id as associated data: a record cannot be replayed under another session.
        ct = self._aead.encrypt(nonce, mapping.to_json().encode("utf-8"), sid.encode("utf-8"))
        self._records[sid] = _Record(ct, nonce, time.time() + self.ttl_seconds)
        return sid

    def get(self, session_id: str) -> Mapping:
        self.purge_expired()
        rec = self._records.get(session_id)
        if rec is None:
            raise MappingExpired(f"no mapping for session {session_id!r}")
        if rec.expires_at < time.time():
            del self._records[session_id]
            raise MappingExpired(f"mapping for session {session_id!r} expired")
        plaintext = self._aead.decrypt(rec.nonce, rec.ciphertext, session_id.encode("utf-8"))
        return Mapping.from_json(plaintext.decode("utf-8"))

    def delete(self, session_id: str) -> None:
        self._records.pop(session_id, None)

    def purge_expired(self) -> int:
        now = time.time()
        dead = [k for k, v in self._records.items() if v.expires_at < now]
        for k in dead:
            del self._records[k]
        return len(dead)

    def __len__(self) -> int:
        return len(self._records)
