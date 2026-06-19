import torch
from torch import uint8, int32, Tensor
import numpy as np


class BitPack:
    """
    Efficient bit-packing for quantized weight storage.
    Packs multiple low-bit integers into a single higher-bit container.

    Supported formats:
      8-bit → uint8  (1:1)
      4-bit → uint8  (2:1 packing)
      3-bit → int32  (10:1 packing — 10 values per 32-bit word)
      2-bit → uint8  (4:1 packing)
    """

    # ------------------------------------------------------------------
    # 8-bit
    # ------------------------------------------------------------------
    @staticmethod
    def pack_8bit_u8(W_q: Tensor) -> Tensor:
        return W_q.to(uint8)

    @staticmethod
    def unpack_8bit_u8(W_q: Tensor, dtype=uint8) -> Tensor:
        return W_q.to(dtype)

    # ------------------------------------------------------------------
    # 4-bit  (two 4-bit values per uint8)
    # ------------------------------------------------------------------
    @staticmethod
    def pack_4bit_u8(W_q: Tensor) -> Tensor:
        W_q = W_q.to(uint8)
        half = W_q.shape[0] // 2
        return (W_q[:half] << 4) | W_q[half:]

    @staticmethod
    def unpack_4bit_u8(W_q: Tensor, dtype=uint8) -> Tensor:
        half = W_q.shape[0]
        out = torch.empty([2 * half, W_q.shape[1]], dtype=dtype, device=W_q.device)
        out[:half] = (W_q & 0b11110000) >> 4
        out[half:] = W_q & 0b00001111
        return out

    # ------------------------------------------------------------------
    # 3-bit  (ten 3-bit values packed into one int32)
    # ------------------------------------------------------------------
    @staticmethod
    def pack_3bit_32(W_q: Tensor) -> Tensor:
        # Pad rows to multiple of 10
        padded_rows = int(10 * np.ceil(W_q.shape[0] / 10.0))
        W = torch.zeros([padded_rows, W_q.shape[1]], device=W_q.device, dtype=int32)
        W[: W_q.shape[0]] = W_q
        s = padded_rows // 10
        return (
            (W[0 * s : 1 * s] << 27)
            | (W[1 * s : 2 * s] << 24)
            | (W[2 * s : 3 * s] << 21)
            | (W[3 * s : 4 * s] << 18)
            | (W[4 * s : 5 * s] << 15)
            | (W[5 * s : 6 * s] << 12)
            | (W[6 * s : 7 * s] << 9)
            | (W[7 * s : 8 * s] << 6)
            | (W[8 * s : 9 * s] << 3)
            | (W[9 * s : 10 * s])
        )

    @staticmethod
    def unpack_3bit_32(W_q: Tensor, dtype=uint8) -> Tensor:
        s = W_q.shape[0]
        out = torch.empty([10 * s, W_q.shape[1]], dtype=dtype, device=W_q.device)
        out[0 * s : 1 * s] = (W_q >> 27) & 0b111
        out[1 * s : 2 * s] = (W_q >> 24) & 0b111
        out[2 * s : 3 * s] = (W_q >> 21) & 0b111
        out[3 * s : 4 * s] = (W_q >> 18) & 0b111
        out[4 * s : 5 * s] = (W_q >> 15) & 0b111
        out[5 * s : 6 * s] = (W_q >> 12) & 0b111
        out[6 * s : 7 * s] = (W_q >> 9) & 0b111
        out[7 * s : 8 * s] = (W_q >> 6) & 0b111
        out[8 * s : 9 * s] = (W_q >> 3) & 0b111
        out[9 * s : 10 * s] = W_q & 0b111
        return out

    # ------------------------------------------------------------------
    # 2-bit  (four 2-bit values per uint8)
    # ------------------------------------------------------------------
    @staticmethod
    def pack_2bit_u8(W_q: Tensor) -> Tensor:
        W_q = W_q.to(uint8)
        q = W_q.shape[0] // 4
        return (
            (W_q[0 * q : 1 * q] << 6)
            | (W_q[1 * q : 2 * q] << 4)
            | (W_q[2 * q : 3 * q] << 2)
            | W_q[3 * q : 4 * q]
        )

    @staticmethod
    def unpack_2bit_u8(W_q: Tensor, dtype=uint8) -> Tensor:
        q = W_q.shape[0]
        out = torch.empty([4 * q, W_q.shape[1]], dtype=dtype, device=W_q.device)
        out[0 * q : 1 * q] = (W_q >> 6) & 0b11
        out[1 * q : 2 * q] = (W_q >> 4) & 0b11
        out[2 * q : 3 * q] = (W_q >> 2) & 0b11
        out[3 * q : 4 * q] = W_q & 0b11
        return out

    # ------------------------------------------------------------------
    # Dispatch helpers
    # ------------------------------------------------------------------
    @classmethod
    def pack(cls, W_q: Tensor, nbits: int) -> Tensor:
        if nbits == 8:
            return cls.pack_8bit_u8(W_q)
        if nbits == 4:
            return cls.pack_4bit_u8(W_q)
        if nbits == 3:
            return cls.pack_3bit_32(W_q)
        if nbits == 2:
            return cls.pack_2bit_u8(W_q)
        raise ValueError(f"Unsupported nbits={nbits}")

    @classmethod
    def unpack(cls, W_q: Tensor, nbits: int, dtype=uint8) -> Tensor:
        if nbits == 8:
            return cls.unpack_8bit_u8(W_q, dtype)
        if nbits == 4:
            return cls.unpack_4bit_u8(W_q, dtype)
        if nbits == 3:
            return cls.unpack_3bit_32(W_q, dtype)
        if nbits == 2:
            return cls.unpack_2bit_u8(W_q, dtype)
        raise ValueError(f"Unsupported nbits={nbits}")
