
from contextlib import contextmanager
from datetime import datetime
from http.client import HTTPMessage
from http.client import parse_headers as parse_http_headers
from argparse import Namespace

import argparse
import base64
import doctest
import functools
import io
import logging
import math
import operator
import os
import secrets
import shutil
import struct
import sys
import typing as t
import unittest

t_wbuf = bytearray | memoryview  # Writable byte buffer
t_buf = bytes | t_wbuf  # Readable byte buffer


###################################################################
###                                               BITS AND BYTES ###


class FixedWordBase:
    "Operations on fixed-size unsigned integers."

    BITS = 0
    POW = 1
    MAX = 0
    BYTES = 0

    def __init_subclass__(cls, /, **kwargs):
        super().__init_subclass__(**kwargs)
        assert cls.__name__.startswith("Word")
        cls.BITS = int(cls.__name__[4:])
        cls.POW = 2**cls.BITS
        cls.MAX = cls.POW - 1
        cls.BYTES = math.ceil(cls.BITS / 8)

    @classmethod
    def rand(cls):
        "Return a genuine random number in the range of this Word."
        return secrets.randbelow(cls.POW)

    @classmethod
    def assert_range(cls, num: int):
        """Ensure that NUM is in the range of this Word.

        >>> Word16.assert_range(Word16.rand())

        """
        assert 0 <= num < cls.POW, hex(num)

    @classmethod
    def add(cls, fst: int, snd: int) -> int:
        """Add fixed-sized unsigned words with appropriate wrap-around.

        >>> Word8.add(100, 200)
        44

        """
        cls.assert_range(fst)
        cls.assert_range(snd)
        return (fst + snd) & cls.MAX

    @classmethod
    def left_rot(cls, num: int, shift: int) -> int:
        """Bitwise left-rotation of NUM by SHIFT.

        >>> for k in range(7):
        ...    z = Word6.left_rot(5, k)
        ...    print(f'{z:06b} {z:02X}')
        000101 05
        001010 0A
        010100 14
        101000 28
        010001 11
        100010 22
        000101 05

        """
        cls.assert_range(num)
        assert 0 <= shift <= cls.BITS
        num <<= shift
        return (num | num >> cls.BITS) & cls.MAX

    @classmethod
    def split(cls, num: int, target) -> t.Iterator[int]:
        """Iterate through smaller chunks of a word, in little-endian order.
        Only supports target sizes that evenly divide our size.

        >>> Word16.hex_grid(Word64.split(0x58edd8db587e3d9f, Word16))
         3d9f 587e d8db 58ed

        """
        assert cls.BITS % target.BITS == 0
        for _ in range(cls.BITS // target.BITS):
            yield num & target.MAX
            num >>= target.BITS

    @classmethod
    def to_bytes(cls, num: int) -> bytes:
        """Pack a word into bytes, in little-endian order.

        >>> Word32.to_bytes(0xa9bfc6d3)
        b'\\xd3\\xc6\\xbf\\xa9'
        """
        cls.assert_range(num)
        return int_to_bytes(num, cls.BYTES)

    @classmethod
    def join(cls, nums: t.Iterable[int]) -> int:
        "Join NUMS into a larger number, in little-endian order."
        result = 0
        k = 0
        for num in nums:
            cls.assert_range(num)
            result |= num << k
            k += cls.BITS
        return result

    @classmethod
    def from_bytes(cls, buf: bytes) -> int:
        "Unpack a word from bytes, in little-endian order."
        assert len(buf) == cls.BYTES
        return int_from_bytes(buf)

    @classmethod
    def hex_grid(
        cls,
        nums: t.Iterable[int],
        cols: t.Optional[int] = None,
        digits: t.Optional[int] = None,
    ):
        """Neatly display the numbers provided by NUMS, using hexadecimal.  Wrap
        the display into COLS columns.  DIGITS specifies the minimum number of
        hexadecimal digits to print for each number.  The defaults are derived
        from the Word size (aiming for 128 bits per line) but can be overridden.

        >>> Word16.hex_grid(0x9c << i & Word16.MAX for i in range(16))
         009c 0138 0270 04e0 09c0 1380 2700 4e00
         9c00 3800 7000 e000 c000 8000 0000 0000

        """
        if digits is None:
            digits = math.ceil(cls.BITS / 4)
        if cols is None:
            cols = math.ceil(32 / digits)
        i = None
        for i, word in enumerate(nums):
            cls.assert_range(word)
            print(
                f" {word:0{digits}x}", end="\n" if i % cols == cols - 1 else ""
            )
        if i % cols != cols - 1:
            print()


class Word6(FixedWordBase):
    "Operations on 6-bit unsigned integers."


class Word8(FixedWordBase):
    "Operations on 8-bit unsigned integers."


class Word16(FixedWordBase):
    "Operations on 16-bit unsigned integers."


class Word32(FixedWordBase):
    "Operations on 32-bit unsigned integers."


class Word64(FixedWordBase):
    "Operations on 64-bit unsigned integers."


class Word70(FixedWordBase):
    "Operations on 70-bit unsigned integers."


class Word128(FixedWordBase):
    "Operations on 128-bit unsigned integers."


class Word256(FixedWordBase):
    "Operations on 256-bit unsigned integers."


class Word512(FixedWordBase):
    "Operations on 512-bit unsigned integers."


def xor_zip(yys: t.Iterable[int], zzs: t.Iterable[int]) -> t.Iterable[int]:
    """Merge two byte streams using XOR, producing a new byte stream.

    Like zip(), when one or the other iterable is exhausted, we stop iterating.
    This example encodes an ASCII byte string by alternating between the bytes
    03 and 04.  You can see that the consecutive 'l' characters in "Hello" are
    encoded differently, as "oh".

    >>> import itertools
    >>> bytes(xor_zip(b"Hello World!", itertools.cycle([3,4])))
    b'Kaohl$Tkqhg%'
    >>> key = [secrets.randbelow(256) for _ in range(6)]
    >>> ctxt = bytes(xor_zip(b"secret message", key))
    >>> [chr(b) for b in xor_zip(key, ctxt)]
    ['s', 'e', 'c', 'r', 'e', 't']

    """
    return (y ^ z for y, z in zip(yys, zzs))


def int_from_bytes(buf: t_buf) -> int:
    "Unpack an integer from bytes, in little-endian order."
    return int.from_bytes(buf, byteorder="little")


def int_to_bytes(val: int, nbytes: int) -> bytes:
    "Pack an integer into bytes, in little-endian order."
    return val.to_bytes(nbytes, byteorder="little")


def xor_bytes_with_int(buf: t_buf, val: int) -> bytes:
    """Merge a byte string with a big integer, convert back to bytes."""
    return int_to_bytes(int_from_bytes(buf) ^ val, len(buf))


####################################################################
###                                            POWERS AND PRIMES ###


def modulo(mod: int):
    """The standard mod operator, but with curried (and flipped) arguments.

    >>> hun = modulo(100)
    >>> hun(2**8)
    56
    """
    return lambda x: x % mod


def fastpow(base: int, exp: int, mod=lambda x: x, mul=operator.mul) -> int:
    """Fast exponentiation, with a customizable multiplication operator.

    >>> fastpow(3, 10)
    59049
    >>> fastpow(7, 12, mod = lambda x: x % 599)
    125
    >>> fastpow(2, 20, modulo(1000))
    576
    >>> [fastpow(2, k, modulo(10)) for k in range(10)]
    [1, 2, 4, 8, 6, 2, 4, 8, 6, 2]
    """
    assert exp >= 0
    result = mod(1)
    while exp > 0:
        if exp & 1:  # Odd: multiply and decrement
            result = mod(mul(result, base))
            exp -= 1
        else:  # Even: square and halve
            base = mod(mul(base, base))
            exp //= 2
    return result


def bin_mul(fst: int, snd: int, add=operator.add, mod=lambda x: x) -> int:
    """The 'peasant' binary multiplication algorithm, with customizable add and
    optional modulus operators."""
    assert fst >= 0
    result = 0
    while fst > 0:
        if fst & 1:
            result = mod(add(result, snd))
        fst = fst >> 1
        snd = mod(snd << 1)
    return result


def xor_mod(mod: int):
    """Perform a type of modulo by 'subtracting' multiples of MOD from VAL using
    XOR."""

    def xor_mod_loop(val: int) -> int:
        mbits = mod.bit_length()
        vbits = val.bit_length()
        while vbits >= mbits:
            factor = mod << (vbits - mbits)
            val ^= factor
            vbits = val.bit_length()
        return val

    return xor_mod_loop


def byte_mul(
    top: bytes, bot: bytes, mul=operator.mul, add=operator.add, mod=lambda x: x
):
    """Customizable byte-by-byte multiplication."""
    return mod(
        functools.reduce(
            add,
            (
                mod(
                    functools.reduce(
                        add,
                        (
                            mul(bval, tval) << 8 * tidx
                            for tidx, tval in enumerate(top)
                        ),
                    )
                    << 8 * bidx
                )
                for bidx, bval in enumerate(bot)
                if bval != 0
            ),
        )
    )


def crack_discrete_log(
    base: int, mod: int, goal: int, progress: int = Word16.POW
) -> int | None:
    """Search for an integer k such that (base**k)%mod == goal.  Each PROGRESS
    iterations, print a status message, then print when found or not found.

    >>> crack_discrete_log(7, 23, 12)
    crack_discrete_log: found 0x8
    8

    >>> crack_discrete_log(2, 164987, 80662)
    crack_discrete_log: check 0x10000...
    crack_discrete_log: found 0x1fa61
    129633

    >>> crack_discrete_log(2, 10, 5)
    crack_discrete_log: not found

    """
    assert 0 < goal < mod
    result = 1
    for k in range(mod):
        if goal == result:
            print(f"crack_discrete_log: found 0x{k:x}")
            return k
        if k % progress == 0 and k > 0:
            print(f"crack_discrete_log: check 0x{k:x}...")
        result = result * base % mod
    print("crack_discrete_log: not found")
    return None


########################################################### AES Code Below

class AES:
    "Pure-python implementation of the Advanced Encryption Standard."

    log = logging.getLogger("AES")
    forward_sbox = bytearray(256)
    inverse_sbox = bytearray(256)

    @staticmethod
    def mul(fst: int, snd: int) -> int:
        "Multiplication in Galois(2**8), with Rijndael polynomial modulus."
        return bin_mul(fst, snd, operator.xor, xor_mod(0x11B))

    @staticmethod
    def round_constant(num: int) -> int:
        """Used in key expansion, round constants are 2**(N-1) in GF(2**8).

        >>> Word8.hex_grid(AES.round_constant(k) for k in range(1,11))
         01 02 04 08 10 20 40 80 1b 36

        """
        assert 1 <= num <= 10
        return fastpow(2, num - 1, mul=AES.mul)

    @staticmethod
    def assert_key_bytes(nbytes: int):
        "Ensure that a key of size NBYTES is compatible with AES."
        assert nbytes in [16, 24, 32]

    @staticmethod
    def sbox_transform(val: int) -> int:
        "The affine transformation applied to inverses to create substitutions."
        return (
            val
            ^ Word8.left_rot(val, 1)
            ^ Word8.left_rot(val, 2)
            ^ Word8.left_rot(val, 3)
            ^ Word8.left_rot(val, 4)
            ^ 0x63
        )

    @classmethod
    def init_substitution_boxes(cls):
        "Initialize the Rijndael substitution boxes."
        cls.forward_sbox[0] = 0x63
        cls.inverse_sbox[0x63] = 0
        ppp = qqq = 1
        for _ in range(255):
            # The bytes 03 and F6 are multiplicative inverses, but also they
            # are generators for the entire field.
            ppp = AES.mul(3, ppp)
            qqq = AES.mul(0xF6, qqq)
            val = cls.sbox_transform(qqq)
            cls.forward_sbox[ppp] = val
            cls.inverse_sbox[val] = ppp
        cls.log.debug("FORWARD S-BOX:")
        cls.show_sbox(cls.forward_sbox, prn=cls.log.debug)
        cls.log.debug("INVERSE S-BOX:")
        cls.show_sbox(cls.inverse_sbox, prn=cls.log.debug)

    @staticmethod
    def show_sbox(box, prn=print):
        "Print out a substitution box with row/column headings."
        assert len(box) == 256
        prn(" " * 5 + " ".join(f"{col:2X}" for col in range(16)))
        for row in range(16):
            prn(
                f"  {row:X}: "
                + " ".join(f"{box[row*16+col]:02x}" for col in range(16))
            )

    def __init__(self, key: t_buf):
        """Constructor for a particular key. Performs immediate expansion.

        Following test is the example from these lecture notes
        https://www.kavaliro.com/wp-content/uploads/2014/03/AES.pdf

        >>> AES(b"Thats my Kung Fu").show_round_keys()
        Round key  0: 54686174 73206d79 204b756e 67204675
        Round key  1: e232fcf1 91129188 b159e4e6 d679a293
        Round key  2: 56082007 c71ab18f 76435569 a03af7fa
        Round key  3: d2600de7 157abc68 6339e901 c3031efb
        Round key  4: a11202c9 b468bea1 d75157a0 1452495b
        Round key  5: b1293b33 05418592 d210d232 c6429b69
        Round key  6: bd3dc287 b87c4715 6a6c9527 ac2e0e4e
        Round key  7: cc96ed16 74eaaa03 1e863f24 b2a8316a
        Round key  8: 8e51ef21 fabb4522 e43d7a06 56954b6c
        Round key  9: bfe2bf90 4559fab2 a16480b4 f7f1cbd8
        Round key 10: 28fddef8 6da4244a ccc0a4fe 3b316f26
        """
        self.assert_key_bytes(len(key))
        self._num_rounds = {16: 10, 24: 12, 32: 14}[len(key)]
        # Key expansion schedule
        key_words = memoryview(key).cast("I")
        kex_buf = memoryview(bytearray(16 * (self._num_rounds + 1)))
        kex_words = kex_buf.cast("I")
        for i, _ in enumerate(kex_words):
            kex_bytes = kex_buf[4 * i :]
            if i < len(key_words):
                kex_words[i] = key_words[i]
            elif i % len(key_words) == 0:
                kex_words[i] = kex_words[i - 1]
                self.rot_word(kex_bytes)
                self.sub_bytes(kex_bytes, self.forward_sbox)
                kex_words[i] ^= kex_words[i - len(key_words)]
                kex_words[i] ^= self.round_constant(i // len(key_words))
            elif len(key_words) > 6 and i % len(key_words) == 4:
                kex_words[i] = kex_words[i - 1]
                self.sub_bytes(kex_bytes, self.forward_sbox)
                kex_words[i] ^= kex_words[i - len(key_words)]
            else:
                kex_words[i] = kex_words[i - 1]
                kex_words[i] ^= kex_words[i - len(key_words)]
        self._round_keys = [k[0] for k in struct.iter_unpack("16s", kex_buf)]
        self._round_keys_long = [k[0] for k in struct.iter_unpack("L", kex_buf)]
        self.show_round_keys(self.log.debug)

    def show_round_keys(self, prn=print):
        "Print out the list of round keys."
        for rnum, rkey in enumerate(self._round_keys):
            prn(f"Round key {rnum:2}: {rkey.hex(' ',4)}")

    @staticmethod
    def rot_word(word: t_wbuf, nbytes: int = 1):
        """Rotate a 4-byte buffer word in-place.

        >>> bs = bytearray(b'FARM.-!')
        >>> AES.rot_word(bs); bs
        bytearray(b'ARMF.-!')
        >>> AES.rot_word(bs, 2); bs
        bytearray(b'MFAR.-!')
        >>> AES.rot_word(bs, 3); bs
        bytearray(b'RMFA.-!')
        """
        assert 0 < nbytes < 4
        front = bytes(word[:nbytes])
        back = bytes(word[nbytes:4])
        word[: 4 - nbytes], word[4 - nbytes : 4] = back, front

    @staticmethod
    def sub_bytes(vec: t_wbuf, sbox: t_buf):
        """In-place substitution of each byte, using sbox.

        >>> bs = bytearray.fromhex("3f1a2b00")
        >>> AES.sub_bytes(bs, AES.forward_sbox); bs.hex()
        '75a2f163'
        >>> AES.sub_bytes(bs, AES.inverse_sbox); bs.hex()
        '3f1a2b00'
        """
        for idx, byte in enumerate(vec):
            vec[idx] = sbox[byte]

    @staticmethod
    def poly_mul(fst: t_buf, snd: t_buf) -> bytes:
        "Polynomial multiplication in GF(2**8)."
        assert len(fst) == len(snd) == 4
        return Word32.to_bytes(
            byte_mul(
                fst,
                snd,
                mul=AES.mul,
                add=operator.xor,
                mod=xor_mod(0x100000001),
            )
        )

    @staticmethod
    def mix_column(vec: t_wbuf):
        """The Rijndael MixColumns multiplication.

        Test vectors are from https://en.wikipedia.org/wiki/Rijndael_MixColumns

        >>> vec = bytearray.fromhex("db135345")
        >>> AES.mix_column(vec); vec.hex()
        '8e4da1bc'
        >>> vec = bytearray.fromhex("f20a225c")
        >>> AES.mix_column(vec); vec.hex()
        '9fdc589d'
        >>> vec = bytearray.fromhex("01010101")
        >>> AES.mix_column(vec); vec.hex()
        '01010101'

        """
        vec[:] = AES.poly_mul(vec, b"\x02\x01\x01\x03")

    @staticmethod
    def unmix_column(vec):
        """Inverse Rijndael MixColumns.

        >>> vec = bytearray.fromhex("c6c6c6c6")
        >>> AES.unmix_column(vec); vec.hex()
        'c6c6c6c6'
        >>> vec = bytearray.fromhex("d5d5d7d6")
        >>> AES.unmix_column(vec); vec.hex()
        'd4d4d4d5'
        >>> vec = bytearray.fromhex("4d7ebdf8")
        >>> AES.unmix_column(vec); vec.hex()
        '2d26314c'
        """
        assert len(vec) == 4
        vec[:] = AES.poly_mul(vec, b"\x0e\x09\x0d\x0b")

    @classmethod
    def key_gen(cls, nbytes=16) -> bytes:
        "Create a uniform-random key compatible with AES."
        cls.assert_key_bytes(nbytes)
        return secrets.token_bytes(nbytes)

    @staticmethod
    def show_block(
        block: bytes | bytearray | memoryview, heading="", prn=print
    ):
        "Print 16 bytes as a matrix, assuming column-major order."
        for row_idx in range(4):
            row = block[row_idx:16:4]
            out = io.StringIO()
            out.write(" [" if row_idx == 0 else "  ")
            out.write(row.hex(" "))
            out.write("]  |" if row_idx == 3 else "   |")
            for byte in row:
                out.write(chr(byte) if 0x20 <= byte < 0x7F else ".")
            out.write("|")
            if row_idx == 0 and len(heading) > 0:
                out.write("  # ")
                out.write(heading)
            prn(out.getvalue())
            out.seek(0)
            out.truncate()

    def enc_block(self, buf):

        """Encrypt one 16-byte block using the key.

        Example from https://www.kavaliro.com/wp-content/uploads/2014/03/AES.pdf

        >>> AES(b"Thats my Kung Fu").enc_block(b"Two One Nine Two").hex()
        '29c3505f571420f6402299b31a02d73a'

        Example from Aumasson book:

        >>> aes = AES(bytes.fromhex("2c6202f9a582668aa96d511862d8a279"))
        >>> aes.enc_block(bytes([0] * 16)).hex()
        '12b620bb5eddcde9a07523e59292a6d7'

        """
        assert len(buf) == 16
        block = memoryview(bytearray(buf))  # Ensure block is mutable
        # These are then mutable slices or views of the block:
        block_longs = block.cast("L")  # as array of 2 longs (64 bits each)
        block_cols = [block[i : i + 4] for i in range(0, 16, 4)]
        block_rows = [block[r::4] for r in range(4)]
        self.show_block(block, "Initial state", self.log.debug)

        for round_num, round_key in enumerate(self._round_keys):
            # Steps: SubBytes, ShiftRows, MixColumns, AddRoundKey.
            # First round (round_num 0) does AddRoundKey only.
            if 0 < round_num:
                self.sub_bytes(block, self.forward_sbox)
                self.show_block(block, f"SubBytes {round_num}", self.log.debug)

                for row_idx, row_slice in enumerate(block_rows[1:], start=1):
                    self.rot_word(row_slice, row_idx)
                self.show_block(block, f"ShiftRows {round_num}", self.log.debug)

            # Last round skips MixColumns.
            if 0 < round_num < self._num_rounds:
                for col in block_cols:
                    self.mix_column(col)
                self.show_block(
                    block, f"MixColumns {round_num}", self.log.debug
                )

            # AddRoundKey: remember that '^=' means "XOR with"
            self.show_block(round_key, f"Key {round_num}", self.log.debug)
            block_longs[0] ^= self._round_keys_long[2 * round_num]  # least
            block_longs[1] ^= self._round_keys_long[2 * round_num + 1]
            self.show_block(block, f"AddRoundKey {round_num}", self.log.debug)

        return bytes(block)

    # def dec_block(self, buf):
        """Decrypt one 16-byte block using the key.

        >>> ciphertext = bytes.fromhex("29c3505f571420f6402299b31a02d73a")
        >>> AES(b"Thats my Kung Fu").dec_block(ciphertext)
        b'Two One Nine Two'
        """
        # TODO:
        #   - Work backwards (main loop counts down, but also steps within loop
        #     are reversed).
        #   - Replace mix_column with unmix_column.
        #   - Rotate row by (3 - row_idx) rather than by row_idx.
        #   - For sub_bytes, provide inverse_sbox rather than forward_sbox.
        #   - To see step-by-step results:
        #     logging.root.setLevel(logging.DEBUG)


    def dec_block(self, buf):
        assert len(buf) == 16
        block = memoryview(bytearray(buf))  # Ensure block is mutable
        # These are then mutable slices or views of the block:
        block_longs = block.cast("L")  # as array of 2 longs (64 bits each)
        block_cols = [block[i: i + 4] for i in range(0, 16, 4)]
        block_rows = [block[r::4] for r in range(4)]
        self.show_block(block, "Initial state", self.log.debug)

        for round_num in reversed(range(self._num_rounds + 1)):
            round_key = self._round_keys[round_num]

            self.show_block(round_key, f"Key {round_num}", self.log.debug)
            block_longs[0] ^= self._round_keys_long[2 * round_num]
            block_longs[1] ^= self._round_keys_long[2 * round_num + 1]
            self.show_block(block, f"AddRoundKey {round_num}", self.log.debug)

            # Inverse MixColumns (skipping for the last round)
            if round_num != 0 and round_num != self._num_rounds:
                for col in block_cols:
                   self.unmix_column(col)
                self.show_block(block, f"Inverse MixColumns {round_num}", self.log.debug)

            # Inverse ShiftRows (except for the first round)
            if round_num != self._num_rounds:
                for row_idx, row_slice in enumerate(block_rows[1:], start=1):
                    self.rot_word(row_slice, 4 - row_idx)
                self.show_block(block, f"Inverse ShiftRows {round_num}", self.log.debug)    

            if 0 < round_num:
                self.sub_bytes(block, self.inverse_sbox)
                self.show_block(block, f"SubBytes {round_num}", self.log.debug)
                
        return bytes(block)

aes = AES(bytes.fromhex('13766cd0e824ebd93d0291d041eaf40f'))

ciphertext = bytes.fromhex('b8c763c8dcb0d4c9793569d40c31ec65')

decrypted = aes.dec_block(ciphertext)

print("Decrypted plaintext:", decrypted.hex())
