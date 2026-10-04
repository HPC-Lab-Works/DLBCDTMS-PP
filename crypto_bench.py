"""Micro-benchmarks of the cryptographic primitives used by DLBCDTMS-PP on this
machine (ECDSA P-256, SHA-256, AES-128-GCM, ECDH-based ECIES) and of
proof-of-work mining at several difficulties. Output: results/crypto.json"""
import json, os, time, hashlib, statistics, platform
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)


def bench(fn, n):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    ts.sort(); k = int(0.05 * n)
    core = ts[k:n - k]  # 5% trimmed
    return {"mean_ms": 1e3 * statistics.mean(core), "std_ms": 1e3 * statistics.stdev(core)}


def main():
    res = {"platform": platform.processor() or platform.machine(), "python": platform.python_version()}
    sk = ec.generate_private_key(ec.SECP256R1()); pk = sk.public_key()
    msg = os.urandom(200)
    sig = sk.sign(msg, ec.ECDSA(hashes.SHA256()))
    res["ecdsa_sign"] = bench(lambda: sk.sign(msg, ec.ECDSA(hashes.SHA256())), 3000)
    res["ecdsa_verify"] = bench(lambda: pk.verify(sig, msg, ec.ECDSA(hashes.SHA256())), 3000)
    res["sha256_200B"] = bench(lambda: hashlib.sha256(msg).digest(), 20000)
    key = AESGCM.generate_key(128); aes = AESGCM(key); nonce = os.urandom(12)
    ct = aes.encrypt(nonce, msg, None)
    res["aesgcm_enc_200B"] = bench(lambda: aes.encrypt(nonce, msg, None), 20000)
    res["aesgcm_dec_200B"] = bench(lambda: aes.decrypt(nonce, ct, None), 20000)
    rsu = ec.generate_private_key(ec.SECP256R1()); rpk = rsu.public_key()

    def ecies_enc():
        e = ec.generate_private_key(ec.SECP256R1())
        s = e.exchange(ec.ECDH(), rpk)
        k = HKDF(hashes.SHA256(), 16, None, b"tpid").derive(s)
        AESGCM(k).encrypt(nonce, msg[:48], None)
    eph = ec.generate_private_key(ec.SECP256R1()); epk = eph.public_key()

    def ecies_dec():
        s = rsu.exchange(ec.ECDH(), epk)
        k = HKDF(hashes.SHA256(), 16, None, b"tpid").derive(s)
    res["ecies_enc"] = bench(ecies_enc, 2000)
    res["ecies_dec"] = bench(ecies_dec, 2000)

    # Proof-of-work: mean time to find a nonce with d leading zero bits (SHA-256)
    pow_res = {}
    for d in (12, 14, 16, 18):
        target = 1 << (256 - d); times = []
        for trial in range(30 if d < 18 else 15):
            hdr = os.urandom(80); n = 0; t0 = time.perf_counter()
            while True:
                h = int.from_bytes(hashlib.sha256(hdr + n.to_bytes(8, "little")).digest(), "big")
                if h < target: break
                n += 1
            times.append(time.perf_counter() - t0)
        pow_res[str(d)] = {"mean_s": statistics.mean(times), "std_s": statistics.stdev(times)}
    res["pow"] = pow_res
    json.dump(res, open(os.path.join(OUT, "crypto.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
