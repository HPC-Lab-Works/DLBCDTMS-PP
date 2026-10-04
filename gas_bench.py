"""Compile the DLBCDTMS-PP smart contracts with solcjs and measure gas on an
in-memory EVM (eth-tester / py-evm). Output: results/gas.json"""
import json, os, subprocess, tempfile, random
from web3 import Web3, EthereumTesterProvider

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)


def compile_contracts():
    src = os.path.join(HERE, "contracts", "DLBC.sol")
    tmp = tempfile.mkdtemp()
    subprocess.run(["solcjs", "--bin", "--abi", "--optimize", "-o", tmp, src], check=True,
                   capture_output=True)
    out = {}
    for name in ("Registry", "FASC", "TESC"):
        f = [x for x in os.listdir(tmp) if x.endswith(f"_{name}.bin")][0]
        a = [x for x in os.listdir(tmp) if x.endswith(f"_{name}.abi")][0]
        out[name] = (open(os.path.join(tmp, a)).read(), open(os.path.join(tmp, f)).read())
    return out


def main():
    c = compile_contracts()
    w3 = Web3(EthereumTesterProvider())
    ra, rsu = w3.eth.accounts[0], w3.eth.accounts[1]
    res = {}

    def deploy(name, *args, frm=ra):
        abi, bin_ = c[name]
        C = w3.eth.contract(abi=json.loads(abi), bytecode=bin_)
        tx = C.constructor(*args).transact({"from": frm, "gas": 8_000_000})
        r = w3.eth.wait_for_transaction_receipt(tx)
        res[f"deploy_{name}"] = r.gasUsed
        return w3.eth.contract(address=r.contractAddress, abi=json.loads(abi))

    def gas(fn, frm):
        r = w3.eth.wait_for_transaction_receipt(fn.transact({"from": frm, "gas": 30_000_000}))
        return r.gasUsed

    reg = deploy("Registry")
    fasc = deploy("FASC", reg.address, 50)
    S = 10**6
    tesc = deploy("TESC", reg.address, int(1.0 * S), int(2.0 * S), int(0.53 * S), int(2 * S))
    gas(reg.functions.addRSU(rsu), ra)

    ids = [Web3.keccak(text=f"tpid{i}") for i in range(400)]
    g = [gas(reg.functions.registerVehicle(t), ra) for t in ids]
    res["registerVehicle"] = sum(g) / len(g)

    rnd = random.Random(1)
    g = []
    for k in range(200):
        a, b = rnd.sample(ids, 2)
        g.append(gas(fasc.functions.submitRating(a, b, Web3.keccak(text=f"e{k}"), 1 if k % 3 else -1, 20000), rsu))
    res["submitRating"] = sum(g) / len(g)

    for n in (10, 50, 100):
        raters = rnd.sample(ids, n); reps = [rnd.choice(ids) for _ in range(n)]
        e = Web3.keccak(text=f"batch{n}")
        gb = gas(fasc.functions.submitBatch(raters, reps, e, [1] * n, [10000] * n), rsu)
        res[f"submitBatch_{n}"] = gb
        res[f"submitBatch_{n}_per_rating"] = gb / n

    res["sealBlock"] = gas(fasc.functions.sealBlock(Web3.keccak(text="root")), rsu)

    for n in (10, 50, 100, 200):
        sel = ids[:n]
        gc = gas(tesc.functions.commitEpoch(1, Web3.keccak(text="r1"), [(x, 800000, 100000, 3000000, 200000) for x in sel]), rsu)
        res[f"commitEpoch_first_{n}"] = gc
        gc2 = gas(tesc.functions.commitEpoch(3, Web3.keccak(text="r3"), [(x, 800000, 100000, 3000000, 200000) for x in sel]), rsu)
        res[f"commitEpoch_update_{n}"] = gc2
        res[f"commitEpoch_update_{n}_per_vehicle"] = gc2 / n
    res["revokeVehicle"] = gas(reg.functions.revokeVehicle(ids[0]), ra)
    # trust query is a view call (no gas when called locally); report estimated gas
    res["trustOf_estimate"] = tesc.functions.trustOf(ids[1]).estimate_gas()
    with open(os.path.join(OUT, "gas.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
