// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title Registry: membership of RSUs and registration of trust pseudonyms (TPIDs)
/// @notice Written only by the Registration Authority (RA).
contract Registry {
    address public ra;
    mapping(address => bool) public isRSU;
    mapping(bytes32 => bool) public registered;   // TPID -> registered
    mapping(bytes32 => bool) public revoked;      // TPID -> revoked

    event VehicleRegistered(bytes32 indexed tpid);
    event VehicleRevoked(bytes32 indexed tpid);

    modifier onlyRA() { require(msg.sender == ra, "RA only"); _; }

    constructor() { ra = msg.sender; }

    function addRSU(address rsu) external onlyRA { isRSU[rsu] = true; }

    function registerVehicle(bytes32 tpid) external onlyRA {
        require(!registered[tpid], "exists");
        registered[tpid] = true;
        emit VehicleRegistered(tpid);
    }

    function revokeVehicle(bytes32 tpid) external onlyRA {
        revoked[tpid] = true;
        emit VehicleRevoked(tpid);
    }

    function isValid(bytes32 tpid) external view returns (bool) {
        return registered[tpid] && !revoked[tpid];
    }
}

/// @title FASC: Feedback Aggregation Smart Contract (Layer 1, executed by the zone RSU)
/// @notice Validates rating transactions (registration, de-duplication, per-rater quota)
///         and accumulates the weighted positive and negative rating mass per reporter.
contract FASC {
    Registry public reg;
    uint32 public quota;                                  // max ratings per rater per block
    uint64 public currentBlock;                           // Layer-1 block index
    mapping(bytes32 => uint64) public posMass;            // reporter TPID -> scaled mass (1e6)
    mapping(bytes32 => uint64) public negMass;
    mapping(bytes32 => bool) public seen;                 // H(rater, reporter, eventId) -> used
    mapping(bytes32 => uint32) public used;               // H(rater, block) -> ratings used
    bytes32[] public touched;                             // reporters touched in this block
    mapping(bytes32 => bool) public isTouched;

    event RatingAccepted(bytes32 indexed reporter, bytes32 indexed eventId, int8 rating);
    event BlockSealed(uint64 indexed blockIndex, bytes32 root, uint256 nReporters);

    constructor(address registry, uint32 q) { reg = Registry(registry); quota = q; }

    function submitRating(bytes32 rater, bytes32 reporter, bytes32 eventId,
                          int8 rating, uint32 raterWeight) public {
        require(reg.isRSU(msg.sender), "RSU only");
        require(reg.isValid(rater) && reg.isValid(reporter), "invalid TPID");
        require(rating == 1 || rating == -1, "rating");
        bytes32 k = keccak256(abi.encodePacked(rater, reporter, eventId));
        require(!seen[k], "duplicate");                    // replay and DoS de-duplication
        seen[k] = true;
        bytes32 q = keccak256(abi.encodePacked(rater, currentBlock));
        require(used[q] < quota, "quota");                 // rating-flood limit
        used[q] += 1;
        if (rating > 0) posMass[reporter] += raterWeight; else negMass[reporter] += raterWeight;
        if (!isTouched[reporter]) { isTouched[reporter] = true; touched.push(reporter); }
        emit RatingAccepted(reporter, eventId, rating);
    }

    function submitBatch(bytes32[] calldata raters, bytes32[] calldata reporters,
                         bytes32 eventId, int8[] calldata ratings,
                         uint32[] calldata weights) external {
        for (uint256 i = 0; i < raters.length; i++) {
            submitRating(raters[i], reporters[i], eventId, ratings[i], weights[i]);
        }
    }

    /// @notice Seal the Layer-1 block: returns the touched reporters with their masses
    ///         and resets the per-block accumulators. The Merkle root is computed off-chain
    ///         over the block's rating transactions and anchored on Layer 2 by TESC.
    function sealBlock(bytes32 root) external returns (bytes32[] memory ids,
            uint64[] memory pos, uint64[] memory neg) {
        require(reg.isRSU(msg.sender), "RSU only");
        uint256 n = touched.length;
        ids = new bytes32[](n); pos = new uint64[](n); neg = new uint64[](n);
        for (uint256 i = 0; i < n; i++) {
            bytes32 r = touched[i];
            ids[i] = r; pos[i] = posMass[r]; neg[i] = negMass[r];
            posMass[r] = 0; negMass[r] = 0; isTouched[r] = false;
        }
        delete touched;
        emit BlockSealed(currentBlock, root, n);
        currentBlock += 1;
    }
}

/// @title TESC: Trust Evaluation Smart Contract (Layer 2, consortium chain of RSUs)
/// @notice Applies the forgetting-factor beta-reputation update with asymmetric penalty
///         to the aggregated reporting evidence (a, b) and to the feedback-agreement
///         evidence (c, e) of every vehicle, and exposes trust values and flags.
contract TESC {
    Registry public reg;
    uint64 public constant S = 1e6;           // fixed-point scale
    uint64 public lambda;                     // forgetting factor (scaled)
    uint64 public eta;                        // asymmetric penalty (scaled)
    uint64 public theta;                      // malicious threshold (scaled)
    uint64 public nMin;                       // minimum evidence (scaled)

    struct Rec { uint64 a; uint64 b; uint64 c; uint64 e; uint64 lastEpoch; bool flagged; }
    mapping(bytes32 => Rec) public rec;
    mapping(uint64 => mapping(address => bytes32)) public anchor; // epoch, RSU -> Layer-1 root

    event TrustUpdated(bytes32 indexed tpid, uint64 trust, uint64 feedbackCred);
    event FlagChanged(bytes32 indexed tpid, bool flagged, uint64 trust);

    constructor(address registry, uint64 lam, uint64 et, uint64 th, uint64 nm) {
        reg = Registry(registry); lambda = lam; eta = et; theta = th; nMin = nm;
    }

    function _decay(uint64 x, uint64 k) internal view returns (uint64) {
        if (lambda == S) return x;
        for (uint64 i = 0; i < k && i < 64; i++) { x = (x * lambda) / S; }
        return x;
    }

    /// @notice One aggregate record per vehicle and epoch, produced by FASC:
    ///   pos, neg  rater-normalised positive and negative reporting evidence (scaled)
    ///   agr, dis  ratings by the vehicle that agreed or disagreed with the consensus (scaled)
    struct Agg { bytes32 id; uint64 pos; uint64 neg; uint64 agr; uint64 dis; }

    function commitEpoch(uint64 epoch, bytes32 root, Agg[] calldata recs) external {
        require(reg.isRSU(msg.sender), "RSU only");
        anchor[epoch][msg.sender] = root;
        for (uint256 i = 0; i < recs.length; i++) {
            _apply(recs[i], epoch);
        }
    }

    function _apply(Agg calldata g, uint64 epoch) internal {
        Rec storage r = rec[g.id];
        uint64 k = epoch > r.lastEpoch ? epoch - r.lastEpoch : 0;
        r.a = _decay(r.a, k) + g.pos;
        r.b = _decay(r.b, k) + (g.neg * eta) / S;
        r.c = _decay(r.c, k) + g.agr;
        r.e = _decay(r.e, k) + g.dis;
        r.lastEpoch = epoch;
        uint64 t = trustOf(g.id);
        emit TrustUpdated(g.id, t, feedbackCredOf(g.id));
        bool f = (r.a + r.b >= nMin) && (t < theta);
        if (f != r.flagged) { r.flagged = f; emit FlagChanged(g.id, f, t); }
    }

    function trustOf(bytes32 tpid) public view returns (uint64) {
        Rec storage r = rec[tpid];
        return ((r.a + S) * S) / (r.a + r.b + 2 * S);
    }

    function feedbackCredOf(bytes32 tpid) public view returns (uint64) {
        Rec storage r = rec[tpid];
        return ((r.c + S) * S) / (r.c + r.e + 2 * S);
    }
}
