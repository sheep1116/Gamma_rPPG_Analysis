from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class MethodSpec:
    requested_name: str
    normalized_name: str
    device_type: str
    function_name: str
    params: Dict[str, object]


def normalize_method_name(name: str) -> str:
    n = name.strip().upper()
    aliases = {
        "GREEN": "GREEN",
        "G": "GREEN",
        "CHROM": "CHROM",
        "POS": "POS",
        "LGI": "LGI",
        "PBV": "PBV",
        "OMIT": "OMIT",
    }
    if n not in aliases:
        raise ValueError(f"Unsupported method '{name}'. Supported: GREEN, CHROM, POS, LGI, PBV, OMIT")
    return aliases[n]


def default_params(method: str) -> Dict[str, object]:
    # pyVHR POS requires fps. RGB_sig_to_BVP replaces 'adaptive' by the actual fps.
    if method == "POS":
        return {"fps": "adaptive"}
    return {}


def resolve_method(vhr_bvp_module, method_name: str, prefer_cuda: bool = True) -> MethodSpec:
    """Resolve a pyVHR BVP method function.

    In current pyVHR master, CHROM/POS have cupy implementations, whereas GREEN/LGI/PBV/OMIT
    are CPU implementations. This resolver tries the requested backend first and falls back safely.
    """
    normalized = normalize_method_name(method_name)
    candidates: List[Tuple[str, str]] = []
    if prefer_cuda:
        candidates.append(("cuda", f"cupy_{normalized}"))
    candidates.append(("cpu", f"cpu_{normalized}"))
    # Last-resort raw name fallback for customized pyVHR installs.
    candidates.append(("cpu", normalized))

    for device, func_name in candidates:
        if hasattr(vhr_bvp_module, func_name):
            return MethodSpec(
                requested_name=method_name,
                normalized_name=normalized,
                device_type=device,
                function_name=func_name,
                params=default_params(normalized),
            )
    tried = ", ".join(f for _, f in candidates)
    raise AttributeError(f"Cannot find pyVHR method for {method_name}. Tried: {tried}")
