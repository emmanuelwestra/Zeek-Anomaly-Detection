from __future__ import annotations

from dataclasses import dataclass, field
from ipaddress import ip_address
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_NUMERIC = ["orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts", "orig_ip_bytes", "resp_ip_bytes"]
DERIVED_NUMERIC = [
    "bytes_per_packet", "packets_per_second", "bytes_per_second", "orig_to_resp_byte_ratio",
    "orig_to_total_bytes", "resp_to_total_bytes", "orig_to_total_pkts", "resp_to_total_pkts",
    "resp_bytes_present", "duration_is_zero", "packet_symmetry", "byte_symmetry",
    "log_total_bytes", "log_total_pkts", "src_is_private", "dest_is_private",
    "internal_to_internal", "internal_to_external", "external_to_internal",
    "dest_port_is_dns", "dest_port_is_web", "dest_port_is_admin", "dest_port_is_file_sharing",
    "src_port_is_ephemeral", "dest_port_is_ephemeral",
]
CATEGORICAL = [
    "protocol", "service", "conn_state", "history", "local_orig", "local_resp",
    "src_port_bucket", "dest_port_bucket", "src_port_bucket_detail",
    "dest_port_bucket_detail", "dest_port_topk",
]
ALIASES = {
    "protocol": ("protocol", "proto"),
    "src_ip": ("src_ip", "src_ip_zeek"),
    "dest_ip": ("dest_ip", "dest_ip_zeek"),
    "src_port": ("src_port", "src_port_zeek"),
    "dest_port": ("dest_port", "dest_port_zeek"),
}
LOG_COLUMNS = set(BASE_NUMERIC) | {
    "bytes_per_packet", "packets_per_second", "bytes_per_second", "orig_to_resp_byte_ratio"
}
SCALED_COLUMNS = LOG_COLUMNS | {"log_total_bytes", "log_total_pkts"}


def _encoder() -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


@dataclass
class FlowPreprocessor:
    """The frozen 31-numeric/11-categorical UWF feature contract."""

    explicit_unknown: bool = False
    top_k_dest_ports: int = 32
    unknown_token: str = "<unknown>"
    numeric_imputer: SimpleImputer = field(default_factory=lambda: SimpleImputer(strategy="median"))
    categorical_imputer: SimpleImputer = field(
        default_factory=lambda: SimpleImputer(strategy="constant", fill_value="<missing>")
    )
    scaler: StandardScaler = field(default_factory=StandardScaler)
    encoder: OneHotEncoder = field(default_factory=_encoder)
    top_ports_: set[str] = field(default_factory=set)
    feature_names_: list[str] = field(default_factory=list)
    fitted_: bool = False

    @classmethod
    def for_model(cls, model: str, config: dict[str, Any]) -> FlowPreprocessor:
        if model not in {"ocsvm", "ae"}:
            raise ValueError("model must be 'ocsvm' or 'ae'")
        return cls(
            explicit_unknown=model == "ae",
            top_k_dest_ports=int(config.get("top_k_dest_ports", 32)),
            unknown_token=str(config.get("unknown_category_token", "<unknown>")),
        )

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        ports = pd.to_numeric(_column(frame, "dest_port"), errors="coerce").dropna().astype(int).astype(str)
        self.top_ports_ = set(ports.value_counts().head(self.top_k_dest_ports).index)
        numeric, categorical = self._blocks(frame)
        numeric_values = self.numeric_imputer.fit_transform(numeric)
        scaled = numeric_values.copy()
        indices = [index for index, name in enumerate(numeric.columns) if name in SCALED_COLUMNS]
        scaled[:, indices] = self.scaler.fit_transform(scaled[:, indices])
        categories = self.categorical_imputer.fit_transform(
            _categorical_input(categorical, explicit_unknown=self.explicit_unknown)
        )
        fit_categories = categories
        if self.explicit_unknown:
            if np.any(categories == self.unknown_token):
                raise ValueError(f"Training data contains reserved token {self.unknown_token!r}")
            synthetic = []
            for index in range(categories.shape[1]):
                row = categories[0].copy()
                row[index] = self.unknown_token
                synthetic.append(row)
            fit_categories = np.vstack([categories, *synthetic])
        top_index = CATEGORICAL.index("dest_port_topk")
        if not np.any(fit_categories[:, top_index] == "dest_port_rare"):
            rare = fit_categories[0].copy()
            rare[top_index] = "dest_port_rare"
            fit_categories = np.vstack([fit_categories, rare])
        self.encoder.fit(fit_categories)
        encoded = self.encoder.transform(categories)
        self.feature_names_ = list(numeric.columns) + self.encoder.get_feature_names_out(CATEGORICAL).tolist()
        self.fitted_ = True
        return np.hstack([scaled, encoded]).astype(np.float32)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("FlowPreprocessor must be fitted before transform")
        numeric, categorical = self._blocks(frame)
        values = self.numeric_imputer.transform(numeric)
        indices = [index for index, name in enumerate(numeric.columns) if name in SCALED_COLUMNS]
        values[:, indices] = self.scaler.transform(values[:, indices])
        categories = self.categorical_imputer.transform(
            _categorical_input(categorical, explicit_unknown=self.explicit_unknown)
        )
        if self.explicit_unknown:
            categories = np.asarray(categories, dtype=object).copy()
            for index, known in enumerate(self.encoder.categories_):
                categories[~np.isin(categories[:, index], known), index] = self.unknown_token
        return np.hstack([values, self.encoder.transform(categories)]).astype(np.float32)

    def reconstruction_schema(self) -> dict[str, Any]:
        if not self.fitted_:
            raise RuntimeError("FlowPreprocessor must be fitted first")
        offset = len(BASE_NUMERIC) + len(DERIVED_NUMERIC)
        groups = []
        for name, labels in zip(CATEGORICAL, self.encoder.categories_, strict=True):
            stop = offset + len(labels)
            groups.append({"name": name, "start": offset, "stop": stop, "labels": list(labels)})
            offset = stop
        return {
            "version": 1,
            "input_width": offset,
            "numeric_count": len(BASE_NUMERIC) + len(DERIVED_NUMERIC),
            "numeric_names": BASE_NUMERIC + DERIVED_NUMERIC,
            "categorical_groups": groups,
            "unknown_category_token": self.unknown_token if self.explicit_unknown else None,
        }

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output)

    @classmethod
    def load(cls, path: str | Path) -> FlowPreprocessor:
        value = joblib.load(path)
        if not isinstance(value, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(value).__name__}")
        return value

    def _blocks(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        features = derive_features(frame, self.top_ports_)
        numeric = features.reindex(columns=BASE_NUMERIC + DERIVED_NUMERIC).apply(pd.to_numeric, errors="coerce")
        for column in LOG_COLUMNS:
            numeric[column] = np.log1p(numeric[column].clip(lower=0))
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        categorical = features.reindex(columns=CATEGORICAL).astype(object)
        categorical = categorical.where(pd.notna(categorical), None)
        return numeric, categorical


def derive_features(frame: pd.DataFrame, top_ports: set[str]) -> pd.DataFrame:
    result = frame.copy()
    for canonical, aliases in ALIASES.items():
        if canonical not in result:
            source = next((name for name in aliases if name in frame), None)
            if source is not None:
                result[canonical] = frame[source]
    orig_bytes = _number(frame, "orig_bytes")
    resp_bytes = _number(frame, "resp_bytes")
    orig_pkts = _number(frame, "orig_pkts")
    resp_pkts = _number(frame, "resp_pkts")
    duration = _number(frame, "duration")
    total_bytes = orig_bytes + resp_bytes
    total_pkts = orig_pkts + resp_pkts
    safe_bytes = total_bytes.replace(0, np.nan)
    safe_pkts = total_pkts.replace(0, np.nan)
    safe_duration = duration.replace(0, np.nan)
    safe_resp = resp_bytes.replace(0, np.nan)
    result["bytes_per_packet"] = (total_bytes / safe_pkts).fillna(0)
    result["packets_per_second"] = (total_pkts / safe_duration).fillna(0)
    result["bytes_per_second"] = (total_bytes / safe_duration).fillna(0)
    result["orig_to_resp_byte_ratio"] = (orig_bytes / safe_resp).fillna(0)
    result["orig_to_total_bytes"] = (orig_bytes / safe_bytes).fillna(0)
    result["resp_to_total_bytes"] = (resp_bytes / safe_bytes).fillna(0)
    result["orig_to_total_pkts"] = (orig_pkts / safe_pkts).fillna(0)
    result["resp_to_total_pkts"] = (resp_pkts / safe_pkts).fillna(0)
    result["resp_bytes_present"] = resp_bytes.gt(0)
    result["duration_is_zero"] = duration.eq(0)
    result["packet_symmetry"] = (1 - (orig_pkts - resp_pkts).abs() / safe_pkts).fillna(0)
    result["byte_symmetry"] = (1 - (orig_bytes - resp_bytes).abs() / safe_bytes).fillna(0)
    result["log_total_bytes"] = np.log1p(total_bytes.clip(lower=0))
    result["log_total_pkts"] = np.log1p(total_pkts.clip(lower=0))
    src_private = _column(frame, "src_ip").map(_private)
    dest_private = _column(frame, "dest_ip").map(_private)
    result["src_is_private"] = src_private
    result["dest_is_private"] = dest_private
    result["internal_to_internal"] = src_private & dest_private
    result["internal_to_external"] = src_private & ~dest_private
    result["external_to_internal"] = ~src_private & dest_private
    src_port = _column(frame, "src_port")
    dest_port = _column(frame, "dest_port")
    result["src_port_bucket"] = src_port.map(_port_bucket)
    result["dest_port_bucket"] = dest_port.map(_port_bucket)
    result["src_port_bucket_detail"] = src_port.map(_detailed_port_bucket)
    result["dest_port_bucket_detail"] = dest_port.map(_detailed_port_bucket)
    result["dest_port_topk"] = dest_port.map(lambda value: _top_port(value, top_ports))
    dest = pd.to_numeric(dest_port, errors="coerce")
    src = pd.to_numeric(src_port, errors="coerce")
    result["dest_port_is_dns"] = dest.eq(53)
    result["dest_port_is_web"] = dest.isin([80, 443, 8080, 8443])
    result["dest_port_is_admin"] = dest.isin([22, 3389, 5900])
    result["dest_port_is_file_sharing"] = dest.isin([137, 138, 139, 445])
    result["src_port_is_ephemeral"] = src.between(49152, 65535, inclusive="both")
    result["dest_port_is_ephemeral"] = dest.between(49152, 65535, inclusive="both")
    return result


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    source = next((candidate for candidate in ALIASES.get(name, (name,)) if candidate in frame), None)
    return frame[source] if source else pd.Series(index=frame.index, dtype=object)


def _number(frame: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(_column(frame, name), errors="coerce").fillna(0)


def _private(value: object) -> bool:
    try:
        return ip_address(str(value)).is_private
    except ValueError:
        return False


def _coerce_port(value: object) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _port_bucket(value: object) -> str:
    port = _coerce_port(value)
    if port is None:
        return "missing"
    if not 0 <= port <= 65535:
        return "invalid"
    if port <= 1023:
        return "well_known"
    return "registered" if port <= 49151 else "dynamic"


def _detailed_port_bucket(value: object) -> str:
    port = _coerce_port(value)
    if port is None:
        return "missing"
    special = {
        3: "icmp_code", 53: "dns_53", 67: "dhcp_67_68", 68: "dhcp_67_68",
        123: "ntp_123", 22: "ssh_22", 445: "smb_445", 5353: "mdns_5353", 5355: "llmnr_5355",
    }
    if port in special:
        return special[port]
    if port in {80, 443, 8080, 8443}:
        return "web"
    if port in {137, 138, 139}:
        return "netbios_137_139"
    if not 0 <= port <= 65535:
        return "invalid"
    if port <= 1023:
        return "system_0_1023"
    return "registered_1024_49151" if port <= 49151 else "ephemeral_49152_65535"


def _top_port(value: object, top_ports: set[str]) -> str:
    port = _coerce_port(value)
    if port is None:
        return "dest_port_missing"
    return f"dest_port_{port}" if str(port) in top_ports else "dest_port_rare"


def _categorical_input(frame: pd.DataFrame, *, explicit_unknown: bool) -> pd.DataFrame:
    if not explicit_unknown:
        return frame
    result = frame.copy()
    for column in result:
        missing = result[column].isna()
        result.loc[~missing, column] = result.loc[~missing, column].map(str)
        result.loc[missing, column] = np.nan
    return result
