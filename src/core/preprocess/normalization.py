from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any, Dict, List, Mapping, Optional, Tuple

from domain.models.infobox import InfoboxRow
from domain.models.normalized_field import NormalizedField


_key_non_alnum_re = re.compile(r"[^0-9a-zA-Z]+")
_number_re = re.compile(r"[-+]?(?:\d{1,3}(?:[,\u00A0]\d{3})+|\d+)(?:\.\d+)?")
_year_re = re.compile(r"\b(19|20)\d{2}\b")
_rank_re = re.compile(r"\b(\d+)(st|nd|rd|th)\b", re.IGNORECASE)
_coord_re = re.compile(r"\b(\d{1,2}°\d{1,2}′[NS])\b|\b(\d{1,3}°\d{1,2}′[EW])\b")
_currency_code_re = re.compile(r"\b([A-Z]{3})\b")
_rank_paren_re = re.compile(r"\s*\(\s*\d+(st|nd|rd|th)\s*\)\s*$", re.IGNORECASE)


def normalize_key(label: str) -> str:
    s = label.strip().strip(":")
    s = s.lower()
    s = _key_non_alnum_re.sub("_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    return s or "field"


def _tokenize(text: str) -> List[str]:
    tokens = re.split(r"[^0-9A-Za-z]+", text)
    return [t for t in tokens if t]


def _parse_numbers(text: str) -> List[float]:
    results: List[float] = []
    for match in _number_re.finditer(text):
        token = match.group(0)
        cleaned = token.replace(",", "").replace("\u00A0", "")
        try:
            if "." in cleaned:
                results.append(float(cleaned))
            else:
                results.append(int(cleaned))
        except ValueError:
            continue
    return results


def normalize_rows(rows: List[InfoboxRow]) -> Mapping[str, NormalizedField]:
    result: Dict[str, NormalizedField] = {}
    current_gdp_context: Optional[str] = None

    for row in rows:
        base_key = normalize_key(row.label)

        if base_key in {"gdp_ppp", "gdp_nominal"}:
            current_gdp_context = base_key
            key = base_key
        else:
            label_stripped = row.label.strip()
            is_bullet = label_stripped.startswith("•")

            if current_gdp_context and is_bullet:
                bullet_key = normalize_key(label_stripped.lstrip("•").strip())
                if bullet_key in {"total", "per_capita"}:
                    key = f"{current_gdp_context}_{bullet_key}"
                else:
                    key = base_key
            else:
                key = base_key

            if not is_bullet and base_key not in {"gdp_ppp", "gdp_nominal"}:
                current_gdp_context = None

        text = row.value_text.strip()
        tokens = _tokenize(text)
        numbers = _parse_numbers(text)

        result[key] = NormalizedField(
            raw_label=row.label,
            text=text,
            tokens=tokens,
            numbers=numbers,
        )

    return result


def normalized_fields_to_dict(fields: Mapping[str, NormalizedField]) -> Dict[str, dict]:
    return {k: asdict(v) for k, v in fields.items()}


def _first_year(text: str) -> Optional[int]:
    m = _year_re.search(text)
    return int(m.group(0)) if m else None


def _first_rank(text: str) -> Optional[int]:
    m = _rank_re.search(text)
    return int(m.group(1)) if m else None


def _extract_capital_name(text: str) -> str:
    cutoff_tokens = ["Country:"]
    for tok in cutoff_tokens:
        idx = text.find(tok)
        if idx != -1:
            text = text[:idx].strip()
    text = re.sub(r"\s+\d{1,3}°.*$", "", text).strip()
    coord_idx = text.find("°")
    if coord_idx != -1:
        text = text[:coord_idx].strip()
    text = re.sub(r"\s+\d+$", "", text).strip()
    return re.sub(r"\s+", " ", text).strip()


def _extract_coords(text: str) -> Tuple[Optional[str], Optional[str]]:
    lat = None
    lon = None
    for m in _coord_re.finditer(text):
        token = m.group(0)
        if token.endswith(("N", "S")) and lat is None:
            lat = token
        elif token.endswith(("E", "W")) and lon is None:
            lon = token
        if lat and lon:
            break
    return lat, lon


def _parse_currency(text: str) -> Tuple[Optional[str], Optional[str]]:
    name = None
    code = None
    if "(" in text and ")" in text:
        before = text.split("(", 1)[0].strip()
        inside = text.split("(", 1)[1].split(")", 1)[0]
        m = _currency_code_re.search(inside)
        if before:
            name = before
        if m:
            code = m.group(1)
    if name is None:
        name = text.strip() or None
    return name, code


def _digits_only(text: str) -> Optional[str]:
    digits = re.sub(r"\D+", "", text or "")
    return digits or None


def _split_languages(text: str) -> List[str]:
    if not text:
        return []
    s = re.sub(r"\s+", " ", text).strip()
    if any(sep in s for sep in [",", ";", "·", " / "]):
        parts = re.split(r"\s*[;,·]|\s+/\s+", s)
    else:
        parts = [s]
    return [p.strip() for p in parts if p.strip()]


def _split_tlds(text: str) -> List[str]:
    if not text:
        return []
    tokens = re.split(r"\s+|[,;]", text.strip())
    return [t for t in tokens if t]


def _strip_rank_paren(text: str) -> str:
    return _rank_paren_re.sub("", text or "").strip()


def build_comparison_fields(
    fields: Mapping[str, NormalizedField],
    *,
    country_name: Optional[str] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    if country_name:
        out["country_name"] = country_name
    if "official_name" in fields:
        out["official_name"] = fields["official_name"].text

    cap_key = None
    for k in ("capital", "capital_and_largest_city"):
        if k in fields:
            cap_key = k
            break
    if cap_key:
        cap_text = fields[cap_key].text
        out["capital"] = _extract_capital_name(cap_text) or cap_text
        if "largest_city" in fields:
            out["largest_city"] = _extract_capital_name(fields["largest_city"].text)
        else:
            out["largest_city"] = out["capital"]
        lat, lon = _extract_coords(cap_text)
        if lat or lon:
            out["coordinates"] = {"latitude": lat, "longitude": lon}

    langs: Dict[str, Any] = {}
    if "official_languages" in fields:
        langs["official"] = _split_languages(fields["official_languages"].text)
    for k in ("recognised_minority_language", "recognised_minority_languages"):
        if k in fields:
            langs["recognized_minority"] = _split_languages(fields[k].text)
            break
    if langs:
        out["languages"] = langs

    gov: Dict[str, Any] = {}
    if "government" in fields:
        gov["type"] = fields["government"].text
    for k in ("president", "monarch", "king", "queen", "emir", "supreme_leader"):
        if k in fields:
            gov["head_of_state"] = fields[k].text
            break
    for k in ("prime_minister", "chancellor", "premier"):
        if k in fields:
            gov["head_of_government"] = fields[k].text
            break
    for k in ("speaker_of_parliament", "parliament_speaker"):
        if k in fields:
            gov["speaker_of_parliament"] = fields[k].text
            break
    if gov:
        out["government"] = gov
    if "legislature" in fields:
        out["legislature"] = fields["legislature"].text

    area: Dict[str, Any] = {}
    if "area_total" in fields:
        nums = fields["area_total"].numbers
        if nums:
            area["total_km2"] = int(nums[0])
    if "area_water" in fields or "area_water_percent" in fields:
        key = "area_water_percent" if "area_water_percent" in fields else "area_water"
        nums = fields[key].numbers
        if nums:
            area["water_percent"] = float(nums[0])
    if area:
        out["area"] = area

    pop: Dict[str, Any] = {}
    pop_year = None
    pop_total = None
    pop_rank = None
    pop_total_candidates = [k for k in fields.keys() if k.startswith("population_") and ("estimate" in k or "census" in k)]
    if pop_total_candidates:
        def key_year(k: str) -> int:
            ys = [int(y) for y in _year_re.findall(k)]
            return max(ys) if ys else 0
        chosen = sorted(pop_total_candidates, key=key_year, reverse=True)[0]
        pop_total_field = fields[chosen]
        pop_year = _first_year(pop_total_field.raw_label) or _first_year(pop_total_field.text) or _first_year(chosen)
        for n in pop_total_field.numbers:
            if isinstance(n, (int, float)) and n >= 10000:
                pop_total = int(n)
                break
        pop_rank = _first_rank(pop_total_field.text)

    if pop_year is not None:
        pop["year"] = pop_year
    if pop_total is not None:
        pop["total"] = pop_total
    if pop_rank is not None:
        pop["rank"] = pop_rank
    if "population_density" in fields:
        nums = fields["population_density"].numbers
        if nums:
            pop["density_per_km2"] = float(nums[0])
    if pop:
        out["population"] = pop

    economy: Dict[str, Any] = {}
    if "gdp_nominal" in fields:
        economy["gdp_nominal_year"] = _first_year(fields["gdp_nominal"].text) or _first_year(fields["gdp_nominal"].raw_label)
    if "gdp_ppp" in fields:
        economy["gdp_ppp_year"] = _first_year(fields["gdp_ppp"].text) or _first_year(fields["gdp_ppp"].raw_label)
    if "gdp_ppp_total" in fields:
        economy["gdp_ppp_total"] = _strip_rank_paren(fields["gdp_ppp_total"].text)
    else:
        for f in fields.values():
            if "GDP ( PPP" in f.raw_label and "Total" in f.raw_label:
                economy["gdp_ppp_total"] = _strip_rank_paren(f.text)
                break
    if "gdp_ppp_per_capita" in fields:
        economy["gdp_ppp_per_capita"] = _strip_rank_paren(fields["gdp_ppp_per_capita"].text)
    else:
        for f in fields.values():
            if "GDP ( PPP" in f.raw_label and "Per capita" in f.raw_label:
                economy["gdp_ppp_per_capita"] = _strip_rank_paren(f.text)
                break
    if "gdp_nominal_total" in fields:
        economy["gdp_nominal_total"] = _strip_rank_paren(fields["gdp_nominal_total"].text)
    else:
        for f in fields.values():
            if "GDP (nominal" in f.raw_label and "Total" in f.raw_label:
                economy["gdp_nominal_total"] = _strip_rank_paren(f.text)
                break
    if "gdp_nominal_per_capita" in fields:
        economy["gdp_nominal_per_capita"] = _strip_rank_paren(fields["gdp_nominal_per_capita"].text)
    else:
        for f in fields.values():
            if "GDP (nominal" in f.raw_label and "Per capita" in f.raw_label:
                economy["gdp_nominal_per_capita"] = _strip_rank_paren(f.text)
                break
    if economy:
        out["economy"] = economy

    dev: Dict[str, Any] = {}
    if "hdi" in fields:
        dev["hdi"] = {"year": _first_year(fields["hdi"].raw_label) or _first_year(fields["hdi"].text), "value": fields["hdi"].numbers[0] if fields["hdi"].numbers else None, "rank": _first_rank(fields["hdi"].text)}
    else:
        hdi_keys = [k for k in fields.keys() if k.startswith("hdi_")]
        if hdi_keys:
            k = sorted(hdi_keys)[-1]
            dev["hdi"] = {"year": _first_year(k) or _first_year(fields[k].raw_label), "value": fields[k].numbers[0] if fields[k].numbers else None, "rank": _first_rank(fields[k].text)}
    gini_keys = [k for k in fields.keys() if k.startswith("gini")]
    if gini_keys:
        k = sorted(gini_keys)[-1]
        dev["gini"] = {"year": _first_year(k) or _first_year(fields[k].raw_label), "value": fields[k].numbers[0] if fields[k].numbers else None}
    if dev:
        out["development"] = dev

    if "currency" in fields:
        name, code = _parse_currency(fields["currency"].text)
        cur: Dict[str, Any] = {}
        if name:
            cur["name"] = name
        if code:
            cur["code"] = code
        if cur:
            out["currency"] = cur
    if "calling_code" in fields:
        out["calling_code"] = _digits_only(fields["calling_code"].text) or fields["calling_code"].text
    if "iso_3166_code" in fields:
        out["iso_3166_code"] = fields["iso_3166_code"].text
    if "internet_tld" in fields:
        out["internet_tld"] = _split_tlds(fields["internet_tld"].text)
    if "time_zone" in fields:
        tz_text = fields["time_zone"].text
        m = re.search(r"UTC\s*[+-]?\d+(?::\d+)?", tz_text)
        standard = m.group(0).replace(" ", "") if m else tz_text
        tz_dict: Dict[str, Any] = {"standard": standard}
        for k in ("summer_dst", "time_zone_dst"):
            if k in fields:
                dst_text = fields[k].text
                m2 = re.search(r"UTC\s*[+-]?\d+(?::\d+)?", dst_text)
                tz_dict["dst"] = m2.group(0).replace(" ", "") if m2 else dst_text
                break
        out["time_zone"] = tz_dict
    if "demonym" in fields:
        out["demonym"] = fields["demonym"].text

    return out
