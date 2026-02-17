from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Boarding Pass Dashboard", layout="wide")


DATA_FILE_DEFAULT = Path("Boarding_Pass.xlsx")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    renamed = {
        "Departure Date": "Departure",
        "Return Date": "Return",
        "Status \nReturn": "Status Return",
        "To and From Station2": "Return Station",
        "PNR Number2": "Return PNR",
        "Return Train": "Return Train Number",
        "Coach2": "Return Coach",
        "Seat2": "Return Seat",
    }
    df = df.rename(columns=renamed)
    return df


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=True)


def _clean_id_like_value(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return pd.NA
    text = text.replace(",", "")
    # Remove trailing ".0" only when it follows a digit token.
    text = re.sub(r"(?<=\d)\.0(?=\D|$)", "", text)
    return text


def _table_image_bytes(df: pd.DataFrame, image_format: str = "png") -> bytes:
    render_df = df.fillna("").astype(str)
    n_rows, n_cols = render_df.shape
    col_char_sizes = [max([len(str(c))] + render_df.iloc[:, i].map(len).tolist()) for i, c in enumerate(render_df.columns)]
    total_chars = max(1, sum(col_char_sizes))
    col_widths = [max(0.04, c / total_chars) for c in col_char_sizes]
    width_scale = sum(col_char_sizes) * 0.16
    fig_w = max(12, min(40, width_scale))
    fig_h = max(3, min(40, (n_rows + 1) * 0.35))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=render_df.values,
        colLabels=render_df.columns,
        loc="center",
        cellLoc="left",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)

    buf = BytesIO()
    save_format = "jpeg" if image_format.lower() in {"jpg", "jpeg"} else "png"
    fig.savefig(buf, format=save_format, bbox_inches="tight", dpi=220)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _drop_empty_rows(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    existing = [c for c in key_cols if c in df.columns]
    if not existing:
        return df
    probe = df[existing].replace(r"^\s*$", pd.NA, regex=True)
    return df[probe.notna().any(axis=1)].copy()


def _table_height(row_count: int) -> int:
    # Keep the table compact to avoid visual blank rows when result size is small.
    return min(520, max(120, 38 + (row_count * 35)))


def _has_text(value: object) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip()
    return bool(text and text.lower() != "nan")


def _clean_age_value(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return pd.NA
    try:
        num = float(text)
        if num.is_integer():
            return str(int(num))
        return str(num)
    except ValueError:
        return text


def _safe_filename_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned.lower() or "guest"


def _jpeg_filename(*parts: object) -> str:
    safe_parts: list[str] = []
    for part in parts:
        if part is None:
            continue
        text = str(part).strip()
        if not text:
            continue
        safe_parts.append(_safe_filename_part(text))
    base = "_".join(safe_parts) or "download"
    return f"{base}.jpeg"


def _room_sort_key(value: object) -> tuple[int, int | float, str]:
    text = str(value).strip()
    lower = text.lower()
    shubh_num_match = re.match(r"^shubh\s*0*(\d+)$", lower)
    if shubh_num_match:
        return (0, int(shubh_num_match.group(1)), lower)
    if re.match(r"^shubh\s*allot$", lower):
        return (0, 10_000, lower)
    if lower.startswith("shubh"):
        return (0, 20_000, lower)
    if text.isdigit():
        return (1, int(text), text)
    return (2, float("inf"), lower)


def _room_group_label(value: object) -> str:
    text = str(value).strip()
    if text == "TRUSTEE OFFICE 1st Floor":
        return "Named Rooms"
    normalized = re.sub(r"[^a-z0-9]+", "", text.lower())
    named_room_markers = [
        "anupama",
        "kumarpal",
        "kumarbhai",
        "mahakali",
        "mayna",
        "shripalmaharaja",
    ]
    if any(marker in normalized for marker in named_room_markers):
        return "Named Rooms"
    if text.isdigit():
        num = int(text)
        if num >= 100:
            return f"{(num // 100) * 100} Range"
        return "1-99 Range"
    alpha_prefix = re.match(r"^[A-Za-z]+", text)
    if alpha_prefix:
        normalized_prefix = alpha_prefix.group(0).capitalize()
        return f"Starting with {normalized_prefix}"
    return "Other"


def _room_group_sort_key(label: str) -> tuple[int, int | float, str]:
    range_match = re.match(r"^(\d+)\s+Range$", label)
    if range_match:
        return (0, int(range_match.group(1)), label)
    if label == "1-99 Range":
        return (0, 1, label)
    if label.startswith("Starting with "):
        return (1, float("inf"), label.lower())
    return (2, float("inf"), label.lower())


def _coach_type(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    if not text or text == "NAN":
        return None
    if "SL" in text:
        return "SL"
    if "AC" in text:
        return "AC"
    if "3E" in text:
        return "AC"
    first = text[0]
    if first == "S":
        return "SL"
    if first in {"A", "B", "M"}:
        return "AC"
    return None


def _coach_bucket(coach_value: object, status_value: object = pd.NA) -> str | None:
    coach_text = "" if pd.isna(coach_value) else str(coach_value).strip().upper()
    status_text = "" if pd.isna(status_value) else str(status_value).strip().upper()
    if not coach_text and not status_text:
        return None
    if coach_text == "NAN":
        coach_text = ""
    if status_text == "NAN":
        status_text = ""
    probe_text = f"{coach_text} {status_text}".strip()
    if not probe_text:
        return None
    has_wl = "WL" in probe_text
    has_rac = "RAC" in probe_text
    base = _coach_type(coach_text if coach_text else probe_text)
    if base == "SL":
        if has_rac:
            return "SL-RAC"
        return "SL-WL" if has_wl else "SL"
    if base == "AC":
        if has_rac:
            return "AC-RAC"
        return "AC-WL" if has_wl else "AC"
    return None


def _format_output_dates_and_age(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Departure" in out.columns:
        out["Departure"] = pd.to_datetime(out["Departure"], errors="coerce").dt.strftime("%d/%m/%Y")
    if "Return" in out.columns:
        out["Return"] = pd.to_datetime(out["Return"], errors="coerce").dt.strftime("%d/%m/%Y")
    out = out.rename(columns={"Departure": "Departure Date", "Return": "Return Date"})
    if "Age" in out.columns:
        out["Age"] = out["Age"].map(_clean_age_value)
    out = out.replace(r"^\s*$", pd.NA, regex=True)
    out = out.replace({"nan": pd.NA, "NaN": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    return out.fillna("NA")


def _to_phone_href(value: object) -> str | None:
    if pd.isna(value):
        return None
    raw = str(value).strip()
    if not raw or raw.lower() == "nan":
        return None
    cleaned = re.sub(r"[^\d+]", "", raw)
    if not cleaned:
        return None
    if cleaned.startswith("+"):
        normalized = "+" + re.sub(r"[^\d]", "", cleaned[1:])
    else:
        normalized = re.sub(r"[^\d]", "", cleaned)
    if not normalized:
        return None
    return f"tel:{normalized}"


@st.cache_data(show_spinner=False)
def load_data(file_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    master = pd.read_excel(file_path, sheet_name="MASTER")
    extra_return = pd.read_excel(file_path, sheet_name="EXTRA RETURN TICKET")

    master = _normalize_columns(master)
    extra_return = _normalize_columns(extra_return)

    for col in ["Departure", "Return"]:
        if col in master.columns:
            master[col] = _to_datetime(master[col])
        if col in extra_return.columns:
            extra_return[col] = _to_datetime(extra_return[col])

    if "Gender" in master.columns:
        master["Gender"] = master["Gender"].astype(str).str.upper().str.strip()

    if "BUS TRAVEL" in master.columns:
        master["Bus Required"] = master["BUS TRAVEL"].astype(str).str.contains(
            "BUS", case=False, na=False
        )
    else:
        master["Bus Required"] = False

    # Excel often loads identifier-like cells as floats (e.g. "12345.0").
    # Normalize these across common ID/contact/seat/group fields.
    id_like_cols = [
        "PNR Number",
        "Train Number",
        "Return PNR",
        "Return Train Number",
        "Mobile",
        "Phone",
        "Phone Number",
        "Group",
        "STAY",
        "Seat",
        "Return Seat",
        "Coach",
        "Return Coach",
    ]
    for col in id_like_cols:
        if col in master.columns:
            master[col] = master[col].map(_clean_id_like_value)
        if col in extra_return.columns:
            extra_return[col] = extra_return[col].map(_clean_id_like_value)

    return master, extra_return


def render_guest_page(df: pd.DataFrame) -> None:
    st.caption("Enter your Name")
    f1 = st.columns(1)[0]
    name_options = sorted(
        [v for v in df["Name"].dropna().astype(str).str.strip().unique().tolist() if v and v.lower() != "nan"]
    )

    with f1:
        name_query = st.selectbox(
            "Name",
            options=name_options,
            index=None,
            placeholder="Type to search name",
        )
    matched = df.copy()
    if not name_query:
        st.info("Select a name to search.")
        st.stop()

    if name_query:
        matched = matched[matched["Name"].astype(str).str.strip() == name_query]
    match_count = len(matched)
    st.caption(f"Matched People: {match_count}")

    if match_count == 0:
        st.warning("No match found for the selected filters.")
        st.stop()
    base_token = name_query if name_query else str(matched.iloc[0].get("Name", "guest"))
    base_filename = f"{_safe_filename_part(str(base_token))}_travel_data"
    table_cols = [
        "Group",
        "Name",
        "NAME ON TICKET-DEPARTURE",
        "Gender",
        "Age",
        "STAY",
        "Departure",
        "Status Departure",
        "Station Name",
        "PNR Number",
        "Train Number",
        "Coach",
        "Seat",
        "Return",
        "NAME ON TICKET-RETURN",
        "Status Return",
        "Return Station",
        "Return PNR",
        "Return Train Number",
        "Return Coach",
        "Return Seat",
        "BUS TRAVEL",
    ]
    table_cols = [c for c in table_cols if c in matched.columns]

    stay_values = [v for v in matched["STAY"].dropna().astype(str).str.strip().unique().tolist() if v]
    if not stay_values:
        st.info("Matched person has no STAY value, so no stay-group records can be shown.")
        st.stop()
    st.caption(f"Stay values found: {', '.join(stay_values)}")
    output_source = df[df["STAY"].astype(str).str.strip().isin(stay_values)].copy()
    output_source["_searched_name_first"] = (
        output_source["Name"].astype(str).str.strip() == str(name_query).strip()
    )
    output_source = output_source.sort_values(by=["STAY", "Group", "Name"], na_position="last")
    output_source = output_source.sort_values(by="_searched_name_first", ascending=False, kind="stable")
    output_source = output_source.drop(columns=["_searched_name_first"])

    output_df = _format_output_dates_and_age(output_source[table_cols].copy())

    departure_cols = [
        "Group",
        "Name",
        "NAME ON TICKET-DEPARTURE",
        "Gender",
        "Age",
        "STAY",
        "Departure Date",
        "Status Departure",
        "Station Name",
        "PNR Number",
        "Train Number",
        "Coach",
        "Seat",
    ]
    return_cols = [
        "Group",
        "Name",
        "NAME ON TICKET-RETURN",
        "Gender",
        "Age",
        "STAY",
        "Return Date",
        "Status Return",
        "Return Station",
        "Return PNR",
        "Return Train Number",
        "Return Coach",
        "Return Seat",
        "BUS TRAVEL",
    ]
    departure_cols = [c for c in departure_cols if c in output_df.columns]
    return_cols = [c for c in return_cols if c in output_df.columns]
    departure_df = output_df[departure_cols].copy()
    return_df = output_df[return_cols].copy()

    st.subheader("Travel Data")

    dep_header_col, dep_btn_col, _ = st.columns([2, 1, 6])
    with dep_header_col:
        st.markdown(
            "<div style='display:inline-block;background:#ffec99;color:#1f2937;border:1px solid #d4a017;"
            "padding:6px 10px;border-radius:6px;font-size:1.1rem;font-weight:700;'>"
            "Departure Table (Towards Gujarat)</div>",
            unsafe_allow_html=True,
        )
    with dep_btn_col:
        dep_image_bytes = _table_image_bytes(departure_df, image_format="jpeg")
        st.download_button(
            "Download Departure (JPEG)",
            data=dep_image_bytes,
            file_name=_jpeg_filename(base_filename, "departure", "towards_gujarat"),
            mime="image/jpeg",
        )
    st.dataframe(
        departure_df,
        use_container_width=True,
        hide_index=True,
        height=_table_height(len(departure_df)),
    )

    ret_header_col, ret_btn_col, _ = st.columns([2, 1, 6])
    with ret_header_col:
        st.markdown(
            "<div style='display:inline-block;background:#ffec99;color:#1f2937;border:1px solid #d4a017;"
            "padding:6px 10px;border-radius:6px;font-size:1.1rem;font-weight:700;'>"
            "Return Table (Towards Mumbai)</div>",
            unsafe_allow_html=True,
        )
    with ret_btn_col:
        ret_image_bytes = _table_image_bytes(return_df, image_format="jpeg")
        st.download_button(
            "Download Return (JPEG)",
            data=ret_image_bytes,
            file_name=_jpeg_filename(base_filename, "return", "towards_mumbai"),
            mime="image/jpeg",
        )
    st.dataframe(
        return_df,
        use_container_width=True,
        hide_index=True,
        height=_table_height(len(return_df)),
    )


def render_admin_page(df: pd.DataFrame) -> None:
    def journey_columns(journey_type: str) -> tuple[str, str, str, str, str, str, str, str]:
        if journey_type == "Departure":
            return (
                "Departure",
                "Train Number",
                "Status Departure",
                "Station Name",
                "PNR Number",
                "Coach",
                "Seat",
                "Departure Date",
            )
        return (
            "Return",
            "Return Train Number",
            "Status Return",
            "Return Station",
            "Return PNR",
            "Return Coach",
            "Return Seat",
            "Return Date",
        )

    def render_admin_result_block(
        filtered: pd.DataFrame,
        date_col: str,
        train_col: str,
        status_col: str,
        station_col: str,
        pnr_col: str,
        coach_col: str,
        seat_col: str,
        result_title: str,
        include_bus_travel: bool,
        show_coach_counts: bool = True,
        key_prefix: str = "admin",
        download_name_parts: tuple[object, ...] | None = None,
    ) -> None:
        filtered = filtered.sort_values(by=["Group", "Name"], na_position="last")
        filtered_view = filtered
        st.markdown(f"**{result_title}**")
        if show_coach_counts:
            coach_bucket_series = filtered.apply(
                lambda row: _coach_bucket(row.get(coach_col), row.get(status_col)),
                axis=1,
            )
            coach_counts = (
                coach_bucket_series
                .dropna()
                .value_counts()
                .reindex(["SL", "SL-WL", "SL-RAC", "AC", "AC-WL", "AC-RAC"], fill_value=0)
            )
            selected_bucket_key = f"{key_prefix}_coach_bucket"
            if selected_bucket_key not in st.session_state:
                st.session_state[selected_bucket_key] = "ALL"

            bc0, bc1, bc2, bc3, bc4, bc5, bc6 = st.columns(7)
            if bc0.button("All", key=f"{key_prefix}_bucket_all"):
                st.session_state[selected_bucket_key] = "ALL"
            if bc1.button(f"SL ({int(coach_counts.get('SL', 0))})", key=f"{key_prefix}_bucket_sl"):
                st.session_state[selected_bucket_key] = "SL"
            if bc2.button(f"SL-WL ({int(coach_counts.get('SL-WL', 0))})", key=f"{key_prefix}_bucket_sl_wl"):
                st.session_state[selected_bucket_key] = "SL-WL"
            if bc3.button(f"SL-RAC ({int(coach_counts.get('SL-RAC', 0))})", key=f"{key_prefix}_bucket_sl_rac"):
                st.session_state[selected_bucket_key] = "SL-RAC"
            if bc4.button(f"AC ({int(coach_counts.get('AC', 0))})", key=f"{key_prefix}_bucket_ac"):
                st.session_state[selected_bucket_key] = "AC"
            if bc5.button(f"AC-WL ({int(coach_counts.get('AC-WL', 0))})", key=f"{key_prefix}_bucket_ac_wl"):
                st.session_state[selected_bucket_key] = "AC-WL"
            if bc6.button(f"AC-RAC ({int(coach_counts.get('AC-RAC', 0))})", key=f"{key_prefix}_bucket_ac_rac"):
                st.session_state[selected_bucket_key] = "AC-RAC"

            selected_bucket = st.session_state[selected_bucket_key]
            st.caption(f"Selected coach filter: {selected_bucket}")
            if selected_bucket != "ALL":
                filtered_view = filtered[
                    filtered.apply(
                        lambda row: _coach_bucket(row.get(coach_col), row.get(status_col)) == selected_bucket,
                        axis=1,
                    )
                ].copy()

        output_cols = [
            "Group",
            "Name",
            "Gender",
            "Age",
            "STAY",
            date_col,
            status_col,
            station_col,
            pnr_col,
            train_col,
            coach_col,
            seat_col,
        ]
        if date_col == "Departure":
            output_cols.insert(2, "NAME ON TICKET-DEPARTURE")
        if date_col == "Return":
            output_cols.insert(2, "NAME ON TICKET-RETURN")
        if include_bus_travel:
            output_cols.append("BUS TRAVEL")
        output_cols = [c for c in output_cols if c in filtered_view.columns]
        output_df = _format_output_dates_and_age(filtered_view[output_cols].copy())

        header_col, dl_col, _ = st.columns([2, 1, 6])
        with header_col:
            st.caption(f"Passengers found: {len(output_df)}")
        with dl_col:
            admin_image_bytes = _table_image_bytes(output_df, image_format="jpeg")
            file_name = (
                _jpeg_filename(*download_name_parts)
                if download_name_parts
                else _jpeg_filename(result_title)
            )
            st.download_button(
                "Download (JPEG)",
                data=admin_image_bytes,
                file_name=file_name,
                mime="image/jpeg",
                key=f"{key_prefix}_download_jpeg",
            )
        st.dataframe(
            output_df,
            use_container_width=True,
            hide_index=True,
            height=_table_height(len(output_df)),
        )

    st.subheader("1. Filter by Date and Train")
    d1, d2, d3 = st.columns(3)
    with d1:
        j1 = st.selectbox("Journey Type", options=["Departure", "Return"], index=0, key="admin_journey_1")
    date_col, train_col, status_col, station_col, pnr_col, coach_col, seat_col, date_label = journey_columns(j1)

    valid_journey_rows = df[
        df[train_col].map(_has_text)
        | df[pnr_col].map(_has_text)
        | df[coach_col].map(_has_text)
        | df[seat_col].map(_has_text)
    ].copy()
    date_values = sorted(valid_journey_rows[date_col].dropna().dt.date.unique().tolist())
    with d2:
        selected_date = st.selectbox(
            "Date",
            options=date_values,
            index=None,
            placeholder=f"Select {date_label.lower()}",
            format_func=lambda d: d.strftime("%d/%m/%Y"),
            key="admin_date_1",
        )

    selected_train = None
    if selected_date:
        date_filtered = valid_journey_rows[valid_journey_rows[date_col].dt.date == selected_date].copy()
        train_options = sorted(
            [
                v
                for v in date_filtered[train_col].dropna().astype(str).str.strip().unique().tolist()
                if v and v.lower() != "nan"
            ]
        )
    else:
        date_filtered = pd.DataFrame(columns=df.columns)
        train_options = []

    with d3:
        selected_train = st.selectbox(
            "Train Number",
            options=train_options,
            index=None,
            placeholder="Select train number",
            key="admin_train_1",
        )

    if selected_date and selected_train:
        filtered_dt = date_filtered[date_filtered[train_col].astype(str).str.strip() == selected_train].copy()
        render_admin_result_block(
            filtered_dt,
            date_col,
            train_col,
            status_col,
            station_col,
            pnr_col,
            coach_col,
            seat_col,
            "Train Passenger List",
            include_bus_travel=(j1 == "Return"),
            show_coach_counts=True,
            key_prefix="admin_dt",
            download_name_parts=("admin", "date_train", j1, selected_date, selected_train),
        )
    else:
        st.info("Select journey type, date, and train number to view passengers.")

    st.divider()
    st.subheader("2. Search by Name (Stay Group)")
    admin_name_options5 = sorted(
        [v for v in df["Name"].dropna().astype(str).str.strip().unique().tolist() if v and v.lower() != "nan"]
    )
    selected_name5 = st.selectbox(
        "Name",
        options=admin_name_options5,
        index=None,
        placeholder="Type to search name",
        key="admin_name_5",
    )
    if not selected_name5:
        st.info("Select a name to view stay-group passengers.")
    else:
        matched5 = df[df["Name"].astype(str).str.strip() == selected_name5].copy()
        stay_values5 = [v for v in matched5["STAY"].dropna().astype(str).str.strip().unique().tolist() if v]
        if not stay_values5:
            st.info("Selected person has no STAY value, so no stay-group records can be shown.")
        else:
            output_source5 = df[df["STAY"].astype(str).str.strip().isin(stay_values5)].copy()
            output_source5["_searched_name_first"] = (
                output_source5["Name"].astype(str).str.strip() == str(selected_name5).strip()
            )
            output_source5 = output_source5.sort_values(by=["STAY", "Group", "Name"], na_position="last")
            output_source5 = output_source5.sort_values(by="_searched_name_first", ascending=False, kind="stable")
            output_source5 = output_source5.drop(columns=["_searched_name_first"])

            phone_col5 = None
            for candidate_col in ["Mobile", "Phone", "Phone Number"]:
                if candidate_col in output_source5.columns:
                    phone_col5 = candidate_col
                    break

            table_cols5 = [
                "Group",
                "Name",
                "Gender",
                "Age",
                "STAY",
            ]
            departure_cols5 = [
                *table_cols5,
                "NAME ON TICKET-DEPARTURE",
                "Departure",
                "Status Departure",
                "Station Name",
                "PNR Number",
                "Train Number",
                "Coach",
                "Seat",
            ]
            return_cols5 = [
                *table_cols5,
                "NAME ON TICKET-RETURN",
                "Return",
                "Status Return",
                "Return Station",
                "Return PNR",
                "Return Train Number",
                "Return Coach",
                "Return Seat",
                "BUS TRAVEL",
            ]
            if phone_col5:
                departure_cols5.insert(2, phone_col5)
                return_cols5.insert(2, phone_col5)
            departure_cols5 = [c for c in departure_cols5 if c in output_source5.columns]
            return_cols5 = [c for c in return_cols5 if c in output_source5.columns]

            departure_df5 = _format_output_dates_and_age(output_source5[departure_cols5].copy())
            return_df5 = _format_output_dates_and_age(output_source5[return_cols5].copy())
            if phone_col5 and phone_col5 in departure_df5.columns:
                departure_df5[phone_col5] = departure_df5[phone_col5].map(
                    lambda value: _to_phone_href(value) if str(value).strip().upper() != "NA" else None
                )
            if phone_col5 and phone_col5 in return_df5.columns:
                return_df5[phone_col5] = return_df5[phone_col5].map(
                    lambda value: _to_phone_href(value) if str(value).strip().upper() != "NA" else None
                )

            st.caption(f"Stay values found: {', '.join(stay_values5)}")
            st.caption(f"Passengers found: {len(departure_df5)}")
            column_config5 = {}
            if phone_col5 and phone_col5 in departure_df5.columns:
                column_config5[phone_col5] = st.column_config.LinkColumn(
                    phone_col5,
                    help="Tap to call",
                    validate=r"^tel:\+?\d+$",
                    display_text=r"tel:(.*)",
                )

            st.markdown("**Departure Details**")
            st.dataframe(
                departure_df5,
                use_container_width=True,
                hide_index=True,
                height=_table_height(len(departure_df5)),
                column_config=column_config5 if column_config5 else None,
            )
            st.markdown("**Return Details**")
            st.dataframe(
                return_df5,
                use_container_width=True,
                hide_index=True,
                height=_table_height(len(return_df5)),
                column_config=column_config5 if column_config5 else None,
            )

    st.divider()
    st.subheader("3. Filter by PNR")
    dep_pnr_options = [
        v for v in df["PNR Number"].dropna().astype(str).str.strip().unique().tolist() if v and v.lower() != "nan"
    ]
    ret_pnr_options = [
        v
        for v in df["Return PNR"].dropna().astype(str).str.strip().unique().tolist()
        if v and v.lower() != "nan"
    ]
    pnr_options2 = sorted(set(dep_pnr_options) | set(ret_pnr_options))
    selected_pnr2 = st.selectbox(
        "PNR",
        options=pnr_options2,
        index=None,
        placeholder="Select PNR number",
        key="admin_pnr_2",
    )

    if selected_pnr2:
        dep_filtered = df[df["PNR Number"].astype(str).str.strip() == selected_pnr2].copy()
        ret_filtered = df[df["Return PNR"].astype(str).str.strip() == selected_pnr2].copy()
        has_dep = not dep_filtered.empty
        has_ret = not ret_filtered.empty

        if has_dep:
            date_col2, train_col2, status_col2, station_col2, pnr_col2, coach_col2, seat_col2, _ = journey_columns(
                "Departure"
            )
            title = "PNR Passenger List (Departure)"
            if has_ret:
                st.caption("PNR found in both departure and return data.")
            render_admin_result_block(
                dep_filtered,
                date_col2,
                train_col2,
                status_col2,
                station_col2,
                pnr_col2,
                coach_col2,
                seat_col2,
                title,
                include_bus_travel=False,
                show_coach_counts=False,
                key_prefix="admin_pnr_dep",
                download_name_parts=("admin", "pnr", selected_pnr2, "departure"),
            )

        if has_ret:
            date_col2, train_col2, status_col2, station_col2, pnr_col2, coach_col2, seat_col2, _ = journey_columns(
                "Return"
            )
            render_admin_result_block(
                ret_filtered,
                date_col2,
                train_col2,
                status_col2,
                station_col2,
                pnr_col2,
                coach_col2,
                seat_col2,
                "PNR Passenger List (Return)",
                include_bus_travel=True,
                show_coach_counts=False,
                key_prefix="admin_pnr_ret",
                download_name_parts=("admin", "pnr", selected_pnr2, "return"),
            )

        if not has_dep and not has_ret:
            st.warning("PNR not found in departure or return data.")
    else:
        st.info("Select a PNR to view passengers.")

    st.divider()
    st.subheader("4. Filter by Bus Date")
    if "BUS TRAVEL" not in df.columns:
        st.info("Bus travel column not available in data.")
    else:
        bus_rows = df[df["BUS TRAVEL"].astype(str).str.contains("BUS", case=False, na=False)].copy()
        bus_rows = bus_rows[bus_rows["Return"].notna()].copy()
        if bus_rows.empty:
            st.info("No bus travel records with return date found.")
        else:
            bus_dates = sorted(bus_rows["Return"].dt.date.unique().tolist())
            selected_bus_date = st.selectbox(
                "Bus Travel Date (Return Date)",
                options=bus_dates,
                index=None,
                placeholder="Select bus travel date",
                format_func=lambda d: d.strftime("%d/%m/%Y"),
                key="admin_bus_date_3",
            )
            if not selected_bus_date:
                st.info("Select a bus travel date to view passengers.")
            else:
                bus_filtered = bus_rows[bus_rows["Return"].dt.date == selected_bus_date].copy()
                date_col3, train_col3, status_col3, station_col3, pnr_col3, coach_col3, seat_col3, _ = journey_columns(
                    "Return"
                )
                render_admin_result_block(
                    bus_filtered,
                    date_col3,
                    train_col3,
                    status_col3,
                    station_col3,
                    pnr_col3,
                    coach_col3,
                    seat_col3,
                    "Bus Passenger List",
                    include_bus_travel=True,
                    show_coach_counts=False,
                    key_prefix="admin_bus",
                    download_name_parts=("admin", "bus", selected_bus_date, "return"),
                )

    st.divider()
    st.subheader("5. Filter by Room Number")
    if "STAY" not in df.columns:
        st.info("Room column (STAY) not available in data.")
    else:
        room_options = sorted(
            [
                v
                for v in df["STAY"].dropna().astype(str).str.strip().unique().tolist()
                if v and v.lower() not in {"nan", "cancel"}
            ],
            key=_room_sort_key,
        )
        room_groups: dict[str, list[str]] = {}
        for room_value in room_options:
            group_label = _room_group_label(room_value)
            room_groups.setdefault(group_label, []).append(room_value)

        room_group_options = sorted(room_groups.keys(), key=_room_group_sort_key)
        selected_group_key = "admin_room_group_5_selected"
        selected_room_key = "admin_room_5_selected"
        if selected_group_key not in st.session_state:
            st.session_state[selected_group_key] = None
        if selected_room_key not in st.session_state:
            st.session_state[selected_room_key] = None

        st.markdown("**Room Category**")
        per_row_group = 5
        for row_start in range(0, len(room_group_options), per_row_group):
            row_groups = room_group_options[row_start: row_start + per_row_group]
            group_cols = st.columns(len(row_groups))
            for idx, group_value in enumerate(row_groups):
                with group_cols[idx]:
                    button_text = group_value
                    if st.session_state[selected_group_key] == group_value:
                        button_text = f"[{group_value}]"
                    if st.button(
                        button_text,
                        key=f"admin_room_group_btn_{row_start + idx}",
                        use_container_width=True,
                    ):
                        st.session_state[selected_group_key] = group_value
                        st.session_state[selected_room_key] = None

        selected_room_group = st.session_state[selected_group_key]
        if selected_room_group:
            st.markdown(f"**Room Options in {selected_room_group}**")
            room_values = room_groups[selected_room_group]
            per_row_room = 12
            for row_start in range(0, len(room_values), per_row_room):
                row_rooms = room_values[row_start: row_start + per_row_room]
                room_cols = st.columns(len(row_rooms))
                for idx, room_value in enumerate(row_rooms):
                    with room_cols[idx]:
                        button_text = room_value
                        if st.session_state[selected_room_key] == room_value:
                            button_text = f"[{room_value}]"
                        if st.button(
                            button_text,
                            key=f"admin_room_btn_{selected_room_group}_{row_start + idx}",
                            use_container_width=True,
                        ):
                            st.session_state[selected_room_key] = room_value
            selected_room = st.session_state[selected_room_key]
            st.caption(f"Selected room: {selected_room}" if selected_room else "No room selected")
        else:
            selected_room = None
            st.info("Select a room category to view room options.")

        if selected_room:
            room_filtered = df[df["STAY"].astype(str).str.strip() == selected_room].copy()
            dep_date_col, dep_train_col, dep_status_col, dep_station_col, dep_pnr_col, dep_coach_col, dep_seat_col, _ = (
                journey_columns("Departure")
            )
            ret_date_col, ret_train_col, ret_status_col, ret_station_col, ret_pnr_col, ret_coach_col, ret_seat_col, _ = (
                journey_columns("Return")
            )

            render_admin_result_block(
                room_filtered,
                dep_date_col,
                dep_train_col,
                dep_status_col,
                dep_station_col,
                dep_pnr_col,
                dep_coach_col,
                dep_seat_col,
                f"Room {selected_room} Passenger List (Departure)",
                include_bus_travel=False,
                show_coach_counts=False,
                key_prefix="admin_room_dep",
                download_name_parts=("admin", "room", selected_room, "departure"),
            )

            render_admin_result_block(
                room_filtered,
                ret_date_col,
                ret_train_col,
                ret_status_col,
                ret_station_col,
                ret_pnr_col,
                ret_coach_col,
                ret_seat_col,
                f"Room {selected_room} Passenger List (Return)",
                include_bus_travel=True,
                show_coach_counts=False,
                key_prefix="admin_room_ret",
                download_name_parts=("admin", "room", selected_room, "return"),
            )
        else:
            st.info("Select a room number to view passengers.")

    st.divider()
    st.subheader("6. Filter by Group")
    group_options4 = sorted(
        [v for v in df["Group"].dropna().astype(str).str.strip().unique().tolist() if v and v.lower() != "nan"]
    )
    selected_group4 = st.selectbox(
        "Group",
        options=group_options4,
        index=None,
        placeholder="Select group",
        key="admin_group_4",
    )

    if selected_group4:
        group_filtered = df[df["Group"].astype(str).str.strip() == selected_group4].copy()
        dep_date_col, dep_train_col, dep_status_col, dep_station_col, dep_pnr_col, dep_coach_col, dep_seat_col, _ = (
            journey_columns("Departure")
        )
        ret_date_col, ret_train_col, ret_status_col, ret_station_col, ret_pnr_col, ret_coach_col, ret_seat_col, _ = (
            journey_columns("Return")
        )

        render_admin_result_block(
            group_filtered,
            dep_date_col,
            dep_train_col,
            dep_status_col,
            dep_station_col,
            dep_pnr_col,
            dep_coach_col,
            dep_seat_col,
            "Group Passenger List (Departure)",
            include_bus_travel=False,
            show_coach_counts=False,
            key_prefix="admin_group_dep",
            download_name_parts=("admin", "group", selected_group4, "departure"),
        )

        render_admin_result_block(
            group_filtered,
            ret_date_col,
            ret_train_col,
            ret_status_col,
            ret_station_col,
            ret_pnr_col,
            ret_coach_col,
            ret_seat_col,
            "Group Passenger List (Return)",
            include_bus_travel=True,
            show_coach_counts=False,
            key_prefix="admin_group_ret",
            download_name_parts=("admin", "group", selected_group4, "return"),
        )
    else:
        st.info("Select a group to view passengers.")

def main() -> None:
    st.title("Shailiben Diksha Mahotsav Guest Travel Dashboard")
    st.markdown(
        """પ્રણામ,

🌸 શૈલીબેન દીક્ષા મહોત્સવ – રૂમ અને ટિકિટ એલોટમેન્ટ માહિતી 🌸

શૈલીબેન દીક્ષા મહોત્સવ પ્રસંગે પધારનાર તમામ શ્રાવક-શ્રાવિકાઓને વિનંતી છે કે રૂમ એલોટમેન્ટ અને પ્રવાસની વિગતો નીચે ચેક કરે.

આપનું નામ દાખલ કરીને સર્ચ કરશો. નામ દાખલ કર્યા પછી આપના રૂમના તમામ સભ્યોની પ્રવાસ તેમજ રૂમ સંબંધિત સંપૂર્ણ વિગતો જોઈ શકાશે.

જો કોઈ વ્યક્તિ આવવાના હોવા છતાં યાદીમાં નામ દેખાતું ન હોય, અથવા દર્શાવેલ વિગતો ખોટી / અધૂરી / સ્પષ્ટ ન દેખાતી હોય, તો નીચે આપેલ નંબર પર સંપર્ક કરશો.

Jitu bhai: <a href="tel:9821133347">9821133347</a><br>
Malay: <a href="tel:9969521053">9969521053</a><br>
Chirag: <a href="tel:9604980800">9604980800</a>"""
        ,
        unsafe_allow_html=True,
    )
    if not DATA_FILE_DEFAULT.exists():
        st.error("Boarding_Pass.xlsx not found in current folder.")
        st.stop()

    try:
        df, _ = load_data(str(DATA_FILE_DEFAULT))
    except Exception as exc:
        st.exception(exc)
        st.stop()

    page = st.radio("Page", options=["Guest", "Admin"], horizontal=True)
    if page == "Guest":
        render_guest_page(df)
        return

    admin_passcode = st.text_input("Admin Access Code", type="password")
    expected_passcode = os.environ.get("ADMIN_DASHBOARD_PASSCODE", "Shaili*26")
    if admin_passcode != expected_passcode:
        st.warning("Admin access required.")
        st.stop()
    render_admin_page(df)


if __name__ == "__main__":
    main()
