import streamlit as st
import re
import sys
import os
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, List, Optional

st.set_page_config(layout="wide")

st.title("TG Sales App")

# Special handling for PyInstaller
if getattr(sys, 'frozen', False):
    # When running as executable
    BASE_DIR = os.path.dirname(sys.executable)
else:
    # When running as script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Pre-compiled Regex Patterns ---
SALES_PERCENT_PATTERN = re.compile(r"(?:📈-W:|W:).*?([-+]?\d+\.\d{2})%\s*sales")
TOTAL_SALES_PATTERN = re.compile(r"TOTAL SALES:\s+([\d,\.]+)\s+SR")
WEEKLY_COMP_PATTERN = re.compile(r"Sales\s+([-+]?\d+\.\d+)%")
OVERALL_SALES_PATTERN = re.compile(r"(\d+\.\d+)%\s*sales")
CHANNEL_SPLIT_PATTERN = re.compile(r"-{35}|={35}")

st.title("📊 DFC Sales Summary Generator")

# --- Session init for presets (per report) ---
for i in range(1, 4):
    preset_key = f"preset_choice_{i}"
    if preset_key not in st.session_state:
        st.session_state[preset_key] = "All channels (default)"

def make_update_preset_from_title(report_id: int):
    """
    Creates a callback that auto-sets the preset for a specific report
    based on the report title (Maestro / New Brands / Pinzatta / MAD).
    """
    def _cb():
        title_key = f"report_title_{report_id}"
        preset_key = f"preset_choice_{report_id}"
        title_val = st.session_state.get(title_key, "")
        lower = title_val.lower()

        preset = "All channels (default)"
        if "maestro" in lower:
            preset = "Maestro Channels"
        elif (
            "new brands" in lower
            or "new brand" in lower
            or "pinzatta" in lower
            or "mad" in lower
        ):
            preset = "New Brands Channels"

        st.session_state[preset_key] = preset
    return _cb

# --- Helper Functions ---
@lru_cache(maxsize=32)
def format_sales_input(value: str, is_lw_sales: bool = False) -> str:
    try:
        num = float(value)
        
        if num >= 999_500:
            value = round(num/1_000_000, 2)
            formatted = f"{value:.2f}"
            return f"{formatted[:-1] if formatted.endswith('0') else formatted} Mn"
        elif num >= 100_000:
            return f"{int(round(num/1_000))}K"
        return f"{round(num/1_000, 1):g}K"
    except:
        return "-"

@st.cache_data
def extract_channel_blocks(text: str) -> List[Tuple[str, str]]:
    clean_text = re.sub(r'^.*?Sales for All branches are:\s*', '', text, flags=re.DOTALL)
    sections = CHANNEL_SPLIT_PATTERN.split(clean_text)
    
    channel_data = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        first_line = section.split('\n')[0]
        if ':' in first_line and 'orders' in first_line.lower():
            channel_name = first_line.split(':')[0].strip()
            channel_data.append((channel_name, section))
    return channel_data

@lru_cache(maxsize=128)
def extract_weekly_sales_delta(block: str) -> float:
    clean_text = re.sub(r'[^\x00-\x7F]+', '', block)
    match = SALES_PERCENT_PATTERN.search(clean_text)
    return float(match.group(1).strip()) if match else 0

@lru_cache(maxsize=128)
def extract_overall_sales_percent(block: str) -> float:
    match = OVERALL_SALES_PATTERN.search(block)
    return float(match.group(1)) if match else 0

@lru_cache(maxsize=32)
def extract_total_sales(text: str) -> Optional[float]:
    match = TOTAL_SALES_PATTERN.search(text)
    return float(match.group(1).replace(",", "")) if match else None

@lru_cache(maxsize=32)
def extract_weekly_comparison(text: str) -> float:
    match = WEEKLY_COMP_PATTERN.search(text)
    return float(match.group(1)) if match else 0

@lru_cache(maxsize=128)
def get_indicator(delta: float) -> str:
    return "🟢" if delta > 0 else "🔴" if delta <= -30 else "🟡"

def process_channel(args: Tuple[str, str]) -> Tuple[str, float, float]:
    name, block = args
    weekly_delta = extract_weekly_sales_delta(block)
    percent_of_overall = extract_overall_sales_percent(block)
    return name, weekly_delta, percent_of_overall

# --- One reusable "report" block ---
def render_report(report_id: int):
    st.markdown(f"### Report {report_id}")

    title_key = f"report_title_{report_id}"
    lw_key = f"lw_sales_{report_id}"
    raw_key = f"raw_input_{report_id}"
    preset_key = f"preset_choice_{report_id}"

    # Title input with per-report callback
    title = st.text_input(
        "Report Title (e.g., Maestro, MAD, Pinzatta)",
        "",
        key=title_key,
        on_change=make_update_preset_from_title(report_id),
    )

    lw_sales_input = st.text_input(
        "Last Week Sales (enter plain number)",
        "",
        key=lw_key,
    )
    raw_input = st.text_area(
        "Paste the raw sales report text below:",
        key=raw_key,
    )

    if not raw_input or not lw_sales_input:
        st.info("Enter Last Week Sales and paste the raw report above to generate this report.")
        return

    formatted_lw = format_sales_input(lw_sales_input, is_lw_sales=True)
    st.caption(f"Formatted as: {formatted_lw}")

    blocks = extract_channel_blocks(raw_input)
    available_channels = []
    original_to_display = {}

    for name, _ in blocks:
        display_name = name.replace("HungerStation", "Hunger")
        available_channels.append(display_name)
        original_to_display[name] = display_name

    display_to_original = {v: k for k, v in original_to_display.items()}

    # --- Presets section (per report) ---
    preset_choice = st.selectbox(
        "Optional: choose a channel preset",
        [
            "All channels (default)",
            "Maestro Channels",
            "New Brands Channels",
            "Custom (start with all, then edit)"
        ],
        key=preset_key,
    )

    # Decide what the multiselect should start with
    if preset_choice == "Maestro Channels":
        maestro_list = ["App", "Web", "Delivery", "Hunger", "Jahez", "Keeta"]
        default_selection = [ch for ch in maestro_list if ch in available_channels]

    elif preset_choice == "New Brands Channels":
        new_brands_list = ["Hunger", "Jahez", "Keeta", "The Chefz", "ToYou"]
        default_selection = [ch for ch in new_brands_list if ch in available_channels]

    elif preset_choice == "All channels (default)":
        default_selection = available_channels  # ordered as they appear in file

    else:  # Custom
        default_selection = available_channels

    # You can STILL change/add/remove after choosing a preset 👇
    selected_channels = st.multiselect(
        "Select channels and their display order",
        available_channels,
        default=default_selection,
        key=f"selected_channels_{report_id}",
    )

    # Core calculations
    lw_sales = float(lw_sales_input) if lw_sales_input.replace('.', '').isdigit() else None
    total_sales = extract_total_sales(raw_input)
    weekly_comparison = extract_weekly_comparison(raw_input)

    if lw_sales and total_sales:
        delta = weekly_comparison
        overall_indicator = get_indicator(delta)
        est_sales = lw_sales * (1 + delta/100)

        st.markdown(f"[{title}]\n")
        st.markdown(
            f"Overall Sales {format_sales_input(total_sales)} SR | "
            f"{abs(delta):.0f}% {overall_indicator}"
        )
        st.markdown(
            f"Est Sales: {format_sales_input(est_sales)}, "
            f"LW Sales: {formatted_lw}\n"
        )

        # Parallel processing for channels
        with ThreadPoolExecutor() as executor:
            processed_channels = list(executor.map(
                process_channel,
                [(name, block) for name, block in blocks if name in display_to_original.values()]
            ))

        breakdown = []
        perc_summary = []
        significant_declines = []

        for display_ch in selected_channels:
            original_ch = display_to_original.get(display_ch)
            if not original_ch:
                continue
                
            for name, weekly_delta, percent_of_overall in processed_channels:
                if name == original_ch:
                    indicator = get_indicator(weekly_delta)
                    breakdown.append(f"{display_ch} {abs(weekly_delta):.0f}% {indicator}")
                    perc_summary.append(f"{display_ch} {percent_of_overall:.0f}%")
                    
                    if weekly_delta <= -30:
                        significant_declines.append(display_ch)
                    break

        # Build one clean text block with correct spacing
        main_lines = []

        # 1) Breakdown list
        main_lines.extend(breakdown)

        # One blank line
        main_lines.append("")

        # 2) Decline sentence (if exists)
        if significant_declines:
            decline_text = ", ".join(significant_declines[:-1])
            if len(significant_declines) > 1:
                decline_text += f" and {significant_declines[-1]}"
            else:
                decline_text = significant_declines[0]

            # Make bold for WhatsApp
            main_lines.append(f"*{decline_text} witnessed a significant decline*")

            # One blank line after sentence
            main_lines.append("")

        # 3) % of overall
        main_lines.append(f"% of overall - {', '.join(perc_summary)}")

        # Output as ONE text block
        st.text("\n".join(main_lines))
    else:
        st.error("❗ Invalid sales input or missing 'TOTAL SALES' in raw data")

# --- Render 3 independent reports SIDE BY SIDE ---
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📄 Report 1")
    render_report(1)

with col2:
    st.markdown("### 📄 Report 2")
    render_report(2)

with col3:
    st.markdown("### 📄 Report 3")
    render_report(3)
    
