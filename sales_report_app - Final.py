import streamlit as st
import re
import sys
import os
import threading
import time
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, List, Dict, Optional

st.title("TG Sales App")

# Special handling for PyInstaller
if getattr(sys, 'frozen', False):
    # When running as executable
    BASE_DIR = os.path.dirname(sys.executable)
else:
    # When running as script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Use os.path.join(BASE_DIR, ...) for all file operatio

# --- Pre-compiled Regex Patterns ---
SALES_PERCENT_PATTERN = re.compile(r"(?:📈-W:|W:).*?([-+]?\d+\.\d{2})%\s*sales")
TOTAL_SALES_PATTERN = re.compile(r"TOTAL SALES:\s+([\d,\.]+)\s+SR")
WEEKLY_COMP_PATTERN = re.compile(r"Sales\s+([-+]?\d+\.\d+)%")
OVERALL_SALES_PATTERN = re.compile(r"(\d+\.\d+)%\s*sales")
CHANNEL_SPLIT_PATTERN = re.compile(r"-{35}|={35}")

st.title("📊 Maestro Sales Summary Generator")

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

# --- Main Logic ---
title = st.text_input("Report Title (e.g., Maestro)", "")
lw_sales_input = st.text_input("Last Week Sales (enter plain number)", "")
raw_input = st.text_area("Paste the raw sales report text below:")

if not raw_input or not lw_sales_input:
    st.error("Missing required inputs")
    st.stop()

formatted_lw = format_sales_input(lw_sales_input, is_lw_sales=True)
st.caption(f"Formatted as: {formatted_lw}")

blocks = extract_channel_blocks(raw_input)
available_channels = []
original_to_display = {}

for name, _ in blocks:
    display_name = name.replace("HungerStation", "Hunger")
    available_channels.append(display_name)
    original_to_display[name] = display_name

display_to_original = {v:k for k,v in original_to_display.items()}
selected_channels = st.multiselect(
    "Select channels and their display order",
    available_channels,
    default=available_channels
)

lw_sales = float(lw_sales_input) if lw_sales_input.replace('.', '').isdigit() else None
total_sales = extract_total_sales(raw_input)
weekly_comparison = extract_weekly_comparison(raw_input)

if lw_sales and total_sales:
    delta = weekly_comparison
    overall_indicator = get_indicator(delta)
    est_sales = lw_sales * (1 + delta/100)

    st.markdown(f"[{title}]\n")
    st.markdown(f"Overall Sales {format_sales_input(total_sales)} SR | {abs(delta):.0f}% {overall_indicator}")
    st.markdown(f"Est Sales: {format_sales_input(est_sales)}, LW Sales: {formatted_lw}\n")

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

        main_lines.append(f"{decline_text} witnessed a significant decline")

        # One blank line after sentence
        main_lines.append("")

    # 3) % of overall
    main_lines.append(f"% of overall - {', '.join(perc_summary)}")

    # Output as ONE text block
    st.text("\n".join(main_lines))
else:

    st.error("❗ Invalid sales input or missing 'TOTAL SALES' in raw data")
