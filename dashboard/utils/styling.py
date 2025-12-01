"""
Styling utilities for the Agent Monitoring Dashboard.

This module provides functions for applying custom CSS, loading indicators,
and visual enhancements to the dashboard.
"""

import streamlit as st
import os
from pathlib import Path
from typing import Optional


def load_custom_css():
    """
    Load custom CSS file for enhanced styling.
    
    This function loads the custom.css file and injects it into the Streamlit app
    for improved visual appearance and consistent styling.
    """
    # Get path to CSS file
    css_file = Path(__file__).parent.parent / "assets" / "custom.css"
    
    if css_file.exists():
        with open(css_file, "r") as f:
            css_content = f.read()
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    else:
        st.warning("Custom CSS file not found. Using default styling.")


def apply_custom_theme():
    """
    Apply custom theme colors and styling.
    
    This function applies additional inline CSS for theme customization
    beyond what's in the CSS file.
    """
    theme_css = """
    <style>
        /* Additional theme customizations */
        :root {
            --primary-color: #FF4B4B;
            --secondary-color: #F0F2F6;
            --text-color: #262730;
            --success-color: #28A745;
            --warning-color: #FFC107;
            --danger-color: #DC3545;
            --info-color: #17A2B8;
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: #F0F2F6;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #C0C0C0;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #A0A0A0;
        }
    </style>
    """
    st.markdown(theme_css, unsafe_allow_html=True)


def render_loading_spinner(text: str = "Loading..."):
    """
    Render a custom loading spinner with text.
    
    Args:
        text: Loading message to display
    """
    spinner_html = f"""
    <div style="
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 2rem;
    ">
        <div style="
            border: 4px solid #F0F2F6;
            border-top: 4px solid #FF4B4B;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
        "></div>
        <p style="
            margin-top: 1rem;
            color: #6C757D;
            font-weight: 500;
        ">{text}</p>
    </div>
    <style>
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
    </style>
    """
    st.markdown(spinner_html, unsafe_allow_html=True)


def render_status_badge(status: str, text: Optional[str] = None) -> str:
    """
    Render a status badge with appropriate styling.
    
    Args:
        status: Status type (executing, completed, failed, idle)
        text: Optional custom text (defaults to status)
        
    Returns:
        HTML string for the status badge
    """
    display_text = text or status.upper()
    
    colors = {
        "executing": {"bg": "#FFF3CD", "text": "#856404"},
        "completed": {"bg": "#D4EDDA", "text": "#155724"},
        "failed": {"bg": "#F8D7DA", "text": "#721C24"},
        "idle": {"bg": "#E9ECEF", "text": "#6C757D"}
    }
    
    color = colors.get(status.lower(), colors["idle"])
    
    badge_html = f"""
    <span style="
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        background: {color['bg']};
        color: {color['text']};
    ">{display_text}</span>
    """
    
    return badge_html


def render_progress_card(
    title: str,
    current: int,
    total: int,
    color: str = "#FF4B4B"
):
    """
    Render a progress card with visual progress bar.
    
    Args:
        title: Card title
        current: Current progress value
        total: Total value
        color: Progress bar color
    """
    percentage = (current / total * 100) if total > 0 else 0
    
    card_html = f"""
    <div style="
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid #E8E8E8;
    ">
        <h4 style="margin: 0 0 1rem 0; color: #262730;">{title}</h4>
        <div style="
            background: #E9ECEF;
            border-radius: 10px;
            height: 12px;
            overflow: hidden;
        ">
            <div style="
                background: {color};
                height: 100%;
                width: {percentage}%;
                transition: width 0.3s ease;
            "></div>
        </div>
        <p style="
            margin: 0.5rem 0 0 0;
            color: #6C757D;
            font-size: 0.875rem;
        ">{current} / {total} ({percentage:.1f}%)</p>
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)


def render_metric_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    delta_color: str = "normal"
):
    """
    Render a custom metric card with enhanced styling.
    
    Args:
        label: Metric label
        value: Metric value
        delta: Optional delta value
        delta_color: Delta color (normal, inverse, off)
    """
    delta_html = ""
    if delta:
        delta_colors = {
            "normal": "#28A745",
            "inverse": "#DC3545",
            "off": "#6C757D"
        }
        delta_color_value = delta_colors.get(delta_color, delta_colors["normal"])
        
        delta_html = f"""
        <div style="
            font-size: 0.875rem;
            font-weight: 500;
            color: {delta_color_value};
            margin-top: 0.25rem;
        ">{delta}</div>
        """
    
    card_html = f"""
    <div style="
        background: linear-gradient(135deg, #F8F9FA 0%, #FFFFFF 100%);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #E8E8E8;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    ">
        <div style="
            font-size: 0.875rem;
            font-weight: 500;
            color: #6C757D;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        ">{label}</div>
        <div style="
            font-size: 1.75rem;
            font-weight: 700;
            color: #212529;
        ">{value}</div>
        {delta_html}
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)


def render_info_box(
    title: str,
    content: str,
    icon: str = "ℹ️",
    color: str = "#17A2B8"
):
    """
    Render an information box with icon and styling.
    
    Args:
        title: Box title
        content: Box content
        icon: Icon emoji
        color: Border color
    """
    box_html = f"""
    <div style="
        background: linear-gradient(135deg, #F8F9FA 0%, #FFFFFF 100%);
        border-left: 4px solid {color};
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    ">
        <div style="
            display: flex;
            align-items: center;
            margin-bottom: 0.5rem;
        ">
            <span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>
            <h4 style="margin: 0; color: #262730;">{title}</h4>
        </div>
        <p style="margin: 0; color: #495057; line-height: 1.6;">{content}</p>
    </div>
    """
    
    st.markdown(box_html, unsafe_allow_html=True)


def render_divider(text: Optional[str] = None):
    """
    Render a styled divider with optional text.
    
    Args:
        text: Optional text to display in divider
    """
    if text:
        divider_html = f"""
        <div style="
            display: flex;
            align-items: center;
            margin: 2rem 0;
        ">
            <div style="flex: 1; height: 2px; background: #E0E0E0;"></div>
            <span style="
                padding: 0 1rem;
                color: #6C757D;
                font-weight: 500;
                font-size: 0.875rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            ">{text}</span>
            <div style="flex: 1; height: 2px; background: #E0E0E0;"></div>
        </div>
        """
    else:
        divider_html = """
        <div style="
            height: 2px;
            background: #E0E0E0;
            margin: 2rem 0;
        "></div>
        """
    
    st.markdown(divider_html, unsafe_allow_html=True)


def render_alert(
    message: str,
    alert_type: str = "info",
    dismissible: bool = False
):
    """
    Render a styled alert message.
    
    Args:
        message: Alert message
        alert_type: Type of alert (success, info, warning, error)
        dismissible: Whether alert can be dismissed
    """
    alert_styles = {
        "success": {
            "bg": "linear-gradient(135deg, #D4EDDA 0%, #C3E6CB 100%)",
            "border": "#28A745",
            "text": "#155724",
            "icon": "✓"
        },
        "info": {
            "bg": "linear-gradient(135deg, #D1ECF1 0%, #BEE5EB 100%)",
            "border": "#17A2B8",
            "text": "#0C5460",
            "icon": "ℹ"
        },
        "warning": {
            "bg": "linear-gradient(135deg, #FFF3CD 0%, #FFE69C 100%)",
            "border": "#FFC107",
            "text": "#856404",
            "icon": "⚠"
        },
        "error": {
            "bg": "linear-gradient(135deg, #F8D7DA 0%, #F5C6CB 100%)",
            "border": "#DC3545",
            "text": "#721C24",
            "icon": "✗"
        }
    }
    
    style = alert_styles.get(alert_type, alert_styles["info"])
    
    alert_html = f"""
    <div style="
        background: {style['bg']};
        border-left: 4px solid {style['border']};
        border-radius: 8px;
        padding: 1rem;
        color: {style['text']};
        margin: 1rem 0;
        display: flex;
        align-items: center;
    ">
        <span style="
            font-size: 1.5rem;
            margin-right: 0.75rem;
            font-weight: bold;
        ">{style['icon']}</span>
        <span style="flex: 1;">{message}</span>
    </div>
    """
    
    st.markdown(alert_html, unsafe_allow_html=True)


def add_animation_class(element_id: str, animation: str = "fadeIn"):
    """
    Add animation class to an element.
    
    Args:
        element_id: ID of element to animate
        animation: Animation name (fadeIn, slideInLeft, slideInRight, pulse)
    """
    animation_css = f"""
    <style>
        #{element_id} {{
            animation: {animation} 0.5s ease;
        }}
    </style>
    """
    st.markdown(animation_css, unsafe_allow_html=True)


def initialize_styling():
    """
    Initialize all styling for the dashboard.
    
    This function should be called once at the start of the dashboard
    to load all custom CSS and apply theme settings.
    """
    # Load custom CSS
    load_custom_css()
    
    # Apply custom theme
    apply_custom_theme()
    
    # Hide only footer, keep MainMenu visible for settings
    hide_streamlit_style = """
    <style>
        footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
