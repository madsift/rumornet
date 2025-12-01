"""
Markdown export functionality for the Agent Monitoring Dashboard.

This module provides components for generating, previewing, and downloading
markdown reports from analysis results.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
"""

import streamlit as st
from typing import Dict, Any, Optional
from datetime import datetime
import io

from utils.markdown_generator import MarkdownGenerator


def generate_markdown_with_metadata(
    report: Dict[str, Any],
    execution_id: Optional[str] = None,
    include_metadata: bool = True
) -> str:
    """
    Generate markdown report with metadata and timestamps.
    
    Creates a comprehensive markdown report from analysis results,
    optionally including metadata for traceability.
    
    Args:
        report: Complete analysis report from orchestrator
        execution_id: Optional execution ID for traceability
        include_metadata: Whether to include metadata section
        
    Returns:
        Formatted markdown string with metadata
        
    Requirements: 5.1, 5.5
    """
    generator = MarkdownGenerator()
    
    # Generate base markdown
    markdown = generator.generate_markdown_report(report)
    
    # Add metadata section if requested
    if include_metadata:
        metadata_section = _generate_metadata_section(report, execution_id)
        # Insert metadata after title
        lines = markdown.split('\n')
        if lines:
            # Find the first empty line after title
            insert_pos = 1
            for i, line in enumerate(lines[1:], 1):
                if line.strip() == '':
                    insert_pos = i + 1
                    break
            
            lines.insert(insert_pos, metadata_section)
            markdown = '\n'.join(lines)
    
    return markdown


def _generate_metadata_section(
    report: Dict[str, Any],
    execution_id: Optional[str] = None
) -> str:
    """Generate metadata section for markdown report."""
    metadata_lines = [
        "## Report Metadata",
        "",
        f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    
    if execution_id:
        metadata_lines.append(f"- **Execution ID:** {execution_id}")
    
    # Add report timestamp if available
    exec_summary = report.get("executive_summary", {})
    if "report_generated" in exec_summary:
        metadata_lines.append(f"- **Analysis Timestamp:** {exec_summary['report_generated']}")
    
    # Add version info if available
    if "version" in report:
        metadata_lines.append(f"- **Report Version:** {report['version']}")
    
    metadata_lines.extend(["", "---", ""])
    
    return '\n'.join(metadata_lines)


def render_markdown_preview(
    report: Dict[str, Any],
    max_height: int = 600
):
    """
    Render markdown preview in the dashboard.
    
    Displays a preview of the generated markdown report with
    scrollable view and syntax highlighting.
    
    Args:
        report: Complete analysis report
        max_height: Maximum height for preview area in pixels
        
    Requirements: 5.2, 5.3
    """
    st.subheader("📝 Markdown Preview")
    
    # Generate markdown
    markdown = generate_markdown_with_metadata(report)
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Rendered View", "Raw Markdown"])
    
    with tab1:
        # Render the markdown
        st.markdown(markdown)
    
    with tab2:
        # Show raw markdown in code block
        st.code(markdown, language="markdown")
    
    # Show character count
    st.caption(f"Total characters: {len(markdown):,}")


def render_markdown_download_button(
    report: Dict[str, Any],
    filename: Optional[str] = None,
    execution_id: Optional[str] = None,
    button_label: str = "📥 Download Markdown Report"
) -> bool:
    """
    Render download button for markdown report.
    
    Creates a download button that allows users to save the
    markdown report as a .md file.
    
    Args:
        report: Complete analysis report
        filename: Optional custom filename (without extension)
        execution_id: Optional execution ID for metadata
        button_label: Label for the download button
        
    Returns:
        True if download button was clicked, False otherwise
        
    Requirements: 5.4
    """
    # Generate markdown with metadata
    markdown = generate_markdown_with_metadata(
        report,
        execution_id=execution_id,
        include_metadata=True
    )
    
    # Generate filename if not provided
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"misinformation_analysis_{timestamp}"
    
    # Ensure .md extension
    if not filename.endswith('.md'):
        filename += '.md'
    
    # Create download button
    clicked = st.download_button(
        label=button_label,
        data=markdown,
        file_name=filename,
        mime="text/markdown",
        help="Download the analysis report as a Markdown file"
    )
    
    return clicked


def render_markdown_export_options(
    report: Dict[str, Any],
    execution_id: Optional[str] = None
):
    """
    Render markdown export options panel.
    
    Provides a comprehensive export interface with options for
    customizing the markdown output and download.
    
    Args:
        report: Complete analysis report
        execution_id: Optional execution ID for metadata
        
    Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
    """
    st.subheader("📤 Export Options")
    
    # Export settings
    col1, col2 = st.columns(2)
    
    with col1:
        include_metadata = st.checkbox(
            "Include metadata",
            value=True,
            help="Add report metadata and timestamps"
        )
    
    with col2:
        custom_filename = st.text_input(
            "Custom filename (optional)",
            placeholder="my_report",
            help="Leave empty for auto-generated filename"
        )
    
    # Preview toggle
    show_preview = st.checkbox(
        "Show preview before download",
        value=False,
        help="Preview the markdown report before downloading"
    )
    
    st.divider()
    
    # Generate markdown based on settings
    if include_metadata:
        markdown = generate_markdown_with_metadata(
            report,
            execution_id=execution_id,
            include_metadata=True
        )
    else:
        generator = MarkdownGenerator()
        markdown = generator.generate_markdown_report(report)
    
    # Show preview if requested
    if show_preview:
        with st.expander("📄 Preview", expanded=True):
            st.markdown(markdown)
    
    # Download button
    filename = custom_filename if custom_filename else None
    
    if render_markdown_download_button(
        report,
        filename=filename,
        execution_id=execution_id,
        button_label="📥 Download Markdown Report"
    ):
        st.success("✅ Markdown report downloaded successfully!")


def render_complete_markdown_export(
    report: Dict[str, Any],
    execution_id: Optional[str] = None
):
    """
    Render complete markdown export interface.
    
    Provides a comprehensive interface for generating, previewing,
    and downloading markdown reports with all available options.
    
    Args:
        report: Complete analysis report
        execution_id: Optional execution ID for metadata
        
    Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
    """
    st.title("📝 Markdown Export")
    
    if not report:
        st.warning("No report data available for export.")
        return
    
    # Create tabs for different export features
    tab1, tab2, tab3 = st.tabs([
        "📥 Quick Export",
        "📝 Preview & Customize",
        "ℹ️ Export Info"
    ])
    
    with tab1:
        st.subheader("Quick Export")
        st.write("Download the markdown report with default settings.")
        
        # Quick download button
        if render_markdown_download_button(
            report,
            execution_id=execution_id,
            button_label="📥 Download Report Now"
        ):
            st.success("✅ Report downloaded successfully!")
        
        # Show basic stats
        st.divider()
        st.write("**Report Statistics:**")
        
        exec_summary = report.get("executive_summary", {})
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Posts", exec_summary.get("total_posts_analyzed", 0))
        
        with col2:
            st.metric("Misinformation", exec_summary.get("misinformation_detected", 0))
        
        with col3:
            st.metric("High Risk", exec_summary.get("high_risk_posts", 0))
    
    with tab2:
        st.subheader("Preview & Customize")
        
        # Export options
        render_markdown_export_options(report, execution_id)
        
        st.divider()
        
        # Full preview
        st.subheader("Full Preview")
        render_markdown_preview(report)
    
    with tab3:
        st.subheader("Export Information")
        
        st.write("""
        ### About Markdown Export
        
        The markdown export feature generates a comprehensive report in Markdown format
        that can be easily shared, version controlled, or converted to other formats.
        
        **Features:**
        - Executive summary with key metrics
        - High-priority posts with detailed analysis
        - Top offenders with statistics
        - Pattern breakdown and occurrence data
        - Topic analysis with misinformation rates
        - Temporal trends and activity patterns
        
        **Metadata:**
        - Generation timestamp
        - Execution ID (if available)
        - Analysis timestamp
        - Report version
        
        **File Format:**
        - Standard Markdown (.md)
        - Compatible with GitHub, GitLab, and other platforms
        - Can be converted to HTML, PDF, or other formats using tools like Pandoc
        
        **Best Practices:**
        - Include metadata for traceability
        - Use descriptive filenames
        - Preview before downloading for large reports
        - Store reports in version control for historical tracking
        """)
        
        # Show report sections
        st.divider()
        st.write("**Report Sections:**")
        
        sections = []
        if "executive_summary" in report:
            sections.append("✅ Executive Summary")
        if "high_priority_posts" in report and report["high_priority_posts"]:
            sections.append(f"✅ High-Priority Posts ({len(report['high_priority_posts'])})")
        if "top_offenders" in report and report["top_offenders"]:
            sections.append(f"✅ Top Offenders ({len(report['top_offenders'])})")
        if "pattern_breakdown" in report and report["pattern_breakdown"]:
            sections.append(f"✅ Pattern Breakdown ({len(report['pattern_breakdown'])})")
        if "topic_analysis" in report:
            topics = report["topic_analysis"].get("topics", [])
            sections.append(f"✅ Topic Analysis ({len(topics)} topics)")
        if "temporal_analysis" in report:
            sections.append("✅ Temporal Trends")
        
        for section in sections:
            st.write(f"- {section}")


def export_markdown_to_string(
    report: Dict[str, Any],
    include_metadata: bool = True,
    execution_id: Optional[str] = None
) -> str:
    """
    Export markdown report as a string.
    
    Utility function for programmatic access to markdown generation.
    
    Args:
        report: Complete analysis report
        include_metadata: Whether to include metadata
        execution_id: Optional execution ID
        
    Returns:
        Markdown report as string
        
    Requirements: 5.1, 5.5
    """
    if include_metadata:
        return generate_markdown_with_metadata(
            report,
            execution_id=execution_id,
            include_metadata=True
        )
    else:
        generator = MarkdownGenerator()
        return generator.generate_markdown_report(report)


def save_markdown_to_file(
    report: Dict[str, Any],
    filepath: str,
    include_metadata: bool = True,
    execution_id: Optional[str] = None
) -> bool:
    """
    Save markdown report to a file.
    
    Utility function for saving markdown reports to disk.
    
    Args:
        report: Complete analysis report
        filepath: Path where to save the file
        include_metadata: Whether to include metadata
        execution_id: Optional execution ID
        
    Returns:
        True if save was successful, False otherwise
        
    Requirements: 5.4
    """
    try:
        markdown = export_markdown_to_string(
            report,
            include_metadata=include_metadata,
            execution_id=execution_id
        )
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(markdown)
        
        return True
    except Exception as e:
        st.error(f"Failed to save markdown file: {e}")
        return False


def get_markdown_stats(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get statistics about the markdown report.
    
    Returns information about the report size, sections, and content.
    
    Args:
        report: Complete analysis report
        
    Returns:
        Dictionary with report statistics
        
    Requirements: 5.5
    """
    generator = MarkdownGenerator()
    markdown = generator.generate_markdown_report(report)
    
    # Count sections
    sections = []
    if "executive_summary" in report:
        sections.append("Executive Summary")
    if "high_priority_posts" in report and report["high_priority_posts"]:
        sections.append("High-Priority Posts")
    if "top_offenders" in report and report["top_offenders"]:
        sections.append("Top Offenders")
    if "pattern_breakdown" in report and report["pattern_breakdown"]:
        sections.append("Pattern Breakdown")
    if "topic_analysis" in report:
        sections.append("Topic Analysis")
    if "temporal_analysis" in report:
        sections.append("Temporal Trends")
    
    # Calculate stats
    lines = markdown.split('\n')
    words = len(markdown.split())
    
    return {
        "total_characters": len(markdown),
        "total_lines": len(lines),
        "total_words": words,
        "sections": sections,
        "section_count": len(sections),
        "estimated_read_time_minutes": max(1, words // 200)  # Assuming 200 words per minute
    }
