"""
Results display and visualization components for the Agent Monitoring Dashboard.

This module provides comprehensive visualization components for displaying
analysis results including executive summary, high-priority posts, top offenders,
pattern breakdown, topic analysis, and temporal trends.

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
"""

import streamlit as st
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime
from components.filtering import (
    create_filter_panel,
    apply_all_filters,
    update_visualizations_with_filters,
    display_filter_summary
)


def render_executive_summary_display(summary: Dict[str, Any]):
    """
    Render executive summary display with key metrics.
    
    Displays comprehensive overview of analysis results including
    total posts, misinformation detected, risk levels, and patterns.
    
    Args:
        summary: Executive summary data from orchestrator report
        
    Requirements: 4.1
    """
    st.header("📋 Executive Summary")
    
    # First row - main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Posts Analyzed",
            summary.get("total_posts_analyzed", 0),
            help="Total number of posts processed in this analysis"
        )
    
    with col2:
        misinfo_count = summary.get("misinformation_detected", 0)
        total = summary.get("total_posts_analyzed", 1)
        misinfo_rate = (misinfo_count / total * 100) if total > 0 else 0
        st.metric(
            "Misinformation Detected",
            misinfo_count,
            delta=f"{misinfo_rate:.1f}%",
            delta_color="inverse",
            help="Number and percentage of posts flagged as misinformation"
        )
    
    with col3:
        st.metric(
            "High Risk Posts",
            summary.get("high_risk_posts", 0),
            help="Posts requiring immediate attention"
        )
    
    with col4:
        st.metric(
            "Critical Posts",
            summary.get("critical_posts", 0),
            help="Critical posts requiring urgent action"
        )
    
    # Second row - user and pattern metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Unique Users",
            summary.get("unique_users", 0),
            help="Total number of unique users in the dataset"
        )
    
    with col2:
        st.metric(
            "Users w/ Misinfo",
            summary.get("users_posting_misinfo", 0),
            help="Number of users who posted misinformation"
        )
    
    with col3:
        st.metric(
            "Patterns Detected",
            summary.get("patterns_detected", 0),
            help="Number of distinct misinformation patterns identified"
        )
    
    with col4:
        st.metric(
            "Topics Identified",
            summary.get("topics_identified", 0),
            help="Number of distinct topics analyzed"
        )
    
    st.divider()


def render_high_priority_posts_table(
    posts: List[Dict[str, Any]],
    sortable: bool = True,
    max_display: int = 50
):
    """
    Render high-priority posts table with sorting capabilities.
    
    Displays posts flagged as high-priority with detailed information
    including risk level, confidence, patterns, and recommended actions.
    
    Args:
        posts: List of high-priority post data
        sortable: Whether to enable sorting functionality
        max_display: Maximum number of posts to display
        
    Requirements: 4.2
    """
    st.header("🚨 High-Priority Posts")
    
    if not posts:
        st.info("No high-priority posts detected in this analysis.")
        return
    
    st.write(f"Showing {min(len(posts), max_display)} of {len(posts)} high-priority posts")
    
    # Prepare data for table
    table_data = []
    for post in posts[:max_display]:
        metadata = post.get("metadata", {})
        analysis = post.get("analysis", {})
        
        # Format verdict
        verdict = analysis.get("verdict")
        if verdict is False:
            verdict_str = "❌ MISINFO"
        elif verdict is True:
            verdict_str = "✅ TRUE"
        else:
            verdict_str = "❓ UNCERTAIN"
        
        # Format patterns
        patterns = analysis.get("patterns", [])
        if isinstance(patterns, list):
            patterns_str = ", ".join(patterns[:2])
            if len(patterns) > 2:
                patterns_str += f" (+{len(patterns) - 2})"
        else:
            patterns_str = "None"
        
        table_data.append({
            "Post ID": post.get("post_id", metadata.get("post_id", "N/A")),
            "User": post.get("username", metadata.get("username", "N/A")),
            "Platform": metadata.get("platform", "N/A"),
            "Verdict": verdict_str,
            "Confidence": f"{analysis.get('confidence', 0):.2f}",
            "Risk Level": analysis.get("risk_level", "N/A"),
            "Patterns": patterns_str,
            "Timestamp": metadata.get("timestamp", "N/A")
        })
    
    # Display as dataframe
    df = pd.DataFrame(table_data)
    
    if sortable:
        # Add sorting controls
        col1, col2 = st.columns([3, 1])
        with col1:
            sort_by = st.selectbox(
                "Sort by",
                options=["Risk Level", "Confidence", "Timestamp", "Post ID"],
                key="high_priority_sort"
            )
        with col2:
            sort_order = st.radio(
                "Order",
                options=["Descending", "Ascending"],
                key="high_priority_order",
                horizontal=True
            )
        
        # Apply sorting
        ascending = (sort_order == "Ascending")
        if sort_by in df.columns:
            df = df.sort_values(by=sort_by, ascending=ascending)
    
    # Display table
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )
    
    # Expandable detailed view
    with st.expander("📄 View Detailed Post Information"):
        for i, post in enumerate(posts[:max_display], 1):
            st.subheader(f"Post {i}: {post.get('post_id', 'N/A')}")
            
            metadata = post.get("metadata", {})
            analysis = post.get("analysis", {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**User:** {post.get('username', metadata.get('username', 'N/A'))}")
                st.write(f"**Platform:** {metadata.get('platform', 'N/A')}")
                st.write(f"**Timestamp:** {metadata.get('timestamp', 'N/A')}")
                
                if post.get("subreddit"):
                    st.write(f"**Subreddit:** {post.get('subreddit')}")
            
            with col2:
                st.write(f"**Risk Level:** {analysis.get('risk_level', 'N/A')}")
                st.write(f"**Confidence:** {analysis.get('confidence', 0):.2f}")
                st.write(f"**Language:** {analysis.get('detected_language', 'N/A')}")
            
            # Engagement metrics
            engagement = post.get("engagement", {})
            if engagement:
                st.write(f"**Engagement:** {engagement.get('upvotes', 0)} upvotes, "
                        f"{engagement.get('comments', 0)} comments, "
                        f"{engagement.get('shares', 0)} shares")
            
            # Patterns
            patterns = analysis.get("patterns", [])
            if patterns:
                st.write("**Detected Patterns:**")
                for pattern in patterns:
                    st.write(f"- {pattern}")
            
            # Text preview
            text_preview = post.get("text_preview", "")
            if text_preview:
                st.write("**Text Preview:**")
                st.text_area(
                    "Content",
                    text_preview,
                    height=100,
                    key=f"text_preview_{i}",
                    disabled=True
                )
            
            # Recommended action
            action = post.get("recommended_action", "")
            if action:
                st.info(f"**Recommended Action:** {action}")
            
            st.divider()


def render_top_offenders_display(users: List[Dict[str, Any]], max_display: int = 20):
    """
    Render top offenders display with statistics.
    
    Displays users with highest misinformation rates including
    detailed statistics, patterns used, and recommended actions.
    
    Args:
        users: List of top offender user data
        max_display: Maximum number of users to display
        
    Requirements: 4.3
    """
    st.header("👤 Top Offenders")
    
    if not users:
        st.info("No offenders identified in this analysis.")
        return
    
    st.write(f"Showing {min(len(users), max_display)} of {len(users)} users")
    
    # Prepare data for table
    table_data = []
    for user in users[:max_display]:
        stats = user.get("statistics", {})
        
        table_data.append({
            "User ID": user.get("user_id", "N/A"),
            "Username": user.get("username", "N/A"),
            "Total Posts": stats.get("total_posts", 0),
            "Misinfo Posts": stats.get("misinformation_posts", 0),
            "Misinfo Rate": stats.get("misinformation_rate", "0%"),
            "Avg Confidence": f"{stats.get('avg_confidence', 0):.2f}",
            "High Confidence": stats.get("high_confidence_misinfo", 0)
        })
    
    # Display as dataframe
    df = pd.DataFrame(table_data)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )
    
    # Expandable detailed view
    with st.expander("📊 View Detailed User Profiles"):
        for i, user in enumerate(users[:max_display], 1):
            st.subheader(f"{i}. {user.get('username', 'N/A')} ({user.get('user_id', 'N/A')})")
            
            stats = user.get("statistics", {})
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Posts", stats.get("total_posts", 0))
                st.metric("Misinfo Posts", stats.get("misinformation_posts", 0))
            
            with col2:
                st.metric("Misinfo Rate", stats.get("misinformation_rate", "0%"))
                st.metric("Avg Confidence", f"{stats.get('avg_confidence', 0):.2f}")
            
            with col3:
                st.metric("High Confidence", stats.get("high_confidence_misinfo", 0))
            
            # Activity period
            activity = user.get("activity_period", {})
            if activity:
                st.write(f"**Activity Period:** {activity.get('first_seen', 'N/A')} to {activity.get('last_seen', 'N/A')}")
            
            # Patterns used
            patterns = user.get("patterns_used", [])
            if patterns:
                st.write("**Patterns Used:**")
                for pattern in patterns:
                    st.write(f"- {pattern}")
            
            # Languages
            languages = user.get("languages", [])
            if languages:
                st.write(f"**Languages:** {', '.join(languages)}")
            
            # Recommended action
            action = user.get("recommended_action", "")
            if action:
                st.warning(f"**Recommended Action:** {action}")
            
            st.divider()


def render_pattern_breakdown_visualization(patterns: List[Dict[str, Any]]):
    """
    Render pattern breakdown visualization.
    
    Displays detected misinformation patterns with occurrence counts,
    unique users, and temporal information.
    
    Args:
        patterns: List of pattern data
        
    Requirements: 4.4
    """
    st.header("🔍 Pattern Breakdown")
    
    if not patterns:
        st.info("No patterns detected in this analysis.")
        return
    
    # Prepare data for visualization
    pattern_data = []
    for pattern in patterns:
        pattern_data.append({
            "Pattern": pattern.get("pattern_name", "Unknown"),
            "Occurrences": pattern.get("total_occurrences", 0),
            "Unique Users": pattern.get("unique_users", 0),
            "First Seen": pattern.get("first_seen", "N/A"),
            "Last Seen": pattern.get("last_seen", "N/A")
        })
    
    df = pd.DataFrame(pattern_data)
    
    # Display summary table
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )
    
    # Bar chart of pattern occurrences
    if len(patterns) > 0:
        st.subheader("Pattern Occurrence Distribution")
        chart_data = df.set_index("Pattern")["Occurrences"]
        st.bar_chart(chart_data)
    
    # Expandable detailed view
    with st.expander("📋 View Detailed Pattern Information"):
        for i, pattern in enumerate(patterns, 1):
            st.subheader(f"{i}. {pattern.get('pattern_name', 'Unknown')}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Occurrences", pattern.get("total_occurrences", 0))
            
            with col2:
                st.metric("Unique Users", pattern.get("unique_users", 0))
            
            with col3:
                first_seen = pattern.get("first_seen", "N/A")
                if isinstance(first_seen, str) and len(first_seen) > 10:
                    first_seen = first_seen[:10]
                st.metric("First Seen", first_seen)
            
            # Recent examples
            examples = pattern.get("recent_examples", [])
            if examples:
                st.write("**Recent Examples:**")
                example_data = []
                for example in examples[:5]:
                    example_data.append({
                        "Post ID": example.get("post_id", "N/A"),
                        "User ID": example.get("user_id", "N/A"),
                        "Timestamp": example.get("timestamp", "N/A"),
                        "Confidence": f"{example.get('confidence', 0):.2f}"
                    })
                
                if example_data:
                    st.dataframe(
                        pd.DataFrame(example_data),
                        use_container_width=True,
                        hide_index=True
                    )
            
            st.divider()


def render_topic_analysis_display(topic_data: Dict[str, Any]):
    """
    Render topic analysis display.
    
    Displays topic-based analysis including misinformation rates per topic,
    keywords, and top users for each topic.
    
    Args:
        topic_data: Topic analysis data from orchestrator
        
    Requirements: 4.5
    """
    st.header("📚 Topic Analysis")
    
    # Check if we have topics
    topics = topic_data.get("topics", [])
    if not topics:
        status = topic_data.get("status", "no_data")
        if status == "no_data":
            st.info("No topic data available for this analysis.")
        else:
            st.info(f"Topic analysis status: {status}")
        return
    
    # Prepare data for table
    topic_table_data = []
    for topic in topics:
        keywords = topic.get("keywords", [])
        keywords_str = ", ".join(keywords[:3])
        if len(keywords) > 3:
            keywords_str += "..."
        
        # Handle misinformation_rate - it might already be a string with %
        misinfo_rate = topic.get('misinformation_rate', 0)
        if isinstance(misinfo_rate, str):
            misinfo_rate_display = misinfo_rate  # Already formatted
        else:
            misinfo_rate_display = f"{misinfo_rate:.1f}%"
        
        topic_table_data.append({
            "Topic": topic.get("topic_name", "Unknown"),
            "Total Posts": topic.get("total_posts", 0),
            "Misinfo Posts": topic.get("misinformation_posts", 0),
            "Misinfo Rate": misinfo_rate_display,
            "Avg Confidence": f"{topic.get('avg_confidence', 0):.2f}",
            "Top Keywords": keywords_str
        })
    
    df = pd.DataFrame(topic_table_data)
    
    # Display summary table
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )
    
    # Bar chart of misinformation rates by topic
    if len(topics) > 0:
        st.subheader("Misinformation Rate by Topic")
        chart_data = pd.DataFrame({
            "Topic": [t.get("topic_name", "Unknown") for t in topics],
            "Misinfo Rate (%)": [t.get("misinformation_rate", 0) for t in topics]
        }).set_index("Topic")
        st.bar_chart(chart_data)
    
    # Expandable detailed view
    with st.expander("📖 View Detailed Topic Information"):
        for i, topic in enumerate(topics, 1):
            st.subheader(f"{i}. {topic.get('topic_name', 'Unknown')}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Posts", topic.get("total_posts", 0))
                st.metric("Misinfo Posts", topic.get("misinformation_posts", 0))
            
            with col2:
                # Handle misinformation_rate - might already be a string
                misinfo_rate = topic.get('misinformation_rate', 0)
                if isinstance(misinfo_rate, str):
                    st.metric("Misinfo Rate", misinfo_rate)
                else:
                    st.metric("Misinfo Rate", f"{misinfo_rate:.1f}%")
                
                st.metric("Avg Confidence", f"{topic.get('avg_confidence', 0):.2f}")
            
            with col3:
                keywords = topic.get("keywords", [])
                st.write(f"**Keywords ({len(keywords)}):**")
                st.write(", ".join(keywords[:10]))
            
            # Top users
            top_users = topic.get("top_users", [])
            if top_users:
                st.write("**Top Users:**")
                for user in top_users[:5]:
                    st.write(f"- {user}")
            
            # Common patterns
            patterns = topic.get("patterns", [])
            if patterns:
                st.write("**Common Patterns:**")
                for pattern in patterns:
                    st.write(f"- {pattern}")
            
            st.divider()


def render_temporal_trends_visualization(trends: Dict[str, Any]):
    """
    Render temporal trends visualization.
    
    Displays temporal analysis including activity patterns, peak hours/days,
    misinformation trends over time, and detected activity bursts.
    
    Args:
        trends: Temporal analysis data from orchestrator
        
    Requirements: 4.5
    """
    st.header("📈 Temporal Trends")
    
    # Check if we have trend data
    if not trends or trends.get("status") == "no_data":
        st.info("No temporal trend data available for this analysis.")
        return
    
    # Time period summary
    if "time_period" in trends:
        period = trends["time_period"]
        st.write(f"**Analysis Period:** {period.get('start', 'N/A')} to {period.get('end', 'N/A')}")
        st.divider()
    
    # Activity patterns
    if "activity_patterns" in trends:
        patterns = trends["activity_patterns"]
        
        st.subheader("Activity Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if "peak_hours" in patterns and patterns["peak_hours"]:
                st.write("**Peak Activity Hours:**")
                peak_hours = patterns["peak_hours"]
                st.write(", ".join(map(str, peak_hours)))
        
        with col2:
            if "peak_days" in patterns and patterns["peak_days"]:
                st.write("**Peak Activity Days:**")
                peak_days = patterns["peak_days"]
                st.write(", ".join(peak_days))
        
        st.divider()
    
    # Misinformation trends
    if "misinformation_trends" in trends:
        misinfo_trends = trends["misinformation_trends"]
        
        st.subheader("Misinformation Trends")
        
        # Hourly rate chart
        if "hourly_rate" in misinfo_trends and misinfo_trends["hourly_rate"]:
            st.write("**Hourly Misinformation Rate:**")
            hourly_data = pd.DataFrame({
                "Hour": list(misinfo_trends["hourly_rate"].keys()),
                "Misinfo Rate (%)": list(misinfo_trends["hourly_rate"].values())
            }).set_index("Hour")
            st.line_chart(hourly_data)
        
        # Daily rate chart
        if "daily_rate" in misinfo_trends and misinfo_trends["daily_rate"]:
            st.write("**Daily Misinformation Rate:**")
            daily_data = pd.DataFrame({
                "Day": list(misinfo_trends["daily_rate"].keys()),
                "Misinfo Rate (%)": list(misinfo_trends["daily_rate"].values())
            }).set_index("Day")
            st.bar_chart(daily_data)
        
        st.divider()
    
    # Burst detection
    if "bursts_detected" in trends and trends["bursts_detected"]:
        bursts = trends["bursts_detected"]
        
        st.subheader("Detected Activity Bursts")
        st.write(f"Found {len(bursts)} activity burst(s)")
        
        burst_data = []
        for burst in bursts:
            # Handle misinfo_rate - might already be a string
            misinfo_rate_burst = burst.get('misinfo_rate', 0)
            if isinstance(misinfo_rate_burst, str):
                misinfo_rate_str = misinfo_rate_burst
            else:
                misinfo_rate_str = f"{misinfo_rate_burst:.1f}%"
            
            burst_data.append({
                "Time": burst.get("time", "N/A"),
                "Posts": burst.get("post_count", 0),
                "Users": burst.get("user_count", 0),
                "Misinfo Rate": misinfo_rate_str
            })
        
        if burst_data:
            st.dataframe(
                pd.DataFrame(burst_data),
                use_container_width=True,
                hide_index=True
            )


def render_complete_results_display(report: Dict[str, Any]):
    """
    Render complete results display with all sections and filtering.
    
    Displays all analysis results in a comprehensive view including
    executive summary, high-priority posts, top offenders, patterns,
    topics, and temporal trends. Includes filtering and search functionality.
    
    Args:
        report: Complete orchestrator report
        
    Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 9.1, 9.2, 9.3, 9.4, 9.5
    """
    # Initialize filter state in session state if not present
    if "results_filters_applied" not in st.session_state:
        st.session_state.results_filters_applied = False
    if "filtered_report" not in st.session_state:
        st.session_state.filtered_report = None
    
    # Create filter panel in sidebar
    with st.sidebar:
        st.divider()
        filters = create_filter_panel()
    
    # Determine which report to display
    display_report = report
    
    # Apply filters if requested
    if filters.get("apply"):
        st.session_state.results_filters_applied = True
        st.session_state.filtered_report = update_visualizations_with_filters(report, filters)
        display_report = st.session_state.filtered_report
        
        # Show filter summary
        if "high_priority_posts" in report:
            original_count = len(report["high_priority_posts"])
            filtered_count = len(display_report["high_priority_posts"])
            display_filter_summary(original_count, filtered_count)
    
    # Clear filters if requested
    elif filters.get("clear"):
        st.session_state.results_filters_applied = False
        st.session_state.filtered_report = None
        display_report = report
        st.rerun()
    
    # Use filtered report if filters were previously applied
    elif st.session_state.results_filters_applied and st.session_state.filtered_report:
        display_report = st.session_state.filtered_report
        
        # Show filter summary
        if "high_priority_posts" in report:
            original_count = len(report["high_priority_posts"])
            filtered_count = len(display_report["high_priority_posts"])
            display_filter_summary(original_count, filtered_count)
    
    # Executive summary
    if "executive_summary" in display_report:
        render_executive_summary_display(display_report["executive_summary"])
    
    # High-priority posts
    if "high_priority_posts" in display_report and display_report["high_priority_posts"]:
        render_high_priority_posts_table(display_report["high_priority_posts"])
        st.divider()
    
    # Top offenders
    if "top_offenders" in display_report and display_report["top_offenders"]:
        render_top_offenders_display(display_report["top_offenders"])
        st.divider()
    
    # Pattern breakdown
    if "pattern_breakdown" in display_report and display_report["pattern_breakdown"]:
        render_pattern_breakdown_visualization(display_report["pattern_breakdown"])
        st.divider()
    
    # Topic analysis
    if "topic_analysis" in display_report:
        render_topic_analysis_display(display_report["topic_analysis"])
        st.divider()
    
    # Temporal trends
    if "temporal_analysis" in display_report:
        render_temporal_trends_visualization(display_report["temporal_analysis"])
