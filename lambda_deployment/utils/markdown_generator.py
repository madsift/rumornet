"""
Markdown report generator for the Agent Monitoring Dashboard.

This module generates comprehensive markdown reports from analysis results,
formatting executive summaries, tables, and detailed findings.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime


class MarkdownGenerator:
    """
    Generate comprehensive markdown reports from analysis results.
    
    Formats analysis data into readable markdown with tables, sections,
    and proper formatting for export and sharing.
    """
    
    def __init__(self):
        """Initialize the markdown generator."""
        self.logger = logging.getLogger(f"{__name__}.markdown_generator")
    
    def generate_markdown_report(self, report: Dict[str, Any]) -> str:
        """
        Generate complete markdown report from analysis results.
        
        Args:
            report: Complete analysis report from orchestrator
            
        Returns:
            Formatted markdown string
        """
        self.logger.info("Generating markdown report")
        
        sections = []
        
        # Title and metadata
        sections.append(self._format_header(report))
        
        # Executive summary
        if "executive_summary" in report:
            sections.append(self.format_executive_summary(report["executive_summary"]))
        
        # High-priority posts
        if "high_priority_posts" in report and report["high_priority_posts"]:
            sections.append(self.format_high_priority_posts(report["high_priority_posts"]))
        
        # Top offenders
        if "top_offenders" in report and report["top_offenders"]:
            sections.append(self.format_top_offenders(report["top_offenders"]))
        
        # Pattern breakdown
        if "pattern_breakdown" in report and report["pattern_breakdown"]:
            sections.append(self.format_pattern_breakdown(report["pattern_breakdown"]))
        
        # Topic analysis
        if "topic_analysis" in report:
            sections.append(self.format_topic_analysis(report["topic_analysis"]))
        
        # Temporal trends
        if "temporal_analysis" in report:
            sections.append(self.format_temporal_trends(report["temporal_analysis"]))
        
        # Join all sections
        markdown = "\n\n".join(sections)
        
        self.logger.info("Markdown report generated successfully")
        return markdown
    
    def _format_header(self, report: Dict[str, Any]) -> str:
        """Format report header with title and metadata."""
        timestamp = report.get("executive_summary", {}).get(
            "report_generated",
            datetime.now().isoformat()
        )
        
        return f"""# Misinformation Detection Analysis Report

**Generated:** {timestamp}

---
"""
    
    def format_executive_summary(self, summary: Dict[str, Any]) -> str:
        """
        Format executive summary section.
        
        Args:
            summary: Executive summary data
            
        Returns:
            Formatted markdown string
        """
        return f"""## Executive Summary

- **Total Posts Analyzed:** {summary.get('total_posts_analyzed', 0)}
- **Misinformation Detected:** {summary.get('misinformation_detected', 0)}
- **High Risk Posts:** {summary.get('high_risk_posts', 0)}
- **Critical Posts:** {summary.get('critical_posts', 0)}
- **Unique Users:** {summary.get('unique_users', 0)}
- **Users Posting Misinformation:** {summary.get('users_posting_misinfo', 0)}
- **Patterns Detected:** {summary.get('patterns_detected', 0)}
- **Topics Identified:** {summary.get('topics_identified', 0)}"""
    
    def format_high_priority_posts(self, posts: List[Dict[str, Any]]) -> str:
        """
        Format high-priority posts table.
        
        Args:
            posts: List of high-priority post data
            
        Returns:
            Formatted markdown string
        """
        lines = ["## High-Priority Posts", ""]
        
        if not posts:
            lines.append("*No high-priority posts detected.*")
            return "\n".join(lines)
        
        # Table header
        lines.append("| Post ID | User | Risk Level | Confidence | Patterns | Action |")
        lines.append("|---------|------|------------|------------|----------|--------|")
        
        # Table rows
        for post in posts:
            post_id = post.get("post_id", "N/A")
            username = post.get("username", "N/A")
            risk_level = post.get("analysis", {}).get("risk_level", "N/A")
            confidence = post.get("analysis", {}).get("confidence", 0)
            patterns = post.get("analysis", {}).get("patterns", [])
            action = post.get("recommended_action", "N/A")
            
            # Format patterns (limit to 2 for table readability)
            patterns_str = ", ".join(patterns[:2])
            if len(patterns) > 2:
                patterns_str += f" (+{len(patterns) - 2} more)"
            if not patterns_str:
                patterns_str = "None"
            
            # Truncate action for table
            action_short = action[:50] + "..." if len(action) > 50 else action
            
            lines.append(
                f"| {post_id} | {username} | {risk_level} | {confidence:.2f} | "
                f"{patterns_str} | {action_short} |"
            )
        
        # Add detailed post information
        lines.append("")
        lines.append("### Detailed Post Information")
        lines.append("")
        
        for i, post in enumerate(posts, 1):
            lines.append(f"#### {i}. Post {post.get('post_id', 'N/A')}")
            lines.append("")
            lines.append(f"**User:** {post.get('username', 'N/A')} ({post.get('user_id', 'N/A')})")
            lines.append(f"**Platform:** {post.get('platform', 'N/A')}")
            
            if post.get("subreddit"):
                lines.append(f"**Subreddit:** {post.get('subreddit')}")
            
            lines.append(f"**Timestamp:** {post.get('timestamp', 'N/A')}")
            lines.append("")
            
            # Engagement metrics
            engagement = post.get("engagement", {})
            lines.append(f"**Engagement:** {engagement.get('upvotes', 0)} upvotes, "
                        f"{engagement.get('comments', 0)} comments, "
                        f"{engagement.get('shares', 0)} shares")
            lines.append("")
            
            # Analysis results
            analysis = post.get("analysis", {})
            lines.append(f"**Verdict:** {analysis.get('verdict', 'N/A')}")
            lines.append(f"**Confidence:** {analysis.get('confidence', 0):.2f}")
            lines.append(f"**Risk Level:** {analysis.get('risk_level', 'N/A')}")
            lines.append(f"**Language:** {analysis.get('detected_language', 'N/A')}")
            lines.append("")
            
            # Patterns
            patterns = analysis.get("patterns", [])
            if patterns:
                lines.append("**Detected Patterns:**")
                for pattern in patterns:
                    lines.append(f"- {pattern}")
                lines.append("")
            
            # Manipulation tactics
            tactics = analysis.get("manipulation_tactics", [])
            if tactics:
                lines.append("**Manipulation Tactics:**")
                for tactic in tactics:
                    lines.append(f"- {tactic}")
                lines.append("")
            
            # Specific examples
            examples = analysis.get("specific_examples", [])
            if examples:
                lines.append("**Specific Examples:**")
                for example in examples:
                    lines.append(f"- {example}")
                lines.append("")
            
            # Text preview
            text_preview = post.get("text_preview", "")
            if text_preview:
                lines.append("**Text Preview:**")
                lines.append(f"> {text_preview}")
                lines.append("")
            
            # Recommended action
            lines.append(f"**Recommended Action:** {post.get('recommended_action', 'N/A')}")
            lines.append("")
        
        return "\n".join(lines)
    
    def format_top_offenders(self, users: List[Dict[str, Any]]) -> str:
        """
        Format top offenders table.
        
        Args:
            users: List of user profile data
            
        Returns:
            Formatted markdown string
        """
        lines = ["## Top Offenders", ""]
        
        if not users:
            lines.append("*No offenders identified.*")
            return "\n".join(lines)
        
        # Table header
        lines.append("| User ID | Username | Total Posts | Misinfo Posts | Misinfo Rate | Action |")
        lines.append("|---------|----------|-------------|---------------|--------------|--------|")
        
        # Table rows
        for user in users:
            user_id = user.get("user_id", "N/A")
            username = user.get("username", "N/A")
            stats = user.get("statistics", {})
            total_posts = stats.get("total_posts", 0)
            misinfo_posts = stats.get("misinformation_posts", 0)
            misinfo_rate = stats.get("misinformation_rate", "0%")
            action = user.get("recommended_action", "N/A")
            
            # Truncate action for table
            action_short = action[:40] + "..." if len(action) > 40 else action
            
            lines.append(
                f"| {user_id} | {username} | {total_posts} | {misinfo_posts} | "
                f"{misinfo_rate} | {action_short} |"
            )
        
        # Add detailed user information
        lines.append("")
        lines.append("### Detailed User Profiles")
        lines.append("")
        
        for i, user in enumerate(users, 1):
            lines.append(f"#### {i}. {user.get('username', 'N/A')} ({user.get('user_id', 'N/A')})")
            lines.append("")
            
            # Statistics
            stats = user.get("statistics", {})
            lines.append("**Statistics:**")
            lines.append(f"- Total Posts: {stats.get('total_posts', 0)}")
            lines.append(f"- Misinformation Posts: {stats.get('misinformation_posts', 0)}")
            lines.append(f"- High Confidence Misinfo: {stats.get('high_confidence_misinfo', 0)}")
            lines.append(f"- Misinformation Rate: {stats.get('misinformation_rate', '0%')}")
            lines.append(f"- Average Confidence: {stats.get('avg_confidence', 0):.2f}")
            lines.append("")
            
            # Activity period
            activity = user.get("activity_period", {})
            if activity:
                lines.append("**Activity Period:**")
                lines.append(f"- First Seen: {activity.get('first_seen', 'N/A')}")
                lines.append(f"- Last Seen: {activity.get('last_seen', 'N/A')}")
                lines.append("")
            
            # Patterns used
            patterns = user.get("patterns_used", [])
            if patterns:
                lines.append("**Patterns Used:**")
                for pattern in patterns:
                    lines.append(f"- {pattern}")
                lines.append("")
            
            # Manipulation tactics
            tactics = user.get("manipulation_tactics", [])
            if tactics:
                lines.append("**Manipulation Tactics:**")
                for tactic in tactics:
                    lines.append(f"- {tactic}")
                lines.append("")
            
            # Languages
            languages = user.get("languages", [])
            if languages:
                lines.append(f"**Languages:** {', '.join(languages)}")
                lines.append("")
            
            # Recommended action
            lines.append(f"**Recommended Action:** {user.get('recommended_action', 'N/A')}")
            lines.append("")
        
        return "\n".join(lines)
    
    def format_topic_analysis(self, topic_data: Dict[str, Any]) -> str:
        """
        Format topic analysis section.
        
        Args:
            topic_data: Topic analysis data
            
        Returns:
            Formatted markdown string
        """
        lines = ["## Topic Analysis", ""]
        
        # Check if we have topics
        topics = topic_data.get("topics", [])
        if not topics:
            status = topic_data.get("status", "no_data")
            if status == "no_data":
                lines.append("*No topic data available.*")
            else:
                lines.append(f"*Topic analysis status: {status}*")
            return "\n".join(lines)
        
        # Summary table
        lines.append("| Topic | Total Posts | Misinfo Posts | Misinfo Rate | Top Keywords |")
        lines.append("|-------|-------------|---------------|--------------|--------------|")
        
        for topic in topics:
            topic_name = topic.get("topic_name", "Unknown")
            total_posts = topic.get("total_posts", 0)
            misinfo_posts = topic.get("misinformation_posts", 0)
            misinfo_rate = topic.get("misinformation_rate", 0)
            keywords = topic.get("keywords", [])
            
            # Format keywords (limit to 3)
            keywords_str = ", ".join(keywords[:3])
            if len(keywords) > 3:
                keywords_str += "..."
            
            # Handle misinfo_rate - might already be a string
            if isinstance(misinfo_rate, str):
                misinfo_rate_str = misinfo_rate
            else:
                misinfo_rate_str = f"{misinfo_rate:.1f}%"
            
            lines.append(
                f"| {topic_name} | {total_posts} | {misinfo_posts} | "
                f"{misinfo_rate_str} | {keywords_str} |"
            )
        
        # Detailed topic information
        lines.append("")
        lines.append("### Detailed Topic Breakdown")
        lines.append("")
        
        for i, topic in enumerate(topics, 1):
            lines.append(f"#### {i}. {topic.get('topic_name', 'Unknown')}")
            lines.append("")
            
            lines.append(f"**Total Posts:** {topic.get('total_posts', 0)}")
            lines.append(f"**Misinformation Posts:** {topic.get('misinformation_posts', 0)}")
            
            # Handle misinformation_rate - might already be a string
            misinfo_rate_val = topic.get('misinformation_rate', 0)
            if isinstance(misinfo_rate_val, str):
                lines.append(f"**Misinformation Rate:** {misinfo_rate_val}")
            else:
                lines.append(f"**Misinformation Rate:** {misinfo_rate_val:.1f}%")
            
            lines.append(f"**Average Confidence:** {topic.get('avg_confidence', 0):.2f}")
            lines.append("")
            
            # Keywords
            keywords = topic.get("keywords", [])
            if keywords:
                lines.append(f"**Keywords:** {', '.join(keywords)}")
                lines.append("")
            
            # Top users
            top_users = topic.get("top_users", [])
            if top_users:
                lines.append("**Top Users:**")
                for user in top_users[:5]:
                    lines.append(f"- {user}")
                lines.append("")
            
            # Common patterns
            patterns = topic.get("patterns", [])
            if patterns:
                lines.append("**Common Patterns:**")
                for pattern in patterns:
                    lines.append(f"- {pattern}")
                lines.append("")
        
        return "\n".join(lines)
    
    def format_temporal_trends(self, trends: Dict[str, Any]) -> str:
        """
        Format temporal trends section.
        
        Args:
            trends: Temporal analysis data
            
        Returns:
            Formatted markdown string
        """
        lines = ["## Temporal Trends", ""]
        
        # Check if we have trend data
        if not trends or trends.get("status") == "no_data":
            lines.append("*No temporal trend data available.*")
            return "\n".join(lines)
        
        # Time period summary
        if "time_period" in trends:
            period = trends["time_period"]
            lines.append(f"**Analysis Period:** {period.get('start', 'N/A')} to {period.get('end', 'N/A')}")
            lines.append("")
        
        # Activity patterns
        if "activity_patterns" in trends:
            patterns = trends["activity_patterns"]
            lines.append("### Activity Patterns")
            lines.append("")
            
            if "peak_hours" in patterns:
                peak_hours = patterns["peak_hours"]
                lines.append(f"**Peak Activity Hours:** {', '.join(map(str, peak_hours))}")
                lines.append("")
            
            if "peak_days" in patterns:
                peak_days = patterns["peak_days"]
                lines.append(f"**Peak Activity Days:** {', '.join(peak_days)}")
                lines.append("")
        
        # Misinformation trends
        if "misinformation_trends" in trends:
            misinfo_trends = trends["misinformation_trends"]
            lines.append("### Misinformation Trends")
            lines.append("")
            
            if "hourly_rate" in misinfo_trends:
                lines.append("**Hourly Misinformation Rate:**")
                for hour, rate in misinfo_trends["hourly_rate"].items():
                    rate_str = rate if isinstance(rate, str) else f"{rate:.1f}%"
                    lines.append(f"- Hour {hour}: {rate_str}")
                lines.append("")
            
            if "daily_rate" in misinfo_trends:
                lines.append("**Daily Misinformation Rate:**")
                for day, rate in misinfo_trends["daily_rate"].items():
                    rate_str = rate if isinstance(rate, str) else f"{rate:.1f}%"
                    lines.append(f"- {day}: {rate_str}")
                lines.append("")
        
        # Burst detection
        if "bursts_detected" in trends:
            bursts = trends["bursts_detected"]
            if bursts:
                lines.append("### Detected Activity Bursts")
                lines.append("")
                
                for i, burst in enumerate(bursts, 1):
                    lines.append(f"**Burst {i}:**")
                    lines.append(f"- Time: {burst.get('time', 'N/A')}")
                    lines.append(f"- Posts: {burst.get('post_count', 0)}")
                    lines.append(f"- Users: {burst.get('user_count', 0)}")
                    
                    # Handle misinfo_rate
                    misinfo_rate_burst = burst.get('misinfo_rate', 0)
                    if isinstance(misinfo_rate_burst, str):
                        lines.append(f"- Misinformation Rate: {misinfo_rate_burst}")
                    else:
                        lines.append(f"- Misinformation Rate: {misinfo_rate_burst:.1f}%")
                    lines.append("")
        
        return "\n".join(lines)
    
    def format_pattern_breakdown(self, patterns: List[Dict[str, Any]]) -> str:
        """
        Format pattern breakdown section.
        
        Args:
            patterns: List of pattern data
            
        Returns:
            Formatted markdown string
        """
        lines = ["## Pattern Breakdown", ""]
        
        if not patterns:
            lines.append("*No patterns detected.*")
            return "\n".join(lines)
        
        # Summary table
        lines.append("| Pattern | Occurrences | Unique Users | First Seen | Last Seen |")
        lines.append("|---------|-------------|--------------|------------|-----------|")
        
        for pattern in patterns:
            pattern_name = pattern.get("pattern_name", "Unknown")
            occurrences = pattern.get("total_occurrences", 0)
            unique_users = pattern.get("unique_users", 0)
            first_seen = pattern.get("first_seen", "N/A")
            last_seen = pattern.get("last_seen", "N/A")
            
            # Truncate timestamps for table
            if isinstance(first_seen, str) and len(first_seen) > 16:
                first_seen = first_seen[:16]
            if isinstance(last_seen, str) and len(last_seen) > 16:
                last_seen = last_seen[:16]
            
            lines.append(
                f"| {pattern_name} | {occurrences} | {unique_users} | "
                f"{first_seen} | {last_seen} |"
            )
        
        # Detailed pattern information
        lines.append("")
        lines.append("### Detailed Pattern Information")
        lines.append("")
        
        for i, pattern in enumerate(patterns, 1):
            lines.append(f"#### {i}. {pattern.get('pattern_name', 'Unknown')}")
            lines.append("")
            
            lines.append(f"**Total Occurrences:** {pattern.get('total_occurrences', 0)}")
            lines.append(f"**Unique Users:** {pattern.get('unique_users', 0)}")
            lines.append(f"**First Seen:** {pattern.get('first_seen', 'N/A')}")
            lines.append(f"**Last Seen:** {pattern.get('last_seen', 'N/A')}")
            lines.append("")
            
            # Recent examples
            examples = pattern.get("recent_examples", [])
            if examples:
                lines.append("**Recent Examples:**")
                lines.append("")
                lines.append("| Post ID | User ID | Timestamp | Confidence |")
                lines.append("|---------|---------|-----------|------------|")
                
                for example in examples:
                    post_id = example.get("post_id", "N/A")
                    user_id = example.get("user_id", "N/A")
                    timestamp = example.get("timestamp", "N/A")
                    confidence = example.get("confidence", 0)
                    
                    # Truncate timestamp
                    if isinstance(timestamp, str) and len(timestamp) > 16:
                        timestamp = timestamp[:16]
                    
                    lines.append(f"| {post_id} | {user_id} | {timestamp} | {confidence:.2f} |")
                
                lines.append("")
        
        return "\n".join(lines)
