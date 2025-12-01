"""
Data transformer for converting various post formats to the standard format.

This module handles transformation of different data formats (Reddit, Twitter, etc.)
into the standard format expected by the orchestrator.
"""

from typing import List, Dict, Any
from datetime import datetime


def transform_reddit_post(reddit_post: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform Reddit post format to standard format.
    
    Input format:
    {
        "submission_id": "15uzos2",
        "author_name": "user_42486",
        "posts": "Brazil arrests police officials...",
        "score": 25,
        "num_comments": 1,
        "upvote_ratio": 0.91,
        "created_utc": "2023-08-19 07:20:07",
        "subreddit": "GlobalTalk"
    }
    
    Output format:
    {
        "post_id": "15uzos2",
        "user_id": "user_42486",
        "username": "user_42486",
        "text": "Brazil arrests police officials...",
        "timestamp": "2023-08-19 07:20:07",
        "platform": "reddit",
        "subreddit": "GlobalTalk",
        "upvotes": 25,
        "comments": 1,
        "shares": 0,
        "text_length": 42,
        "metadata": {
            "upvote_ratio": 0.91,
            "score": 25
        }
    }
    """
    text = reddit_post.get("posts", "")
    
    return {
        "post_id": reddit_post.get("submission_id", ""),
        "user_id": reddit_post.get("author_name", ""),
        "username": reddit_post.get("author_name", ""),
        "text": text,
        "timestamp": reddit_post.get("created_utc", ""),
        "platform": "reddit",
        "subreddit": reddit_post.get("subreddit", ""),
        "upvotes": reddit_post.get("score", 0),
        "comments": reddit_post.get("num_comments", 0),
        "shares": 0,  # Reddit doesn't have shares
        "text_length": len(text),
        "metadata": {
            "upvote_ratio": reddit_post.get("upvote_ratio", 0.0),
            "score": reddit_post.get("score", 0),
            "submission_id": reddit_post.get("submission_id", ""),
            "created_utc": reddit_post.get("created_utc", "")
        }
    }


def extract_posts_from_nested_structure(data: Any) -> List[Dict[str, Any]]:
    """
    Extract posts from nested data structures.
    
    Handles formats like:
    {
        "user_id_1": {"posts": [...]},
        "user_id_2": {"posts": [...]}
    }
    
    Args:
        data: Data in various formats
        
    Returns:
        Flat list of posts
    """
    posts = []
    
    # If it's already a list, return it
    if isinstance(data, list):
        return data
    
    # If it's a dict with a 'posts' key at top level
    if isinstance(data, dict) and 'posts' in data:
        return data['posts']
    
    # If it's a nested dict with user IDs as keys
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                # Check if this dict has a 'posts' key
                if 'posts' in value and isinstance(value['posts'], list):
                    posts.extend(value['posts'])
                # Check if this dict itself looks like a post
                elif 'submission_id' in value or 'post_id' in value:
                    posts.append(value)
    
    return posts


def transform_posts_batch(posts: List[Dict[str, Any]], format_type: str = "auto") -> List[Dict[str, Any]]:
    """
    Transform a batch of posts to standard format.
    
    Args:
        posts: List of posts in various formats
        format_type: Format type ("reddit", "twitter", "auto")
        
    Returns:
        List of posts in standard format
    """
    if not posts:
        return []
    
    # Auto-detect format if needed
    if format_type == "auto":
        first_post = posts[0]
        if "submission_id" in first_post and "author_name" in first_post:
            format_type = "reddit"
        elif "post_id" in first_post and "user_id" in first_post:
            # Already in standard format
            return posts
        else:
            # Unknown format, return as-is
            return posts
    
    # Transform based on format
    if format_type == "reddit":
        return [transform_reddit_post(post) for post in posts]
    
    # Default: return as-is
    return posts


def validate_post_format(post: Dict[str, Any]) -> bool:
    """
    Validate that a post has the required fields.
    
    Args:
        post: Post dictionary to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ["post_id", "text"]
    return all(field in post for field in required_fields)


def enrich_post_metadata(post: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enrich post with additional metadata fields.
    
    Args:
        post: Post dictionary
        
    Returns:
        Enriched post dictionary
    """
    # Add text_length if not present
    if "text_length" not in post and "text" in post:
        post["text_length"] = len(post["text"])
    
    # Add engagement metrics if not present
    if "engagement" not in post:
        post["engagement"] = {
            "upvotes": post.get("upvotes", 0),
            "comments": post.get("comments", 0),
            "shares": post.get("shares", 0)
        }
    
    # Ensure metadata exists
    if "metadata" not in post:
        post["metadata"] = {}
    
    # Add platform info to metadata
    if "platform" in post:
        post["metadata"]["platform"] = post["platform"]
    
    if "subreddit" in post:
        post["metadata"]["subreddit"] = post["subreddit"]
    
    return post
