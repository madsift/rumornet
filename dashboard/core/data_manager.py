"""
Data manager for execution history persistence.

This module handles saving and loading execution results to/from disk,
providing persistent storage for analysis history.
"""

import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from models.data_models import ExecutionResult, AgentStatus

# Import boto3 exceptions at module level for proper scoping
try:
    import boto3
    from botocore.exceptions import NoCredentialsError, ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    NoCredentialsError = Exception  # Fallback
    ClientError = Exception  # Fallback


class DataManager:
    """
    Manages execution history persistence using JSON files.
    
    Provides methods for saving, loading, and managing execution results
    with automatic cleanup and history management.
    """
    
    def __init__(self, data_dir: str = "dashboard/data/history"):
        """
        Initialize the data manager.
        
        Args:
            data_dir: Directory path for storing execution history files
        """
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(f"{__name__}.data_manager")
        
        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"DataManager initialized with directory: {self.data_dir}")
    
    def save_execution_result(self, result: ExecutionResult) -> bool:
        """
        Save an execution result to disk.
        
        Args:
            result: ExecutionResult to save
            
        Returns:
            True if save was successful, False otherwise
        """
        try:
            # Create filename from execution ID
            filename = f"{result.execution_id}.json"
            filepath = self.data_dir / filename
            
            # Convert to dictionary
            result_dict = result.to_dict()
            
            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved execution result: {result.execution_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save execution result {result.execution_id}: {e}")
            return False
    
    def load_execution_result(self, execution_id: str, try_s3: bool = True) -> Optional[ExecutionResult]:
        """
        Load a specific execution result by ID from local storage or S3.
        
        Args:
            execution_id: ID of the execution to load
            try_s3: Whether to try loading from S3 if not found locally
            
        Returns:
            ExecutionResult if found, None otherwise
        """
        try:
            # Try local file first
            filename = f"{execution_id}.json"
            filepath = self.data_dir / filename
            
            if filepath.exists():
                # Read from local file
                with open(filepath, 'r', encoding='utf-8') as f:
                    result_dict = json.load(f)
                
                # Convert to ExecutionResult
                result = ExecutionResult.from_dict(result_dict)
                
                self.logger.info(f"Loaded execution result from local: {execution_id}")
                return result
            
            # If not found locally and S3 is enabled, try S3
            if try_s3 and BOTO3_AVAILABLE:
                try:
                    s3_client = boto3.client('s3')
                    bucket = 'rumornet'
                    
                    # Try to find the file in S3 with the execution_id
                    # S3 files might be named differently, so we need to search
                    prefix = 'misinformation-detection/reports/'
                    
                    # List objects to find matching execution_id
                    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
                    
                    if 'Contents' in response:
                        for obj in response['Contents']:
                            key = obj['Key']
                            # Check if this key contains our execution_id
                            if execution_id in key and key.endswith('.json'):
                                # Download and parse
                                s3_response = s3_client.get_object(Bucket=bucket, Key=key)
                                result_dict = json.loads(s3_response['Body'].read())
                                
                                # Convert S3 structure to ExecutionResult
                                report = result_dict.get("report", {})
                                exec_summary = report.get("executive_summary", {})
                                
                                # Generate markdown from report
                                markdown_report = ""
                                if report:
                                    try:
                                        from utils.markdown_generator import MarkdownGenerator
                                        generator = MarkdownGenerator()
                                        markdown_report = generator.generate_markdown_report(report)
                                    except Exception as e:
                                        self.logger.warning(f"Failed to generate markdown for S3 result: {e}")
                                
                                execution_result = ExecutionResult(
                                    execution_id=result_dict.get("job_id", execution_id),
                                    timestamp=datetime.fromisoformat(result_dict.get("stored_at", result_dict.get("completed_at", datetime.now().isoformat()))),
                                    total_posts=result_dict.get("total_posts", 0),
                                    posts_analyzed=exec_summary.get("total_posts_analyzed", 0),
                                    misinformation_detected=exec_summary.get("misinformation_detected", 0),
                                    high_risk_posts=exec_summary.get("high_risk_posts", 0),
                                    execution_time_ms=result_dict.get("execution_time_ms", 0),
                                    agent_statuses={},
                                    full_report=report,
                                    markdown_report=markdown_report
                                )
                                
                                self.logger.info(f"Loaded execution result from S3: {execution_id}")
                                return execution_result
                    
                    self.logger.warning(f"Execution result not found in S3: {execution_id}")
                    
                except (NoCredentialsError, ClientError) as e:
                    self.logger.warning(f"Could not access S3 for execution {execution_id}: {e}")
                except Exception as e:
                    self.logger.warning(f"Failed to load from S3 for execution {execution_id}: {e}")
            
            self.logger.warning(f"Execution result not found: {execution_id}")
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to load execution result {execution_id}: {e}")
            return None
    
    def load_all_execution_results(self, load_from_s3: bool = True) -> List[ExecutionResult]:
        """
        Load all execution results from disk and optionally from S3.
        
        Args:
            load_from_s3: Whether to also load results from S3
        
        Returns:
            List of ExecutionResult objects, sorted by timestamp (newest first)
        """
        results = []
        
        try:
            # Load from local disk
            json_files = list(self.data_dir.glob("*.json"))
            self.logger.info(f"Found {len(json_files)} local execution result files")
            
            for filepath in json_files:
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        result_dict = json.load(f)
                    
                    result = ExecutionResult.from_dict(result_dict)
                    results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"Failed to load file {filepath}: {e}")
                    continue
            
            # Load from S3 if enabled
            if load_from_s3 and BOTO3_AVAILABLE:
                try:
                    # Try to create S3 client - will use environment variables or IAM role
                    s3_client = boto3.client('s3')
                    bucket = 'rumornet'
                    prefix = 'misinformation-detection/reports/'
                    
                    # List all objects in the reports folder
                    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
                    
                    if 'Contents' in response:
                        s3_files = response['Contents']
                        self.logger.info(f"Found {len(s3_files)} S3 report files")
                        
                        for obj in s3_files:
                            key = obj['Key']
                            if key.endswith('.json'):
                                try:
                                    # Download and parse
                                    s3_response = s3_client.get_object(Bucket=bucket, Key=key)
                                    result_dict = json.loads(s3_response['Body'].read())
                                    
                                    # Convert to ExecutionResult
                                    # S3 results have different structure, adapt it
                                    report = result_dict.get("report", {})
                                    exec_summary = report.get("executive_summary", {})
                                    
                                    # Note: We don't generate markdown here for performance
                                    # It will be generated on-demand when viewing details
                                    execution_result = ExecutionResult(
                                        execution_id=result_dict.get("job_id", key.split('/')[-1].replace('.json', '')),
                                        timestamp=datetime.fromisoformat(result_dict.get("stored_at", result_dict.get("completed_at", datetime.now().isoformat()))),
                                        total_posts=result_dict.get("total_posts", 0),
                                        posts_analyzed=exec_summary.get("total_posts_analyzed", 0),
                                        misinformation_detected=exec_summary.get("misinformation_detected", 0),
                                        high_risk_posts=exec_summary.get("high_risk_posts", 0),
                                        execution_time_ms=result_dict.get("execution_time_ms", 0),
                                        agent_statuses={},
                                        full_report=report,
                                        markdown_report=""  # Generated on-demand in load_execution_result
                                    )
                                    
                                    results.append(execution_result)
                                    
                                except Exception as e:
                                    self.logger.error(f"Failed to load S3 file {key}: {e}")
                                    continue
                    else:
                        self.logger.info("No S3 reports found")
                
                except NoCredentialsError:
                    self.logger.warning("AWS credentials not found - S3 history disabled. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables to enable.")
                except ClientError as e:
                    self.logger.warning(f"S3 access error (continuing with local only): {e}")
                except Exception as e:
                    self.logger.warning(f"Failed to load from S3 (continuing with local only): {e}")
            
            # Sort by timestamp (newest first)
            results.sort(key=lambda x: x.timestamp, reverse=True)
            
            self.logger.info(f"Loaded {len(results)} total execution results")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to load execution results: {e}")
            return []
    
    def get_execution_by_id(self, execution_id: str) -> Optional[ExecutionResult]:
        """
        Get a specific execution result by ID.
        
        This is an alias for load_execution_result for consistency with
        the state manager interface.
        
        Args:
            execution_id: ID of the execution to retrieve
            
        Returns:
            ExecutionResult if found, None otherwise
        """
        return self.load_execution_result(execution_id)
    
    def delete_execution_result(self, execution_id: str) -> bool:
        """
        Delete a specific execution result.
        
        Args:
            execution_id: ID of the execution to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            filename = f"{execution_id}.json"
            filepath = self.data_dir / filename
            
            if not filepath.exists():
                self.logger.warning(f"Execution result not found: {execution_id}")
                return False
            
            filepath.unlink()
            self.logger.info(f"Deleted execution result: {execution_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete execution result {execution_id}: {e}")
            return False
    
    def clear_all_history(self) -> bool:
        """
        Clear all execution history.
        
        Returns:
            True if clearing was successful, False otherwise
        """
        try:
            # Get all JSON files
            json_files = list(self.data_dir.glob("*.json"))
            
            # Delete each file
            for filepath in json_files:
                try:
                    filepath.unlink()
                except Exception as e:
                    self.logger.error(f"Failed to delete file {filepath}: {e}")
            
            self.logger.info(f"Cleared {len(json_files)} execution results")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clear history: {e}")
            return False
    
    def cleanup_old_results(self, max_age_days: int = 30) -> int:
        """
        Clean up execution results older than specified days.
        
        Args:
            max_age_days: Maximum age in days for keeping results
            
        Returns:
            Number of results deleted
        """
        deleted_count = 0
        
        try:
            # Get all results
            results = self.load_all_execution_results()
            
            # Calculate cutoff date
            from datetime import timedelta
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            
            # Delete old results
            for result in results:
                if result.timestamp < cutoff_date:
                    if self.delete_execution_result(result.execution_id):
                        deleted_count += 1
            
            self.logger.info(f"Cleaned up {deleted_count} old execution results")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old results: {e}")
            return deleted_count
    
    def limit_history_size(self, max_items: int = 50) -> int:
        """
        Limit the number of stored execution results.
        
        Keeps only the most recent max_items results, deleting older ones.
        
        Args:
            max_items: Maximum number of results to keep
            
        Returns:
            Number of results deleted
        """
        deleted_count = 0
        
        try:
            # Get all results (sorted by timestamp, newest first)
            results = self.load_all_execution_results()
            
            # If we have more than max_items, delete the oldest ones
            if len(results) > max_items:
                results_to_delete = results[max_items:]
                
                for result in results_to_delete:
                    if self.delete_execution_result(result.execution_id):
                        deleted_count += 1
                
                self.logger.info(f"Limited history to {max_items} items, deleted {deleted_count}")
            
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Failed to limit history size: {e}")
            return deleted_count
    
    def get_history_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the execution history.
        
        Returns:
            Dictionary with history statistics
        """
        try:
            results = self.load_all_execution_results()
            
            if not results:
                return {
                    "total_executions": 0,
                    "oldest_execution": None,
                    "newest_execution": None,
                    "total_posts_analyzed": 0,
                    "total_misinformation_detected": 0
                }
            
            # Calculate statistics
            total_posts = sum(r.total_posts for r in results)
            total_misinfo = sum(r.misinformation_detected for r in results)
            
            return {
                "total_executions": len(results),
                "oldest_execution": results[-1].timestamp.isoformat(),
                "newest_execution": results[0].timestamp.isoformat(),
                "total_posts_analyzed": total_posts,
                "total_misinformation_detected": total_misinfo,
                "storage_path": str(self.data_dir)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get history summary: {e}")
            return {
                "total_executions": 0,
                "error": str(e)
            }
    
    def export_history_to_json(self, output_path: str) -> bool:
        """
        Export all execution history to a single JSON file.
        
        Args:
            output_path: Path for the output JSON file
            
        Returns:
            True if export was successful, False otherwise
        """
        try:
            results = self.load_all_execution_results()
            
            # Convert all results to dictionaries
            results_dict = [r.to_dict() for r in results]
            
            # Write to output file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Exported {len(results)} results to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export history: {e}")
            return False
    
    def get_storage_size(self) -> Dict[str, Any]:
        """
        Get information about storage usage.
        
        Returns:
            Dictionary with storage statistics
        """
        try:
            total_size = 0
            file_count = 0
            
            # Calculate total size
            for filepath in self.data_dir.glob("*.json"):
                total_size += filepath.stat().st_size
                file_count += 1
            
            # Convert to human-readable format
            size_mb = total_size / (1024 * 1024)
            
            return {
                "total_files": file_count,
                "total_size_bytes": total_size,
                "total_size_mb": round(size_mb, 2),
                "storage_path": str(self.data_dir)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get storage size: {e}")
            return {
                "total_files": 0,
                "total_size_bytes": 0,
                "error": str(e)
            }
