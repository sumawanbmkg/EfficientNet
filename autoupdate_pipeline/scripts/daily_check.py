#!/usr/bin/env python
"""
Daily Pipeline Check Script

Script untuk pengecekan harian status pipeline dan trigger otomatis.
Dapat dijalankan via cron job atau Windows Task Scheduler.

Usage:
    python scripts/daily_check.py              # Check status only
    python scripts/daily_check.py --run        # Run pipeline if conditions met
    python scripts/daily_check.py --run --auto-deploy  # Auto-deploy if challenger wins
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, load_registry, check_trigger_conditions, log_pipeline_event

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("daily_check")


def check_status() -> dict:
    """Check pipeline status and return summary."""
    config = load_config()
    registry = load_registry()
    
    champion = registry.get('champion', {})
    pending = registry.get('pending_events', {})
    validated = registry.get('validated_events', [])
    history = registry.get('pipeline_history', {})
    
    # Check trigger conditions
    trigger = check_trigger_conditions(registry, config)
    
    status = {
        "timestamp": datetime.now().isoformat(),
        "champion": {
            "version": champion.get('version', 'N/A'),
            "magnitude_accuracy": champion.get('metrics', {}).get('magnitude_accuracy', 0),
            "azimuth_accuracy": champion.get('metrics', {}).get('azimuth_accuracy', 0),
            "deployed_at": champion.get('deployed_at', 'N/A')
        },
        "events": {
            "pending": pending.get('count', 0),
            "validated": len(validated),
            "min_required": trigger.get('min_required', 5)
        },
        "trigger": {
            "should_trigger": trigger.get('should_trigger', False),
            "reasons": trigger.get('reasons', []),
            "days_since_last_run": trigger.get('days_since_last_run', 0)
        },
        "history": {
            "total_runs": history.get('total_runs', 0),
            "successful_updates": history.get('successful_updates', 0),
            "last_run": history.get('last_run', 'Never')
        }
    }
    
    return status


def print_status(status: dict):
    """Print status in human-readable format."""
    print("\n" + "=" * 60)
    print(f"📊 DAILY PIPELINE CHECK - {status['timestamp'][:10]}")
    print("=" * 60)
    
    # Champion info
    champ = status['champion']
    print(f"\n🏆 Champion Model: v{champ['version']}")
    print(f"   Magnitude Accuracy: {champ['magnitude_accuracy']:.2f}%")
    print(f"   Azimuth Accuracy:   {champ['azimuth_accuracy']:.2f}%")
    
    # Events
    events = status['events']
    print(f"\n📥 Events Status:")
    print(f"   Pending:   {events['pending']}")
    print(f"   Validated: {events['validated']}")
    print(f"   Required:  {events['min_required']}")
    
    # Trigger status
    trigger = status['trigger']
    trigger_icon = "✅" if trigger['should_trigger'] else "⏳"
    print(f"\n⚡ Trigger Status: {trigger_icon}")
    print(f"   Ready to Run: {'YES' if trigger['should_trigger'] else 'NO'}")
    if trigger['reasons']:
        print(f"   Reasons: {', '.join(trigger['reasons'])}")
    if trigger['days_since_last_run']:
        print(f"   Days Since Last Run: {trigger['days_since_last_run']}")
    
    # History
    hist = status['history']
    print(f"\n📈 Pipeline History:")
    print(f"   Total Runs: {hist['total_runs']}")
    print(f"   Successful: {hist['successful_updates']}")
    print(f"   Last Run:   {hist['last_run']}")
    
    print("\n" + "=" * 60)


def run_pipeline_if_ready(auto_deploy: bool = False, force: bool = False) -> bool:
    """Run pipeline if trigger conditions are met."""
    status = check_status()
    
    if not status['trigger']['should_trigger'] and not force:
        logger.info("Pipeline not ready to run. Use --force to override.")
        return False
    
    logger.info("Trigger conditions met. Starting pipeline...")
    
    # Import and run pipeline
    from run_pipeline import PipelineRunner
    
    runner = PipelineRunner()
    results = runner.run(force=force, auto_deploy=auto_deploy)
    
    return results.get('success', False)


def main():
    parser = argparse.ArgumentParser(description="Daily pipeline check")
    parser.add_argument("--run", action="store_true", 
                       help="Run pipeline if conditions are met")
    parser.add_argument("--auto-deploy", action="store_true",
                       help="Auto-deploy if challenger wins")
    parser.add_argument("--force", action="store_true",
                       help="Force run even if conditions not met")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Quiet mode - minimal output")
    
    args = parser.parse_args()
    
    # Check status
    status = check_status()
    
    if not args.quiet:
        print_status(status)
    
    # Log daily check
    log_pipeline_event("daily_check", {
        "trigger_ready": status['trigger']['should_trigger'],
        "validated_events": status['events']['validated'],
        "pending_events": status['events']['pending']
    })
    
    # Run pipeline if requested
    if args.run:
        success = run_pipeline_if_ready(
            auto_deploy=args.auto_deploy,
            force=args.force
        )
        return 0 if success else 1
    
    # Return exit code based on trigger status
    return 0 if status['trigger']['should_trigger'] else 1


if __name__ == "__main__":
    sys.exit(main())
