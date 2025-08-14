#!/usr/bin/env python3
"""
Batch evaluation script for multiple Hugging Face models in the LLM competition.
This script can evaluate multiple models listed in a configuration file.
"""

import argparse
import json
import os
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List
import subprocess
import sys
from datetime import datetime
import uuid
import getpass

def load_model_config(config_path: str) -> Dict:
    """Load model configuration from YAML or JSON file."""
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(f)
        else:
            config = json.load(f)
    
    # Handle new format with individual model configs
    if 'model_configs' in config:
        # New format: load individual model configs
        models = []
        for model_config_path in config['model_configs']:
            with open(model_config_path, 'r') as f:
                if model_config_path.endswith('.yaml') or model_config_path.endswith('.yml'):
                    model_config = yaml.safe_load(f)
                else:
                    model_config = json.load(f)
                
                # Extract model and evaluation settings
                models.append(model_config['model'])
                # Use the first model's evaluation settings as base
                if not models or len(models) == 1:
                    evaluation_settings = model_config['evaluation_settings']
        
        # Return in the old format for compatibility
        return {
            'models': models,
            'evaluation_settings': evaluation_settings,
            'batch_settings': config.get('batch_settings', {})
        }
    else:
        # Legacy format: return as-is
        return config

def create_sample_config(output_path: str = "model_config.yaml"):
    """Create a sample configuration file for model evaluation with individual model config directories."""
    from datetime import datetime
    
    # Create main config directory if it doesn't exist
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    sample_models = [
        {
            'name': 'team1/llama2-7b-chat-finetune',
            'description': 'Team 1 fine-tuned LLaMA2-7B-Chat model',
            'system_prompt': None  # Use default
        },
        {
            'name': 'team2/mistral-7b-instruct-v0.2',
            'description': 'Team 2 Mistral-7B-Instruct model',
            'system_prompt': 'Custom system prompt here'
        }
    ]
    
    # Create individual config files for each model
    model_config_paths = []
    
    for model in sample_models:
        model_name = model['name']
        # Sanitize model name for directory/file name
        safe_model_name = model_name.replace('/', '_').replace('\\', '_').replace(':', '_')
        
        # Always generate UUID prefix for uniqueness and reproducibility
        unique_id = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
        safe_model_name_with_uuid = f"{unique_id}_{safe_model_name}"
        model_dir = config_dir / safe_model_name_with_uuid
        
        model_dir.mkdir(exist_ok=True)
        
        # Create individual model config
        model_config = {
            'model': model,
            'evaluation_settings': {
                'dataset_path': '../../datasets/Instruction/do_not_answer_en.csv',
                'output_dir': f'./evaluation_results/{safe_model_name}',
                'max_questions': None,  # null for all questions
                'wandb_project': 'llm-competition-do-not-answer',
                'log_wandb': True
            }
        }
        
        # Save individual model config
        model_config_file = model_dir / "config.yaml"
        with open(model_config_file, 'w') as f:
            yaml.dump(model_config, f, default_flow_style=False, indent=2)
        
        model_config_paths.append(str(model_config_file))
        print(f"Model config created: {model_config_file}")
        
        # Always notify the user about UUID assignment
        current_user = getpass.getuser()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"\n📋 UUID Assignment Notice")
        print(f"Model: {model['name']}")
        print(f"UUID: {unique_id}")
        print(f"Directory: {model_dir}")
        print(f"User: {current_user}")
        print(f"📝 Please record this UUID for future reference!")
        
        # Log the assignment
        log_uuid_assignment(model['name'], unique_id, model_dir, current_user, timestamp)
    
    # Create main batch config that references individual configs
    batch_config = {
        'model_configs': model_config_paths,
        'batch_settings': {
            'parallel_execution': False,
            'skip_failed': True,
            'timeout_hours': 2
        }
    }
    
    # Save main batch config
    with open(output_path, 'w') as f:
        yaml.dump(batch_config, f, default_flow_style=False, indent=2)
    
    print(f"\nMain batch configuration created at: {output_path}")
    print(f"Individual model configs created in: {config_dir}")
    print(f"Total model configs: {len(model_config_paths)}")
    
    return batch_config

def add_model_config(model_name: str, description: str = "", system_prompt: str = None):
    """Add a new model configuration to the config directory."""
    from datetime import datetime
    
    # Create main config directory if it doesn't exist
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    # Sanitize model name for directory/file name
    safe_model_name = model_name.replace('/', '_').replace('\\', '_').replace(':', '_')
    
    # Always generate UUID prefix for uniqueness and reproducibility
    unique_id = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
    safe_model_name_with_uuid = f"{unique_id}_{safe_model_name}"
    model_dir = config_dir / safe_model_name_with_uuid
    
    model_dir.mkdir(exist_ok=True)
    
    # Create individual model config
    model_config = {
        'model': {
            'name': model_name,
            'description': description or f'Configuration for {model_name}',
            'system_prompt': system_prompt
        },
        'evaluation_settings': {
            'dataset_path': '../../datasets/Instruction/do_not_answer_en.csv',
            'output_dir': f'./evaluation_results/{safe_model_name}',
            'max_questions': None,  # null for all questions
            'wandb_project': 'llm-competition-do-not-answer',
            'log_wandb': True
        }
    }
    
    # Save individual model config
    model_config_file = model_dir / "config.yaml"
    with open(model_config_file, 'w') as f:
        yaml.dump(model_config, f, default_flow_style=False, indent=2)
    
    print(f"Model config created: {model_config_file}")
    
    # Always notify the user about UUID assignment
    current_user = getpass.getuser()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"\n{'='*60}")
    print(f"📋 UUID Assignment Notice")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"UUID: {unique_id}")
    print(f"Directory: {model_dir}")
    print(f"User: {current_user}")
    print(f"Timestamp: {timestamp}")
    print(f"{'='*60}")
    print(f"📝 Please record this UUID for future reference!")
    print(f"   This ensures you can identify your model configuration.")
    print(f"{'='*60}\n")
    
    # Always log UUID assignment to a file
    log_uuid_assignment(model_name, unique_id, model_dir, current_user, timestamp)
    
    return str(model_config_file)

def log_uuid_assignment(model_name: str, unique_id: str, model_dir: Path, user: str, timestamp: str):
    """Log UUID assignment to a file for record keeping."""
    log_file = Path("config") / "uuid_assignments.log"
    
    log_entry = f"{timestamp} | User: {user} | Model: {model_name} | UUID: {unique_id} | Directory: {model_dir}\n"
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_entry)
    
    print(f"📋 UUID assignment logged to: {log_file}")

def list_uuid_assignments():
    """List all UUID assignments from the log file."""
    log_file = Path("config") / "uuid_assignments.log"
    
    if not log_file.exists():
        print("No UUID assignments found. Log file does not exist.")
        return
    
    print(f"\n{'='*80}")
    print(f"UUID ASSIGNMENTS LOG")
    print(f"{'='*80}")
    
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if not lines:
        print("No UUID assignments recorded.")
        return
    
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if line:
            parts = line.split(' | ')
            if len(parts) >= 5:
                timestamp = parts[0]
                user = parts[1].replace('User: ', '')
                model = parts[2].replace('Model: ', '')
                uuid_val = parts[3].replace('UUID: ', '')
                directory = parts[4].replace('Directory: ', '')
                
                print(f"{i:2d}. {timestamp}")
                print(f"    User: {user}")
                print(f"    Model: {model}")
                print(f"    UUID: {uuid_val}")
                print(f"    Directory: {directory}")
                print()
    
    print(f"{'='*80}")
    print(f"Total assignments: {len([l for l in lines if l.strip()])}")
    print(f"{'='*80}\n")

def run_evaluation(model_config: Dict, evaluation_settings: Dict) -> Dict:
    """Run evaluation for a single model."""
    model_name = model_config['name']
    print(f"\n{'='*60}")
    print(f"Evaluating model: {model_name}")
    print(f"{'='*60}")
    
    # Prepare command
    cmd = [
        sys.executable, 'evaluate_huggingface_models.py',
        '--model_name', model_name,
        '--dataset_path', evaluation_settings['dataset_path'],
        '--output_dir', evaluation_settings['output_dir'],
        '--wandb_project', evaluation_settings['wandb_project']
    ]
    
    if model_config.get('system_prompt'):
        cmd.extend(['--system_prompt', model_config['system_prompt']])
    
    if evaluation_settings.get('max_questions'):
        cmd.extend(['--max_questions', str(evaluation_settings['max_questions'])])
    
    if evaluation_settings.get('log_wandb', False):
        cmd.append('--log_wandb')
    
    # Run evaluation
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hour timeout
        
        if result.returncode == 0:
            print(f"✅ Successfully evaluated {model_name}")
            status = 'success'
            error_message = None
        else:
            print(f"❌ Error evaluating {model_name}")
            print(f"STDERR: {result.stderr}")
            status = 'error'
            error_message = result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout evaluating {model_name}")
        status = 'timeout'
        error_message = 'Evaluation timed out after 2 hours'
    except Exception as e:
        print(f"❌ Exception evaluating {model_name}: {e}")
        status = 'exception'
        error_message = str(e)
    
    return {
        'model_name': model_name,
        'status': status,
        'error_message': error_message,
        'timestamp': datetime.now().isoformat()
    }

def generate_summary_report(results: List[Dict], output_dir: str):
    """Generate a summary report of all evaluations."""
    summary_data = []
    
    for result in results:
        model_name = result['model_name']
        model_name_safe = model_name.replace('/', '_').replace('\\', '_')
        
        # Try to load evaluation results
        eval_file = os.path.join(output_dir, f"evaluation_{model_name_safe}.json")
        
        summary_entry = {
            'model_name': model_name,
            'status': result['status'],
            'timestamp': result['timestamp']
        }
        
        if result['status'] == 'success' and os.path.exists(eval_file):
            try:
                with open(eval_file, 'r') as f:
                    eval_data = json.load(f)
                
                summary_entry['total_questions'] = eval_data.get('total_questions', 'N/A')
                
                if 'gpt4_evaluation' in eval_data:
                    gpt4_results = eval_data['gpt4_evaluation']
                    summary_entry['safety_rate'] = gpt4_results.get('safety_rate', 'N/A')
                    summary_entry['harmful_responses'] = gpt4_results.get('harmful_responses', 'N/A')
                else:
                    summary_entry['safety_rate'] = 'No GPT-4 eval'
                    summary_entry['harmful_responses'] = 'No GPT-4 eval'
                    
            except Exception as e:
                summary_entry['error'] = f"Error loading results: {e}"
        else:
            summary_entry['safety_rate'] = 'Failed'
            summary_entry['harmful_responses'] = 'Failed'
            if result['error_message']:
                summary_entry['error'] = result['error_message']
        
        summary_data.append(summary_entry)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary
    summary_file = os.path.join(output_dir, 'evaluation_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    
    # Create leaderboard
    leaderboard_data = []
    for entry in summary_data:
        if entry['status'] == 'success' and isinstance(entry.get('safety_rate'), (int, float)):
            leaderboard_data.append({
                'rank': 0,  # Will be filled after sorting
                'model_name': entry['model_name'],
                'safety_rate': entry['safety_rate'],
                'harmful_responses': entry['harmful_responses']
            })
    
    # Sort by safety rate (descending)
    leaderboard_data.sort(key=lambda x: x['safety_rate'], reverse=True)
    
    # Add ranks
    for i, entry in enumerate(leaderboard_data):
        entry['rank'] = i + 1
    
    leaderboard_df = pd.DataFrame(leaderboard_data)
    leaderboard_file = os.path.join(output_dir, 'leaderboard.csv')
    leaderboard_df.to_csv(leaderboard_file, index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("BATCH EVALUATION SUMMARY")
    print("="*80)
    
    successful = len([r for r in results if r['status'] == 'success'])
    total = len(results)
    
    print(f"Total models evaluated: {total}")
    print(f"Successful evaluations: {successful}")
    print(f"Failed evaluations: {total - successful}")
    
    if leaderboard_data:
        print(f"\nTop 3 models by safety rate:")
        for entry in leaderboard_data[:3]:
            print(f"  {entry['rank']}. {entry['model_name']}: {entry['safety_rate']:.2f}%")
    
    print(f"\nDetailed results saved to: {summary_file}")
    print(f"Leaderboard saved to: {leaderboard_file}")

def main():
    parser = argparse.ArgumentParser(description="Batch evaluate multiple Hugging Face models")
    parser.add_argument(
        "--config",
        default="model_config.yaml",
        help="Path to model configuration file (YAML or JSON)"
    )
    parser.add_argument(
        "--create_sample_config",
        action="store_true",
        help="Create a sample configuration file and exit"
    )
    parser.add_argument(
        "--output_dir",
        help="Override output directory from config"
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        help="Override max questions from config"
    )
    parser.add_argument(
        "--skip_failed",
        action="store_true",
        help="Skip models that have already been evaluated successfully"
    )
    parser.add_argument(
        "--add_model",
        help="Add a new model configuration (format: 'model_name,description,system_prompt')"
    )
    parser.add_argument(
        "--list_uuids",
        action="store_true",
        help="List all UUID assignments from the log file"
    )
    
    args = parser.parse_args()
    
    if args.create_sample_config:
        create_sample_config(args.config)
        return
    
    if args.list_uuids:
        list_uuid_assignments()
        return
    
    if args.add_model:
        parts = args.add_model.split(',', 2)
        if len(parts) < 1:
            print("Error: Invalid format for --add_model. Use: 'model_name,description,system_prompt'")
            return
        
        model_name = parts[0].strip()
        description = parts[1].strip() if len(parts) > 1 else ""
        system_prompt = parts[2].strip() if len(parts) > 2 else None
        
        if not system_prompt or system_prompt.lower() == 'none':
            system_prompt = None
        
        config_file = add_model_config(model_name, description, system_prompt)
        print(f"Model configuration added successfully: {config_file}")
        return
    
    # Load configuration
    config = load_model_config(args.config)
    models = config['models']
    evaluation_settings = config['evaluation_settings']
    
    # Override settings from command line
    if args.output_dir:
        evaluation_settings['output_dir'] = args.output_dir
    if args.max_questions:
        evaluation_settings['max_questions'] = args.max_questions
    
    # Create output directory
    os.makedirs(evaluation_settings['output_dir'], exist_ok=True)
    
    print(f"Starting batch evaluation of {len(models)} models")
    print(f"Output directory: {evaluation_settings['output_dir']}")
    
    # Run evaluations
    results = []
    
    for i, model_config in enumerate(models, 1):
        model_name = model_config['name']
        
        # Skip if already evaluated successfully
        if args.skip_failed:
            model_name_safe = model_name.replace('/', '_').replace('\\', '_')
            eval_file = os.path.join(evaluation_settings['output_dir'], f"evaluation_{model_name_safe}.json")
            if os.path.exists(eval_file):
                print(f"Skipping {model_name} (already evaluated)")
                results.append({
                    'model_name': model_name,
                    'status': 'skipped',
                    'error_message': None,
                    'timestamp': datetime.now().isoformat()
                })
                continue
        
        print(f"\nProgress: {i}/{len(models)}")
        result = run_evaluation(model_config, evaluation_settings)
        results.append(result)
    
    # Generate summary report
    generate_summary_report(results, evaluation_settings['output_dir'])
    
    print(f"\nBatch evaluation completed!")
    print(f"Check {evaluation_settings['output_dir']} for detailed results")

if __name__ == "__main__":
    main()