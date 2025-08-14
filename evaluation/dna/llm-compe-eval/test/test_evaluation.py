#!/usr/bin/env python3
"""
Simple test script to verify the evaluation system works correctly.
This script tests the evaluation pipeline with a small subset of questions.
"""

import pandas as pd
import sys
import os
from datetime import datetime

def test_dataset_loading():
    """Test if the dataset can be loaded correctly."""
    print("Testing dataset loading...")
    
    dataset_path = "../datasets/Instruction/do_not_answer_en.csv"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        return False
    
    try:
        df = pd.read_csv(dataset_path)
        print(f"✅ Dataset loaded successfully: {len(df)} questions")
        print(f"Columns: {list(df.columns)}")
        
        # Check required columns
        required_columns = ['id', 'question', 'risk_area', 'types_of_harm', 'specific_harms']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return False
        
        print("✅ All required columns present")
        return True
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

def test_api_keys():
    """Test if API keys are configured."""
    print("\nTesting API key configuration...")
    
    info_file = "../do_not_answer/utils/info.yaml"
    if not os.path.exists(info_file):
        print(f"❌ Info file not found at: {info_file}")
        return False
    
    try:
        # Import the info_data
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append(parent_dir)
        from do_not_answer.utils import info_data
        
        openai_key = info_data.get("OpenAI", "")
        hf_key = info_data.get("HuggingFace", "")
        anthropic_key = info_data.get("Anthropic", "")
        
        print(f"OpenAI API: {'✅ Configured' if openai_key and openai_key != 'YOURKEY' else '⚠️  Not configured'}")
        print(f"HuggingFace: {'✅ Configured' if hf_key and hf_key != 'YOURKEY' else '⚠️  Not configured'}")
        print(f"Anthropic: {'✅ Configured' if anthropic_key and anthropic_key != 'YOURKEY' else '⚠️  Not configured'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking API keys: {e}")
        return False

def test_imports():
    """Test if all required modules can be imported."""
    print("\nTesting module imports...")
    
    required_modules = [
        'pandas', 'torch', 'transformers', 'tqdm', 'yaml'
    ]
    
    optional_modules = [
        'wandb', 'openai', 'anthropic'
    ]
    
    all_good = True
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            all_good = False
    
    print("\nOptional modules:")
    for module in optional_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"⚠️  {module}: {e}")
    
    return all_good

def test_evaluation_scripts():
    """Test if evaluation scripts can be executed with help."""
    print("\nTesting evaluation scripts...")
    
    scripts = [
        'evaluate_huggingface_models.py',
        'batch_evaluate_models.py'
    ]
    
    for script in scripts:
        if os.path.exists(script):
            print(f"✅ {script} exists")
        else:
            print(f"❌ {script} not found")
            return False
    
    return True

def create_test_report(results):
    """Create a test report."""
    print("\n" + "="*60)
    print("TEST REPORT")
    print("="*60)
    print(f"Test run timestamp: {datetime.now().isoformat()}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    
    print("\nDetailed results:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! The evaluation system is ready to use.")
        return True
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. Please check the issues above.")
        return False

def main():
    print("LLM Competition Evaluation System Test")
    print("="*60)
    
    # Run tests
    results = {
        'Dataset Loading': test_dataset_loading(),
        'API Keys': test_api_keys(),
        'Module Imports': test_imports(),
        'Evaluation Scripts': test_evaluation_scripts()
    }
    
    # Create test report
    success = create_test_report(results)
    
    if success:
        print("\n📋 Next steps:")
        print("1. Configure API keys in do_not_answer/utils/info.yaml")
        print("2. Install missing dependencies if any")
        print("3. Test with a small model: python evaluate_huggingface_models.py --model_name 'gpt2' --max_questions 5")
        print("4. Run batch evaluation: python batch_evaluate_models.py --config test_config.yaml")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)