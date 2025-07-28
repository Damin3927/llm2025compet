#!/usr/bin/env python3
"""
Test script for length_selector.py to verify it correctly preserves all fields.
"""

import json
import os
import tempfile
import subprocess
import sys

def create_test_data():
    """Create test data with question, answer, and solution fields."""
    test_data = [
        {
            "question": "What is 2 + 2?",
            "answer": "4",
            "solution": "Add 2 and 2 to get 4.",  # Length: 21
            "extra_field": "This should be ignored"
        },
        {
            "question": "What is the square root of 16?",
            "answer": "4",
            "solution": "The square root of 16 is 4 because 4 × 4 = 16.",  # Length: 50
            "other_field": "Also ignored"
        },
        {
            "question": "Solve for x: 2x + 5 = 15",
            "answer": "5",
            "solution": "Subtract 5 from both sides: 2x = 10. Divide by 2: x = 5.",  # Length: 61
            "metadata": "Not included"
        },
        {
            "question": "What is the area of a circle with radius 3?",
            "answer": "9π",
            "solution": "The area formula is A = πr². With r = 3: A = π(3)² = 9π.",  # Length: 63
            "category": "geometry"
        },
        {
            "question": "What is the derivative of x²?",
            "answer": "2x",
            "solution": "Using the power rule: d/dx(x²) = 2x^(2-1) = 2x.",  # Length: 53
            "difficulty": "basic"
        }
    ]
    return test_data

def test_length_selector():
    """Test the length_selector script."""
    print("=== Testing length_selector.py ===\n")
    
    # Create test data
    test_data = create_test_data()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Write test data to file
        input_file = os.path.join(temp_dir, "test_input.json")
        output_file = os.path.join(temp_dir, "test_output.json")
        
        with open(input_file, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        print(f"Created test input with {len(test_data)} items")
        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")
        
        # Run length_selector
        cmd = [
            sys.executable, "length_selector.py",
            "--input", input_file,
            "--output", output_file,
            "--question_field", "question",
            "--answer_field", "answer", 
            "--solution_field", "solution",
            "--total_samples", "3",
            "--id_header", "test"
        ]
        
        print(f"\nRunning command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            
            print(f"\nReturn code: {result.returncode}")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            
            if result.returncode == 0 and os.path.exists(output_file):
                # Read and verify output
                with open(output_file, 'r') as f:
                    output_data = json.load(f)
                
                print(f"\n=== OUTPUT VERIFICATION ===")
                print(f"Output contains {len(output_data)} items")
                
                if output_data:
                    print(f"\nSample output item:")
                    sample = output_data[0]
                    print(json.dumps(sample, indent=2))
                    
                    # Verify the required output format
                    required_fields = ["id", "question", "output", "answer"]
                    missing_fields = [f for f in required_fields if f not in sample]
                    extra_fields = [f for f in sample.keys() if f not in required_fields]
                    
                    # Check if id follows the expected format (id_header_index)
                    sample_id = sample.get("id", "")
                    id_valid = isinstance(sample_id, str) and sample_id.startswith("test_") and sample_id.split("_")[-1].isdigit()
                    
                    if not missing_fields and not extra_fields and id_valid:
                        print("✅ SUCCESS: Correct output format with id, question, output, answer")
                    else:
                        if missing_fields:
                            print(f"❌ ERROR: Missing fields: {missing_fields}")
                        if extra_fields:
                            print(f"⚠️  WARNING: Extra fields found: {extra_fields}")
                        if not id_valid:
                            print(f"❌ ERROR: Invalid id field: {sample.get('id')}")
                    
                    # Verify the field mapping
                    print(f"\nField mapping verification:")
                    print(f"  id: {sample.get('id')} (should be test_1, test_2, etc.)")
                    print(f"  question: '{sample.get('question', 'MISSING')[:30]}...'")
                    print(f"  output: '{sample.get('output', 'MISSING')[:30]}...'")
                    print(f"  answer: '{sample.get('answer', 'MISSING')}'")
                    
                    # Check if all items have sequential IDs with correct header
                    ids = [item.get("id", "") for item in output_data]
                    expected_ids = [f"test_{i}" for i in range(1, len(output_data) + 1)]
                    if ids == expected_ids:
                        print(f"✅ Sequential IDs correct: {ids}")
                    else:
                        print(f"❌ Sequential IDs incorrect: got {ids}, expected {expected_ids}")
            else:
                print("❌ ERROR: Script failed or output file not created")
                
        except Exception as e:
            print(f"❌ ERROR running test: {e}")

if __name__ == "__main__":
    test_length_selector()