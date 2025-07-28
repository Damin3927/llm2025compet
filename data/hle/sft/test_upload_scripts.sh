#!/bin/bash

# Test script to demonstrate upload_data.py bash scripts
# This creates sample data and shows how to use the scripts

echo "=== Testing Upload Data Scripts ==="
echo ""

# Create test directory structure
TEST_DIR="./test_upload_data"
mkdir -p "$TEST_DIR/train"
mkdir -p "$TEST_DIR/validation"

# Create sample JSON data
cat > "$TEST_DIR/train/sample1.json" << 'EOF'
[
  {
    "id": "train_1",
    "question": "What is 2+2?",
    "output": "To solve 2+2, I add the two numbers: 2 + 2 = 4",
    "answer": "4"
  },
  {
    "id": "train_2", 
    "question": "What is 3*5?",
    "output": "To solve 3*5, I multiply: 3 × 5 = 15",
    "answer": "15"
  }
]
EOF

cat > "$TEST_DIR/validation/sample1.json" << 'EOF'
[
  {
    "id": "val_1",
    "question": "What is 10-3?",
    "output": "To solve 10-3, I subtract: 10 - 3 = 7",
    "answer": "7"
  }
]
EOF

echo "✅ Created test data structure:"
echo "📁 $TEST_DIR/"
echo "├── 📁 train/"
echo "│   └── 📄 sample1.json (2 items)"
echo "└── 📁 validation/"
echo "    └── 📄 sample1.json (1 item)"
echo ""

echo "=== Script Usage Examples ==="
echo ""

echo "1️⃣  Basic SLURM script (run_upload_data.sh):"
echo "   Edit the configuration variables in the script, then run:"
echo "   sbatch run_upload_data.sh"
echo ""

echo "2️⃣  Universal script (works locally and on SLURM):"
echo "   ./universal_upload_data.sh \"$TEST_DIR\" \"your-username/test-dataset\" true"
echo ""

echo "3️⃣  Dynamic SLURM submission:"
echo "   ./submit_upload_data.sh \"$TEST_DIR\" \"your-username/test-dataset\" true"
echo ""

echo "=== Script Validation Test ==="
echo ""

# Test the universal script validation
echo "Testing universal script validation..."
./universal_upload_data.sh "$TEST_DIR" "invalid-repo-format" true 2>/dev/null
if [ $? -ne 0 ]; then
    echo "✅ Validation correctly caught invalid repository format"
else
    echo "❌ Validation failed to catch invalid repository format"
fi

# Test with non-existent path
./universal_upload_data.sh "./non-existent-path" "user/dataset" false 2>/dev/null
if [ $? -ne 0 ]; then
    echo "✅ Validation correctly caught non-existent path"
else
    echo "❌ Validation failed to catch non-existent path"
fi

echo ""
echo "=== Configuration Examples ==="
echo ""

echo "📝 Common configuration patterns:"
echo ""
echo "For OpenMathReasoning data:"
echo "  DATASET_PATH=\"./results/selected_data/OpenMathReasoning_train\""
echo "  REPO_ID=\"your-username/openmath-sft-data\""
echo ""
echo "For multiple splits:"
echo "  DATASET_PATH=\"./results/processed_dataset\"  # Contains train/, val/, test/"
echo "  REPO_ID=\"your-username/multi-split-dataset\""
echo ""

echo "=== Expected Data Structure ==="
echo ""
echo "Your dataset_path should look like:"
echo "📁 dataset_path/"
echo "├── 📁 train/          # Becomes 'train' split"
echo "│   ├── 📄 file1.json"
echo "│   └── 📄 file2.json"
echo "├── 📁 validation/     # Becomes 'validation' split"
echo "│   └── 📄 file1.json"
echo "└── 📁 test/           # Becomes 'test' split (optional)"
echo "    └── 📄 file1.json"
echo ""

echo "=== Next Steps ==="
echo ""
echo "1. Replace 'your-username/dataset-name' with your actual HF repository"
echo "2. Ensure you're logged in: huggingface-cli login"
echo "3. Create the repository if it doesn't exist:"
echo "   huggingface-cli repo create --type dataset your-username/dataset-name"
echo "4. Run one of the scripts above"
echo ""

echo "🧹 Cleaning up test data..."
rm -rf "$TEST_DIR"
echo "✅ Test completed!"