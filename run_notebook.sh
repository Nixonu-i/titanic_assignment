#!/bin/bash
# Launch Jupyter Notebook from the project root for correct relative paths

cd "$(dirname "$0")"

echo "Starting Jupyter Notebook from project root..."
echo "This ensures '../data/train.csv' paths work correctly."
echo ""

# Activate virtual environment if it exists
if [ -d "jupyter_env/bin" ]; then
    source jupyter_env/bin/activate
    echo "✓ Activated virtual environment: jupyter_env"
fi

# Start Jupyter
jupyter notebook notebooks/Titanic_Feature_Engineering.ipynb
