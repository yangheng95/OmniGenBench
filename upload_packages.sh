#!/bin/bash
# -*- coding: utf-8 -*-
# file: upload_packages.sh
# Upload both omnigenbench and omnigenbench packages to PyPI

set -e  # Exit on any error

echo "🚀 Starting package upload process..."

# Function to upload a package
upload_package() {
    local setup_file=$1
    local package_name=$2

    echo "📦 Building and uploading $package_name..."

    # Clean previous builds
    rm -rf build/ dist/ *.egg-info/

    # Build the package
    python $setup_file sdist bdist_wheel

    # Upload to PyPI (use --repository testpypi for testing)
    python -m twine upload dist/*

    # Clean up
    rm -rf build/ dist/ *.egg-info/

    echo "✅ $package_name uploaded successfully!"
}

# Upload omnigenbench package
upload_package "setup_omnigenome.py" "omnigenome"

# Upload omnigenbench package
upload_package "setup_omnigenbench.py" "omnigenbench"

echo "🎉 All packages uploaded successfully!"
