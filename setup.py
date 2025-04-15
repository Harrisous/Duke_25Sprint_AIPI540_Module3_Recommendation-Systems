import os
import pathlib
import sys
import subprocess
import shutil

def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking dependencies...")
    try:
        import pandas
        import torch
        import fastapi
        import uvicorn
        import google.genai
        import tenacity
        import tqdm
        import sklearn
        print("All required packages are installed.")
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully.")

def create_directories():
    """Create necessary directories if they don't exist"""
    print("Setting up project directories...")
    
    # Define directories to create
    directories = [
        "data/raw",
        "data/processed",
        "data/outputs",
        "models/dnn/models",
        "notebooks",
        "scripts"
    ]
    
    # Create directories
    for directory in directories:
        dir_path = pathlib.Path(directory)
        if not dir_path.exists():
            print(f"Creating directory: {directory}")
            dir_path.mkdir(parents=True, exist_ok=True)

def check_env_file():
    """Check if .env file exists, create from example if it doesn't"""
    if not os.path.exists(".env") and os.path.exists(".env.example"):
        print("Creating .env file from .env.example")
        shutil.copy(".env.example", ".env")
        print("Please edit the .env file to add your API keys")
    elif not os.path.exists(".env"):
        print("WARNING: .env file not found and no .env.example to copy from.")
        print("You will need to create a .env file with your API keys.")
        with open(".env", "w") as f:
            f.write("GOOGLE_API_KEY=your_google_api_key_here\n")

def setup_data():
    """Check if data files exist and provide instructions to download if not"""
    data_dir = pathlib.Path("data/raw/ml-100k")
    if not data_dir.exists() or not any(data_dir.iterdir()):
        print("MovieLens 100K dataset not found.")
        print("Please download the dataset from: https://grouplens.org/datasets/movielens/100k/")
        print(f"Extract it to: {data_dir.absolute()}")
        # Create the directory in case it doesn't exist
        data_dir.mkdir(parents=True, exist_ok=True)

def main():
    """Main setup function"""
    print("Setting up Hybrid Recommendation System project...")
    
    # Check and install dependencies
    check_dependencies()
    
    # Create necessary directories
    create_directories()
    
    # Check .env file
    check_env_file()
    
    # Setup data
    setup_data()
    
    print("\nSetup completed successfully!")
    print("\nNext steps:")
    print("1. Make sure you have the MovieLens 100K dataset in data/raw/ml-100k/")
    print("2. Edit the .env file to add your Google API key")
    print("3. Run the preprocessing scripts: python scripts/make_dataset.py")
    print("4. Train the model: python models/dnn/model_v2.py")
    print("5. Start the API server: python models/dnn/api_server.py")

if __name__ == "__main__":
    main()
