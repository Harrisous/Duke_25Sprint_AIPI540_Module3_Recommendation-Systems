"""
Main entry point for the Hybrid Recommendation System
This script provides a command line interface to run different components of the system
"""
import argparse
import pathlib
import sys
import os

# Define paths
CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
SCRIPTS_DIR = CURRENT_DIR / "scripts"
MODELS_DIR = CURRENT_DIR / "models" / "dnn"

def setup_project():
    """Set up the project structure and dependencies"""
    print("Setting up project environment...")
    try:
        from setup import main as setup_main
        setup_main()
    except ImportError:
        print("Error: Could not import setup module.")
        print("Make sure setup.py is in the project root directory.")
        sys.exit(1)

def prepare_data():
    """Download and prepare the dataset"""
    print("Preparing dataset...")
    try:
        sys.path.append(str(SCRIPTS_DIR))
        from make_dataset import main as make_dataset_main
        make_dataset_main()
    except ImportError:
        print("Error: Could not import make_dataset module.")
        print("Make sure scripts/make_dataset.py exists.")
        sys.exit(1)

def build_features():
    """Generate features for the recommendation system"""
    print("Building features...")
    try:
        sys.path.append(str(SCRIPTS_DIR))
        from build_features import main as build_features_main
        build_features_main()
    except ImportError:
        print("Error: Could not import build_features module.")
        print("Make sure scripts/build_features.py exists.")
        sys.exit(1)

def train_model():
    """Train the recommendation model"""
    print("Training model...")
    try:
        sys.path.append(str(SCRIPTS_DIR))
        from model import main as train_model_main
        train_model_main()
    except ImportError:
        print("Error: Could not import model module.")
        print("Make sure scripts/model.py exists.")
        sys.exit(1)

def run_api():
    """Run the API server for recommendations"""
    print("Starting API server...")
    try:
        # Add model directory to path
        sys.path.append(str(MODELS_DIR))
        
        # Import the server module
        from api_server import app
        import uvicorn
        
        # Run the server
        uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
    except ImportError as e:
        print(f"Error: {e}")
        print("Make sure models/dnn/api_server.py exists and all dependencies are installed.")
        sys.exit(1)

def run_inference(user_id, user_text=None, num_recommendations=10):
    """Run inference to get recommendations for a user"""
    print(f"Getting recommendations for user {user_id}...")
    try:
        # Add model directory to path
        sys.path.append(str(MODELS_DIR))
        
        # Import inference module
        from inference_v2 import get_recommendations
        
        # Get recommendations
        recommendations = get_recommendations(
            user_id=user_id,
            user_text=user_text,
            num_recommendations=num_recommendations
        )
        
        # Print recommendations
        print("\nTop recommendations:")
        for i, (movie_id, rating, desc) in enumerate(recommendations, 1):
            print(f"{i}. Movie ID: {movie_id}, Predicted Rating: {rating:.2f}")
            print(f"   Description: {desc}")
            print()
        
        return recommendations
    except ImportError as e:
        print(f"Error: {e}")
        print("Make sure models/dnn/inference_v2.py exists and all dependencies are installed.")
        sys.exit(1)

def main():
    """Main function to parse arguments and run commands"""
    parser = argparse.ArgumentParser(description="Hybrid Recommendation System")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Set up project environment")
    
    # Data preparation command
    data_parser = subparsers.add_parser("data", help="Prepare dataset")
    
    # Feature building command
    features_parser = subparsers.add_parser("features", help="Build features")
    
    # Model training command
    train_parser = subparsers.add_parser("train", help="Train model")
    
    # API server command
    api_parser = subparsers.add_parser("api", help="Run API server")
    
    # Inference command
    inference_parser = subparsers.add_parser("recommend", help="Get recommendations for a user")
    inference_parser.add_argument("--user_id", type=int, help="User ID")
    inference_parser.add_argument("--user_text", type=str, help="User profile text (for cold start)")
    inference_parser.add_argument("--num", type=int, default=10, help="Number of recommendations")
    
    # Pipeline command to run all steps
    pipeline_parser = subparsers.add_parser("pipeline", help="Run the full pipeline")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run selected command
    if args.command == "setup":
        setup_project()
    elif args.command == "data":
        prepare_data()
    elif args.command == "features":
        build_features()
    elif args.command == "train":
        train_model()
    elif args.command == "api":
        run_api()
    elif args.command == "recommend":
        if not args.user_id and not args.user_text:
            print("Error: Either user_id or user_text must be provided.")
            sys.exit(1)
        run_inference(args.user_id, args.user_text, args.num)
    elif args.command == "pipeline":
        setup_project()
        prepare_data()
        build_features()
        train_model()
        run_api()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
