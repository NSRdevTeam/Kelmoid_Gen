#!/usr/bin/env python3
"""
Environment Setup Script for Natural Language Geometry Engine
Helps configure API keys and environment variables for the project
"""

import os
import sys
from pathlib import Path


def check_env_file():
    """Check if .env file exists and is configured"""
    env_path = Path('.env')
    env_example_path = Path('.env.example')
    
    if not env_path.exists():
        print("❌ .env file not found!")
        if env_example_path.exists():
            print("📝 Creating .env from .env.example template...")
            with open(env_example_path, 'r') as example_file:
                content = example_file.read()
            with open(env_path, 'w') as env_file:
                env_file.write(content)
            print("✅ .env file created! Please edit it with your API keys.")
        else:
            print("❌ .env.example template not found!")
            return False
    else:
        print("✅ .env file exists")
    
    # Check if API keys are configured
    with open(env_path, 'r') as f:
        content = f.read()
        
    if 'your_openai_api_key_here' in content:
        print("⚠️  OpenAI API key not configured in .env file")
        print("   Visit: https://platform.openai.com/api-keys")
    else:
        print("✅ OpenAI API key appears to be configured")
        
    if 'your_huggingface_token_here' in content:
        print("⚠️  Hugging Face token not configured in .env file")
        print("   Visit: https://huggingface.co/settings/tokens")
    else:
        print("✅ Hugging Face token appears to be configured")
    
    return True


def check_dependencies():
    """Check if required Python packages are installed"""
    required_packages = [
        'python-dotenv',
        'openai',
        'transformers',
        'torch',
        'gradio',
        'cadquery'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Install missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        print(f"\nOr install all requirements:")
        print("pip install -r requirements.txt")
        return False
    
    return True


def test_env_loading():
    """Test loading environment variables"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        openai_key = os.getenv('OPENAI_API_KEY')
        hf_token = os.getenv('HUGGINGFACE_API_TOKEN')
        
        print("\n🔑 Environment Variables Test:")
        if openai_key and openai_key != 'your_openai_api_key_here':
            print(f"✅ OPENAI_API_KEY loaded (***{openai_key[-4:]})")
        else:
            print("❌ OPENAI_API_KEY not properly configured")
            
        if hf_token and hf_token != 'your_huggingface_token_here':
            print(f"✅ HUGGINGFACE_API_TOKEN loaded (***{hf_token[-4:]})")
        else:
            print("❌ HUGGINGFACE_API_TOKEN not properly configured")
            
        return True
        
    except ImportError:
        print("❌ python-dotenv not installed")
        return False


def create_directories():
    """Create necessary directories if they don't exist"""
    dirs_to_create = [
        'logs',
        'cache',
        'temp', 
        'output',
        'fine_tuning_datasets'
    ]
    
    for dir_name in dirs_to_create:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created directory: {dir_name}")
        else:
            print(f"✅ Directory exists: {dir_name}")


def main():
    """Main setup function"""
    print("🚀 Natural Language Geometry Engine - Environment Setup")
    print("=" * 60)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Check .env file
    print("\n1. Checking environment configuration...")
    env_ok = check_env_file()
    
    # Check dependencies  
    print("\n2. Checking Python dependencies...")
    deps_ok = check_dependencies()
    
    # Test environment loading
    if env_ok and deps_ok:
        print("\n3. Testing environment variable loading...")
        test_env_loading()
    
    # Create directories
    print("\n4. Creating necessary directories...")
    create_directories()
    
    print("\n" + "=" * 60)
    print("🎯 Next Steps:")
    print("1. Edit the .env file with your actual API keys")
    print("2. Install missing dependencies if any: pip install -r requirements.txt")
    print("3. Run the application: python app.py")
    print("4. Generate fine-tuning datasets using the NLG Engine tab")
    
    if env_ok and deps_ok:
        print("\n✅ Setup completed successfully!")
    else:
        print("\n⚠️  Please resolve the issues above before running the application.")


if __name__ == "__main__":
    main()
