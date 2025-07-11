"""
Setup script for Dell Organizational Operations Chatbot
Automated installation and configuration
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_banner():
    """Print welcome banner"""
    print("=" * 60)
    print("🏢 DELL ORGANIZATIONAL OPERATIONS CHATBOT")
    print("=" * 60)
    print("Executive Analysis • Error Prevention • Strategic Insights")
    print("Built for CEO and Board of Directors")
    print("=" * 60)

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8+ required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        print("   Please upgrade Python and try again")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True

def install_requirements():
    """Install required Python packages"""
    print("\n📦 Installing required packages...")
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ Error: requirements.txt not found")
        return False
    
    try:
        # Install requirements
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ All packages installed successfully")
            return True
        else:
            print("❌ Error installing packages:")
            print(result.stderr)
            return False
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        "./data",
        "./data/chroma_db",
        "./logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created")

def setup_environment():
    """Setup environment variables"""
    print("\n🔧 Environment Setup...")
    
    env_file = Path(".env")
    
    if not env_file.exists():
        print("Creating .env file...")
        
        # Get OpenAI API key from user
        print("\n🔑 OpenAI API Key Setup (Optional)")
        print("   - Required for advanced AI analysis")
        print("   - System works in fallback mode without it")
        print("   - You can add it later in .env file")
        
        api_key = input("\nEnter OpenAI API Key (or press Enter to skip): ").strip()
        
        env_content = "# Dell Operations Chatbot Environment Variables\n"
        if api_key:
            env_content += f"OPENAI_API_KEY={api_key}\n"
        else:
            env_content += "# OPENAI_API_KEY=your-api-key-here\n"
        
        env_content += "# CHROME_DB_CONNECTION=sqlite:///dell_operations.db\n"
        
        with open(".env", "w") as f:
            f.write(env_content)
        
        print("✅ Environment file created")
    else:
        print("✅ Environment file already exists")

def run_system_test():
    """Run basic system test"""
    print("\n🧪 Running system test...")
    
    try:
        # Test imports
        from config import DellConfig
        from data_manager import DataManager
        from chatbot_engine import DellOperationsChatbot
        
        print("✅ All modules imported successfully")
        
        # Test basic functionality
        current_fy = DellConfig.get_current_fiscal_year()
        fy_range = DellConfig.get_fiscal_year_range()
        
        print(f"✅ Fiscal year calculation: FY{current_fy}")
        print(f"✅ Analysis range: FY{fy_range[0]} - FY{fy_range[1]}")
        
        return True
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False

def display_next_steps():
    """Display next steps for the user"""
    print("\n🎯 SETUP COMPLETE!")
    print("=" * 40)
    
    print("\n📋 Next Steps:")
    print("\n1. Start the Web Interface:")
    print("   streamlit run app.py")
    
    print("\n2. Or run the Command Line Demo:")
    print("   python demo.py")
    
    print("\n3. Add your data using one of these methods:")
    print("   • Manual entry through web interface")
    print("   • CSV file upload")
    print("   • Database connection")
    print("   • Load sample data for testing")
    
    print("\n💡 Tips:")
    print("   • Use sample data to test the system first")
    print("   • Focus on Dell's last 3 fiscal years (Feb-Jan cycle)")
    print("   • Ask executive-level questions for best results")
    
    print("\n🔗 Quick Commands:")
    print("   Web App:    streamlit run app.py")
    print("   Demo:       python demo.py")
    print("   Help:       python -c \"from config import *; help()\"")
    
    print("\n📚 Documentation:")
    print("   See README.md for detailed usage instructions")

def main():
    """Main setup function"""
    print_banner()
    
    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Step 2: Install requirements
    if not install_requirements():
        print("\n❌ Setup failed during package installation")
        sys.exit(1)
    
    # Step 3: Create directories
    create_directories()
    
    # Step 4: Setup environment
    setup_environment()
    
    # Step 5: Run system test
    if not run_system_test():
        print("\n⚠️  Setup completed with warnings")
        print("   Some features may not work correctly")
        print("   Check error messages above")
    
    # Step 6: Display next steps
    display_next_steps()
    
    print("\n🎉 Welcome to Dell Operations Chatbot!")
    print("   Ready to help CEO and Board with operational analysis")

if __name__ == "__main__":
    main()