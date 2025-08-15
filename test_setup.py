"""
Test script to validate the setup and API keys.
Run this before using the main application.
"""

import os
import sys
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported."""
    print("🔍 Testing imports...")
    
    try:
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__}")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import langchain
        print(f"✅ LangChain {langchain.__version__}")
    except ImportError as e:
        print(f"❌ LangChain import failed: {e}")
        return False
    
    try:
        import langgraph
        print(f"✅ LangGraph (version check not available)")
    except ImportError as e:
        print(f"❌ LangGraph import failed: {e}")
        return False
        
    try:
        import mistralai
        print(f"✅ Mistral AI client")
    except ImportError as e:
        print(f"❌ Mistral AI import failed: {e}")
        return False
    
    try:
        from brave_search_python_client import BraveSearch
        print(f"✅ Brave Search client")
    except ImportError as e:
        print(f"❌ Brave Search import failed: {e}")
        return False
    
    try:
        import faiss
        print(f"✅ FAISS")
    except ImportError as e:
        print(f"❌ FAISS import failed: {e}")
        return False
    
    try:
        from bs4 import BeautifulSoup
        print(f"✅ BeautifulSoup")
    except ImportError as e:
        print(f"❌ BeautifulSoup import failed: {e}")
        return False
    
    return True

def test_environment():
    """Test environment variables."""
    print("\n🔑 Testing environment variables...")
    
    # Load .env if it exists
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Loaded .env file")
    except:
        print("⚠️  No .env file found or python-dotenv not installed")
    
    required_keys = ['MISTRAL_API_KEY', 'BRAVE_SEARCH_API_KEY']
    optional_keys = ['LANGSMITH_API_KEY', 'LANGSMITH_PROJECT', 'LANGSMITH_TRACING']
    
    missing_required = []
    
    for key in required_keys:
        value = os.getenv(key)
        if value:
            print(f"✅ {key}: {'*' * 8}{value[-4:] if len(value) >= 4 else '****'}")
        else:
            print(f"❌ {key}: Not set")
            missing_required.append(key)
    
    for key in optional_keys:
        value = os.getenv(key)
        if value:
            print(f"✅ {key}: {value}")
        else:
            print(f"⚠️  {key}: Not set (optional)")
    
    return len(missing_required) == 0

def test_api_connections():
    """Test API connections."""
    print("\n🌐 Testing API connections...")
    
    # Add src to path
    current_dir = Path(__file__).parent
    src_dir = current_dir / "src"
    sys.path.insert(0, str(src_dir))
    
    # Test Mistral AI
    try:
        from src.components.llm_engine import MistralLLMEngine
        llm = MistralLLMEngine()
        if llm.validate_api_key():
            print("✅ Mistral AI connection successful")
        else:
            print("❌ Mistral AI connection failed")
            return False
    except Exception as e:
        print(f"❌ Mistral AI test failed: {e}")
        return False
    
    # Test Brave Search
    try:
        from src.components.search_engine import BraveSearchEngine
        search = BraveSearchEngine()
        if search.validate_api_key():
            print("✅ Brave Search connection successful")
        else:
            print("❌ Brave Search connection failed")
            return False
    except Exception as e:
        print(f"❌ Brave Search test failed: {e}")
        return False
    
    return True

def test_workflow():
    """Test the complete workflow with a simple query."""
    print("\n🔄 Testing complete workflow...")
    
    try:
        # Add src to path
        current_dir = Path(__file__).parent
        src_dir = current_dir / "src"
        sys.path.insert(0, str(src_dir))
        
        from src.components.orchestrator import PerplexityWorkflow
        
        workflow = PerplexityWorkflow()
        
        # Run a simple test query
        result = workflow.run("What is Python?")
        
        if result.get('error'):
            print(f"❌ Workflow test failed: {result['error']}")
            return False
        
        if result.get('formatted_response'):
            print("✅ Workflow test successful")
            print(f"   Response length: {len(result['formatted_response'])} characters")
            print(f"   Sources found: {len(result.get('sources', []))}")
            return True
        else:
            print("❌ Workflow test failed: No response generated")
            return False
            
    except Exception as e:
        print(f"❌ Workflow test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Perplexity Clone Setup Test")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 4
    
    # Test imports
    if test_imports():
        tests_passed += 1
    
    # Test environment
    if test_environment():
        tests_passed += 1
    
    # Test API connections (only if env vars are set)
    if os.getenv('MISTRAL_API_KEY') and os.getenv('BRAVE_SEARCH_API_KEY'):
        if test_api_connections():
            tests_passed += 1
        
        # Test complete workflow
        if test_workflow():
            tests_passed += 1
    else:
        print("\n⚠️  Skipping API and workflow tests (missing API keys)")
        total_tests = 2
    
    # Summary
    print("\n" + "=" * 50)
    print(f"🎯 Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! You're ready to run the app.")
        print("\n🚀 Run: streamlit run app.py")
    else:
        print("❌ Some tests failed. Please check the issues above.")
        
        if not os.getenv('MISTRAL_API_KEY') or not os.getenv('BRAVE_SEARCH_API_KEY'):
            print("\n💡 Don't forget to:")
            print("   1. Copy .env.example to .env")
            print("   2. Add your API keys to .env")
            print("   3. Run this test again")

if __name__ == "__main__":
    main()
