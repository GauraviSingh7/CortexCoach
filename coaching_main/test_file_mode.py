"""
Test script to process a WAV file through the coaching system
Usage: python test_file_mode.py
"""

import requests
import time
import json

API_BASE_URL = "http://localhost:8000"
FILE_PATH = r"C:\Users\ADMIN\Desktop\AI COACH OBSERVER - Copy\coaching_main\how-to-use-the-grow-model-coaching-demonstration.wav"

def test_file_session():
    """Test processing a coaching session from a WAV file"""
    
    print("🎵 Testing file mode with WAV file...")
    print(f"📁 File: {FILE_PATH}")
    
    # Start session in file mode
    print("\n1️⃣ Starting session...")
    response = requests.post(
        f"{API_BASE_URL}/session/start",
        json={
            "session_type": "file",
            "file_path": FILE_PATH
        }
    )
    
    if response.status_code != 200:
        print(f"❌ Failed to start session: {response.text}")
        return
    
    data = response.json()
    session_id = data["session_id"]
    print(f"✅ Session started: {session_id}")
    
    # Monitor progress
    print("\n2️⃣ Monitoring processing...")
    print("⏳ Processing file (this may take 1-2 minutes)...")
    
    last_chunks = 0
    while True:
        time.sleep(3)
        
        # Check status
        status_response = requests.get(f"{API_BASE_URL}/session/status")
        if status_response.status_code == 200:
            status = status_response.json()
            chunks = status.get("chunks_processed", 0)
            
            if chunks > last_chunks:
                print(f"📊 Processed {chunks} chunks...")
                last_chunks = chunks
            
            # Check if session is still active
            if not status.get("active", False):
                print("ℹ️ Session no longer active")
                break
        
        # Wait reasonable time (2 minutes max)
        if last_chunks > 0:
            # Give it 30 more seconds after last update
            time.sleep(30)
            break
    
    # Stop session and get report
    print("\n3️⃣ Stopping session and generating report...")
    stop_response = requests.post(f"{API_BASE_URL}/session/stop")
    
    if stop_response.status_code != 200:
        print(f"❌ Failed to stop session: {stop_response.text}")
        return
    
    report = stop_response.json()
    
    # Display report
    print("\n" + "="*80)
    print("📋 SESSION REPORT")
    print("="*80)
    
    print(f"\n📌 Session ID: {report['session_id']}")
    print(f"⏱️ Duration: {report['duration_minutes']:.1f} minutes")
    print(f"💬 Total Chunks: {last_chunks}")
    
    print("\n📊 Coaching Effectiveness:")
    effectiveness = report.get('coaching_effectiveness', {})
    print(f"  • Overall: {effectiveness.get('overall', 0):.2f}")
    print(f"  • Questioning: {effectiveness.get('questioning', 0):.2f}")
    print(f"  • Listening: {effectiveness.get('listening', 0):.2f}")
    
    print("\n🔍 Key Insights:")
    for insight in report.get('key_insights', []):
        print(f"  • {insight}")
    
    print("\n💡 Recommendations:")
    for rec in report.get('recommendations', []):
        print(f"  • {rec}")
    
    print("\n📝 Summary:")
    print(f"  {report.get('transcript_summary', 'No summary available')}")
    
    print("\n" + "="*80)
    print("✅ Test completed!")
    
    # Save full report to file
    output_file = "coaching_report.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"💾 Full report saved to: {output_file}")


if __name__ == "__main__":
    try:
        test_file_session()
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()