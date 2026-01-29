"""
Debug script to test audio capture and AssemblyAI connection
Run this separately to diagnose audio issues
"""

import asyncio
import logging
import os
from dotenv import load_dotenv

from backend.models.audio_capture import AudioCaptureSystem
from backend.models.audio_processor import AudioProcessor

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

async def test_audio():
    """Test audio capture and AssemblyAI connection"""
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    
    if not api_key:
        print("❌ ASSEMBLYAI_API_KEY not found in environment")
        return
    
    print("🔧 Testing audio capture system...")
    
    # Test audio devices
    capture = AudioCaptureSystem()
    devices = capture.get_available_devices()
    print(f"\n📱 Found {len(devices)} audio input devices:")
    for dev in devices:
        print(f"  [{dev['index']}] {dev['name']} - {dev['sample_rate']}Hz")
    
    # Test processor
    print("\n🔧 Initializing AudioProcessor...")
    processor = AudioProcessor(api_key)
    
    # Create a dummy queue for testing
    import asyncio
    test_queue = asyncio.Queue()
    
    print("🔧 Starting transcription...")
    try:
        await processor.start_live_transcription(test_queue)
        print("✅ AudioProcessor started successfully")
        
        # Test sending some dummy audio (silence)
        print("🔧 Testing audio streaming...")
        dummy_audio = b'\x00' * 3200  # 200ms of silence at 16kHz
        
        for i in range(10):
            await processor.send_audio(dummy_audio)
            print(f"  Sent chunk {i+1}/10")
            await asyncio.sleep(0.2)
        
        print("\n✅ Audio streaming test completed")
        print("Check backend logs for transcription events")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await processor.stop_transcription()
        print("✅ Cleaned up")

if __name__ == "__main__":
    asyncio.run(test_audio())
