"""
Complete WAV File Analysis Test - Shows ALL coaching insights
Usage: python test_wav_file.py
"""

import requests
import time
import json
from pathlib import Path
import textwrap

API_BASE_URL = "http://localhost:8000"
WAV_FILE = r"C:\Users\ADMIN\Desktop\AI COACH OBSERVER - Copy\coaching_main\how-to-use-the-grow-model-coaching-demonstration.wav"

def print_section(title, char="="):
    """Print a formatted section header"""
    width = 80
    print("\n" + char * width)
    print(f"{title.center(width)}")
    print(char * width + "\n")

def print_subsection(title):
    """Print a subsection"""
    print(f"\n{'─' * 80}")
    print(f"  {title}")
    print('─' * 80)

def wrap_text(text, width=76, indent="  "):
    """Wrap text with indentation"""
    wrapped = textwrap.fill(text, width=width)
    return "\n".join([indent + line for line in wrapped.split("\n")])

def test_wav_file():
    """Test processing a coaching WAV file with FULL analysis"""
    
    print_section("🎯 AI COACHING OBSERVER - COMPLETE WAV FILE ANALYSIS", "=")
    
    # Check file exists
    if not Path(WAV_FILE).exists():
        print(f"❌ File not found: {WAV_FILE}")
        return
    
    print(f"📁 File: {Path(WAV_FILE).name}")
    print(f"📂 Location: {WAV_FILE}\n")
    
    # Start session
    print("🚀 Starting file processing session...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/session/start",
            json={"session_type": "file", "file_path": WAV_FILE},
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"❌ Failed: {response.text}")
            return
        
        session_id = response.json()["session_id"]
        print(f"✅ Session started: {session_id}\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Monitor progress
    print_subsection("📊 Processing Status")
    print("⏳ Uploading to AssemblyAI and transcribing...")
    print("   This may take 1-3 minutes for a typical coaching session\n")
    
    last_chunks = 0
    start_time = time.time()
    dots = 0
    
    for i in range(180):  # 3 minutes max
        time.sleep(1)
        
        # Show progress dots
        if i % 5 == 0:
            dots = (dots + 1) % 4
            print(f"\r   Processing{'.' * dots}{' ' * (3-dots)}", end='', flush=True)
        
        try:
            status = requests.get(f"{API_BASE_URL}/session/status", timeout=2).json()
            chunks = status.get("chunks_processed", 0)
            
            if chunks > last_chunks:
                print(f"\r   ✅ Processed {chunks} utterances ({status.get('duration', 0):.1f} min)")
                last_chunks = chunks
        except:
            pass
        
        # Check for completion
        if i > 30 and last_chunks > 0:
            time.sleep(10)  # Wait 10 more seconds
            try:
                status = requests.get(f"{API_BASE_URL}/session/status").json()
                if status.get("chunks_processed", 0) == last_chunks:
                    print(f"\n\n✅ Processing complete! Analyzed {last_chunks} conversational turns")
                    break
            except:
                pass
    
    print(f"\n⏱️ Total processing time: {time.time() - start_time:.1f} seconds\n")
    
    # Stop session and get report
    print_subsection("📋 Generating Comprehensive Analysis")
    try:
        report = requests.post(f"{API_BASE_URL}/session/stop", timeout=60).json()
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        return
    
    # Display FULL report
    print_section("📊 COMPREHENSIVE COACHING ANALYSIS REPORT", "=")
    
    # Session Overview
    print_subsection("📌 Session Overview")
    print(f"  Session ID: {report['session_id']}")
    print(f"  Duration: {report['duration_minutes']:.1f} minutes")
    print(f"  Total Interactions: {last_chunks}")
    
    # Participant Analysis
    print_subsection("👥 Participant Analysis")
    participants = report.get('participants', {})
    
    for role, stats in participants.items():
        print(f"\n  {role.upper()}:")
        print(f"    • Total Turns: {stats.get('total_turns', 0)}")
        print(f"    • Average Engagement: {stats.get('engagement_avg', 0):.2f} ({int(stats.get('engagement_avg', 0)*100)}%)")
        if 'avg_quality' in stats:
            print(f"    • Average Quality: {stats.get('avg_quality', 0):.2f}")
    
    # GROW Model Analysis
    print_subsection("🎯 GROW Model Phase Analysis")
    grow_phases = report.get('grow_phases', [])
    
    if grow_phases:
        phase_dist = {}
        for phase_data in grow_phases:
            phase = phase_data.get('phase', 'Unknown')
            phase_dist[phase] = phase_dist.get(phase, 0) + 1
        
        total_phases = len(grow_phases)
        print(f"\n  Total Phase Detections: {total_phases}\n")
        
        for phase, count in sorted(phase_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_phases) * 100
            bar = "█" * int(percentage / 2)
            print(f"  {phase:15} {bar:50} {percentage:5.1f}% ({count} times)")
        
        # Phase progression
        print(f"\n  Phase Progression:")
        phase_sequence = [p.get('phase') for p in grow_phases[:10]]  # First 10
        print(f"    {' → '.join(phase_sequence)}")
        
        # Best phase
        if grow_phases:
            best_phase = max(grow_phases, key=lambda x: x.get('confidence', 0))
            print(f"\n  Strongest Phase Detection:")
            print(f"    Phase: {best_phase.get('phase')}")
            print(f"    Confidence: {best_phase.get('confidence', 0):.2f}")
            print(f"    Reasoning: {best_phase.get('reasoning', 'N/A')}")
    else:
        print("  No GROW phase data available")
    
    # Emotional Journey
    print_subsection("😊 Emotional Journey Analysis")
    emotional_journey = report.get('emotional_journey', {})
    
    for speaker, journey in emotional_journey.items():
        if journey:
            print(f"\n  {speaker.upper()}'s Emotional Trajectory:")
            
            # Get all unique emotions
            all_emotions = {}
            for moment in journey:
                for emotion, score in moment.get('emotions', {}).items():
                    all_emotions[emotion] = all_emotions.get(emotion, 0) + score
            
            # Sort and display top emotions
            if all_emotions:
                sorted_emotions = sorted(all_emotions.items(), key=lambda x: x[1], reverse=True)
                for emotion, total_score in sorted_emotions[:5]:
                    avg_score = total_score / len(journey)
                    bar = "▓" * int(avg_score * 30)
                    print(f"    {emotion:12} {bar:30} {avg_score:.2f}")
    
    # Learning Style Analysis
    print_subsection("🎓 Learning Style Analysis (VAK Model)")
    learning_style = report.get('learning_style_analysis', {})
    
    if learning_style:
        print("\n  Detected Learning Preferences:")
        sorted_styles = sorted(learning_style.items(), key=lambda x: x[1], reverse=True)
        
        for style, score in sorted_styles:
            percentage = score * 100
            bar = "●" * int(percentage / 5)
            emoji = {"visual": "👁️", "auditory": "👂", "kinesthetic": "✋"}.get(style.lower(), "")
            print(f"    {emoji} {style.capitalize():12} {bar:20} {percentage:5.1f}%")
        
        # Primary style
        primary = max(learning_style.items(), key=lambda x: x[1])
        print(f"\n  Primary Learning Style: {primary[0].upper()} ({primary[1]*100:.0f}%)")
    
    # Coaching Effectiveness
    print_subsection("📊 Coaching Effectiveness Metrics")
    effectiveness = report.get('coaching_effectiveness', {})
    
    metrics = [
        ("Overall Effectiveness", effectiveness.get('overall', 0)),
        ("Questioning Quality", effectiveness.get('questioning', 0)),
        ("Listening Quality", effectiveness.get('listening', 0)),
        ("GROW Alignment", effectiveness.get('grow_alignment', 0))
    ]
    
    print()
    for metric, score in metrics:
        percentage = score * 100
        bar = "█" * int(percentage / 2)
        rating = "Excellent" if score > 0.8 else "Good" if score > 0.6 else "Fair" if score > 0.4 else "Needs Improvement"
        print(f"  {metric:25} {bar:50} {percentage:5.1f}% ({rating})")
    
    # Key Insights
    print_subsection("🔍 Key Insights & Observations")
    insights = report.get('key_insights', [])
    
    if insights:
        for i, insight in enumerate(insights, 1):
            print(f"\n  {i}. {wrap_text(insight, width=74, indent='     ').strip()}")
    else:
        print("  No specific insights generated")
    
    # Recommendations
    print_subsection("💡 Coaching Recommendations")
    recommendations = report.get('recommendations', [])
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"\n  {i}. {wrap_text(rec, width=74, indent='     ').strip()}")
    else:
        print("  No specific recommendations generated")
    
    # Session Summary
    print_subsection("📝 Session Summary & Commentary")
    summary = report.get('transcript_summary', 'No summary available')
    print(wrap_text(summary, width=76, indent="  "))
    
    # Save report
    print_subsection("💾 Saving Report")
    output_file = "coaching_analysis_full_report.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
        print(f"  ✅ Full report saved to: {output_file}")
        print(f"  📂 Location: {Path(output_file).absolute()}")
    except Exception as e:
        print(f"  ⚠️ Could not save report: {e}")
    
    # Final summary
    print_section("✅ ANALYSIS COMPLETE", "=")
    
    print("SUMMARY SCORES:")
    print(f"  • Overall Coaching Quality: {effectiveness.get('overall', 0)*100:.0f}%")
    print(f"  • Average Engagement: {participants.get('coachee', {}).get('engagement_avg', 0)*100:.0f}%")
    print(f"  • Primary GROW Phase: {max(phase_dist.items(), key=lambda x: x[1])[0] if grow_phases else 'Unknown'}")
    print(f"  • Learning Style: {max(learning_style.items(), key=lambda x: x[1])[0].upper() if learning_style else 'Unknown'}")
    print(f"\n  📊 Total Insights: {len(insights)}")
    print(f"  💡 Total Recommendations: {len(recommendations)}")
    print(f"  📁 Full report saved to: {output_file}\n")
    
    print("=" * 80)


if __name__ == "__main__":
    print("\n" + "🎯" * 40)
    print("\n  ENSURE BACKEND IS RUNNING: python -m backend.main\n")
    print("🎯" * 40 + "\n")
    
    try:
        # Check backend
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is running and healthy\n")
            test_wav_file()
        else:
            print("❌ Backend is not responding properly")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend!")
        print("   Start it with: python -m backend.main")
    except requests.exceptions.Timeout:
        print("❌ Backend is slow to respond - it may be loading models")
        print("   Wait a moment and try again")
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()