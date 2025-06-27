from core.orchestrator import SystemOrchestrator

if __name__ == "__main__":
    orchestrator = SystemOrchestrator()
    orchestrator.initialize_all()
    orchestrator.check_health()
    print("🎉 System is ready to launch.")