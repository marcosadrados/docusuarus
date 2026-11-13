"""Trading loop gateway for managing the orchestrator."""

class Gateway:
    def __init__(self, config=None):
        self.config = config or {}
        self.orchestrator = None

    def start(self):
        self.orchestrator = Orchestrator(self.config)
        self.orchestrator.start()

    def stop(self):
        if self.orchestrator:
            self.orchestrator.stop()
