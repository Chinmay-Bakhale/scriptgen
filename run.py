"""Entry point for running the research workflow."""
from dotenv import load_dotenv
from scriptgen.core.workflow import MultiAgentResearchSystem

load_dotenv()

if __name__ == "__main__":
    system = MultiAgentResearchSystem()
    system.run()
