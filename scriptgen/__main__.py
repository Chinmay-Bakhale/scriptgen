"""Main entry point for scriptgen package."""
from scriptgen.core.workflow import MultiAgentResearchSystem


def main():
    """Run the research system."""
    research_system = MultiAgentResearchSystem()
    research_system.run()


if __name__ == "__main__":
    main()
