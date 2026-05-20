"""Legacy entry point — delegates to agent.py for the full pipeline."""

import sys

if __name__ == "__main__":
    # Forward to the subagent entry point so both entry points work.
    from agent import main
    main()
