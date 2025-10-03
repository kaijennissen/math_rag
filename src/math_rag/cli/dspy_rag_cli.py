"""
CLI for DSPy-based mathematical RAG system.

This module provides a command-line interface for the DSPy-based single-agent
RAG approach with document grading and simplified workflow.
"""

import os

from rich.console import Console

from math_rag.config import RagChatSettings, settings_provider
from math_rag.rag_agents.dspy_agent import run_rag_chat, setup_dspy_rag_chat


def run_dspy_chat_interface(settings: RagChatSettings):
    """Run the DSPy-based RAG chat interface."""
    # Create Rich console with force_terminal=True to ensure color output
    console = Console(force_terminal=True)

    # Setup DSPy RAG system
    console.print(
        "[bold blue]Setting up DSPy-based Mathematical RAG system...[/bold blue]"
    )
    rag_module = setup_dspy_rag_chat(
        openai_api_key=settings.openai_api_key,
        neo4j_uri=settings.neo4j_uri,
        neo4j_username=settings.neo4j_username,
        neo4j_password=settings.neo4j_password,
        neo4j_database=settings.neo4j_database,
        agent_config_path=settings.agent_config_path,
        model_id=settings.model_id,
        api_base=settings.api_base,
        huggingface_api_key=settings.huggingface_api_key,
    )

    try:
        # Welcome message
        console.print(
            "\n[bold green]Welcome to DSPy Mathematical RAG Chat![/bold green]"
        )
        console.print(
            "[bold white]Using single-agent approach with document grading[/bold white]"
        )
        console.print(
            "Type [bold red]exit[/bold red], [bold red]quit[/bold red], or "
            "[bold red]q[/bold red] to end the session"
        )
        console.print("Type [bold yellow]clear[/bold yellow] to clear the screen\n")

        # Chat loop
        chat_history = []
        while True:
            try:
                # Get user input
                console.print("[bold cyan]You:[/bold cyan]", end=" ")
                user_input = input().strip()

                # Check for exit command
                if user_input.lower() in ["exit", "quit", "q"]:
                    console.print("\n[bold green]Goodbye![/bold green]")
                    break

                # Check for clear command
                if user_input.lower() == "clear":
                    os.system("cls" if os.name == "nt" else "clear")
                    console.print("[bold green]Welcome to DSPy RAG Chat![/bold green]")
                    console.print(
                        "[bold white]Using single-agent approach with "
                        "document grading[/bold white]"
                    )
                    continue

                # Skip empty input
                if not user_input:
                    continue

                # Process question
                console.print("\n[bold yellow]Processing with DSPy...[/bold yellow]")
                response = run_rag_chat(rag_module, user_input)

                # Display response
                console.print("\n[bold magenta]Assistant:[/bold magenta]")
                console.print(response)
                console.print("\n" + "-" * 50 + "\n")

                # Add to history
                chat_history.append({"user": user_input, "assistant": response})

            except KeyboardInterrupt:
                console.print("\n[bold green]Goodbye![/bold green]")
                break
            except Exception as e:
                console.print(f"\n[bold red]Error:[/bold red] {str(e)}")

    except KeyboardInterrupt:
        console.print("\n[bold green]Goodbye![/bold green]")
    except Exception as e:
        console.print(f"\n[bold red]Fatal Error:[/bold red] {str(e)}")


if __name__ == "__main__":
    settings = settings_provider.get_settings(RagChatSettings)
    run_dspy_chat_interface(settings)
