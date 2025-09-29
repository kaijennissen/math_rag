import os

from rich.console import Console

from math_rag.config import RagChatSettings, settings_provider
from math_rag.rag_agents.agents import setup_rag_chat


def run_chat_interface(settings: RagChatSettings):
    # Create Rich console with force_terminal=True to ensure color output
    console = Console(force_terminal=True)

    # Setup agent with graph retrieval
    console.print(
        "[bold blue]Setting up Mathematical RAG system with Neo4j Knowledge Graph..."
        "[/bold blue]"
    )
    agent, mcp_client = setup_rag_chat(
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
        agent.visualize()

        # Welcome message
        console.print("\n[bold green]Welcome to Mathematical RAG Chat![/bold green]")
        console.print(
            "[bold white]Using Neo4j Knowledge Graph for mathematical content retrieval"
            "[/bold white]"
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
                # Get user input - use console.print for prompt and regular input for
                # user entry
                console.print("[bold cyan]You:[/bold cyan]", end=" ")
                user_input = input().strip()

                # Check for exit command
                if user_input.lower() in ["exit", "quit", "q"]:
                    console.print("\n[bold green]Goodbye![/bold green]")
                    break

                # Check for clear command
                if user_input.lower() == "clear":
                    os.system("cls" if os.name == "nt" else "clear")
                    console.print(
                        "[bold green]Welcome to Mathematical RAG Chat![/bold green]"
                    )
                    console.print(
                        "[bold white]Using Neo4j Knowledge Graph for "
                        "mathematical content retrieval[/bold white]"
                    )
                    continue

                # Skip empty input
                if not user_input:
                    continue

                # Process question
                console.print("\n[bold yellow]Thinking...[/bold yellow]")
                response = agent.run(user_input)

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

    finally:
        # Ensure MCP client is disconnected
        console.print("[bold blue]Closing connection to Neo4j...[/bold blue]")
        mcp_client.disconnect()


if __name__ == "__main__":
    settings = settings_provider.get_settings(RagChatSettings)
    run_chat_interface(settings)
