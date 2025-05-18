from math_rag.agent_system.agents import setup_rag_chat
from rich.console import Console
import os


def run_chat_interface():
    # Create Rich console with force_terminal=True to ensure color output
    console = Console(force_terminal=True)

    # Setup agent with graph retrieval
    console.print(
        "[bold blue]Setting up Mathematical RAG system with Neo4j Knowledge Graph...[/bold blue]"
    )
    agent = setup_rag_chat()
    agent.visualize()

    # Welcome message
    console.print("\n[bold green]Welcome to Mathematical RAG Chat![/bold green]")
    console.print(
        "[bold white]Using Neo4j Knowledge Graph for mathematical content retrieval[/bold white]"
    )
    console.print(
        "Type [bold red]exit[/bold red], [bold red]quit[/bold red], or [bold red]q[/bold red] to end the session"
    )
    console.print("Type [bold yellow]clear[/bold yellow] to clear the screen\n")

    # Chat loop
    chat_history = []
    while True:
        try:
            # Get user input - use console.print for prompt and regular input for user entry
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
                    "[bold white]Using Neo4j Knowledge Graph for mathematical content retrieval[/bold white]"
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

            # Add to history (could be used for context in more advanced implementations)
            chat_history.append({"user": user_input, "assistant": response})

        except KeyboardInterrupt:
            console.print("\n[bold green]Goodbye![/bold green]")
            break
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {str(e)}")


if __name__ == "__main__":
    run_chat_interface()
