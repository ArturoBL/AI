import openai  # pip install openai
import typer  # pip install "typer[all]"
from rich import print  # pip install rich
from rich.table import Table

"""
Webs de inter茅s:
- M贸dulo OpenAI: https://github.com/openai/openai-python
- Documentaci贸n API ChatGPT: https://platform.openai.com/docs/api-reference/chat
- Typer: https://typer.tiangolo.com
- Rich: https://rich.readthedocs.io/en/stable/
"""


def main():

    openai.api_key = "API KEY"

    print("馃挰 [bold green]ChatGPT API en Python[/bold green]")

    table = Table("Comando", "Descripci贸n")
    table.add_row("exit", "Salir de la aplicaci贸n")
    table.add_row("new", "Crear una nueva conversaci贸n")

    print(table)

    # Contexto del asistente
    context = {"role": "system",
               "content": "Eres un asistente muy 煤til."}
    messages = [context]

    while True:

        content = __prompt()

        if content == "new":
            print("馃啎 Nueva conversaci贸n creada")
            messages = [context]
            content = __prompt()

        messages.append({"role": "user", "content": content})

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages)

        response_content = response.choices[0].message.content

        messages.append({"role": "assistant", "content": response_content})

        print(f"[bold green]> [/bold green] [green]{response_content}[/green]")


def __prompt() -> str:
    prompt = typer.prompt("\n驴Sobre qu茅 quieres hablar? ")

    if prompt == "exit":
        exit = typer.confirm("鉁?驴Est谩s seguro?")
        if exit:
            print("馃憢 隆Hasta luego!")
            raise typer.Abort()

        return __prompt()

    return prompt


if __name__ == "__main__":
    typer.run(main)