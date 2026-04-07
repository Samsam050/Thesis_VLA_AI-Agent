# THe main structure of the project
import logging
import sys

from dotenv import load_dotenv
load_dotenv()

from model import OpenRouterModel


def run_cli():
    model = OpenRouterModel(log_level=logging.DEBUG)
    messages = []
    while True:
        user_input = input("User: ")
        messages.append({"role": "user", "content": user_input})
        response = model.generate_response(messages)
        messages.append({"role": "assistant", "content": response})
        print("Bot:", response)


def run_whatsapp():
    from whatsapp import run_bot
    print("Starting WhatsApp bot — make sure the bridge is running:")
    print("  cd bridge && node dist/index.js")
    run_bot()


if __name__ == "__main__":
    if "--whatsapp" in sys.argv:
        run_whatsapp()
    else:
        run_cli()