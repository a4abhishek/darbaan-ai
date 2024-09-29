import requests

from config import config

if __name__ == '__main__':
    bot_token = config['telegram']['bot_token']
    response = requests.get(f'https://api.telegram.org/bot{bot_token}/getUpdates').json()
    print(response)
