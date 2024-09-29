.PHONY: encrypt-config decrypt-config

encrypt-config:
	@if [ -f config.py ]; then \
		echo "Remember the password you set for decryption. Default: 12345"; \
		ansible-vault encrypt config.py --output=config.py.vault; \
		echo "Encrypted config.py to config.py.vault"; \
	else \
		echo "config.py not found!"; \
		exit 1; \
	fi

decrypt-config:
	@if [ -f config.py.vault ]; then \
		echo "Use the password set during encryption. Default: 12345"; \
		ansible-vault decrypt config.py.vault --output=config.py; \
		echo "Decrypted config.py.vault to config.py"; \
	else \
		echo "config.py.vault not found!"; \
		exit 1; \
	fi
