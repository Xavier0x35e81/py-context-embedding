# py-context-embedding

## Python Version

First, you must have the 3.11 version of Python installed:

https://www.python.org/downloads/

This is very important, as only a few versions of Python support LangChain and
OpenAI.

## Pipenv Installation and Configuration

```zsh
pip install pipenv

# or

pip3 install pipenv
```

## Install dependencies

```zsh
pipenv install
```

## Create and enter a new environment:

```zsh
pipenv shell
```

After doing this your terminal will now be running commands in this new
environment managed by Pipenv.

Once inside this shell, you can run Python commands. eg:

```zsh
python main.py
```

## package, env vars / keys update

1. package update (if needed)

```zsh
pipenv install <new-package>
```

2. exit the shell with `exit` command
3. re-enter using the `pipenv shell` command

## generating requirement.txt after package update

```zsh
pipenv run pip freeze > requirements.txt
```
