# Development Setup (Windows + WSL2 + Ubuntu + Python)

This guide walks you through setting up a modern Python development environment on **Windows** using **WSL2**, **Ubuntu**, **VS Code**, and **pyenv**.

> ⚠️ **Follow the steps in order — skipping ahead can break things later.**

---

## 1. GitHub Account

You'll need a GitHub account to store and share your code.

1. Create an account: https://github.com/join
2. Upload a profile picture and verify your name: https://github.com/settings/profile
3. Enable Two-Factor Authentication (2FA) — this protects your account:
   https://docs.github.com/en/authentication/securing-your-account-with-two-factor-authentication-2fa/configuring-two-factor-authentication

---

## 2. Check Your Windows Version

You need **Windows 10 (version 2004 or later)** or **Windows 11** for WSL2 to work.

**How to check:**
- Press `Windows + R`
- Type `winver`
- Press `Enter`

If your Windows 10 version is **below 2004**, update it:
- Press `Windows + R`
- Type `ms-settings:windowsupdate`
- Press `Enter`
- Click **Check for updates**

---

## 3. Enable Virtualization

WSL2 requires virtualization to be enabled in your CPU settings.

**How to check:**
- Press `Windows + R`
- Type `taskmgr`
- Press `Enter`
- Go to **Performance → CPU**

You should see: **Virtualization: Enabled**

If it says **Disabled**, you'll need to enable it in your BIOS/UEFI settings (look for **Intel VT-x**, **AMD-V**, or **SVM**). Google your laptop model + "enable virtualization BIOS" if you're unsure how.

---

## 4. Before You Continue — Quick Checklist

Make sure you've done these before moving forward:

- ✅ You're signed in to a **Microsoft account** on Windows
- ✅ **Windows is activated** (Settings → System → Activation)

---

## 5. Install Windows Terminal *(optional but recommended)*

Windows Terminal gives you a much nicer experience than the default Command Prompt.

Open **Command Prompt** and run:

```powershell
winget install Microsoft.WindowsTerminal --source winget
```

---

## 6. Install VS Code

VS Code is the code editor you'll use throughout the course.

Download it at: https://code.visualstudio.com/download

Or install via command line:

```bash
winget install Microsoft.VisualStudioCode --source winget
```

---

## 7. Install WSL2 + Ubuntu

WSL2 lets you run a real Linux environment inside Windows.

**Open Command Prompt as Administrator:**
- Press `Windows + R`
- Type `cmd`
- Press `Ctrl + Shift + Enter` *(this runs it as Administrator)*

Run:

```powershell
wsl --install
```

> This installs WSL2 and Ubuntu automatically. It may take a few minutes.

### Verify WSL version

```bash
wsl -l -v
```

> You should see Ubuntu listed with **VERSION 2**. If it shows version 1, ask for help.

If Ubuntu wasn't installed automatically, run:

```powershell
wsl --install Ubuntu
```

### First Launch

When you open Ubuntu for the first time, it will ask you to create a user account:

- **Username:** use lowercase letters, no spaces (e.g. `alice`)
- **Password:** nothing will appear as you type — that's normal, just type your password and press Enter

> 💡 Remember this password! You'll need it whenever you use `sudo`.

**Restart your computer when finished.**

### Set Ubuntu as your default terminal

After restarting, open Windows Terminal, press `Ctrl + ,` to open Settings, and set **Ubuntu** as your default profile. Then close and reopen the terminal.

---

## 8. Check Your Linux Username

Open Ubuntu and run:

```bash
whoami
```

> It should print your username (e.g. `alice`). If it prints `root`, **stop here** and ask for help — running as root can cause permission issues.

---

## 9. Connect VS Code to WSL

This lets you edit Linux files directly from VS Code.

From Ubuntu, run:

```bash
code --install-extension ms-vscode-remote.remote-wsl
code .
```

After it opens, check the **bottom-left corner of VS Code** — it should say `WSL: Ubuntu`. That means it's connected.

---

## 10. Install Essential Tools

These are common command-line tools you'll use during the course.

```bash
sudo apt update
sudo apt install -y curl git imagemagick jq unzip vim zsh tree
```

> `sudo apt update` refreshes the list of available packages. Always run it before installing.

---

## 11. Install GitHub CLI

The GitHub CLI (`gh`) lets you interact with GitHub from your terminal — much easier than copy-pasting tokens.

```bash
# Remove any conflicting package first
sudo apt remove -y gitsome

# Add GitHub's official package source
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install -y gh
```

Verify it installed correctly:

```bash
gh --version
```

Log in to GitHub (using SSH — the most secure option):

```bash
gh auth login -s 'user:email' -w --git-protocol ssh
```

> Follow the prompts in your browser to authorize. When asked about protocol, SSH is already selected.

Confirm it worked:

```bash
gh auth status
```

---

## 12. Install Oh My Zsh *(optional)*

Oh My Zsh makes your terminal shell more powerful and easier to use.

```bash
sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

> If it asks you to change your default shell to `zsh`, say **yes**.

---

## 13. Install pyenv

`pyenv` lets you install and manage multiple Python versions easily.

```bash
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
exec zsh
```

Install the libraries Python needs to compile:

```bash
sudo apt-get update
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev sqlite3 libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev python3-dev
```

---

## 14. Install Locale Settings

This makes sure your terminal handles text encoding correctly (avoids weird character bugs).

```bash
sudo apt update
sudo apt install locales
sudo locale-gen en_US.UTF-8
sudo update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
```

---

## 15. Create the SSH Key Folder

This folder is needed for secure connections to GitHub and servers.

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
```

---

## 16. Configure `.zshrc`

`.zshrc` is the configuration file for your Zsh shell — it loads every time you open a terminal.

Open it in VS Code:

```bash
cd
code .zshrc
```

**Replace the entire contents** with the following:

```zsh
ZSH=$HOME/.oh-my-zsh

# You can change the theme with another one from https://github.com/robbyrussell/oh-my-zsh/wiki/themes
ZSH_THEME="robbyrussell"

# Useful oh-my-zsh plugins for Le Wagon bootcamps
plugins=(git gitfast last-working-dir common-aliases history-substring-search ssh-agent)

# (macOS-only) Prevent Homebrew from reporting - https://github.com/Homebrew/brew/blob/master/docs/Analytics.md
export HOMEBREW_NO_ANALYTICS=1

# Disable warning about insecure completion-dependent directories
ZSH_DISABLE_COMPFIX=true

# Actually load Oh-My-Zsh
source "${ZSH}/oh-my-zsh.sh"
unalias rm # No interactive rm by default (brought by plugins/common-aliases)
unalias lt # we need `lt` for https://github.com/localtunnel/localtunnel


# Load pyenv (to manage your Python versions)
export PYENV_VIRTUALENV_DISABLE_PROMPT=1

# Load nvm (to manage your node versions)
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion

# Call `nvm use` automatically in a directory with a `.nvmrc` file
autoload -U add-zsh-hook
load-nvmrc() {
  if nvm -v &> /dev/null; then
    local node_version="$(nvm version)"
    local nvmrc_path="$(nvm_find_nvmrc)"

    if [ -n "$nvmrc_path" ]; then
      local nvmrc_node_version=$(nvm version "$(cat "${nvmrc_path}")")

      if [ "$nvmrc_node_version" = "N/A" ]; then
        nvm install
      elif [ "$nvmrc_node_version" != "$node_version" ]; then
        nvm use --silent
      fi
    elif [ "$node_version" != "$(nvm version default)" ]; then
      nvm use default --silent
    fi
  fi
}
type -a nvm > /dev/null && add-zsh-hook chpwd load-nvmrc
type -a nvm > /dev/null && load-nvmrc

# Rails and Ruby uses the local `bin` folder to store binstubs.
# So instead of running `bin/rails` like the doc says, just run `rails`
# Same for `./node_modules/.bin` and nodejs
export PATH="./bin:./node_modules/.bin:${PATH}:/usr/local/sbin"

# Store your own aliases in the ~/.aliases file and load the here.
[[ -f "$HOME/.aliases" ]] && source "$HOME/.aliases"

# Encoding stuff for the terminal
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

export BUNDLER_EDITOR=code
export EDITOR=code

# Set ipdb as the default Python debugger
export PYTHONBREAKPOINT=ipdb.set_trace
export BUNDLER_EDITOR="subl $@ >/dev/null 2>&1"

# Load pyenv automatically by adding
# the following to ~/.zshrc:

export PATH="$HOME/.pyenv/bin:$PATH"


eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
type -a pyenv > /dev/null && eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init - 2> /dev/null)" && RPROMPT+='[🐍 $(pyenv version-name)]'

export BROWSER=google-chrome

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
```

Save the file, then **reload your shell** so the changes take effect:

```bash
source ~/.zshrc
```

---

## 17. Add Your GitHub Username

Replace `your_github_username` with your actual GitHub username:

```zsh
export COURSE_USERNAME="your_github_username"
```

---

## 18. Install Python

Now install Python 3.12.9 using pyenv:

```bash
pyenv install 3.12.9
pyenv global 3.12.9
exec zsh
```

> This may take a few minutes — Python is being compiled from source.

Verify it worked:

```bash
python --version
```

> You should see `Python 3.12.9`. If you see a different version or an error, ask for help.

---

## 19. Install pyenv-virtualenv

This plugin lets you create isolated Python environments per project (so packages don't conflict).

```bash
git clone https://github.com/pyenv/pyenv-virtualenv.git "$(pyenv root)/plugins/pyenv-virtualenv"
exec zsh
```

---

## 20. Create a Virtual Environment

Create an environment called `myenv` and set it as the global default:

```bash
pyenv virtualenv 3.12.9 myenv
pyenv global myenv
```

Upgrade `pip` (the Python package installer) to the latest version:

```bash
pip install --upgrade pip
```

Install some common data science packages to test everything works:

```bash
pip install numpy pandas scikit-learn jupyter
```

---

## 21. Test Jupyter

Launch a Jupyter notebook to verify the setup is working:

```bash
jupyter notebook
```

> A browser window should open with the Jupyter interface. Press `Ctrl + C` in the terminal to stop it when you're done.

---

## 22. Install Node.js

Node.js is needed to view the course slides locally.

```bash
sudo apt update
sudo apt install nodejs npm
```

---

## 23. Clone the Course Repo

Create a folder and clone the course repository:

```bash
cd ~
pip install pytest
gh repo clone smart-tinmen/python_class
cd python_class
```

---

## ✅ Setup Complete!

You're all set. Your environment includes:

- **WSL2 + Ubuntu** — Linux inside Windows
- **VS Code** — connected to WSL
- **pyenv** — managing Python versions
- **Python 3.12.9** — in an isolated virtual environment
- **Jupyter** — for interactive notebooks
- **GitHub CLI** — for repo management

If anything went wrong, re-read the relevant section carefully, and don't hesitate to ask for help. 🚀
