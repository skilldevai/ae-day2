# Applied AI Engineering for the Enterprise - Day 2

## Models & Retrieval-Augmented Generation (RAG)

These instructions will guide you through running the container environment locally in a VS Code instance using Docker Desktop. If you instead need/want a GitHub Codespaces environment, see [README-codespaces.md](./README-codespaces.md) in this repository. 

<br><br>

**1. First, ensure you have the prerequisites below installed and running on your system.**
### System Requirements
- macOS (Intel or Apple Silicon)
- At least 4 CPU cores
- 16GB RAM minimum
- 32GB free disk space

### Required Software

#### 1. Install Docker Desktop
- Download from: https://www.docker.com/products/docker-desktop
- Choose the appropriate version:
  - **Apple Silicon (M1/M2/M3)**: Download "Mac with Apple chip"
  - **Intel Mac**: Download "Mac with Intel chip"
- Install by dragging Docker.app to Applications
- Launch Docker Desktop and complete the setup
- Ensure Docker Desktop is running (you'll see the whale icon in your menu bar)

#### 2. Install Visual Studio Code
- Download from: https://code.visualstudio.com/
- Open the downloaded .zip file
- Drag Visual Studio Code to your Applications folder
- Launch VS Code

#### 3. Install Dev Containers Extension
- Open VS Code
- Press `Cmd+Shift+X` to open Extensions
- Search for "Dev Containers" (published by Microsoft)
- Click **Install**

## Setup Steps

[![Open in Dev Containers](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/skilldevai/ae-day2)

Open in Dev Containers: https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/skilldevai/ae-day2

### 1. Clone the Repository
Open Terminal and run:
```bash
git clone https://github.com/skilldevai/ae-day2.git
cd ai-aip
```


### 2. Open in VS Code
```bash
code .
```

### 3. Reopen in Container
When VS Code opens, you should see a notification asking if you want to "Reopen in Container":
- Click **Reopen in Container**

If you don't see the notification:
- Press `Cmd+Shift+P` to open the Command Palette
- Type "Dev Containers: Reopen in Container"
- Press Enter

### 4. Wait for Initial Setup
The first time you open the container, it will take **10-15 minutes** to:
- Build the Docker container
- Set up the Python environment
- Install all dependencies
- Install Ollama
- Download required AI models

You can monitor progress in the VS Code terminal.



## Troubleshooting

### Docker Desktop Not Running
If you see "Docker is not running" errors:
- Check that Docker Desktop is running (whale icon in menu bar)
- Try restarting Docker Desktop
- Ensure you've completed the Docker Desktop setup wizard

### Container Build Fails
- Ensure you have enough disk space (32GB minimum free)
- Check Docker Desktop has sufficient resources allocated:
  - Open Docker Desktop → Settings → Resources
  - Ensure at least 4 CPUs and 8GB memory are allocated

### Ollama Models Not Downloaded
If models are missing:
```bash
ollama pull llama3.2:latest
ollama pull nomic-embed-text
```

### Port Already in Use
If you see port conflict errors:
- Check what's using the port: `lsof -i :PORT_NUMBER`
- Stop the conflicting service or change the port in `.devcontainer/devcontainer.json`

## Next Steps

Once setup is complete, you're ready to start the workshop! Refer to the main [README.md](README.md) for lab instructions.

## Stopping the Container

When you're done working:
- Close VS Code or
- Press `Cmd+Shift+P` and select "Dev Containers: Reopen Folder Locally"

Docker Desktop can continue running in the background or be quit from the menu bar icon.

![Creating new codespace from button](./images/aia-0-2.png?raw=true "Creating new codespace from button")

This will run for a long time (up to 10 minutes) while it gets everything ready.

After the initial startup, it will run a script to setup the python environment and install needed python pieces. This will take several more minutes to run. It will look like this while this is running.

![Final prep](./images/aia-1-2.png?raw=true "Final prep")

The codespace is ready to use when you see a prompt like the one shown below in its terminal.

![Ready to use](./images/aia-1-1.png?raw=true "Ready to use")

When you see this, just hit *Enter* to get to a prompt.

<br><br>

**4. Open up the *labs.md* file so you can follow along with the labs.**
You can either open it in a separate browser instance or open it in the codespace. 

![Opening labs](./images/aia-0-4.png?raw=true "Opening labs")

<br>

**Now, you are ready for the labs!**

<br><br>



---

## License and Use

These materials are provided as part of the **Applied AI Engineering for the Enterprise** conducted by **TechUpSkills (Brent Laster)**.

Use of this repository is permitted **only for registered workshop participants** for their own personal learning and
practice. Redistribution, republication, or reuse of any part of these materials for teaching, commercial, or derivative
purposes is not allowed without written permission.

© 2026 TechUpSkills / Brent Laster. All rights reserved.
