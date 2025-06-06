# ✨ GenMaster AI Studio ✨

GenMaster AI Studio is a comprehensive, full-stack desktop application for AI image generation, built entirely with Python. It features a modern user interface powered by CustomTkinter, a robust FastAPI backend for handling logic and API requests, and integrates with leading AI services like Stability AI and OpenAI (DALL-E).

The application provides a complete user experience, including user authentication, a subscription and credits system with Stripe integration (stubs included), a personal image gallery, and advanced generation controls.



---

## 🚀 Features

-   **Modern Desktop UI**: A sleek, themeable interface built with `customtkinter`.
-   **Dual AI Engine Support**:
    -   Generate images using various **Stability AI** models (including SDXL).
    -   Generate images using **OpenAI's DALL-E 3**.
-   **Full-Stack Architecture**:
    -   **FastAPI Backend**: Manages all business logic, user data, and external API interactions.
    -   **SQLite Database**: Stores user information, generation history, and payment records.
-   **Complete User Management**:
    -   Secure user registration and login with JWT-based authentication.
    -   Password hashing and secure token management.
    -   Email verification and password reset functionality (stubs included).
    -   Custom user avatars.
-   **Subscription & Credits System**:
    -   Multiple subscription tiers (Free, Basic, Premium, etc.).
    -   One-time credit packs.
    -   **Stripe Integration**: Checkout sessions for purchasing plans and a customer portal for managing subscriptions.
    -   Monthly credit reset logic for subscribers.
-   **Advanced Image Generation Controls**:
    -   Positive and Negative Prompts.
    -   Image-to-Image generation with strength control.
    -   Adjustable parameters: resolution (width/height), steps, CFG scale, seed, and batch size.
-   **Personal Image Gallery**:
    -   View, filter, and sort all your generated images.
    -   Click an image to reload its exact generation parameters into the UI.
    -   Delete images from your gallery.
-   **Theming**: Switch between Light, Dark, and System appearance modes.

---

## 🛠️ Setup and Installation

Follow these steps to get the GenMaster AI Studio running on your local machine.

### Prerequisites

-   Python 3.10 or newer.
-   API key from [Stability AI](https://platform.stability.ai/account/keys).
-   API key from [OpenAI](https://platform.openai.com/api-keys).
-   (Optional) A [Stripe](https://stripe.com/) account for payment processing.

### Step 1: Clone the Repository

Clone this project to your local machine.

```bash
git clone https://github.com/your-username/genmaster-ai-studio.git
cd genmaster-ai-studio
```

### Step 2: Create a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

The project's dependencies are listed below. Install them all using pip:

```bash
pip install fastapi "uvicorn[standard]" customtkinter Pillow python-jose[cryptography] passlib[bcrypt] python-multipart requests
```

Alternatively, you can copy the contents of the `requirements.txt` section below into a file and run `pip install -r requirements.txt`.

<details>
<summary><code>requirements.txt</code></summary>

```text
# FastAPI and Web Server
fastapi
uvicorn[standard]

# GUI Framework
customtkinter
Pillow

# Authentication and Security
python-jose[cryptography]
passlib[bcrypt]
python-multipart

# API Client
requests
```
</details>

### Step 4: Configure Environment Variables

For the application to connect to the AI services, you must provide your API keys. The best way to do this is with an environment file.

1.  Create a new file named `.env` in the root directory of the project.
2.  Copy the content below into your new `.env` file.
3.  Replace the placeholder values with your actual keys and secrets.

<details>
<summary><code>.env</code> file content</summary>

```dotenv
# ===============================================
# GenMaster AI Studio - Environment Configuration
# ===============================================
#
# This file stores your secret keys and configuration.
# Do not share this file or commit it to version control.
#

# --- AI API Keys ---
# Get your API key from https://platform.stability.ai/account/keys
STABILITY_API_KEY="YOUR_STABILITY_AI_API_KEY_HERE"

# Get your API key from https://platform.openai.com/api-keys
OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"

# --- Security (IMPORTANT) ---
# Replace this with a long, random, and secret string.
# You can generate one using: python -c 'import secrets; print(secrets.token_hex(32))'
SECRET_KEY="a_very_secret_key_for_jwt_v3_change_me_now"

# --- Stripe Configuration (Optional) ---
# These are required only if you want to test the payment functionality.
# Get these keys from your Stripe Dashboard (https://dashboard.stripe.com/test/apikeys).
STRIPE_SECRET_KEY="sk_test_YOUR_STRIPE_SECRET_KEY"
STRIPE_PUBLISHABLE_KEY="pk_test_YOUR_STRIPE_PUBLISHABLE_KEY"
# This is the signing secret for your webhook endpoint.
STRIPE_WEBHOOK_SECRET="whsec_YOUR_STRIPE_WEBHOOK_SECRET"
# Get this by creating a new configuration at https://dashboard.stripe.com/test/settings/billing/portal
STRIPE_CUSTOMER_PORTAL_LINK="YOUR_STRIPE_CUSTOMER_PORTAL_CONFIG_LINK_HERE"
```
</details>

### Step 5: Download UI Icons

The application looks for icons in a `ui_icons` directory. You will need to create this directory and add the necessary `.png` icon files. You can find suitable icons from sources like [Google Fonts Icons](https://fonts.google.com/icons).

-   Create a directory named `ui_icons` in the project root.
-   Download and place the following icons (or equivalents) inside it:
    -   `palette_FILL0_wght400_GRAD0_opsz24.png` (for 'generate')
    -   `photo_library_FILL0_wght400_GRAD0_opsz24.png` (for 'gallery')
    -   `workspace_premium_FILL0_wght400_GRAD0_opsz24.png` (for 'subscription')
    -   `account_circle_FILL0_wght400_GRAD0_opsz24.png` (for 'profile')
    -   `logout_FILL0_wght400_GRAD0_opsz24.png` (for 'logout')
    -   `upload_file_FILL0_wght400_GRAD0_opsz24.png` (for 'upload')
    -   `save_FILL0_wght400_GRAD0_opsz24.png` (for 'save')
    -   `delete_FILL0_wght400_GRAD0_opsz24.png` (for 'delete')
    -   `edit_FILL0_wght400_GRAD0_opsz24.png` (for 'edit')

### Step 6: Run the Application

Execute the main Python script. This will automatically start the FastAPI backend server and launch the CustomTkinter GUI.

```bash
python your_main_script_name.py
```

The terminal will show logs from the Uvicorn server, and the desktop application window will appear. You can now register a new user and start generating images!

---

## 📂 Project Structure and Version Control

The project creates several files and directories during operation. It is crucial to prevent sensitive information and generated content from being committed to version control.

### Files to Ignore

Create a `.gitignore` file in the root of your project with the following content. This will ensure that your environment file, database, generated images, and other temporary files are not tracked by Git.

<details>
<summary><code>.gitignore</code></summary>

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class

# Virtual environment
/venv/
/.venv/
/env/
/ENV/
/virtualenv/

# Database
*.db
*.sqlite3

# Environment variables
.env

# Generated Content
/generated_images_v3/
/user_uploads_v3/

# IDE and OS specific
.idea/
.vscode/
.DS_Store
```
</details>

### Directory Overview

-   `your_main_script_name.py`: The single-file source code.
-   `.env`: Local environment configuration file (API keys, secrets).
-   `.gitignore`: Specifies files for Git to ignore.
-   `ai_generator_v3.db`: The SQLite database file (created automatically).
-   `generated_images_v3/`: Directory where generated images are saved.
-   `user_uploads_v3/`: Directory for user-uploaded content like avatars.
-   `ui_icons/`: Directory containing icons for the user interface.

---

## 🤝 Contributing

Contributions are welcome! If you have suggestions or improvements, please feel free to fork the repository, create a new branch, and submit a pull request.

1.  Fork the Project.
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the Branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

---

## 📜 License

This project is licensed under the MIT License.
