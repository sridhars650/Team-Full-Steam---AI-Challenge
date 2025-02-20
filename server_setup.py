import subprocess
import sys
import os
import time

def install_packages():
    """Install packages from requirements.txt if not already installed."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while installing packages: {e}")
        sys.exit(1)

def install_guardrail_pkgs(package):
    retries = 3
    for i in range(retries):
        try:
            subprocess.run(['timeout', '300', 'guardrails', 'hub', 'install', package], check=True)
            print(f"✅ Successfully installed {package}!")
            return  # Success, exit the function
        except subprocess.TimeoutError:
            print(f"❌ Guardrails installation for {package} timed out (attempt {i+1}/{retries}). Retrying in 10 seconds...")
            time.sleep(10)  # Wait before retrying
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing {package}: {e}")
            return # Stop trying other packages if one fails.

    print(f"❌ Guardrails installation for {package} failed after {retries} retries.") # All retries failed


def main():
    """Main server logic."""
    from dotenv import load_dotenv
    load_dotenv()
    print("All requirements are satisfied. Proceeding with the main code...")

    try:
        subprocess.check_call(["guardrails", "configure", "--token", os.getenv('GUARDRAILS_CLI_TOKEN'), "--disable-remote-inferencing", "--disable-metrics"])
    except subprocess.CalledProcessError as e:
        print(f"Guardrails configure failed: {e}")
        sys.exit(1)


    guardrail_pkgs = ["hub://guardrails/ban_list", "hub://guardrails/bias_check", "hub://guardrails/nsfw_text",
                      "hub://guardrails/profanity_free", "hub://guardrails/logic_check", "hub://cartesia/mentions_drugs",
                      "hub://guardrails/politeness_check", "hub://guardrails/toxic_language"]

    for i in guardrail_pkgs:
        install_guardrail_pkgs(i)

    # Clear the terminal (if needed) - might not work in all environments
    # subprocess.call("clear")  # Removed to avoid potential issues in Cloud Build

    print("SERVER SETUP HAS FINISHED. SERVER IS NOW LOADING. PLEASE WAIT.")
    # THIS IS FOR GOOGLE CLOUD
    sys.exit()

if __name__ == "__main__":
    try:
        # Check if requirements are already installed
        subprocess.check_call([sys.executable, "-m", "pip", "check"])
    except subprocess.CalledProcessError:
        print("Requirements are missing or outdated. Installing...")
        install_packages()

    main()