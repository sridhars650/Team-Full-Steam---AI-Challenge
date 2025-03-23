import subprocess
import sys, os,time 
import os

def install_packages():
    """Install packages from requirements.txt if not already installed."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while installing packages: {e}")
        sys.exit(1)
    

def install_guardrail_pkgs(package):
    try:
        subprocess.check_call(["guardrails", "hub", "install", package])
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while installing Guardrail packages: {e}")
        sys.exit(1)

def main():
    """Main server logic."""
    from dotenv import load_dotenv
    load_dotenv() 
    print("All requirements are satisfied. Proceeding with the main code...")
    subprocess.check_call(["guardrails", "configure", "--token", os.getenv('GUARDRAILS_CLI_TOKEN') , "--disable-remote-inferencing", "--disable-metrics"])
    guardrail_pkgs = ["hub://guardrails/ban_list", "hub://guardrails/bias_check", "hub://guardrails/nsfw_text", 
           "hub://guardrails/profanity_free", "hub://guardrails/logic_check", "hub://cartesia/mentions_drugs",
           "hub://guardrails/politeness_check", "hub://guardrails/toxic_language"]
    for i in guardrail_pkgs:
        install_guardrail_pkgs(i)
    subprocess.check_call("clear")
    print("SERVER SETUP HAS FINISHED. SERVER IS NOW LOADING. PLEASE WAIT.")
    subprocess.Popen([sys.executable, "server.py"]) 
    sys.exit()



if __name__ == "__main__":
    try:
        # Check if requirements are already installed
        subprocess.check_call([sys.executable, "-m", "pip", "check"])
        # install_packages()
    except subprocess.CalledProcessError:
        print("Requirements are missing or outdated. Installing...")
        install_packages()

    main()
