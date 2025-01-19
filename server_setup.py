import subprocess
import sys, os,time 

def install_packages():
    """Install packages from requirements.txt if not already installed."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while installing packages: {e}")
        sys.exit(1)
    subprocess.check_call(["pip", "uninstall", "setuptools", "-y"])
    subprocess.check_call(["pip", "install", "setuptools"])

def install_guardrail_pkgs(package):
    try:
        subprocess.check_call(["guardrails", "hub", "install", package])
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while installing Guardrail packages: {e}")
        sys.exit(1)

def main():
    """Main server logic."""
    print("All requirements are satisfied. Proceeding with the main code...")
    subprocess.check_call(["guardrails", "configure"])
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
        install_packages()
    except subprocess.CalledProcessError:
        print("Requirements are missing or outdated. Installing...")
        install_packages()

    main()
