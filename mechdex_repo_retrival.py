import subprocess
from pathlib import Path

def get_latest_commit_hash(repo_path='./mechanics'):
    result = subprocess.run(['git', '-C', repo_path, 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
    return result.stdout.decode('utf-8').strip()

def is_new_commit(repo_path='./mechanics'):
    last_commit_file = Path('mechanics/.last_commit')
    latest_commit = get_latest_commit_hash(repo_path)
    
    if last_commit_file.exists():
        stored_commit = last_commit_file.read_text().strip()
        if stored_commit == latest_commit:
            return False
    
    last_commit_file.write_text(latest_commit)
    return True

if __name__ == "__main__":
	if is_new_commit():
		print("Nouveau commit détecté !")
	else:
		print("Pas de nouveau commit.")
